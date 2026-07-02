# utils/odk_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
from utils.odk_api import (
    AFAR_MENTORSHIP_PROJECT14_FORM_IDS,
    AFAR_MENTORSHIP_ODK_PROJECT_ID,
    AFAR_MENTORSHIP_SECTION_LABEL,
    AFAR_REGION_ID,
    list_forms,
)
from utils.data_service import fetch_odk_data_for_user
from utils.region_mapping import (
    get_odk_region_codes,
    get_region_name_from_database_id,
    get_odk_code_mapping_display,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MENTORSHIP_METRIC_COLORS = [
    "#2563eb",
    "#0ea5e9",
    "#0891b2",
    "#16a34a",
    "#ca8a04",
    "#14b8a6",
    "#7c3aed",
    "#9333ea",
    "#334155",
]

QI_COACHING_INDICATORS = [
    ("part8_project-q801", "qi_wit_functional_pct", "QI Team/WIT Conduct Clinical Audit Regularly"),
    ("part2_qi_structure-q203", "qi_meeting_pct", "QI Team/WIT Conduct Regular Meetings"),
    (
        "part5_project-q502",
        "qi_trained_coach_pct",
        "At Least 1 QIT Member Trained on QI Methods",
    ),
    ("part6_project-q604", "qi_received_coaching_pct", "QIT Receives Regular QI Coaching"),
    ("part3_project-q301", "qi_active_project_pct", "Active QI Projects Available in Unit"),
    ("part3_project-q309", "qi_improvement_signal_pct", "Evidence of Improvement from Tested Change Ideas"),
    ("part4_project-q408", "qi_storyboard_posted_pct", "QI Storyboard Posted on Ward Wall"),
    (
        "part5_project-q504",
        "qi_cop_session_pct",
        "QI Team Participates in Learning Session (Local Govt / SLL Supported)",
    ),
]


# Cache forms listing for 1 hour
@st.cache_data(ttl=3600, show_spinner=False)
def list_forms_cached(odk_project_id: str | int | None = None):
    """Cached version of forms listing"""
    if odk_project_id is None:
        return list_forms()
    return list_forms(odk_project_id=odk_project_id)


@st.cache_data(ttl=300, show_spinner=False)
def load_merged_bmet_data() -> pd.DataFrame:
    """Load merged mentorship dataset from local mentorship folder."""
    merged_path = os.path.join(
        os.path.dirname(__file__), "mentorship", "merged_bmet.csv"
    )
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged file not found: {merged_path}")
    return pd.read_csv(merged_path)


@st.cache_data(ttl=300, show_spinner=False)
def load_merged_skill_data() -> pd.DataFrame:
    """Load merged skill assessment dataset from local mentorship folder."""
    merged_path = os.path.join(
        os.path.dirname(__file__), "mentorship", "merged_skill.csv"
    )
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged file not found: {merged_path}")
    return pd.read_csv(merged_path)


@st.cache_data(ttl=300, show_spinner=False)
def load_merged_qoc_data() -> pd.DataFrame:
    """Load merged QoC mentorship dataset from local mentorship folder."""
    merged_path = os.path.join(
        os.path.dirname(__file__), "mentorship", "merged_qoc.csv"
    )
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged file not found: {merged_path}")
    return pd.read_csv(merged_path)


@st.cache_data(ttl=300, show_spinner=False)
def load_merged_qi_coaching_data() -> pd.DataFrame:
    """Load merged QI coaching dataset from local mentorship folder."""
    merged_path = os.path.join(
        os.path.dirname(__file__), "mentorship", "merged_qi_coaching.csv"
    )
    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f"Merged file not found: {merged_path}. "
            "Run utils/qi_coaching_merge.py to generate it."
        )
    return pd.read_csv(merged_path)


@st.cache_data(ttl=300, show_spinner=False)
def load_cis_bundle_data() -> pd.DataFrame:
    """Load CIS Bundle of Care dataset from local mentorship folder."""
    cis_path = os.path.join(
        os.path.dirname(__file__), "mentorship", "CIS_Bundle_of_Care.csv"
    )
    if not os.path.exists(cis_path):
        raise FileNotFoundError(
            f"CIS Bundle of Care file not found: {cis_path}. "
            "Ensure CIS_Bundle_of_Care.csv is available in utils/mentorship."
        )
    return pd.read_csv(cis_path)


@st.cache_data(ttl=300, show_spinner=False)
def load_data_mentorship_form_data() -> pd.DataFrame:
    """Load merged Data Mentorship dataset from local mentorship folder."""
    data_path = os.path.join(os.path.dirname(__file__), "mentorship", "merged_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Merged mentorship file not found: {data_path}. "
            "Run utils/data_mentorship_merge.py to generate it."
        )
    return pd.read_csv(data_path)


def _yes_no_to_binary(series: pd.Series) -> pd.Series:
    """
    Convert common yes/no and boolean-like values to 1/0.
    Missing/unrecognized values are treated as 0.
    """
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(
        {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
        }
    )
    return mapped.fillna(0).astype(float)


def _blank_like_mask(series: pd.Series) -> pd.Series:
    """Identify blank-like values that should be treated as missing."""
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.isin({"", "nan", "none", "null", "na", "n/a"})


def _to_numeric_indicator_value(series: pd.Series) -> pd.Series:
    """
    Convert responses to numeric values for indicator averaging.
    Yes/no values become 1/0, while numeric responses keep their original value.
    """
    cleaned = series.astype(str).str.strip()
    lowered = cleaned.str.lower()
    result = pd.Series(index=series.index, dtype="float64")

    blank_mask = _blank_like_mask(series)
    result.loc[lowered.isin({"yes", "true", "y"})] = 1.0
    result.loc[lowered.isin({"no", "false", "n"})] = 0.0

    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric_mask = numeric.notna() & ~blank_mask
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype(float)

    return result


def _to_positive_binary_indicator_value(series: pd.Series) -> pd.Series:
    """
    Convert responses to a presence/absence indicator for percentage metrics.
    Positive numeric values count as 1 and zero counts as 0.
    """
    numeric_values = _to_numeric_indicator_value(series)
    binary_values = pd.Series(index=series.index, dtype="float64")

    numeric_mask = numeric_values.notna()
    binary_values.loc[numeric_mask] = (numeric_values.loc[numeric_mask] > 0).astype(float)

    return binary_values


def _normalize_region_code(value) -> str:
    """Normalize region code values to canonical string form like '1'..'6'."""
    if pd.isna(value):
        return ""
    code = str(value).strip()
    if code.endswith(".0"):
        code = code[:-2]
    return code


def _normalize_entity_text(value) -> str:
    """Normalize entity labels/codes for display, removing trailing .0 from numeric ids."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        raw = text[:-2]
        if raw.isdigit():
            return raw
    return text


MENTORSHIP_FACILITY_CODE_TO_NAME = {
    "1": "Adama RH",
    "2": "Olenchiti PH",
    "3": "Batu GH",
    "4": "Meki PH",
    "6": "Felegehiwot CSH",
    "7": "Enjibara GH",
    "8": "Merawi PH",
    "9": "Dangla PH",
    "11": "Ayder CSH",
    "12": "Adigudom PH",
    "13": "Mekelle GH",
    "14": "Hagereselam PH",
    "15": "Adigrat general hospital",
    "16": "Adishu primary hospital",
    "17": "Lemlem Karl general hospital hospital",
    "18": "Mekoni primary hospital",
    "19": "Quiha general hospital",
    "20": "Wukro general hospital",
    "21": "Abyi-adi general hospital",
    "22": "Adwa general hospital",
    "23": "Axum referral hospital",
    "24": "Jimma CSH",
    "25": "Shenan Gibe GH",
    "26": "Limmu Genat GH",
    "27": "Agaro GH",
    "28": "Dr Tsegay G/her primary hospital",
    "29": "Edaga-arbi primary hospital",
    "30": "Freweyni primary hospital",
    "31": "Hawzen primary hospital",
    "32": "Meles primary hospital",
    "33": "Selekleka primary hospital",
    "34": "St. mary general hospital",
    "35": "Suhul general hospital",
    "36": "Wukromaray primary hospital",
    "37": "Yechila primary hospital",
    "38": "Endabaguna primary hospital",
    "40": "Seka Chekorsa Primary Hospital",
    "41": "Dedo Primary Hospital",
    "42": "Asela Teaching Hospital",
    "43": "Bokoji Primary Hospital",
    "44": "Kersa Primary Hospital",
    "45": "Huruta primary hospital",
    "46": "Robe Dida General Hospital",
    "47": "Shashamane Specialized",
    "48": "Melka Oda General Hospital",
    "49": "Dodola General Hospital",
    "50": "Loke Ada Primary Hospital",
    "51": "Arsi Negele Primary Hospital",
    "52": "Ambo university Hospital",
    "53": "Ambo General Hospital",
    "54": "Gedo General Hospital",
    "55": "Ginchi Primary Hospitals",
    "56": "Guder Primary Hospital",
    "58": "Hawassa University CSH",
    "59": "Adare GH",
    "60": "Tula PH",
    "61": "Dorebafano PH",
    "63": "Aleta Wondo PH",
    "64": "Bona GH",
    "65": "Daye GH",
    "66": "Leku GH",
    "67": "Yirgalem GH",
    "68": "Arba Minch General Hospital",
    "69": "Butajira GH",
    "70": "Dilla Referral Hospital",
    "71": "Dr.Bogalech Memorial GH",
    "72": "Gebre Tsadik shawo GH",
    "73": "Halaba GH",
    "74": "Jinka General Hospital",
    "75": "Mizan tepi teacing Hospital",
    "76": "Nigist Eleni M/M/referral Hospital",
    "77": "Sawla General General Hospital",
    "78": "Tepi General Hospital",
    "79": "Tercha General Hospital",
    "80": "Welkite University Specialized Teaching Hospital",
    "81": "Wolayta dsodo T and R Hospital",
    "82": "Worabe CSH",
    "84": "Dubti RH",
    "85": "Ayssaita GH",
    "88": "Tibebe Ghion teaching referral hospital",
    "89": "Addis Alem Primary Hospital",
    "90": "Durbetie Primary hospital",
    "91": "Burie Aserade primary hospital",
    "92": "Fenote selam General Hospital",
    "93": "Chagni Primary Hospital",
    "94": "Debere Markos comprehensive referral hospital",
    "95": "Lumamie Primary Hospital",
    "96": "Dejen Primary Hospital",
    "97": "Debere birhan comprehensive referral hospital",
    "98": "Deberesina Primary Hospital",
    "99": "Deneba Primary Hospital",
    "100": "Deberetabor comprehensive referral hospital",
    "101": "Addis Zemen Primary Hospital",
    "102": "Dessie comprehensive referral hospital",
    "103": "Haik Primary Hospital",
    "104": "Boru meda General Hospital",
    "105": "Woldia comprehensive referral hospital",
    "106": "Habiru Primary Hospital",
    "107": "Kobo Primary Hospital",
    "109": "Korem primary hospital",
    "110": "Alamata general hospital",
    "111": "Samre Primary hospital",
    "112": "Lekatit 11 primary hospital",
    "113": "Mulu Assefa primary hospital",
    "114": "Birshewa primay hospital",
    "115": "Semema primary hospital",
    "116": "Mayni general hospital",
    "117": "May tsebri primary hospital",
    "118": "Dewhan primary hospital",
    "119": "Adi daero primary hospital",
}


LEARNING_FACILITY_REGION_BY_NAME = {
    "Adama RH": "Oromia",
    "Olenchiti PH": "Oromia",
    "Meki PH": "Oromia",
    "Batu GH": "Oromia",
    "Hagereselam PH": "Tigray",
    "Adigudom PH": "Tigray",
    "Mekelle GH": "Tigray",
    "Ayder CSH": "Tigray",
    "Adare GH": "SNNP",
    "Tula PH": "SNNP",
    "Hawassa University CSH": "SNNP",
    "Dorebafano PH": "SNNP",
    "Enjibara GH": "Amhara",
    "Felegehiwot CSH": "Amhara",
    "Dangla PH": "Amhara",
    "Merawi PH": "Amhara",
}


LEARNING_FACILITY_NAMES = set(LEARNING_FACILITY_REGION_BY_NAME)


def _display_mentorship_facility(value) -> str:
    code_or_name = _normalize_entity_text(value)
    if not code_or_name:
        return "Other"
    if code_or_name in MENTORSHIP_FACILITY_CODE_TO_NAME:
        return MENTORSHIP_FACILITY_CODE_TO_NAME[code_or_name]
    if code_or_name.isdigit():
        return "Other"
    return code_or_name


def _filter_learning_facilities(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only configured learning facilities, matched by facility name and region."""
    if df.empty or "hospital" not in df.columns:
        return df.iloc[0:0].copy()
    filtered = df[df["hospital"].isin(LEARNING_FACILITY_NAMES)].copy()
    if "region_label" in filtered.columns:
        expected_region = filtered["hospital"].map(LEARNING_FACILITY_REGION_BY_NAME)
        filtered = filtered[
            filtered["region_label"].astype(str).str.casefold()
            == expected_region.astype(str).str.casefold()
        ]
    return filtered


def _get_region_code_to_name_mapping() -> dict[str, str]:
    """
    Build ODK region code to display name mapping using region_mapping.py.
    Required mapping reference:
    Tigray=3, Afar=5, Amhara=2, Oromia=1, SNNP=4 and 6.
    """
    mapping_from_module = get_odk_code_mapping_display()
    code_to_name = {
        "1": "Oromia",
        "2": "Amhara",
        "3": "Tigray",
        "4": "SNNP",
        "5": "Afar",
        "6": "SNNP",
    }
    for code, names in mapping_from_module.items():
        if code not in code_to_name and names:
            code_to_name[code] = " / ".join(names)
    return code_to_name


def compute_all_or_none(df, bundle_cols=None):
    if bundle_cols is None:
        bundle_cols = ["LD-ld1", "LD-ld2", "LD-ld3", "LD-ld4", "LD-ld5"]
    for col in bundle_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[bundle_cols] = df[bundle_cols].fillna(0)
    df["_yes_count"] = df[bundle_cols].sum(axis=1)
    df["row_all_or_none"] = df[bundle_cols].eq(1).all(axis=1).astype(int)
    entity_cols = [c for c in ["region_label", "hospital"] if c in df.columns]
    group_cols = ["week", "card"] + entity_cols
    idx = df.groupby(group_cols)["_yes_count"].idxmax()
    results = df.loc[idx, group_cols + ["row_all_or_none"]].reset_index(drop=True)
    results = results.rename(columns={"row_all_or_none": "all_or_none"})
    return results


def compute_weekly_percentages(df):
    stats = df.groupby("week")["all_or_none"].agg(total_cards="count", compliant="sum")
    stats["percentage"] = np.where(stats["total_cards"] > 0, (stats["compliant"] / stats["total_cards"]) * 100, 0.0)
    return stats.index.tolist(), stats["percentage"].tolist()


def plot_run_chart(weeks, values):
    if not weeks or not values:
        st.warning("No data to display in run chart.")
        return
    median_val = np.median(values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=values,
        mode="lines+markers",
        name="% Bundle Compliance",
        marker=dict(size=10, symbol="circle", color="#2563eb"),
        line=dict(color="#2563eb", width=2),
    ))
    fig.add_hline(
        y=median_val,
        line=dict(color="#e91e9e", width=2, dash="solid"),
        annotation_text=f"Median: {median_val:.1f}%",
    )
    fig.update_layout(
        title="Percentage of Hypothermia prevention<br>All or None Bundle of care provided",
        xaxis_title="Weeks",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_mentorship_analysis_dashboard():
    """Render mentorship analysis for mentorship datasets."""
    st.markdown(
        """
    <style>
    [class*="st-key-mentorship_filters_card"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 3px solid #1d4ed8;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.12);
        margin-bottom: 10px;
    }
    div[data-baseweb="tab-panel"] {
        padding-top: 0.2rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    def _run_mentorship_pipeline(
        work_df, score_cols, value_label, value_title,
        tab_key, left_col, right_col, is_qoc=False, is_bmet=False, filter_by_unit=False,
    ):
        region_code_map = _get_region_code_to_name_mapping()
        work_df["region_label"] = work_df["region_code"].map(region_code_map).fillna(
            work_df["region_code"].apply(lambda x: f"Unknown ({x})" if x else "Unknown")
        )

        current_user = st.session_state.get("user", {})
        is_regional_user = current_user.get("role") == "regional"
        regional_locked_label = None
        if is_regional_user:
            try:
                region_id = int(current_user.get("region_id"))
            except (TypeError, ValueError):
                region_id = None
            regional_codes = get_odk_region_codes(region_id) if region_id is not None else []
            if regional_codes:
                work_df = work_df[work_df["region_code"].isin([str(c) for c in regional_codes])]
                regional_locked_label = get_region_name_from_database_id(region_id)

        with right_col:
            with st.container(key=f"mentorship_filters_card_{tab_key}"):
                st.markdown('<div class="mentorship-filter-box">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="mentorship-filter-title">Filters</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="mentorship-filter-subtitle">Refine by group and round</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="mentorship-filter-divider"></div>', unsafe_allow_html=True)

                all_rounds_label = "All Rounds"
                round_values = sorted([r for r in work_df["round"].dropna().unique() if r != ""])
                if not round_values:
                    st.warning("No round values found in the data.")
                    return
                round_options = [all_rounds_label] + round_values

                group_mode_options = (
                    ["Regional", "Learning Facilities"]
                    if is_regional_user
                    else ["Multi Regional", "Regional", "Learning Facilities"]
                )
                group_mode = st.radio(
                    "Group By", options=group_mode_options,
                    key=f"mentorship_group_mode_{tab_key}",
                )

                scoped_work_df = (
                    _filter_learning_facilities(work_df)
                    if group_mode == "Learning Facilities"
                    else work_df.copy()
                )
                if scoped_work_df.empty:
                    st.warning("No data available for learning facilities.")
                    return

                selected_region_for_facility = None
                if group_mode == "Multi Regional":
                    entity_col = "region_label"
                    entity_options = sorted([
                        v for v in scoped_work_df[entity_col].dropna().unique().tolist() if str(v).strip()
                    ])
                    all_token = "All Regions"
                    selector_options = [all_token] + entity_options
                    entity_key = f"mentorship_entities_multi_{tab_key}"
                    if entity_key not in st.session_state or not st.session_state.get(entity_key):
                        st.session_state[entity_key] = [all_token]
                    st.multiselect("Select Regions", options=selector_options, key=entity_key)
                    current_selected = st.session_state.get(entity_key, [all_token])
                    effective_selected = (
                        [all_token] if (all_token in current_selected or len(current_selected) == 0)
                        else current_selected
                    )
                    selected_entities = entity_options if all_token in effective_selected else effective_selected
                elif group_mode == "Learning Facilities":
                    entity_col = "hospital"
                    entity_options = sorted([
                        v for v in scoped_work_df[entity_col].dropna().unique().tolist() if str(v).strip()
                    ])
                    all_token = "All Learning Facilities"
                    selector_options = [all_token] + entity_options
                    entity_key = f"mentorship_entities_learning_{tab_key}"
                    if entity_key not in st.session_state or not st.session_state.get(entity_key):
                        st.session_state[entity_key] = [all_token]
                    st.multiselect("Select Learning Facilities", options=selector_options, key=entity_key)
                    current_selected = st.session_state.get(entity_key, [all_token])
                    effective_selected = (
                        [all_token] if (all_token in current_selected or len(current_selected) == 0)
                        else current_selected
                    )
                    selected_entities = entity_options if all_token in effective_selected else effective_selected
                else:
                    if is_regional_user:
                        selected_region_for_facility = regional_locked_label
                        st.caption(f"Regional scope: {selected_region_for_facility}")
                    else:
                        region_options = sorted([
                            v for v in scoped_work_df["region_label"].dropna().unique().tolist() if str(v).strip()
                        ])
                        selected_region_for_facility = st.selectbox(
                            "Select Region", options=region_options,
                            key=f"mentorship_regional_region_{tab_key}",
                        )
                    facilities_in_region = sorted([
                        v for v in scoped_work_df.loc[
                            scoped_work_df["region_label"] == selected_region_for_facility, "hospital"
                        ].dropna().unique().tolist() if str(v).strip()
                    ])
                    entity_col = "hospital"
                    all_token = "All Facilities in Region"
                    selector_options = [all_token] + facilities_in_region
                    entity_key = f"mentorship_entities_facilities_{tab_key}"
                    if entity_key not in st.session_state or not st.session_state.get(entity_key):
                        st.session_state[entity_key] = [all_token]
                    st.multiselect("Select Facilities", options=selector_options, key=entity_key)
                    current_selected = st.session_state.get(entity_key, [all_token])
                    effective_selected = (
                        [all_token] if (all_token in current_selected or len(current_selected) == 0)
                        else current_selected
                    )
                    selected_entities = facilities_in_region if all_token in effective_selected else effective_selected

                selected_round = st.selectbox(
                    "Round", options=round_options, index=0,
                    key=f"mentorship_round_{tab_key}",
                )

                selected_unit = None
                if is_qoc or filter_by_unit:
                    if work_df.empty:
                        st.warning("No unit values found in data.")
                        return
                    unit_display_options = ["1 - Labor and Delivery", "2 - NICU"]
                    selected_unit_display = st.selectbox(
                        "Unit", options=unit_display_options,
                        key=f"mentorship_unit_{tab_key}",
                    )
                    reverse_map = {"1 - Labor and Delivery": "1", "2 - NICU": "2"}
                    selected_unit = reverse_map.get(selected_unit_display, selected_unit_display)

                st.markdown("</div>", unsafe_allow_html=True)

        with left_col:
            filtered_df = scoped_work_df.copy()
            if selected_round != "All Rounds":
                filtered_df = filtered_df[filtered_df["round"] == selected_round]
            if (is_qoc or filter_by_unit) and selected_unit is not None:
                if selected_unit == "1":
                    filtered_df = filtered_df[filtered_df["unit_has_1"]]
                elif selected_unit == "2":
                    filtered_df = filtered_df[filtered_df["unit_has_2"]]
            if group_mode == "Regional" and selected_region_for_facility:
                filtered_df = filtered_df[filtered_df["region_label"] == selected_region_for_facility]
            if selected_entities:
                filtered_df = filtered_df[filtered_df[entity_col].isin(selected_entities)]
            else:
                filtered_df = filtered_df.iloc[0:0]

            if filtered_df.empty:
                st.info("No data for the selected filter combination.")
                return

            hover_entity_label = "Region" if group_mode == "Multi Regional" else "Facility"

            if is_qoc:
                grouped = filtered_df.groupby(entity_col, as_index=False)
                counts_df = grouped.size().rename(columns={"size": "_record_count"})
                sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
                non_null_counts_df = grouped[score_cols].count().rename(
                    columns={col: f"{col}__non_null_count" for col in score_cols}
                )
                agg_df = counts_df.merge(sums_df, on=entity_col, how="left").merge(
                    non_null_counts_df, on=entity_col, how="left"
                )
                qoc_records_df = counts_df.copy()
                qoc_sums_df = sums_df.copy()
                qoc_non_null_counts_df = non_null_counts_df.copy()
                for col in score_cols:
                    denom_col = f"{col}__non_null_count"
                    agg_df[col] = (
                        pd.to_numeric(agg_df[col], errors="coerce").fillna(0)
                        / pd.to_numeric(agg_df[denom_col], errors="coerce").replace(0, pd.NA)
                    ).fillna(0)
                agg_df = agg_df.drop(
                    columns=["_record_count"] + [f"{col}__non_null_count" for col in score_cols]
                ).sort_values(entity_col)
            elif is_bmet:
                grouped = filtered_df.groupby(entity_col, as_index=False)
                sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
                non_null_counts_df = grouped[score_cols].count().rename(
                    columns={col: f"{col}__non_null_count" for col in score_cols}
                )
                agg_df = sums_df.merge(non_null_counts_df, on=entity_col, how="left")
                for col in score_cols:
                    denom_col = f"{col}__non_null_count"
                    agg_df[col] = (
                        pd.to_numeric(agg_df[col], errors="coerce").fillna(0)
                        / pd.to_numeric(agg_df[denom_col], errors="coerce").replace(0, pd.NA)
                    ).fillna(0)
                agg_df = agg_df.drop(
                    columns=[f"{col}__non_null_count" for col in score_cols]
                ).sort_values(entity_col)
            elif tab_key == "skill":
                agg_df = filtered_df.groupby(entity_col, as_index=False)[score_cols].mean().sort_values(entity_col)
            elif tab_key == "qi":
                grouped = filtered_df.groupby(entity_col, as_index=False)
                sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
                non_null_counts_df = grouped[score_cols].count().rename(
                    columns={col: f"{col}__non_null_count" for col in score_cols}
                )
                agg_df = sums_df.merge(non_null_counts_df, on=entity_col, how="left")
                for col in score_cols:
                    denom_col = f"{col}__non_null_count"
                    agg_df[col] = (
                        pd.to_numeric(agg_df[col], errors="coerce").fillna(0)
                        / pd.to_numeric(agg_df[denom_col], errors="coerce").replace(0, pd.NA)
                    ).fillna(0) * 100
                agg_df = agg_df.drop(
                    columns=[f"{col}__non_null_count" for col in score_cols]
                ).sort_values(entity_col)
            elif tab_key == "data":
                grouped = filtered_df.groupby(entity_col, as_index=False)
                counts_df = grouped.size().rename(columns={"size": "_record_count"})
                sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
                agg_df = counts_df.merge(sums_df, on=entity_col, how="left")
                for col in score_cols:
                    agg_df[col] = (
                        pd.to_numeric(agg_df[col], errors="coerce").fillna(0)
                        / pd.to_numeric(agg_df["_record_count"], errors="coerce").replace(0, pd.NA)
                    ).fillna(0)
                agg_df = agg_df.sort_values(entity_col)
            else:
                agg_df = filtered_df.groupby(entity_col, as_index=False)[score_cols].sum().sort_values(entity_col)

            metric_colors = MENTORSHIP_METRIC_COLORS
            entity_tick_values = agg_df[entity_col].astype(str).tolist()
            y_axis_tick_config = {
                "tickmode": "array",
                "tickvals": entity_tick_values,
                "ticktext": entity_tick_values,
            }

            def _update_subplot_yaxis(fig, row, col, font_size=8, title_size=9):
                fig.update_yaxes(
                    tickfont=dict(size=font_size),
                    title_font=dict(size=title_size),
                    automargin=True,
                    showticklabels=(col == 1),
                    row=row,
                    col=col,
                    **y_axis_tick_config,
                )

            def _show_computation_info(title, rows):
                with st.expander(title, expanded=False):
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if tab_key == "skill":
                subplot_titles = [
                    "Point of Care 1", "Point of Care 2", "Point of Care 3",
                    "Point of Care 4", "Point of Care 5", "Point of Care 6",
                    "Point of Care 7", "Point of Care 8", "Overall Point of Care",
                ]
                hover_metric_labels = subplot_titles
                fig = make_subplots(rows=3, cols=3, subplot_titles=subplot_titles,
                                    vertical_spacing=0.08, horizontal_spacing=0.18)
                for idx, metric in enumerate(score_cols):
                    row = idx // 3 + 1
                    col = idx % 3 + 1
                    fig.add_bar(
                        y=agg_df[entity_col], x=agg_df[metric].fillna(0), orientation="h",
                        marker_color=metric_colors[idx % len(metric_colors)],
                        showlegend=False,
                        customdata=agg_df[[entity_col]].values,
                        hovertemplate=(
                            f"{hover_entity_label}: %{{customdata[0]}}"
                            f"<br>{hover_metric_labels[idx]}: %{{x:.2f}}"
                            "<extra></extra>"
                        ),
                        row=row, col=col,
                    )
                    fig.update_xaxes(title_text=value_title, tickfont=dict(size=8), title_font=dict(size=9),
                                     automargin=True, row=row, col=col)
                    _update_subplot_yaxis(fig, row, col)
                skill_chart_height = max(760, min(1350, 620 + len(agg_df) * 14))
                fig.update_layout(template="plotly_white", height=skill_chart_height,
                                  margin=dict(l=150, r=24, t=36, b=8), font=dict(size=9),
                                  hoverlabel=dict(font_size=10), bargap=0.24, bargroupgap=0.08)
                st.plotly_chart(fig, use_container_width=True, key=f"mentorship_skill_bars_{tab_key}_{group_mode}_{selected_round}")
                skill_table_labels = {
                    score_cols[0]: "Point of Care 1 (%)",
                    score_cols[1]: "Point of Care 2 (%)",
                    score_cols[2]: "Point of Care 3 (%)",
                    score_cols[3]: "Point of Care 4 (%)",
                    score_cols[4]: "Point of Care 5 (%)",
                    score_cols[5]: "Point of Care 6 (%)",
                    score_cols[6]: "Point of Care 7 (%)",
                    score_cols[7]: "Point of Care 8 (%)",
                    score_cols[8]: "Overall Point of Care (%)",
                }
                table_df = agg_df.rename(columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **skill_table_labels,
                }).round(2)
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                _show_computation_info(
                    "Skill Assessment Computation Info",
                    [
                        {
                            "Indicator": label.replace(" (%)", ""),
                            "Variable": variable,
                            "Numerator": "Sum of score values",
                            "Denominator": "Number of non-empty submissions for this score",
                        }
                        for variable, label in skill_table_labels.items()
                    ],
                )

            elif tab_key == "qoc":
                subplot_titles = [
                    "Maintaining Warm", "KMC", "CPAP",
                    "Neonatal Jaundice", "IPC Measures", "Overall QoC",
                ]
                hover_metric_labels = subplot_titles
                fig = make_subplots(rows=2, cols=3, subplot_titles=subplot_titles,
                                    vertical_spacing=0.10, horizontal_spacing=0.18)
                for idx, metric in enumerate(score_cols):
                    row = idx // 3 + 1
                    col = idx % 3 + 1
                    fig.add_bar(
                        y=agg_df[entity_col], x=agg_df[metric].fillna(0), orientation="h",
                        marker_color=metric_colors[idx % len(metric_colors)],
                        showlegend=False,
                        customdata=agg_df[[entity_col]].values,
                        hovertemplate=(
                            f"{hover_entity_label}: %{{customdata[0]}}"
                            f"<br>{hover_metric_labels[idx]}: %{{x:.2f}}"
                            "<extra></extra>"
                        ),
                        row=row, col=col,
                    )
                    fig.update_xaxes(title_text=value_title, tickfont=dict(size=8), title_font=dict(size=9),
                                     automargin=True, row=row, col=col)
                    _update_subplot_yaxis(fig, row, col)
                qoc_chart_height = max(640, min(1200, 520 + len(agg_df) * 14))
                fig.update_layout(template="plotly_white", height=qoc_chart_height,
                                  margin=dict(l=150, r=24, t=36, b=8), font=dict(size=9),
                                  hoverlabel=dict(font_size=10), bargap=0.24, bargroupgap=0.08)
                st.plotly_chart(fig, use_container_width=True, key=f"mentorship_qoc_bars_{tab_key}_{group_mode}_{selected_round}_{selected_unit}")
                qoc_table_labels = {
                    score_cols[0]: "Maintaining Warm (%)",
                    score_cols[1]: "KMC (%)",
                    score_cols[2]: "CPAP (%)",
                    score_cols[3]: "Neonatal Jaundice (%)",
                    score_cols[4]: "IPC Measures (%)",
                    score_cols[5]: "Overall QoC (%)",
                }
                table_df = agg_df.rename(columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **qoc_table_labels,
                }).round(2)
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                _show_computation_info(
                    "QoC Computation Info",
                    [
                        {
                            "Indicator": label.replace(" (%)", ""),
                            "Variable": variable,
                            "Numerator": "Sum of score values",
                            "Denominator": "Number of non-empty submissions for this score",
                        }
                        for variable, label in qoc_table_labels.items()
                    ],
                )
                if (
                    qoc_records_df is not None
                    and qoc_sums_df is not None
                    and qoc_non_null_counts_df is not None
                ):
                    with st.expander("QoC Computation Audit (Sum and Record Count)", expanded=False):
                        audit_df = (
                            qoc_records_df.merge(qoc_sums_df, on=entity_col, how="left").merge(
                                qoc_non_null_counts_df, on=entity_col, how="left"
                            )
                        )
                        audit_df = audit_df.rename(columns={
                            entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                            "_record_count": "Filtered Records",
                            "QOC1_b-QOC1_perce": "Maintaining Warm Sum",
                            "QOC2_b-QOC2-QOC2_perce": "KMC Sum",
                            "QOC3_b-QOC3_perce": "CPAP Sum",
                            "QOC4_b-QOC4_perce": "Neonatal Jaundice Sum",
                            "QOC5_b-QOC5_gr5-QOC5_perce": "IPC Measures Sum",
                            "QOC_perce": "Overall QoC Sum",
                            "QOC1_b-QOC1_perce__non_null_count": "Maintaining Warm Non-Null Count",
                            "QOC2_b-QOC2-QOC2_perce__non_null_count": "KMC Non-Null Count",
                            "QOC3_b-QOC3_perce__non_null_count": "CPAP Non-Null Count",
                            "QOC4_b-QOC4_perce__non_null_count": "Neonatal Jaundice Non-Null Count",
                            "QOC5_b-QOC5_gr5-QOC5_perce__non_null_count": "IPC Measures Non-Null Count",
                            "QOC_perce__non_null_count": "Overall QoC Non-Null Count",
                        })
                        st.dataframe(audit_df.round(2), use_container_width=True, hide_index=True)

            elif tab_key == "qi":
                subplot_titles = [spec[2] for spec in QI_COACHING_INDICATORS]
                fig = make_subplots(rows=4, cols=2, subplot_titles=subplot_titles,
                                    vertical_spacing=0.08, horizontal_spacing=0.22)
                for idx, metric in enumerate(score_cols):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    fig.add_bar(
                        y=agg_df[entity_col], x=agg_df[metric].fillna(0), orientation="h",
                        marker_color=metric_colors[idx % len(metric_colors)],
                        showlegend=False,
                        customdata=agg_df[[entity_col]].values,
                        hovertemplate=(
                            f"{hover_entity_label}: %{{customdata[0]}}"
                            f"<br>{subplot_titles[idx]}: %{{x:.2f}}%"
                            "<extra></extra>"
                        ),
                        row=row, col=col,
                    )
                    fig.update_xaxes(title_text=value_title, tickfont=dict(size=8), title_font=dict(size=9),
                                     automargin=True, row=row, col=col)
                    _update_subplot_yaxis(fig, row, col)
                qi_chart_height = max(860, min(1500, 720 + len(agg_df) * 16))
                fig.update_layout(template="plotly_white", height=qi_chart_height,
                                  margin=dict(l=150, r=24, t=36, b=8), font=dict(size=9),
                                  hoverlabel=dict(font_size=10), bargap=0.24, bargroupgap=0.08)
                st.plotly_chart(fig, use_container_width=True, key=f"mentorship_qi_bars_{tab_key}_{group_mode}_{selected_round}")
                qi_table_labels = {
                    "qi_wit_functional_pct": "QI Team/WIT Conduct Clinical Audit Regularly (%)",
                    "qi_meeting_pct": "QI Team/WIT Conduct Regular Meetings (%)",
                    "qi_trained_coach_pct": "At Least 1 QIT Member Trained on QI Methods (%)",
                    "qi_received_coaching_pct": "QIT Receives Regular QI Coaching (%)",
                    "qi_active_project_pct": "Active QI Projects Available in Unit (%)",
                    "qi_improvement_signal_pct": "Evidence of Improvement from Tested Change Ideas (%)",
                    "qi_storyboard_posted_pct": "QI Storyboard Posted on Ward Wall (%)",
                    "qi_cop_session_pct": "QI Team Participates in Learning Session (Local Govt / SLL Supported) (%)",
                }
                table_df = agg_df.rename(columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **qi_table_labels,
                }).round(2)
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                qi_audit_rows = [
                    {
                        "Indicator": label,
                        "Variable": source_col,
                        "Numerator": "Number of yes responses",
                        "Denominator": "Number of non-empty submissions for this question",
                    }
                    for source_col, derived_col, label in QI_COACHING_INDICATORS
                ]
                _show_computation_info("QI Computation Info", qi_audit_rows)

            elif tab_key == "data":
                subplot_titles = [
                    "System Access & User Management",
                    "Tracker Data Entry (IMNID Program)",
                    "Tracker Features & Program Functionality",
                    "Analysis & Visualization (Dashboard Use)",
                ]
                fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles,
                                    vertical_spacing=0.11, horizontal_spacing=0.22)
                for idx, metric in enumerate(score_cols):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    fig.add_bar(
                        y=agg_df[entity_col], x=agg_df[metric].fillna(0), orientation="h",
                        marker_color=metric_colors[idx % len(metric_colors)],
                        showlegend=False,
                        customdata=agg_df[[entity_col]].values,
                        hovertemplate=(
                            f"{hover_entity_label}: %{{customdata[0]}}"
                            f"<br>{subplot_titles[idx]}: %{{x:.2f}}"
                            "<extra></extra>"
                        ),
                        row=row, col=col,
                    )
                    fig.update_xaxes(title_text=value_title, tickfont=dict(size=8), title_font=dict(size=9),
                                     automargin=True, row=row, col=col)
                    _update_subplot_yaxis(fig, row, col)
                mentorship_chart_height = max(620, min(1200, 520 + len(agg_df) * 14))
                fig.update_layout(template="plotly_white", height=mentorship_chart_height,
                                  margin=dict(l=150, r=24, t=36, b=8), font=dict(size=9),
                                  hoverlabel=dict(font_size=10), bargap=0.24, bargroupgap=0.08)
                st.plotly_chart(fig, use_container_width=True, key=f"mentorship_data_form_bars_{tab_key}_{group_mode}_{selected_round}")
                mentorship_table_labels = {
                    score_cols[0]: "System Access & User Management (Avg)",
                    score_cols[1]: "Tracker Data Entry (Avg)",
                    score_cols[2]: "Tracker Features & Program Functionality (Avg)",
                    score_cols[3]: "Analysis & Visualization (Avg)",
                }
                table_df = agg_df.rename(columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **mentorship_table_labels,
                }).round(2)
                table_columns = [
                    "Region" if group_mode == "Multi Regional" else "Facility",
                    "System Access & User Management (Avg)",
                    "Tracker Data Entry (Avg)",
                    "Tracker Features & Program Functionality (Avg)",
                    "Analysis & Visualization (Avg)",
                ]
                table_df = table_df[[c for c in table_columns if c in table_df.columns]]
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                _show_computation_info(
                    "Data Mentorship Computation Info",
                    [
                        {
                            "Indicator": "System Access & User Management",
                            "Variable": "q21-q24",
                            "Numerator": "Sum of yes responses in this question group",
                            "Denominator": "Number of non-empty submissions",
                        },
                        {
                            "Indicator": "Tracker Data Entry",
                            "Variable": "q31-q39 + q391",
                            "Numerator": "Sum of yes responses in this question group",
                            "Denominator": "Number of non-empty submissions",
                        },
                        {
                            "Indicator": "Tracker Features & Program Functionality",
                            "Variable": "q41-q44",
                            "Numerator": "Sum of yes responses in this question group",
                            "Denominator": "Number of non-empty submissions",
                        },
                        {
                            "Indicator": "Analysis & Visualization",
                            "Variable": "q51-q57",
                            "Numerator": "Sum of yes responses in this question group",
                            "Denominator": "Number of non-empty submissions",
                        },
                    ],
                )

            elif tab_key == "bmet":
                fig = go.Figure()
                fig.add_bar(
                    y=agg_df[entity_col], x=agg_df["competency_assessment-Score"],
                    name="Competency Score", marker_color=metric_colors[0], orientation="h",
                    customdata=agg_df[[entity_col]].values,
                    hovertemplate=f"{hover_entity_label}: %{{customdata[0]}}<br>Competency Score: %{{x:.2f}}<extra></extra>",
                )
                fig.add_bar(
                    y=agg_df[entity_col], x=agg_df["facility_assessment-score_fac"],
                    name="Facility Score", marker_color=metric_colors[1], orientation="h",
                    customdata=agg_df[[entity_col]].values,
                    hovertemplate=f"{hover_entity_label}: %{{customdata[0]}}<br>Facility Score: %{{x:.2f}}<extra></extra>",
                )
                is_regional_mode = group_mode == "Regional"
                chart_height = (
                    max(280, min(560, 56 + len(agg_df) * 16))
                    if is_regional_mode
                    else max(260, min(500, 52 + len(agg_df) * 14))
                )
                bar_gap = 0.26 if is_regional_mode else 0.20
                bar_group_gap = 0.08 if is_regional_mode else 0.06
                fig.update_layout(
                    barmode="group", template="plotly_white", height=chart_height,
                    margin=dict(l=120, r=24, t=14, b=6), xaxis_title=value_title,
                    yaxis_title="Region" if group_mode == "Multi Regional" else "Facility",
                    legend_title_text="Indicators", bargap=bar_gap, bargroupgap=bar_group_gap,
                    font=dict(size=10), hoverlabel=dict(font_size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=9)),
                )
                fig.update_xaxes(tickfont=dict(size=9), title_font=dict(size=10), automargin=True, tickformat=".2f")
                fig.update_yaxes(tickfont=dict(size=9), title_font=dict(size=10), automargin=True, **y_axis_tick_config)
                st.plotly_chart(fig, use_container_width=True, key=f"mentorship_bmet_bars_{tab_key}_{group_mode}_{selected_round}")
                table_df = agg_df.rename(columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    "competency_assessment-Score": "Competency Score",
                    "facility_assessment-score_fac": "Facility Score",
                }).round(2)
                st.dataframe(table_df, use_container_width=True, hide_index=True)
                _show_computation_info(
                    "BMET Computation Info",
                    [
                        {
                            "Indicator": "Competency Score",
                            "Variable": "competency_assessment-Score",
                            "Numerator": "Sum of competency score values",
                            "Denominator": "Number of non-empty competency score submissions",
                        },
                        {
                            "Indicator": "Facility Score",
                            "Variable": "facility_assessment-score_fac",
                            "Numerator": "Sum of facility score values",
                            "Denominator": "Number of non-empty facility score submissions",
                        },
                    ],
                )

    tab_bmet, tab_skill, tab_qoc, tab_qi, tab_data, tab_cis = st.tabs([
        "BMET", "Skill Assessment", "Quality of Care", "QI Coaching",
        "Data Mentorship", "CIS Bundle of Care",
    ])

    with tab_bmet:
        try:
            df = load_merged_bmet_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to load merged mentorship data: {exc}")
        else:
            score_cols = ["competency_assessment-Score", "facility_assessment-score_fac"]
            required_columns = ["region", "hospital", "round"] + score_cols
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in merged_bmet.csv: {missing_cols}")
            else:
                work_df = df.copy()
                work_df["region_code"] = work_df["region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["hospital"].apply(_display_mentorship_facility)
                work_df["round"] = work_df["round"].astype(str).str.strip()
                for score_col in score_cols:
                    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")
                left_col, right_col = st.columns([3, 1])
                _run_mentorship_pipeline(
                    work_df, score_cols, "Average Score", "Average Score",
                    tab_key="bmet", left_col=left_col, right_col=right_col, is_bmet=True,
                )

    with tab_skill:
        try:
            df = load_merged_skill_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to load merged skill data: {exc}")
        else:
            score_cols = [
                "POC1-POC1_perc", "POC2-POC2_perc", "POC3-POC3_perc",
                "POC4-POC4_perc", "POC5-POC5_perc", "POC6-POC6_perc",
                "POC7-POC7_perc", "POC8-POC8_perc", "POC_perc",
            ]
            required_columns = ["reg-region", "reg-hospital", "reg-round", "unit"] + score_cols
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in merged_skill.csv: {missing_cols}")
            else:
                work_df = df.copy()
                work_df["region_code"] = work_df["reg-region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["reg-hospital"].apply(_display_mentorship_facility)
                work_df["round"] = work_df["reg-round"].astype(str).str.strip()
                work_df["unit"] = (
                    work_df["unit"].astype(str).str.strip().replace({"1.0": "1", "2.0": "2"})
                )
                work_df["unit_has_1"] = work_df["unit"].astype(str).str.contains("1", regex=False)
                work_df["unit_has_2"] = work_df["unit"].astype(str).str.contains("2", regex=False)
                for score_col in score_cols:
                    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")
                left_col, right_col = st.columns([3, 1])
                _run_mentorship_pipeline(
                    work_df, score_cols, "Average %", "Average Score (%)",
                    tab_key="skill", left_col=left_col, right_col=right_col,
                    filter_by_unit=True,
                )

    with tab_qoc:
        try:
            df = load_merged_qoc_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to load merged QoC data: {exc}")
        else:
            score_cols = [
                "QOC1_b-QOC1_perce", "QOC2_b-QOC2-QOC2_perce",
                "QOC3_b-QOC3_perce", "QOC4_b-QOC4_perce",
                "QOC5_b-QOC5_gr5-QOC5_perce", "QOC_perce",
            ]
            region_col = "region" if "region" in df.columns else "reg-region"
            hospital_col = "hospital" if "hospital" in df.columns else "reg-hospital"
            round_col = "round" if "round" in df.columns else "reg-round"
            unit_col = "unit"
            required_columns = [region_col, hospital_col, round_col, unit_col] + score_cols
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in merged_qoc.csv: {missing_cols}")
            else:
                work_df = df.copy()
                work_df["region_code"] = work_df[region_col].apply(_normalize_region_code)
                work_df["hospital"] = work_df[hospital_col].apply(_display_mentorship_facility)
                work_df["round"] = work_df[round_col].astype(str).str.strip()
                work_df["unit"] = (
                    work_df[unit_col].astype(str).str.strip().replace({"1.0": "1", "2.0": "2"})
                )
                work_df["unit_has_1"] = work_df["unit"].astype(str).str.contains("1", regex=False)
                work_df["unit_has_2"] = work_df["unit"].astype(str).str.contains("2", regex=False)
                for score_col in score_cols:
                    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")
                left_col, right_col = st.columns([3, 1])
                _run_mentorship_pipeline(
                    work_df, score_cols, "Average %", "Average Score (%)",
                    tab_key="qoc", left_col=left_col, right_col=right_col, is_qoc=True,
                )

    with tab_qi:
        try:
            df = load_merged_qi_coaching_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to load merged QI coaching data: {exc}")
        else:
            indicator_source_cols = [spec[0] for spec in QI_COACHING_INDICATORS]
            score_cols = [spec[1] for spec in QI_COACHING_INDICATORS]
            required_columns = ["reg-region", "reg-hospital", "reg-round"] + indicator_source_cols
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in merged_qi_coaching.csv: {missing_cols}")
            else:
                work_df = df.copy()
                work_df["region_code"] = work_df["reg-region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["reg-hospital"].apply(_display_mentorship_facility)
                work_df["round"] = work_df["reg-round"].astype(str).str.strip()
                has_any_indicator_answer = ~work_df[indicator_source_cols].apply(_blank_like_mask)
                work_df = work_df[has_any_indicator_answer.any(axis=1)].copy()
                if work_df.empty:
                    st.warning("No QI Coaching submissions contain answered indicator questions.")
                else:
                    for source_col, metric_col, _ in QI_COACHING_INDICATORS:
                        if source_col == "part5_project-q502":
                            work_df[metric_col] = _to_positive_binary_indicator_value(work_df[source_col])
                        else:
                            work_df[metric_col] = _to_numeric_indicator_value(work_df[source_col])
                    left_col, right_col = st.columns([3, 1])
                    _run_mentorship_pipeline(
                        work_df, score_cols, "Percentage", "Percentage (%)",
                        tab_key="qi", left_col=left_col, right_col=right_col,
                    )

    with tab_data:
        try:
            df = load_data_mentorship_form_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to load merged Data Mentorship form data: {exc}")
        else:
            # DataMentorship uses column names with -summary, -qXX suffixes
            system_access_cols = ["system_access-q21", "system_access-q22", "system_access-q23", "system_access-q24"]
            tracker_data_cols = ["tracker_data-q31", "tracker_data-q32", "tracker_data-q33", "tracker_data-q34", "tracker_data-q35", "tracker_data-q36", "tracker_data-q37", "tracker_data-q38", "tracker_data-q39", "tracker_data-q391"]
            tracker_feature_cols = ["tracker_features-q41", "tracker_features-q42", "tracker_features-q43", "tracker_features-q44"]
            analysis_cols = ["analysis_vis-q51", "analysis_vis-q52", "analysis_vis-q53", "analysis_vis-q54", "analysis_vis-q55", "analysis_vis-q56", "analysis_vis-q57"]
            all_indicator_cols = system_access_cols + tracker_data_cols + tracker_feature_cols + analysis_cols
            required_columns = ["region", "hospital", "round"] + all_indicator_cols
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns in merged_data.csv: {missing_cols}")
            else:
                work_df = df.copy()
                work_df["region_code"] = work_df["region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["hospital"].apply(_display_mentorship_facility)
                work_df["round"] = work_df["round"].astype(str).str.strip()
                cleaned_answers = work_df[all_indicator_cols].apply(
                    lambda col: col.astype(str).str.strip().str.lower()
                )
                has_any_indicator_answer = ~cleaned_answers.isin({"", "nan", "none", "null", "na", "n/a"})
                work_df = work_df[has_any_indicator_answer.any(axis=1)].copy()
                if work_df.empty:
                    st.warning("No Data Mentorship submissions contain answered indicator questions.")
                else:
                    # Convert all indicator values to binary (1/0)
                    for col in all_indicator_cols:
                        work_df[col] = _yes_no_to_binary(work_df[col])
                    # Calculate average scores for each section
                    work_df["system_access_avg_score"] = work_df[system_access_cols].sum(axis=1)
                    work_df["tracker_data_entry_avg_score"] = work_df[tracker_data_cols].sum(axis=1)
                    work_df["tracker_features_avg_score"] = work_df[tracker_feature_cols].sum(axis=1)
                    work_df["analysis_visualization_avg_score"] = work_df[analysis_cols].sum(axis=1)
                    score_cols = [
                        "system_access_avg_score", "tracker_data_entry_avg_score",
                        "tracker_features_avg_score", "analysis_visualization_avg_score",
                    ]
                    left_col, right_col = st.columns([3, 1])
                    _run_mentorship_pipeline(
                        work_df, score_cols, "Average Score", "Average Score per Submission",
                        tab_key="data", left_col=left_col, right_col=right_col,
                    )

    with tab_cis:
        try:
            df = load_cis_bundle_data()
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unable to load CIS Bundle of Care data: {exc}")
        else:
            region_code_map = _get_region_code_to_name_mapping()
            cis_raw = df.copy()
            work_df = cis_raw.copy()
            work_df["region_code"] = work_df["reg-region"].apply(_normalize_region_code)
            work_df["hospital"] = work_df["reg-hospital"].apply(_display_mentorship_facility)
            work_df["region_label"] = work_df["region_code"].map(region_code_map).fillna(
                work_df["region_code"].apply(lambda x: f"Unknown ({x})" if x else "Unknown")
            )
            work_df["round"] = work_df["reg-round"].astype(str).str.strip()
            work_df["unit"] = work_df["unit"].astype(str).str.strip().replace({"1.0": "1", "2.0": "2"})

            cis_current_user = st.session_state.get("user", {})
            if cis_current_user.get("role") == "regional":
                try:
                    cis_region_id = int(cis_current_user.get("region_id"))
                except (TypeError, ValueError):
                    cis_region_id = None
                cis_regional_codes = get_odk_region_codes(cis_region_id) if cis_region_id is not None else []
                if cis_regional_codes:
                    work_df = work_df[work_df["region_code"].isin([str(c) for c in cis_regional_codes])]

            work_df = work_df[work_df["unit"] == "1"].copy()

            required_cols = ["week", "card", "LD-ld1", "LD-ld2", "LD-ld3", "LD-ld4", "LD-ld5"]
            missing = [c for c in required_cols if c not in work_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                work_df["week"] = pd.to_numeric(work_df["week"], errors="coerce")
                work_df = work_df.dropna(subset=["week", "card"])

                bundle_cols = ["LD-ld1", "LD-ld2", "LD-ld3", "LD-ld4", "LD-ld5"]
                for col in bundle_cols:
                    work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
                work_df[bundle_cols] = work_df[bundle_cols].fillna(0)

                cpap_ld_cols = ["cpap_ld-cpap1", "cpap_ld-cpap2", "cpap_ld-cpap3", "cpap_ld-cpap4"]
                cpap_ld_cols_present = [c for c in cpap_ld_cols if c in work_df.columns]
                if cpap_ld_cols_present:
                    for col in cpap_ld_cols_present:
                        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
                    work_df[cpap_ld_cols_present] = work_df[cpap_ld_cols_present].fillna(0)

                all_or_none_df = compute_all_or_none(work_df)
                if all_or_none_df.empty:
                    st.warning("No data available for L&D.")
                else:
                    round_info = work_df[["week", "card", "region_label", "hospital", "round"]].drop_duplicates(subset=["week", "card", "region_label", "hospital"])
                    all_or_none_df = all_or_none_df.merge(round_info, on=["week", "card", "region_label", "hospital"], how="left")

                    ld_tab, nicu_tab = st.tabs(["L&D", "NICU"])

                    with ld_tab:
                        left_col, right_col = st.columns([3, 1])

                        with right_col:
                            with st.container(key="mentorship_filters_card_cis"):
                                st.markdown('<div class="mentorship-filter-box">', unsafe_allow_html=True)
                                st.markdown(
                                    '<div class="mentorship-filter-title">Filters</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    '<div class="mentorship-filter-subtitle">Refine by group and round</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown('<div class="mentorship-filter-divider"></div>', unsafe_allow_html=True)

                                round_values = sorted([r for r in work_df["round"].dropna().unique() if r != ""])
                                all_rounds_label = "All Rounds"
                                round_options = [all_rounds_label] + round_values if round_values else [all_rounds_label]

                                selected_round = st.selectbox(
                                    "Round", options=round_options, index=0,
                                    key="mentorship_cis_round",
                                )

                                current_user = st.session_state.get("user", {})
                                is_regional_user = current_user.get("role") == "regional"
                                group_mode_options = (
                                    ["Regional", "Learning Facilities"]
                                    if is_regional_user
                                    else ["Multi Regional", "Regional", "Learning Facilities"]
                                )
                                group_mode = st.radio(
                                    "Group By", options=group_mode_options,
                                    key="mentorship_cis_group_mode",
                                )
                                cis_all_or_none_df = (
                                    _filter_learning_facilities(all_or_none_df)
                                    if group_mode == "Learning Facilities"
                                    else all_or_none_df.copy()
                                )
                                cis_work_df = (
                                    _filter_learning_facilities(work_df)
                                    if group_mode == "Learning Facilities"
                                    else work_df.copy()
                                )
                                if cis_all_or_none_df.empty or cis_work_df.empty:
                                    st.warning("No data available for learning facilities.")
                                    return

                                if group_mode == "Multi Regional":
                                    entity_col = "region_label"
                                    entity_options = sorted([
                                        v for v in cis_all_or_none_df[entity_col].dropna().unique().tolist()
                                        if str(v).strip()
                                    ])
                                    all_token = "All Regions"
                                    selector_options = [all_token] + entity_options
                                    entity_key = "mentorship_cis_entities_multi"
                                    if entity_key not in st.session_state or not st.session_state.get(entity_key):
                                        st.session_state[entity_key] = [all_token]
                                    st.multiselect("Select Regions", options=selector_options, key=entity_key)
                                    current_selected = st.session_state.get(entity_key, [all_token])
                                    effective_selected = (
                                        [all_token] if (all_token in current_selected or len(current_selected) == 0)
                                        else current_selected
                                    )
                                    selected_entities = entity_options if all_token in effective_selected else effective_selected
                                elif group_mode == "Learning Facilities":
                                    entity_col = "hospital"
                                    entity_options = sorted([
                                        v for v in cis_all_or_none_df[entity_col].dropna().unique().tolist()
                                        if str(v).strip()
                                    ])
                                    all_token = "All Learning Facilities"
                                    selector_options = [all_token] + entity_options
                                    entity_key = "mentorship_cis_entities_learning"
                                    if entity_key not in st.session_state or not st.session_state.get(entity_key):
                                        st.session_state[entity_key] = [all_token]
                                    st.multiselect("Select Learning Facilities", options=selector_options, key=entity_key)
                                    current_selected = st.session_state.get(entity_key, [all_token])
                                    effective_selected = (
                                        [all_token] if (all_token in current_selected or len(current_selected) == 0)
                                        else current_selected
                                    )
                                    selected_entities = entity_options if all_token in effective_selected else effective_selected
                                else:
                                    region_options = sorted([
                                        v for v in cis_all_or_none_df["region_label"].dropna().unique().tolist()
                                        if str(v).strip()
                                    ])
                                    selected_region = st.selectbox(
                                        "Select Region", options=region_options,
                                        key="mentorship_cis_region",
                                    )
                                    facilities_in_region = sorted([
                                        v for v in cis_all_or_none_df.loc[
                                            cis_all_or_none_df["region_label"] == selected_region, "hospital"
                                        ].dropna().unique().tolist() if str(v).strip()
                                    ])
                                    entity_col = "hospital"
                                    all_token = "All Facilities in Region"
                                    selector_options = [all_token] + facilities_in_region
                                    entity_key = "mentorship_cis_entities_facilities"
                                    if entity_key not in st.session_state or not st.session_state.get(entity_key):
                                        st.session_state[entity_key] = [all_token]
                                    st.multiselect("Select Facilities", options=selector_options, key=entity_key)
                                    current_selected = st.session_state.get(entity_key, [all_token])
                                    effective_selected = (
                                        [all_token] if (all_token in current_selected or len(current_selected) == 0)
                                        else current_selected
                                    )
                                    selected_entities = facilities_in_region if all_token in effective_selected else effective_selected

                                st.markdown("</div>", unsafe_allow_html=True)

                        with left_col:
                            hypo_tab, cpap_tab = st.tabs(["Hypothermia Prevention Bundle of Care", "Early CPAP Bundle of Care"])

                            with hypo_tab:
                                all_or_none_tab, care_delivered_tab = st.tabs([
                                    "Percentage of Hypothermia prevention All or None",
                                    "Percentage of care delivered",
                                ])

                                with all_or_none_tab:
                                    filtered = cis_all_or_none_df.copy()
                                    if selected_round != "All Rounds":
                                        filtered = filtered[filtered["round"] == selected_round]
                                    if selected_entities:
                                        filtered = filtered[filtered[entity_col].isin(selected_entities)]
                                    else:
                                        filtered = filtered.iloc[0:0]

                                    if filtered.empty:
                                        st.info("No data for the selected filter combination.")
                                    else:
                                        show_card_details = (group_mode == "Regional" and len(selected_entities) == 1 and entity_col == "hospital")

                                        if show_card_details:
                                            card_hover = filtered.groupby("week").apply(
                                                lambda g: "<br>".join(
                                                    f"Card {int(row['card'])}: {int(row['all_or_none'])}"
                                                    for _, row in g.iterrows()
                                                )
                                            ).reset_index(name="card_details")

                                        weekly_stats = filtered.groupby("week").agg(
                                            total_cards=("all_or_none", "count"),
                                            compliant=("all_or_none", "sum"),
                                        ).reset_index()
                                        weekly_stats["percentage"] = np.where(
                                            weekly_stats["total_cards"] > 0,
                                            (weekly_stats["compliant"] / weekly_stats["total_cards"]) * 100,
                                            0.0,
                                        )
                                        if show_card_details:
                                            weekly_stats = weekly_stats.merge(card_hover, on="week", how="left")
                                        weekly_stats = weekly_stats.sort_values("week")

                                        week_display_map = {w: f"Week {int(w)}" for w in sorted(weekly_stats["week"].unique())}
                                        weekly_stats["week_label"] = weekly_stats["week"].map(week_display_map)

                                        median_val = np.median(weekly_stats["percentage"]) if len(weekly_stats) > 0 else 0

                                        ci_total = 0
                                        ci_compliant = 1
                                        customdata_cols = ["total_cards", "compliant"]
                                        if show_card_details:
                                            customdata_cols = ["card_details", "total_cards", "compliant"]
                                            ci_total = 1
                                            ci_compliant = 2

                                        hover_parts = [
                                            "<b>%{x}</b><br>",
                                            f"Patients with full bundle: %{{customdata[{ci_compliant}]:.0f}}<br>",
                                            f"Total patients observed: %{{customdata[{ci_total}]:.0f}}<br>",
                                            "% of Hypothermia prevention: %{y:.1f}%<br>",
                                            f"Median: {median_val:.1f}%",
                                        ]

                                        if show_card_details:
                                            hover_parts.append("<br><br><b>All-or-None</b><br>%{customdata[0]}")

                                        hover_parts.append("<extra></extra>")

                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=weekly_stats["week_label"],
                                            y=weekly_stats["percentage"],
                                            mode="lines+markers",
                                            name="Hypothermia Prevention All-or-None Bundle",
                                            marker=dict(size=8, symbol="circle", color="#2563eb", line=dict(color="#2563eb", width=1)),
                                            line=dict(color="#2563eb", width=2.5),
                                            customdata=weekly_stats[customdata_cols].fillna("").values,
                                            hovertemplate="".join(hover_parts),
                                        ))
                                        fig.add_hline(
                                            y=median_val,
                                            line=dict(color="#e91e9e", width=2.5, dash="solid"),
                                            annotation_text=f"Median: {median_val:.1f}%",
                                            annotation_font_size=10,
                                            annotation_position="right",
                                        )
                                        fig.add_trace(go.Scatter(
                                            x=[None], y=[None],
                                            mode="lines",
                                            name=f"Median: {median_val:.1f}%",
                                            line=dict(color="#e91e9e", width=2.5),
                                            showlegend=True,
                                        ))

                                        fig.update_layout(
                                            template="plotly_white",
                                            height=290,
                                            margin=dict(l=8, r=8, t=20, b=8),
                                            xaxis_title=dict(text="Week", font=dict(size=11)),
                                            yaxis_title=dict(text="Percentage (%)", font=dict(size=11)),
                                            font=dict(size=10),
                                            hoverlabel=dict(font_size=10, namelength=-1),
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                            yaxis=dict(range=[0, 110], tickformat=".0f", dtick=20),
                                        )
                                        sorted_week_labels = [f"Week {int(w)}" for w in sorted(weekly_stats["week"].unique())]
                                        fig.update_xaxes(tickfont=dict(size=9), automargin=True, categoryorder="array", categoryarray=sorted_week_labels)
                                        fig.update_yaxes(tickfont=dict(size=9), automargin=True, gridcolor="#f0f0f0")

                                        st.plotly_chart(fig, use_container_width=True, key="mentorship_cis_run_chart")

                                        with st.expander("How this is computed", expanded=False):
                                            st.markdown(
                                                """
**Percentage** = (Patients with full bundle ÷ Total patients observed) × 100

**Hypothermia Prevention Bundle of Care – L&D**

- The current labour and delivery ward room temperature is between 25°C – 28°C
- The radiant warmer is on pre-warm mode 5–10 minutes before the delivery
- The baby is dried thoroughly, wet towel discarded, and baby covered by a pre-warmed towel
- Skin-to-skin contact is maintained
- The baby's body temperature is between 36.5 and 37.5°C when leaving the labour ward
                                                """
                                            )

                                with care_delivered_tab:
                                    selected_weeks = st.multiselect(
                                        "Select Weeks",
                                        options=sorted([int(w) for w in cis_work_df["week"].dropna().unique() if str(w).strip()]),
                                        default=sorted([int(w) for w in cis_work_df["week"].dropna().unique() if str(w).strip()]),
                                        key="mentorship_cis_care_weeks",
                                    )

                                    if not selected_weeks:
                                        st.info("Please select at least one week.")
                                    else:
                                        bundle_cols_bar = ["LD-ld1", "LD-ld2", "LD-ld3", "LD-ld4", "LD-ld5"]
                                        bundle_labels = {
                                            "LD-ld1": "Room temp 25-28°C",
                                            "LD-ld2": "Radiant warmer pre-warm",
                                            "LD-ld3": "Dried & covered",
                                            "LD-ld4": "Skin-to-skin",
                                            "LD-ld5": "Temp 36.5-37.5°C",
                                        }
                                        care_entity_col = "hospital"

                                        all_weeks_data = []
                                        for week_val in selected_weeks:
                                            week_raw = cis_work_df[cis_work_df["week"] == week_val].copy()
                                            if selected_round != "All Rounds":
                                                week_raw = week_raw[week_raw["round"] == selected_round]
                                            if selected_entities:
                                                week_raw = week_raw[week_raw[entity_col].isin(selected_entities)]
                                            if week_raw.empty:
                                                continue
                                            week_raw["_yes_count"] = week_raw[bundle_cols_bar].sum(axis=1)
                                            _idx = week_raw.groupby(["card", care_entity_col])["_yes_count"].idxmax()
                                            card_data = week_raw.loc[_idx, ["card", care_entity_col] + bundle_cols_bar].reset_index(drop=True)
                                            card_data["care_pct_actual"] = card_data[bundle_cols_bar].sum(axis=1) / 5 * 100
                                            card_data["care_pct"] = card_data["care_pct_actual"].clip(lower=0.5)
                                            card_data["card_label"] = card_data.apply(
                                                lambda r: f"Card {int(r['card'])}", axis=1
                                            )
                                            card_data["week"] = f"Week {week_val}"

                                            bundle_vals = []
                                            for _, r in card_data.iterrows():
                                                lines = [f"<b>Card {int(r['card'])}</b>"]
                                                for col in bundle_cols_bar:
                                                    val = int(r[col])
                                                    lines.append(f"{bundle_labels[col]}: {'Yes' if val else 'No'}")
                                                bundle_vals.append("<br>".join(lines))
                                            card_data["bundle_detail"] = bundle_vals

                                            all_weeks_data.append(card_data)

                                        if not all_weeks_data:
                                            st.info("No data for the selected filter combination.")
                                        else:
                                            plot_df = pd.concat(all_weeks_data, ignore_index=True)
                                            fig = px.bar(
                                                plot_df,
                                                x=care_entity_col,
                                                y="care_pct",
                                                color="card_label",
                                                barmode="group",
                                                facet_col="week",
                                                facet_col_wrap=2,
                                                color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3,
                                                custom_data=["bundle_detail", "care_pct_actual"],
                                                labels={care_entity_col: "Facility", "care_pct": "%"},
                                            )
                                            fig.update_traces(
                                                hovertemplate=(
                                                    "<b>%{x}</b><br>"
                                                    + "%{customdata[0]}<br>"
                                                    + "Care delivered: %{customdata[1]:.1f}%"
                                                    + "<extra></extra>"
                                                ),
                                                marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
                                            )

                                            fig.update_layout(
                                                template="plotly_white",
                                                height=max(300, len(selected_weeks) * 200),
                                                margin=dict(l=8, r=8, t=48, b=8),
                                                font=dict(size=9),
                                                hoverlabel=dict(font_size=9),
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="left",
                                                    x=0,
                                                    font=dict(size=8),
                                                    title=dict(text="Card", font=dict(size=8)),
                                                ),
                                            )
                                            fig.for_each_annotation(lambda a: a.update(
                                                text=a.text.split("=")[-1].strip(),
                                                font=dict(size=12, color="black"),
                                            ))
                                            fig.update_xaxes(
                                                matches=None,
                                                showticklabels=True,
                                                tickfont=dict(size=8),
                                                automargin=True,
                                                tickangle=45,
                                                showline=True,
                                                linewidth=1,
                                                linecolor="lightgray",
                                                mirror=True,
                                            )
                                            fig.update_yaxes(
                                                range=[0, 110],
                                                tickfont=dict(size=8),
                                                automargin=True,
                                                showline=True,
                                                linewidth=1,
                                                linecolor="lightgray",
                                                mirror=True,
                                            )
                                            st.plotly_chart(fig, use_container_width=True, key="mentorship_cis_care_delivered")

                                        with st.expander("How this is computed", expanded=False):
                                            st.markdown(
                                                f"""
**Percentage of care delivered per patient** = (Bundle items = Yes ÷ 5) × 100

For each patient (card), all 5 bundle items are evaluated. Missing values are treated as **No (0)** for consistency.

**Hypothermia Prevention Bundle Variables – L&D**

| Variable | Description |
|---|---|
| LD-ld1 | The current labour and delivery ward room temperature is between 25°C – 28°C |
| LD-ld2 | The radiant warmer is on pre-warm mode 5–10 minutes before the delivery |
| LD-ld3 | The baby is dried thoroughly, wet towel discarded, and baby covered by a pre-warmed towel |
| LD-ld4 | Skin-to-skin contact is maintained |
| LD-ld5 | The baby's body temperature is between 36.5 and 37.5°C when leaving the labour ward |
                                                """
                                            )

                            with cpap_tab:
                                cpap_ld_cols_present = [c for c in ["cpap_ld-cpap1", "cpap_ld-cpap2", "cpap_ld-cpap3", "cpap_ld-cpap4"] if c in work_df.columns]
                                cpap_all_or_none_tab, cpap_care_delivered_tab = st.tabs([
                                    "Percentage of CPAP All or None Bundle of care provided",
                                    "Percentage of care delivered",
                                ])

                                with cpap_all_or_none_tab:
                                    if len(cpap_ld_cols_present) < 4:
                                        st.info("CPAP columns not fully available in L&D data.")
                                    else:
                                        cpap_round_info = cis_work_df[["week", "card", "region_label", "hospital", "round"]].drop_duplicates(
                                            subset=["week", "card", "region_label", "hospital"]
                                        )
                                        cpap_all_or_none = compute_all_or_none(cis_work_df, bundle_cols=cpap_ld_cols_present)
                                        cpap_all_or_none = cpap_all_or_none.merge(
                                            cpap_round_info, on=["week", "card", "region_label", "hospital"], how="left"
                                        )

                                        cpap_filtered = cpap_all_or_none.copy()
                                        if selected_round != "All Rounds":
                                            cpap_filtered = cpap_filtered[cpap_filtered["round"] == selected_round]
                                        if selected_entities:
                                            cpap_filtered = cpap_filtered[cpap_filtered[entity_col].isin(selected_entities)]
                                        else:
                                            cpap_filtered = cpap_filtered.iloc[0:0]

                                        if cpap_filtered.empty:
                                            st.info("No data for the selected filter combination.")
                                        else:
                                            cpap_show_card_details = (group_mode == "Regional" and len(selected_entities) == 1 and entity_col == "hospital")

                                            if cpap_show_card_details:
                                                cpap_card_hover = cpap_filtered.groupby("week").apply(
                                                    lambda g: "<br>".join(
                                                        f"Card {int(row['card'])}: {int(row['all_or_none'])}"
                                                        for _, row in g.iterrows()
                                                    )
                                                ).reset_index(name="card_details")

                                            cpap_weekly_stats = cpap_filtered.groupby("week").agg(
                                                total_cards=("all_or_none", "count"),
                                                compliant=("all_or_none", "sum"),
                                            ).reset_index()
                                            cpap_weekly_stats["percentage"] = np.where(
                                                cpap_weekly_stats["total_cards"] > 0,
                                                (cpap_weekly_stats["compliant"] / cpap_weekly_stats["total_cards"]) * 100,
                                                0.0,
                                            )
                                            if cpap_show_card_details:
                                                cpap_weekly_stats = cpap_weekly_stats.merge(cpap_card_hover, on="week", how="left")
                                            cpap_weekly_stats = cpap_weekly_stats.sort_values("week")

                                            cpap_week_display_map = {w: f"Week {int(w)}" for w in sorted(cpap_weekly_stats["week"].unique())}
                                            cpap_weekly_stats["week_label"] = cpap_weekly_stats["week"].map(cpap_week_display_map)

                                            cpap_median_val = np.median(cpap_weekly_stats["percentage"]) if len(cpap_weekly_stats) > 0 else 0

                                            cpap_ci_total = 0
                                            cpap_ci_compliant = 1
                                            cpap_customdata_cols = ["total_cards", "compliant"]
                                            if cpap_show_card_details:
                                                cpap_customdata_cols = ["card_details", "total_cards", "compliant"]
                                                cpap_ci_total = 1
                                                cpap_ci_compliant = 2

                                            cpap_hover_parts = [
                                                "<b>%{x}</b><br>",
                                                f"Patients with full bundle: %{{customdata[{cpap_ci_compliant}]:.0f}}<br>",
                                                f"Total patients observed: %{{customdata[{cpap_ci_total}]:.0f}}<br>",
                                                "% of CPAP All or None: %{y:.1f}%<br>",
                                                f"Median: {cpap_median_val:.1f}%",
                                            ]
                                            if cpap_show_card_details:
                                                cpap_hover_parts.append("<br><br><b>All-or-None</b><br>%{customdata[0]}")
                                            cpap_hover_parts.append("<extra></extra>")

                                            cpap_fig = go.Figure()
                                            cpap_fig.add_trace(go.Scatter(
                                                x=cpap_weekly_stats["week_label"],
                                                y=cpap_weekly_stats["percentage"],
                                                mode="lines+markers",
                                                name="CPAP All-or-None Bundle",
                                                marker=dict(size=8, symbol="circle", color="#2563eb", line=dict(color="#2563eb", width=1)),
                                                line=dict(color="#2563eb", width=2.5),
                                                customdata=cpap_weekly_stats[cpap_customdata_cols].fillna("").values,
                                                hovertemplate="".join(cpap_hover_parts),
                                            ))
                                            cpap_fig.add_hline(
                                                y=cpap_median_val,
                                                line=dict(color="#e91e9e", width=2.5, dash="solid"),
                                                annotation_text=f"Median: {cpap_median_val:.1f}%",
                                                annotation_font_size=10,
                                                annotation_position="right",
                                            )
                                            cpap_fig.add_trace(go.Scatter(
                                                x=[None], y=[None],
                                                mode="lines",
                                                name=f"Median: {cpap_median_val:.1f}%",
                                                line=dict(color="#e91e9e", width=2.5),
                                                showlegend=True,
                                            ))

                                            cpap_fig.update_layout(
                                                template="plotly_white",
                                                height=290,
                                                margin=dict(l=8, r=8, t=20, b=8),
                                                xaxis_title=dict(text="Week", font=dict(size=11)),
                                                yaxis_title=dict(text="Percentage (%)", font=dict(size=11)),
                                                font=dict(size=10),
                                                hoverlabel=dict(font_size=10, namelength=-1),
                                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                                yaxis=dict(range=[0, 110], tickformat=".0f", dtick=20),
                                            )
                                            cpap_sorted_week_labels = [f"Week {int(w)}" for w in sorted(cpap_weekly_stats["week"].unique())]
                                            cpap_fig.update_xaxes(tickfont=dict(size=9), automargin=True, categoryorder="array", categoryarray=cpap_sorted_week_labels)
                                            cpap_fig.update_yaxes(tickfont=dict(size=9), automargin=True, gridcolor="#f0f0f0")

                                            st.plotly_chart(cpap_fig, use_container_width=True, key="mentorship_cis_cpap_run_chart")

                                            with st.expander("How this is computed", expanded=False):
                                                st.markdown(
                                                    """
**Percentage** = (Patients with full bundle ÷ Total patients observed) × 100

**Early CPAP Bundle of Care – L&D**

1. bCPAP is readily available at the L&D room immediately before the baby is born for indicated baby
2. bCPAP initiated immediately after delivery with blended oxygen at 5-6cm H20 and Fio2 adjusted according to patient need
3. Baby's airway is patent and correctly positioned
4. CPAP used during transporting neonate to the NICU
                                                    """
                                                )

                                with cpap_care_delivered_tab:
                                    cpap_selected_weeks = st.multiselect(
                                        "Select Weeks",
                                        options=sorted([int(w) for w in cis_work_df["week"].dropna().unique() if str(w).strip()]),
                                        default=sorted([int(w) for w in cis_work_df["week"].dropna().unique() if str(w).strip()]),
                                        key="mentorship_cis_cpap_care_weeks",
                                    )

                                    if not cpap_selected_weeks:
                                        st.info("Please select at least one week.")
                                    else:
                                        cpap_bundle_labels = {
                                            "cpap_ld-cpap1": "bCPAP ready at L&D room",
                                            "cpap_ld-cpap2": "bCPAP initiated with blended O2",
                                            "cpap_ld-cpap3": "Airway patent & positioned",
                                            "cpap_ld-cpap4": "CPAP during transport to NICU",
                                        }
                                        cpap_care_entity_col = "hospital"

                                        cpap_all_weeks_data = []
                                        for week_val in cpap_selected_weeks:
                                            week_raw = cis_work_df[cis_work_df["week"] == week_val].copy()
                                            if selected_round != "All Rounds":
                                                week_raw = week_raw[week_raw["round"] == selected_round]
                                            if selected_entities:
                                                week_raw = week_raw[week_raw[entity_col].isin(selected_entities)]
                                            if week_raw.empty:
                                                continue
                                            week_raw["_yes_count"] = week_raw[cpap_ld_cols_present].sum(axis=1)
                                            _idx = week_raw.groupby(["card", cpap_care_entity_col])["_yes_count"].idxmax()
                                            card_data = week_raw.loc[_idx, ["card", cpap_care_entity_col] + cpap_ld_cols_present].reset_index(drop=True)
                                            card_data["care_pct_actual"] = card_data[cpap_ld_cols_present].sum(axis=1) / 4 * 100
                                            card_data["care_pct"] = card_data["care_pct_actual"].clip(lower=0.5)
                                            card_data["card_label"] = card_data.apply(
                                                lambda r: f"Card {int(r['card'])}", axis=1
                                            )
                                            card_data["week"] = f"Week {week_val}"

                                            bundle_vals = []
                                            for _, r in card_data.iterrows():
                                                lines = [f"<b>Card {int(r['card'])}</b>"]
                                                for col in cpap_ld_cols_present:
                                                    val = int(r[col])
                                                    lines.append(f"{cpap_bundle_labels[col]}: {'Yes' if val else 'No'}")
                                                bundle_vals.append("<br>".join(lines))
                                            card_data["bundle_detail"] = bundle_vals

                                            cpap_all_weeks_data.append(card_data)

                                        if not cpap_all_weeks_data:
                                            st.info("No data for the selected filter combination.")
                                        else:
                                            cpap_plot_df = pd.concat(cpap_all_weeks_data, ignore_index=True)
                                            cpap_fig2 = px.bar(
                                                cpap_plot_df,
                                                x=cpap_care_entity_col,
                                                y="care_pct",
                                                color="card_label",
                                                barmode="group",
                                                facet_col="week",
                                                facet_col_wrap=2,
                                                color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3,
                                                custom_data=["bundle_detail", "care_pct_actual"],
                                                labels={cpap_care_entity_col: "Facility", "care_pct": "%"},
                                            )
                                            cpap_fig2.update_traces(
                                                hovertemplate=(
                                                    "<b>%{x}</b><br>"
                                                    + "%{customdata[0]}<br>"
                                                    + "Care delivered: %{customdata[1]:.1f}%"
                                                    + "<extra></extra>"
                                                ),
                                                marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
                                            )

                                            cpap_fig2.update_layout(
                                                template="plotly_white",
                                                height=max(300, len(cpap_selected_weeks) * 200),
                                                margin=dict(l=8, r=8, t=48, b=8),
                                                font=dict(size=9),
                                                hoverlabel=dict(font_size=9),
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="bottom",
                                                    y=1.02,
                                                    xanchor="left",
                                                    x=0,
                                                    font=dict(size=8),
                                                    title=dict(text="Card", font=dict(size=8)),
                                                ),
                                            )
                                            cpap_fig2.for_each_annotation(lambda a: a.update(
                                                text=a.text.split("=")[-1].strip(),
                                                font=dict(size=12, color="black"),
                                            ))
                                            cpap_fig2.update_xaxes(
                                                matches=None,
                                                showticklabels=True,
                                                tickfont=dict(size=8),
                                                automargin=True,
                                                tickangle=45,
                                                showline=True,
                                                linewidth=1,
                                                linecolor="lightgray",
                                                mirror=True,
                                            )
                                            cpap_fig2.update_yaxes(
                                                range=[0, 110],
                                                tickfont=dict(size=8),
                                                automargin=True,
                                                showline=True,
                                                linewidth=1,
                                                linecolor="lightgray",
                                                mirror=True,
                                            )
                                            st.plotly_chart(cpap_fig2, use_container_width=True, key="mentorship_cis_cpap_care_delivered")

                                        with st.expander("How this is computed", expanded=False):
                                            st.markdown(
                                                f"""
**Percentage of care delivered per patient** = (Bundle items = Yes ÷ 4) × 100

For each patient (card), all 4 bundle items are evaluated. Missing values are treated as **No (0)** for consistency.

**Early CPAP Bundle Variables – L&D**

| Variable | Description |
|---|---|
| cpap_ld-cpap1 | bCPAP is readily available at the L&D room immediately before the baby is born for indicated baby |
| cpap_ld-cpap2 | bCPAP initiated immediately after delivery with blended oxygen at 5-6cm H20 and Fio2 adjusted according to patient need |
| cpap_ld-cpap3 | Baby's airway is patent and correctly positioned |
| cpap_ld-cpap4 | CPAP used during transporting neonate to the NICU |
                                                """
                                            )

                    with nicu_tab:
                        nicu_bundle_cols = ["NICU-nicu1", "NICU-nicu2", "NICU-nicu3", "NICU-nicu4", "NICU-nicu5"]
                        nicu_work_df = cis_raw.copy()
                        nicu_work_df["region_code"] = nicu_work_df["reg-region"].apply(_normalize_region_code)
                        nicu_work_df["hospital"] = nicu_work_df["reg-hospital"].apply(_display_mentorship_facility)
                        nicu_work_df["region_label"] = nicu_work_df["region_code"].map(region_code_map).fillna(
                            nicu_work_df["region_code"].apply(lambda x: f"Unknown ({x})" if x else "Unknown")
                        )
                        nicu_work_df["round"] = nicu_work_df["reg-round"].astype(str).str.strip()
                        nicu_work_df["unit"] = nicu_work_df["unit"].astype(str).str.strip().replace({"1.0": "1", "2.0": "2"})

                        nicu_current_user = st.session_state.get("user", {})
                        if nicu_current_user.get("role") == "regional":
                            try:
                                nicu_region_id = int(nicu_current_user.get("region_id"))
                            except (TypeError, ValueError):
                                nicu_region_id = None
                            nicu_regional_codes = get_odk_region_codes(nicu_region_id) if nicu_region_id is not None else []
                            if nicu_regional_codes:
                                nicu_work_df = nicu_work_df[nicu_work_df["region_code"].isin([str(c) for c in nicu_regional_codes])]

                        nicu_work_df = nicu_work_df[nicu_work_df["unit"] == "2"].copy()

                        missing_nicu = [c for c in nicu_bundle_cols if c not in nicu_work_df.columns]
                        if missing_nicu:
                            st.error(f"Missing required NICU columns: {missing_nicu}")
                        else:
                            nicu_work_df["week"] = pd.to_numeric(nicu_work_df["week"], errors="coerce")
                            nicu_work_df = nicu_work_df.dropna(subset=["week", "card"])

                            for col in nicu_bundle_cols:
                                nicu_work_df[col] = pd.to_numeric(nicu_work_df[col], errors="coerce")
                            nicu_work_df[nicu_bundle_cols] = nicu_work_df[nicu_bundle_cols].fillna(0)

                            nicu_cpap_cols = ["cpap_nicu-cpap1nicu", "cpap_nicu-cpap2nicu", "cpap_nicu-cpap3nicu", "cpap_nicu-cpap4nicu", "cpap_nicu-cpap5nicu", "cpap_nicu-cpap6nicu"]
                            nicu_cpap_cols_present = [c for c in nicu_cpap_cols if c in nicu_work_df.columns]
                            if nicu_cpap_cols_present:
                                for col in nicu_cpap_cols_present:
                                    nicu_work_df[col] = pd.to_numeric(nicu_work_df[col], errors="coerce")
                                nicu_work_df[nicu_cpap_cols_present] = nicu_work_df[nicu_cpap_cols_present].fillna(0)

                            nicu_kmc_cols = ["kmc-kmc1", "kmc-kmc2", "kmc-kmc3", "kmc-kmc4", "kmc-kmc5", "kmc-kmc6"]
                            nicu_kmc_cols_present = [c for c in nicu_kmc_cols if c in nicu_work_df.columns]
                            if nicu_kmc_cols_present:
                                for col in nicu_kmc_cols_present:
                                    nicu_work_df[col] = pd.to_numeric(nicu_work_df[col], errors="coerce")
                                nicu_work_df[nicu_kmc_cols_present] = nicu_work_df[nicu_kmc_cols_present].fillna(0)

                            nicu_nutrition_cols = ["Referral-referral1", "Referral-referral2", "Referral-referral3", "Referral-referral4", "Referral-referral5"]
                            nicu_nutrition_cols_present = [c for c in nicu_nutrition_cols if c in nicu_work_df.columns]
                            if nicu_nutrition_cols_present:
                                for col in nicu_nutrition_cols_present:
                                    nicu_work_df[col] = pd.to_numeric(nicu_work_df[col], errors="coerce")
                                nicu_work_df[nicu_nutrition_cols_present] = nicu_work_df[nicu_nutrition_cols_present].fillna(0)

                            nicu_all_or_none_df = compute_all_or_none(nicu_work_df, bundle_cols=nicu_bundle_cols)
                            if nicu_all_or_none_df.empty:
                                st.warning("No data available for NICU.")
                            else:
                                nicu_round_info = nicu_work_df[["week", "card", "region_label", "hospital", "round"]].drop_duplicates(
                                    subset=["week", "card", "region_label", "hospital"]
                                )
                                nicu_all_or_none_df = nicu_all_or_none_df.merge(
                                    nicu_round_info, on=["week", "card", "region_label", "hospital"], how="left"
                                )

                                nicu_left_col, nicu_right_col = st.columns([3, 1])

                                with nicu_right_col:
                                    with st.container(key="mentorship_filters_card_nicu"):
                                        st.markdown('<div class="mentorship-filter-box">', unsafe_allow_html=True)
                                        st.markdown(
                                            '<div class="mentorship-filter-title">Filters</div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown(
                                            '<div class="mentorship-filter-subtitle">Refine by group and round</div>',
                                            unsafe_allow_html=True,
                                        )
                                        st.markdown('<div class="mentorship-filter-divider"></div>', unsafe_allow_html=True)

                                        nicu_round_values = sorted([r for r in nicu_work_df["round"].dropna().unique() if r != ""])
                                        nicu_all_rounds_label = "All Rounds"
                                        nicu_round_options = [nicu_all_rounds_label] + nicu_round_values if nicu_round_values else [nicu_all_rounds_label]

                                        nicu_selected_round = st.selectbox(
                                            "Round", options=nicu_round_options, index=0,
                                            key="mentorship_nicu_round",
                                        )

                                        current_user = st.session_state.get("user", {})
                                        is_regional_user = current_user.get("role") == "regional"
                                        nicu_group_mode_options = (
                                            ["Regional", "Learning Facilities"]
                                            if is_regional_user
                                            else ["Multi Regional", "Regional", "Learning Facilities"]
                                        )
                                        nicu_group_mode = st.radio(
                                            "Group By", options=nicu_group_mode_options,
                                            key="mentorship_nicu_group_mode",
                                        )
                                        nicu_scoped_all_or_none_df = (
                                            _filter_learning_facilities(nicu_all_or_none_df)
                                            if nicu_group_mode == "Learning Facilities"
                                            else nicu_all_or_none_df.copy()
                                        )
                                        nicu_scoped_work_df = (
                                            _filter_learning_facilities(nicu_work_df)
                                            if nicu_group_mode == "Learning Facilities"
                                            else nicu_work_df.copy()
                                        )
                                        if nicu_scoped_all_or_none_df.empty or nicu_scoped_work_df.empty:
                                            st.warning("No data available for learning facilities.")
                                            return

                                        if nicu_group_mode == "Multi Regional":
                                            nicu_entity_col = "region_label"
                                            nicu_entity_options = sorted([
                                                v for v in nicu_scoped_all_or_none_df[nicu_entity_col].dropna().unique().tolist()
                                                if str(v).strip()
                                            ])
                                            nicu_all_token = "All Regions"
                                            nicu_selector_options = [nicu_all_token] + nicu_entity_options
                                            nicu_entity_key = "mentorship_nicu_entities_multi"
                                            if nicu_entity_key not in st.session_state or not st.session_state.get(nicu_entity_key):
                                                st.session_state[nicu_entity_key] = [nicu_all_token]
                                            st.multiselect("Select Regions", options=nicu_selector_options, key=nicu_entity_key)
                                            nicu_current_selected = st.session_state.get(nicu_entity_key, [nicu_all_token])
                                            nicu_effective_selected = (
                                                [nicu_all_token] if (nicu_all_token in nicu_current_selected or len(nicu_current_selected) == 0)
                                                else nicu_current_selected
                                            )
                                            nicu_selected_entities = nicu_entity_options if nicu_all_token in nicu_effective_selected else nicu_effective_selected
                                        elif nicu_group_mode == "Learning Facilities":
                                            nicu_entity_col = "hospital"
                                            nicu_entity_options = sorted([
                                                v for v in nicu_scoped_all_or_none_df[nicu_entity_col].dropna().unique().tolist()
                                                if str(v).strip()
                                            ])
                                            nicu_all_token = "All Learning Facilities"
                                            nicu_selector_options = [nicu_all_token] + nicu_entity_options
                                            nicu_entity_key = "mentorship_nicu_entities_learning"
                                            if nicu_entity_key not in st.session_state or not st.session_state.get(nicu_entity_key):
                                                st.session_state[nicu_entity_key] = [nicu_all_token]
                                            st.multiselect("Select Learning Facilities", options=nicu_selector_options, key=nicu_entity_key)
                                            nicu_current_selected = st.session_state.get(nicu_entity_key, [nicu_all_token])
                                            nicu_effective_selected = (
                                                [nicu_all_token] if (nicu_all_token in nicu_current_selected or len(nicu_current_selected) == 0)
                                                else nicu_current_selected
                                            )
                                            nicu_selected_entities = nicu_entity_options if nicu_all_token in nicu_effective_selected else nicu_effective_selected
                                        else:
                                            nicu_region_options = sorted([
                                                v for v in nicu_scoped_all_or_none_df["region_label"].dropna().unique().tolist()
                                                if str(v).strip()
                                            ])
                                            nicu_selected_region = st.selectbox(
                                                "Select Region", options=nicu_region_options,
                                                key="mentorship_nicu_region",
                                            )
                                            nicu_facilities_in_region = sorted([
                                                v for v in nicu_scoped_all_or_none_df.loc[
                                                    nicu_scoped_all_or_none_df["region_label"] == nicu_selected_region, "hospital"
                                                ].dropna().unique().tolist() if str(v).strip()
                                            ])
                                            nicu_entity_col = "hospital"
                                            nicu_all_token = "All Facilities in Region"
                                            nicu_selector_options = [nicu_all_token] + nicu_facilities_in_region
                                            nicu_entity_key = "mentorship_nicu_entities_facilities"
                                            if nicu_entity_key not in st.session_state or not st.session_state.get(nicu_entity_key):
                                                st.session_state[nicu_entity_key] = [nicu_all_token]
                                            st.multiselect("Select Facilities", options=nicu_selector_options, key=nicu_entity_key)
                                            nicu_current_selected = st.session_state.get(nicu_entity_key, [nicu_all_token])
                                            nicu_effective_selected = (
                                                [nicu_all_token] if (nicu_all_token in nicu_current_selected or len(nicu_current_selected) == 0)
                                                else nicu_current_selected
                                            )
                                            nicu_selected_entities = nicu_facilities_in_region if nicu_all_token in nicu_effective_selected else nicu_effective_selected

                                        st.markdown("</div>", unsafe_allow_html=True)

                                with nicu_left_col:
                                    nicu_hypo_tab, nicu_cpap_tab, nicu_kmc_tab, nicu_nutrition_tab = st.tabs(["Hypothermia Prevention Bundle of Care", "Early CPAP Bundle of Care", "KMC Bundle of Care", "Nutrition Bundle of Care"])

                                    with nicu_hypo_tab:
                                        nicu_all_or_none_tab, nicu_care_delivered_tab = st.tabs([
                                            "Percentage of Hypothermia prevention All or None",
                                            "Percentage of care delivered",
                                        ])

                                        with nicu_all_or_none_tab:
                                            nicu_filtered = nicu_scoped_all_or_none_df.copy()
                                            if nicu_selected_round != "All Rounds":
                                                nicu_filtered = nicu_filtered[nicu_filtered["round"] == nicu_selected_round]
                                            if nicu_selected_entities:
                                                nicu_filtered = nicu_filtered[nicu_filtered[nicu_entity_col].isin(nicu_selected_entities)]
                                            else:
                                                nicu_filtered = nicu_filtered.iloc[0:0]

                                            if nicu_filtered.empty:
                                                st.info("No data for the selected filter combination.")
                                            else:
                                                nicu_show_card_details = (nicu_group_mode == "Regional" and len(nicu_selected_entities) == 1 and nicu_entity_col == "hospital")

                                                if nicu_show_card_details:
                                                    nicu_card_hover = nicu_filtered.groupby("week").apply(
                                                        lambda g: "<br>".join(
                                                            f"Card {int(row['card'])}: {int(row['all_or_none'])}"
                                                            for _, row in g.iterrows()
                                                        )
                                                    ).reset_index(name="card_details")

                                                nicu_weekly_stats = nicu_filtered.groupby("week").agg(
                                                    total_cards=("all_or_none", "count"),
                                                    compliant=("all_or_none", "sum"),
                                                ).reset_index()
                                                nicu_weekly_stats["percentage"] = np.where(
                                                    nicu_weekly_stats["total_cards"] > 0,
                                                    (nicu_weekly_stats["compliant"] / nicu_weekly_stats["total_cards"]) * 100,
                                                    0.0,
                                                )
                                                if nicu_show_card_details:
                                                    nicu_weekly_stats = nicu_weekly_stats.merge(nicu_card_hover, on="week", how="left")
                                                nicu_weekly_stats = nicu_weekly_stats.sort_values("week")

                                                nicu_week_display_map = {w: f"Week {int(w)}" for w in sorted(nicu_weekly_stats["week"].unique())}
                                                nicu_weekly_stats["week_label"] = nicu_weekly_stats["week"].map(nicu_week_display_map)

                                                nicu_median_val = np.median(nicu_weekly_stats["percentage"]) if len(nicu_weekly_stats) > 0 else 0

                                                nicu_ci_total = 0
                                                nicu_ci_compliant = 1
                                                nicu_customdata_cols = ["total_cards", "compliant"]
                                                if nicu_show_card_details:
                                                    nicu_customdata_cols = ["card_details", "total_cards", "compliant"]
                                                    nicu_ci_total = 1
                                                    nicu_ci_compliant = 2

                                                nicu_hover_parts = [
                                                    "<b>%{x}</b><br>",
                                                    f"Patients with full bundle: %{{customdata[{nicu_ci_compliant}]:.0f}}<br>",
                                                    f"Total patients observed: %{{customdata[{nicu_ci_total}]:.0f}}<br>",
                                                    "% of Hypothermia prevention: %{y:.1f}%<br>",
                                                    f"Median: {nicu_median_val:.1f}%",
                                                ]

                                                if nicu_show_card_details:
                                                    nicu_hover_parts.append("<br><br><b>All-or-None</b><br>%{customdata[0]}")

                                                nicu_hover_parts.append("<extra></extra>")

                                                nicu_fig = go.Figure()
                                                nicu_fig.add_trace(go.Scatter(
                                                    x=nicu_weekly_stats["week_label"],
                                                    y=nicu_weekly_stats["percentage"],
                                                    mode="lines+markers",
                                                    name="Hypothermia Prevention All-or-None Bundle",
                                                    marker=dict(size=8, symbol="circle", color="#2563eb", line=dict(color="#2563eb", width=1)),
                                                    line=dict(color="#2563eb", width=2.5),
                                                    customdata=nicu_weekly_stats[nicu_customdata_cols].fillna("").values,
                                                    hovertemplate="".join(nicu_hover_parts),
                                                ))
                                                nicu_fig.add_hline(
                                                    y=nicu_median_val,
                                                    line=dict(color="#e91e9e", width=2.5, dash="solid"),
                                                    annotation_text=f"Median: {nicu_median_val:.1f}%",
                                                    annotation_font_size=10,
                                                    annotation_position="right",
                                                )
                                                nicu_fig.add_trace(go.Scatter(
                                                    x=[None], y=[None],
                                                    mode="lines",
                                                    name=f"Median: {nicu_median_val:.1f}%",
                                                    line=dict(color="#e91e9e", width=2.5),
                                                    showlegend=True,
                                                ))

                                                nicu_fig.update_layout(
                                                    template="plotly_white",
                                                    height=290,
                                                    margin=dict(l=8, r=8, t=20, b=8),
                                                    xaxis_title=dict(text="Week", font=dict(size=11)),
                                                    yaxis_title=dict(text="Percentage (%)", font=dict(size=11)),
                                                    font=dict(size=10),
                                                    hoverlabel=dict(font_size=10, namelength=-1),
                                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                                    yaxis=dict(range=[0, 110], tickformat=".0f", dtick=20),
                                                )
                                                nicu_sorted_week_labels = [f"Week {int(w)}" for w in sorted(nicu_weekly_stats["week"].unique())]
                                                nicu_fig.update_xaxes(tickfont=dict(size=9), automargin=True, categoryorder="array", categoryarray=nicu_sorted_week_labels)
                                                nicu_fig.update_yaxes(tickfont=dict(size=9), automargin=True, gridcolor="#f0f0f0")

                                                st.plotly_chart(nicu_fig, use_container_width=True, key="mentorship_nicu_run_chart")

                                                with st.expander("How this is computed", expanded=False):
                                                    st.markdown(
                                                        """
**Percentage** = (Patients with full bundle ÷ Total patients observed) × 100

**Hypothermia Prevention Bundle of Care – NICU**

1. The baby is transported and arrived while on skin to skin with the mother/surrogate or using pre-warmed transport incubator or body covered in a plastic bag
2. The room temperature of the newborn unit is maintained at 25 - 28 degree Celsius
3. Baby's received for admission to the newborn unit under a prewarmed radiant warmer
4. Baby's body temperature is taken and recorded at admission and every 3-6 hours in the NBU
5. Baby's body temperature is between 36.5 and 37.5 degree Celsius at admission
                                                        """
                                                    )

                                        with nicu_care_delivered_tab:
                                            nicu_selected_weeks = st.multiselect(
                                                "Select Weeks",
                                                options=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                default=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                key="mentorship_nicu_care_weeks",
                                            )

                                            if not nicu_selected_weeks:
                                                st.info("Please select at least one week.")
                                            else:
                                                nicu_bundle_labels = {
                                                    "NICU-nicu1": "Skin-to-skin / pre-warmed incubator / plastic bag",
                                                    "NICU-nicu2": "Newborn unit room temp 25-28°C",
                                                    "NICU-nicu3": "Admitted under prewarmed radiant warmer",
                                                    "NICU-nicu4": "Temp taken at admission & q3-6h",
                                                    "NICU-nicu5": "Body temp 36.5-37.5°C at admission",
                                                }
                                                nicu_care_entity_col = "hospital"

                                                nicu_all_weeks_data = []
                                                for week_val in nicu_selected_weeks:
                                                    week_raw = nicu_scoped_work_df[nicu_scoped_work_df["week"] == week_val].copy()
                                                    if nicu_selected_round != "All Rounds":
                                                        week_raw = week_raw[week_raw["round"] == nicu_selected_round]
                                                    if nicu_selected_entities:
                                                        week_raw = week_raw[week_raw[nicu_entity_col].isin(nicu_selected_entities)]
                                                    if week_raw.empty:
                                                        continue
                                                    week_raw["_yes_count"] = week_raw[nicu_bundle_cols].sum(axis=1)
                                                    _idx = week_raw.groupby(["card", nicu_care_entity_col])["_yes_count"].idxmax()
                                                    card_data = week_raw.loc[_idx, ["card", nicu_care_entity_col] + nicu_bundle_cols].reset_index(drop=True)
                                                    card_data["care_pct_actual"] = card_data[nicu_bundle_cols].sum(axis=1) / 5 * 100
                                                    card_data["care_pct"] = card_data["care_pct_actual"].clip(lower=0.5)
                                                    card_data["card_label"] = card_data.apply(
                                                        lambda r: f"Card {int(r['card'])}", axis=1
                                                    )
                                                    card_data["week"] = f"Week {week_val}"

                                                    bundle_vals = []
                                                    for _, r in card_data.iterrows():
                                                        lines = [f"<b>Card {int(r['card'])}</b>"]
                                                        for col in nicu_bundle_cols:
                                                            val = int(r[col])
                                                            lines.append(f"{nicu_bundle_labels[col]}: {'Yes' if val else 'No'}")
                                                        bundle_vals.append("<br>".join(lines))
                                                    card_data["bundle_detail"] = bundle_vals

                                                    nicu_all_weeks_data.append(card_data)

                                                if not nicu_all_weeks_data:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_plot_df = pd.concat(nicu_all_weeks_data, ignore_index=True)
                                                    nicu_fig2 = px.bar(
                                                        nicu_plot_df,
                                                        x=nicu_care_entity_col,
                                                        y="care_pct",
                                                        color="card_label",
                                                        barmode="group",
                                                        facet_col="week",
                                                        facet_col_wrap=2,
                                                        color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3,
                                                        custom_data=["bundle_detail", "care_pct_actual"],
                                                        labels={nicu_care_entity_col: "Facility", "care_pct": "%"},
                                                    )
                                                    nicu_fig2.update_traces(
                                                        hovertemplate=(
                                                            "<b>%{x}</b><br>"
                                                            + "%{customdata[0]}<br>"
                                                            + "Care delivered: %{customdata[1]:.1f}%"
                                                            + "<extra></extra>"
                                                        ),
                                                        marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
                                                    )

                                                    nicu_fig2.update_layout(
                                                        template="plotly_white",
                                                        height=max(300, len(nicu_selected_weeks) * 200),
                                                        margin=dict(l=8, r=8, t=48, b=8),
                                                        font=dict(size=9),
                                                        hoverlabel=dict(font_size=9),
                                                        legend=dict(
                                                            orientation="h",
                                                            yanchor="bottom",
                                                            y=1.02,
                                                            xanchor="left",
                                                            x=0,
                                                            font=dict(size=8),
                                                            title=dict(text="Card", font=dict(size=8)),
                                                        ),
                                                    )
                                                    nicu_fig2.for_each_annotation(lambda a: a.update(
                                                        text=a.text.split("=")[-1].strip(),
                                                        font=dict(size=12, color="black"),
                                                    ))
                                                    nicu_fig2.update_xaxes(
                                                        matches=None,
                                                        showticklabels=True,
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        tickangle=45,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    nicu_fig2.update_yaxes(
                                                        range=[0, 110],
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    st.plotly_chart(nicu_fig2, use_container_width=True, key="mentorship_nicu_care_delivered")

                                                with st.expander("How this is computed", expanded=False):
                                                    st.markdown(
                                                        f"""
**Percentage of care delivered per patient** = (Bundle items = Yes ÷ 5) × 100

For each patient (card), all 5 bundle items are evaluated. Missing values are treated as **No (0)** for consistency.

**Hypothermia Prevention Bundle Variables – NICU**

| Variable | Description |
|---|---|
| NICU-nicu1 | The baby is transported and arrived while on skin to skin with the mother/surrogate or using pre-warmed transport incubator or body covered in a plastic bag |
| NICU-nicu2 | The room temperature of the newborn unit is maintained at 25 - 28 degree Celsius |
| NICU-nicu3 | Baby's received for admission to the newborn unit under a prewarmed radiant warmer |
| NICU-nicu4 | Baby's body temperature is taken and recorded at admission and every 3-6 hours in the NBU |
| NICU-nicu5 | Baby's body temperature is between 36.5 and 37.5 degree Celsius at admission |
                                                        """
                                                    )

                                    with nicu_cpap_tab:
                                        nicu_cpap_cols_present = [c for c in ["cpap_nicu-cpap1nicu", "cpap_nicu-cpap2nicu", "cpap_nicu-cpap3nicu", "cpap_nicu-cpap4nicu", "cpap_nicu-cpap5nicu", "cpap_nicu-cpap6nicu"] if c in nicu_work_df.columns]
                                        nicu_cpap_all_or_none_tab, nicu_cpap_care_delivered_tab = st.tabs([
                                            "Percentage of CPAP All or None Bundle of care provided",
                                            "Percentage of care delivered",
                                        ])

                                        with nicu_cpap_all_or_none_tab:
                                            if len(nicu_cpap_cols_present) < 6:
                                                st.info("CPAP columns not fully available in NICU data.")
                                            else:
                                                nicu_cpap_round_info = nicu_scoped_work_df[["week", "card", "region_label", "hospital", "round"]].drop_duplicates(
                                                    subset=["week", "card", "region_label", "hospital"]
                                                )
                                                nicu_cpap_all_or_none = compute_all_or_none(nicu_scoped_work_df, bundle_cols=nicu_cpap_cols_present)
                                                nicu_cpap_all_or_none = nicu_cpap_all_or_none.merge(
                                                    nicu_cpap_round_info, on=["week", "card", "region_label", "hospital"], how="left"
                                                )

                                                nicu_cpap_filtered = nicu_cpap_all_or_none.copy()
                                                if nicu_selected_round != "All Rounds":
                                                    nicu_cpap_filtered = nicu_cpap_filtered[nicu_cpap_filtered["round"] == nicu_selected_round]
                                                if nicu_selected_entities:
                                                    nicu_cpap_filtered = nicu_cpap_filtered[nicu_cpap_filtered[nicu_entity_col].isin(nicu_selected_entities)]
                                                else:
                                                    nicu_cpap_filtered = nicu_cpap_filtered.iloc[0:0]

                                                if nicu_cpap_filtered.empty:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_cpap_show_card_details = (nicu_group_mode == "Regional" and len(nicu_selected_entities) == 1 and nicu_entity_col == "hospital")

                                                    if nicu_cpap_show_card_details:
                                                        nicu_cpap_card_hover = nicu_cpap_filtered.groupby("week").apply(
                                                            lambda g: "<br>".join(
                                                                f"Card {int(row['card'])}: {int(row['all_or_none'])}"
                                                                for _, row in g.iterrows()
                                                            )
                                                        ).reset_index(name="card_details")

                                                    nicu_cpap_weekly_stats = nicu_cpap_filtered.groupby("week").agg(
                                                        total_cards=("all_or_none", "count"),
                                                        compliant=("all_or_none", "sum"),
                                                    ).reset_index()
                                                    nicu_cpap_weekly_stats["percentage"] = np.where(
                                                        nicu_cpap_weekly_stats["total_cards"] > 0,
                                                        (nicu_cpap_weekly_stats["compliant"] / nicu_cpap_weekly_stats["total_cards"]) * 100,
                                                        0.0,
                                                    )
                                                    if nicu_cpap_show_card_details:
                                                        nicu_cpap_weekly_stats = nicu_cpap_weekly_stats.merge(nicu_cpap_card_hover, on="week", how="left")
                                                    nicu_cpap_weekly_stats = nicu_cpap_weekly_stats.sort_values("week")

                                                    nicu_cpap_week_display_map = {w: f"Week {int(w)}" for w in sorted(nicu_cpap_weekly_stats["week"].unique())}
                                                    nicu_cpap_weekly_stats["week_label"] = nicu_cpap_weekly_stats["week"].map(nicu_cpap_week_display_map)

                                                    nicu_cpap_median_val = np.median(nicu_cpap_weekly_stats["percentage"]) if len(nicu_cpap_weekly_stats) > 0 else 0

                                                    nicu_cpap_ci_total = 0
                                                    nicu_cpap_ci_compliant = 1
                                                    nicu_cpap_customdata_cols = ["total_cards", "compliant"]
                                                    if nicu_cpap_show_card_details:
                                                        nicu_cpap_customdata_cols = ["card_details", "total_cards", "compliant"]
                                                        nicu_cpap_ci_total = 1
                                                        nicu_cpap_ci_compliant = 2

                                                    nicu_cpap_hover_parts = [
                                                        "<b>%{x}</b><br>",
                                                        f"Patients with full bundle: %{{customdata[{nicu_cpap_ci_compliant}]:.0f}}<br>",
                                                        f"Total patients observed: %{{customdata[{nicu_cpap_ci_total}]:.0f}}<br>",
                                                        "% of CPAP All or None: %{y:.1f}%<br>",
                                                        f"Median: {nicu_cpap_median_val:.1f}%",
                                                    ]
                                                    if nicu_cpap_show_card_details:
                                                        nicu_cpap_hover_parts.append("<br><br><b>All-or-None</b><br>%{customdata[0]}")
                                                    nicu_cpap_hover_parts.append("<extra></extra>")

                                                    nicu_cpap_fig = go.Figure()
                                                    nicu_cpap_fig.add_trace(go.Scatter(
                                                        x=nicu_cpap_weekly_stats["week_label"],
                                                        y=nicu_cpap_weekly_stats["percentage"],
                                                        mode="lines+markers",
                                                        name="CPAP All-or-None Bundle",
                                                        marker=dict(size=8, symbol="circle", color="#2563eb", line=dict(color="#2563eb", width=1)),
                                                        line=dict(color="#2563eb", width=2.5),
                                                        customdata=nicu_cpap_weekly_stats[nicu_cpap_customdata_cols].fillna("").values,
                                                        hovertemplate="".join(nicu_cpap_hover_parts),
                                                    ))
                                                    nicu_cpap_fig.add_hline(
                                                        y=nicu_cpap_median_val,
                                                        line=dict(color="#e91e9e", width=2.5, dash="solid"),
                                                        annotation_text=f"Median: {nicu_cpap_median_val:.1f}%",
                                                        annotation_font_size=10,
                                                        annotation_position="right",
                                                    )
                                                    nicu_cpap_fig.add_trace(go.Scatter(
                                                        x=[None], y=[None],
                                                        mode="lines",
                                                        name=f"Median: {nicu_cpap_median_val:.1f}%",
                                                        line=dict(color="#e91e9e", width=2.5),
                                                        showlegend=True,
                                                    ))

                                                    nicu_cpap_fig.update_layout(
                                                        template="plotly_white",
                                                        height=290,
                                                        margin=dict(l=8, r=8, t=20, b=8),
                                                        xaxis_title=dict(text="Week", font=dict(size=11)),
                                                        yaxis_title=dict(text="Percentage (%)", font=dict(size=11)),
                                                        font=dict(size=10),
                                                        hoverlabel=dict(font_size=10, namelength=-1),
                                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                                        yaxis=dict(range=[0, 110], tickformat=".0f", dtick=20),
                                                    )
                                                    nicu_cpap_sorted_week_labels = [f"Week {int(w)}" for w in sorted(nicu_cpap_weekly_stats["week"].unique())]
                                                    nicu_cpap_fig.update_xaxes(tickfont=dict(size=9), automargin=True, categoryorder="array", categoryarray=nicu_cpap_sorted_week_labels)
                                                    nicu_cpap_fig.update_yaxes(tickfont=dict(size=9), automargin=True, gridcolor="#f0f0f0")

                                                    st.plotly_chart(nicu_cpap_fig, use_container_width=True, key="mentorship_nicu_cpap_run_chart")

                                                    with st.expander("How this is computed", expanded=False):
                                                        st.markdown(
                                                            """
**Percentage** = (Patients with full bundle ÷ Total patients observed) × 100

**Early CPAP Bundle of Care – NICU**

1. bCPAP initiated at L&D or Operation theatre for inborn baby or within one hour of admission for outborn baby
2. bCPAP initiated at 6cm H20 and Fio2 adjusted according to patient need
3. Baby's vitals (RR, HR, Temp, SPO2) are monitored at 15-20mins after initiation and at every hour until stable then 3-4hourly
4. Baby's airway is patent and correctly positioned
5. Baby's oro-gastric tube in-situ and open (unless used for feeding)
6. Normal saline drops instilled into nostrils at least every 4 hours and recorded
                                                            """
                                                        )

                                        with nicu_cpap_care_delivered_tab:
                                            nicu_cpap_selected_weeks = st.multiselect(
                                                "Select Weeks",
                                                options=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                default=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                key="mentorship_nicu_cpap_care_weeks",
                                            )

                                            if not nicu_cpap_selected_weeks:
                                                st.info("Please select at least one week.")
                                            else:
                                                nicu_cpap_bundle_labels = {
                                                    "cpap_nicu-cpap1nicu": "bCPAP within 1hr of admission",
                                                    "cpap_nicu-cpap2nicu": "bCPAP at 6cm H2O, FiO2 adjusted",
                                                    "cpap_nicu-cpap3nicu": "Vitals monitored q15min→hourly",
                                                    "cpap_nicu-cpap4nicu": "Airway patent & positioned",
                                                    "cpap_nicu-cpap5nicu": "OG tube in-situ & open",
                                                    "cpap_nicu-cpap6nicu": "Normal saline drops q4h",
                                                }
                                                nicu_cpap_care_entity_col = "hospital"

                                                nicu_cpap_all_weeks_data = []
                                                for week_val in nicu_cpap_selected_weeks:
                                                    week_raw = nicu_scoped_work_df[nicu_scoped_work_df["week"] == week_val].copy()
                                                    if nicu_selected_round != "All Rounds":
                                                        week_raw = week_raw[week_raw["round"] == nicu_selected_round]
                                                    if nicu_selected_entities:
                                                        week_raw = week_raw[week_raw[nicu_entity_col].isin(nicu_selected_entities)]
                                                    if week_raw.empty:
                                                        continue
                                                    week_raw["_yes_count"] = week_raw[nicu_cpap_cols_present].sum(axis=1)
                                                    _idx = week_raw.groupby(["card", nicu_cpap_care_entity_col])["_yes_count"].idxmax()
                                                    card_data = week_raw.loc[_idx, ["card", nicu_cpap_care_entity_col] + nicu_cpap_cols_present].reset_index(drop=True)
                                                    card_data["care_pct_actual"] = card_data[nicu_cpap_cols_present].sum(axis=1) / 6 * 100
                                                    card_data["care_pct"] = card_data["care_pct_actual"].clip(lower=0.5)
                                                    card_data["card_label"] = card_data.apply(
                                                        lambda r: f"Card {int(r['card'])}", axis=1
                                                    )
                                                    card_data["week"] = f"Week {week_val}"

                                                    bundle_vals = []
                                                    for _, r in card_data.iterrows():
                                                        lines = [f"<b>Card {int(r['card'])}</b>"]
                                                        for col in nicu_cpap_cols_present:
                                                            val = int(r[col])
                                                            lines.append(f"{nicu_cpap_bundle_labels[col]}: {'Yes' if val else 'No'}")
                                                        bundle_vals.append("<br>".join(lines))
                                                    card_data["bundle_detail"] = bundle_vals

                                                    nicu_cpap_all_weeks_data.append(card_data)

                                                if not nicu_cpap_all_weeks_data:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_cpap_plot_df = pd.concat(nicu_cpap_all_weeks_data, ignore_index=True)
                                                    nicu_cpap_fig2 = px.bar(
                                                        nicu_cpap_plot_df,
                                                        x=nicu_cpap_care_entity_col,
                                                        y="care_pct",
                                                        color="card_label",
                                                        barmode="group",
                                                        facet_col="week",
                                                        facet_col_wrap=2,
                                                        color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3,
                                                        custom_data=["bundle_detail", "care_pct_actual"],
                                                        labels={nicu_cpap_care_entity_col: "Facility", "care_pct": "%"},
                                                    )
                                                    nicu_cpap_fig2.update_traces(
                                                        hovertemplate=(
                                                            "<b>%{x}</b><br>"
                                                            + "%{customdata[0]}<br>"
                                                            + "Care delivered: %{customdata[1]:.1f}%"
                                                            + "<extra></extra>"
                                                        ),
                                                        marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
                                                    )

                                                    nicu_cpap_fig2.update_layout(
                                                        template="plotly_white",
                                                        height=max(300, len(nicu_cpap_selected_weeks) * 200),
                                                        margin=dict(l=8, r=8, t=48, b=8),
                                                        font=dict(size=9),
                                                        hoverlabel=dict(font_size=9),
                                                        legend=dict(
                                                            orientation="h",
                                                            yanchor="bottom",
                                                            y=1.02,
                                                            xanchor="left",
                                                            x=0,
                                                            font=dict(size=8),
                                                            title=dict(text="Card", font=dict(size=8)),
                                                        ),
                                                    )
                                                    nicu_cpap_fig2.for_each_annotation(lambda a: a.update(
                                                        text=a.text.split("=")[-1].strip(),
                                                        font=dict(size=12, color="black"),
                                                    ))
                                                    nicu_cpap_fig2.update_xaxes(
                                                        matches=None,
                                                        showticklabels=True,
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        tickangle=45,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    nicu_cpap_fig2.update_yaxes(
                                                        range=[0, 110],
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    st.plotly_chart(nicu_cpap_fig2, use_container_width=True, key="mentorship_nicu_cpap_care_delivered")

                                                with st.expander("How this is computed", expanded=False):
                                                    st.markdown(
                                                        f"""
**Percentage of care delivered per patient** = (Bundle items = Yes ÷ 6) × 100

For each patient (card), all 6 bundle items are evaluated. Missing values are treated as **No (0)** for consistency.

**Early CPAP Bundle Variables – NICU**

| Variable | Description |
|---|---|
| cpap_nicu-cpap1nicu | bCPAP initiated at L&D or Operation theatre for inborn baby or within one hour of admission for outborn baby |
| cpap_nicu-cpap2nicu | bCPAP initiated at 6cm H20 and Fio2 adjusted according to patient need |
| cpap_nicu-cpap3nicu | Baby's vitals (RR, HR, Temp, SPO2) are monitored at 15-20mins after initiation and at every hour until stable then 3-4hourly |
| cpap_nicu-cpap4nicu | Baby's airway is patent and correctly positioned |
| cpap_nicu-cpap5nicu | Baby's oro-gastric tube in-situ and open (unless used for feeding) |
| cpap_nicu-cpap6nicu | Normal saline drops instilled into nostrils at least every 4 hours and recorded |
                                                        """
                                                    )

                                    with nicu_kmc_tab:
                                        nicu_kmc_tab_cols_present = [c for c in ["kmc-kmc1", "kmc-kmc2", "kmc-kmc3", "kmc-kmc4", "kmc-kmc5", "kmc-kmc6"] if c in nicu_work_df.columns]
                                        nicu_kmc_all_or_none_tab, nicu_kmc_care_delivered_tab = st.tabs([
                                            "Percentage of KMC All or None Bundle of care provided",
                                            "Percentage of care delivered",
                                        ])

                                        with nicu_kmc_all_or_none_tab:
                                            if len(nicu_kmc_tab_cols_present) < 6:
                                                st.info("KMC columns not fully available in NICU data.")
                                            else:
                                                nicu_kmc_round_info = nicu_scoped_work_df[["week", "card", "region_label", "hospital", "round"]].drop_duplicates(
                                                    subset=["week", "card", "region_label", "hospital"]
                                                )
                                                nicu_kmc_all_or_none = compute_all_or_none(nicu_scoped_work_df, bundle_cols=nicu_kmc_tab_cols_present)
                                                nicu_kmc_all_or_none = nicu_kmc_all_or_none.merge(
                                                    nicu_kmc_round_info, on=["week", "card", "region_label", "hospital"], how="left"
                                                )

                                                nicu_kmc_filtered = nicu_kmc_all_or_none.copy()
                                                if nicu_selected_round != "All Rounds":
                                                    nicu_kmc_filtered = nicu_kmc_filtered[nicu_kmc_filtered["round"] == nicu_selected_round]
                                                if nicu_selected_entities:
                                                    nicu_kmc_filtered = nicu_kmc_filtered[nicu_kmc_filtered[nicu_entity_col].isin(nicu_selected_entities)]
                                                else:
                                                    nicu_kmc_filtered = nicu_kmc_filtered.iloc[0:0]

                                                if nicu_kmc_filtered.empty:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_kmc_show_card_details = (nicu_group_mode == "Regional" and len(nicu_selected_entities) == 1 and nicu_entity_col == "hospital")

                                                    if nicu_kmc_show_card_details:
                                                        nicu_kmc_card_hover = nicu_kmc_filtered.groupby("week").apply(
                                                            lambda g: "<br>".join(
                                                                f"Card {int(row['card'])}: {int(row['all_or_none'])}"
                                                                for _, row in g.iterrows()
                                                            )
                                                        ).reset_index(name="card_details")

                                                    nicu_kmc_weekly_stats = nicu_kmc_filtered.groupby("week").agg(
                                                        total_cards=("all_or_none", "count"),
                                                        compliant=("all_or_none", "sum"),
                                                    ).reset_index()
                                                    nicu_kmc_weekly_stats["percentage"] = np.where(
                                                        nicu_kmc_weekly_stats["total_cards"] > 0,
                                                        (nicu_kmc_weekly_stats["compliant"] / nicu_kmc_weekly_stats["total_cards"]) * 100,
                                                        0.0,
                                                    )
                                                    if nicu_kmc_show_card_details:
                                                        nicu_kmc_weekly_stats = nicu_kmc_weekly_stats.merge(nicu_kmc_card_hover, on="week", how="left")
                                                    nicu_kmc_weekly_stats = nicu_kmc_weekly_stats.sort_values("week")

                                                    nicu_kmc_week_display_map = {w: f"Week {int(w)}" for w in sorted(nicu_kmc_weekly_stats["week"].unique())}
                                                    nicu_kmc_weekly_stats["week_label"] = nicu_kmc_weekly_stats["week"].map(nicu_kmc_week_display_map)

                                                    nicu_kmc_median_val = np.median(nicu_kmc_weekly_stats["percentage"]) if len(nicu_kmc_weekly_stats) > 0 else 0

                                                    nicu_kmc_ci_total = 0
                                                    nicu_kmc_ci_compliant = 1
                                                    nicu_kmc_customdata_cols = ["total_cards", "compliant"]
                                                    if nicu_kmc_show_card_details:
                                                        nicu_kmc_customdata_cols = ["card_details", "total_cards", "compliant"]
                                                        nicu_kmc_ci_total = 1
                                                        nicu_kmc_ci_compliant = 2

                                                    nicu_kmc_hover_parts = [
                                                        "<b>%{x}</b><br>",
                                                        f"Patients with full bundle: %{{customdata[{nicu_kmc_ci_compliant}]:.0f}}<br>",
                                                        f"Total patients observed: %{{customdata[{nicu_kmc_ci_total}]:.0f}}<br>",
                                                        "% of KMC All or None: %{y:.1f}%<br>",
                                                        f"Median: {nicu_kmc_median_val:.1f}%",
                                                    ]
                                                    if nicu_kmc_show_card_details:
                                                        nicu_kmc_hover_parts.append("<br><br><b>All-or-None</b><br>%{customdata[0]}")
                                                    nicu_kmc_hover_parts.append("<extra></extra>")

                                                    nicu_kmc_fig = go.Figure()
                                                    nicu_kmc_fig.add_trace(go.Scatter(
                                                        x=nicu_kmc_weekly_stats["week_label"],
                                                        y=nicu_kmc_weekly_stats["percentage"],
                                                        mode="lines+markers",
                                                        name="KMC All-or-None Bundle",
                                                        marker=dict(size=8, symbol="circle", color="#2563eb", line=dict(color="#2563eb", width=1)),
                                                        line=dict(color="#2563eb", width=2.5),
                                                        customdata=nicu_kmc_weekly_stats[nicu_kmc_customdata_cols].fillna("").values,
                                                        hovertemplate="".join(nicu_kmc_hover_parts),
                                                    ))
                                                    nicu_kmc_fig.add_hline(
                                                        y=nicu_kmc_median_val,
                                                        line=dict(color="#e91e9e", width=2.5, dash="solid"),
                                                        annotation_text=f"Median: {nicu_kmc_median_val:.1f}%",
                                                        annotation_font_size=10,
                                                        annotation_position="right",
                                                    )
                                                    nicu_kmc_fig.add_trace(go.Scatter(
                                                        x=[None], y=[None],
                                                        mode="lines",
                                                        name=f"Median: {nicu_kmc_median_val:.1f}%",
                                                        line=dict(color="#e91e9e", width=2.5),
                                                        showlegend=True,
                                                    ))

                                                    nicu_kmc_fig.update_layout(
                                                        template="plotly_white",
                                                        height=290,
                                                        margin=dict(l=8, r=8, t=20, b=8),
                                                        xaxis_title=dict(text="Week", font=dict(size=11)),
                                                        yaxis_title=dict(text="Percentage (%)", font=dict(size=11)),
                                                        font=dict(size=10),
                                                        hoverlabel=dict(font_size=10, namelength=-1),
                                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                                        yaxis=dict(range=[0, 110], tickformat=".0f", dtick=20),
                                                    )
                                                    nicu_kmc_sorted_week_labels = [f"Week {int(w)}" for w in sorted(nicu_kmc_weekly_stats["week"].unique())]
                                                    nicu_kmc_fig.update_xaxes(tickfont=dict(size=9), automargin=True, categoryorder="array", categoryarray=nicu_kmc_sorted_week_labels)
                                                    nicu_kmc_fig.update_yaxes(tickfont=dict(size=9), automargin=True, gridcolor="#f0f0f0")

                                                    st.plotly_chart(nicu_kmc_fig, use_container_width=True, key="mentorship_nicu_kmc_run_chart")

                                                    with st.expander("How this is computed", expanded=False):
                                                        st.markdown(
                                                            """
**Percentage** = (Patients with full bundle ÷ Total patients observed) × 100

**KMC Bundle of Care – NICU**

1. Eligible baby has received skin-to-skin contact on same day of birth at NICU/iKMC/KMC ward.
2. The baby is skin to skin more than 8hours with in 24hrs
3. Baby is secured firmly to the mother's or care giver's chest with a binder that ensures a patent airway.
4. Baby is being breast fed every 2-3hours and a record of this maintained
5. Baby vitals are monitored and recorded at least four times per day
6. Baby is weighed daily, and weight gain progress recorded and analyzed
                                                            """
                                                        )

                                        with nicu_kmc_care_delivered_tab:
                                            nicu_kmc_selected_weeks = st.multiselect(
                                                "Select Weeks",
                                                options=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                default=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                key="mentorship_nicu_kmc_care_weeks",
                                            )

                                            if not nicu_kmc_selected_weeks:
                                                st.info("Please select at least one week.")
                                            else:
                                                nicu_kmc_bundle_labels = {
                                                    "kmc-kmc1": "Skin-to-skin on day of birth",
                                                    "kmc-kmc2": "Skin-to-skin >8hrs in 24hrs",
                                                    "kmc-kmc3": "Secured w/ binder, patent airway",
                                                    "kmc-kmc4": "Breast fed q2-3h, recorded",
                                                    "kmc-kmc5": "Vitals monitored 4x/day",
                                                    "kmc-kmc6": "Daily weight & gain recorded",
                                                }
                                                nicu_kmc_care_entity_col = "hospital"

                                                nicu_kmc_all_weeks_data = []
                                                for week_val in nicu_kmc_selected_weeks:
                                                    week_raw = nicu_scoped_work_df[nicu_scoped_work_df["week"] == week_val].copy()
                                                    if nicu_selected_round != "All Rounds":
                                                        week_raw = week_raw[week_raw["round"] == nicu_selected_round]
                                                    if nicu_selected_entities:
                                                        week_raw = week_raw[week_raw[nicu_entity_col].isin(nicu_selected_entities)]
                                                    if week_raw.empty:
                                                        continue
                                                    week_raw["_yes_count"] = week_raw[nicu_kmc_tab_cols_present].sum(axis=1)
                                                    _idx = week_raw.groupby(["card", nicu_kmc_care_entity_col])["_yes_count"].idxmax()
                                                    card_data = week_raw.loc[_idx, ["card", nicu_kmc_care_entity_col] + nicu_kmc_tab_cols_present].reset_index(drop=True)
                                                    card_data["care_pct_actual"] = card_data[nicu_kmc_tab_cols_present].sum(axis=1) / 6 * 100
                                                    card_data["care_pct"] = card_data["care_pct_actual"].clip(lower=0.5)
                                                    card_data["card_label"] = card_data.apply(
                                                        lambda r: f"Card {int(r['card'])}", axis=1
                                                    )
                                                    card_data["week"] = f"Week {week_val}"

                                                    bundle_vals = []
                                                    for _, r in card_data.iterrows():
                                                        lines = [f"<b>Card {int(r['card'])}</b>"]
                                                        for col in nicu_kmc_tab_cols_present:
                                                            val = int(r[col])
                                                            lines.append(f"{nicu_kmc_bundle_labels[col]}: {'Yes' if val else 'No'}")
                                                        bundle_vals.append("<br>".join(lines))
                                                    card_data["bundle_detail"] = bundle_vals

                                                    nicu_kmc_all_weeks_data.append(card_data)

                                                if not nicu_kmc_all_weeks_data:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_kmc_plot_df = pd.concat(nicu_kmc_all_weeks_data, ignore_index=True)
                                                    nicu_kmc_fig2 = px.bar(
                                                        nicu_kmc_plot_df,
                                                        x=nicu_kmc_care_entity_col,
                                                        y="care_pct",
                                                        color="card_label",
                                                        barmode="group",
                                                        facet_col="week",
                                                        facet_col_wrap=2,
                                                        color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3,
                                                        custom_data=["bundle_detail", "care_pct_actual"],
                                                        labels={nicu_kmc_care_entity_col: "Facility", "care_pct": "%"},
                                                    )
                                                    nicu_kmc_fig2.update_traces(
                                                        hovertemplate=(
                                                            "<b>%{x}</b><br>"
                                                            + "%{customdata[0]}<br>"
                                                            + "Care delivered: %{customdata[1]:.1f}%"
                                                            + "<extra></extra>"
                                                        ),
                                                        marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
                                                    )
                                                    nicu_kmc_fig2.update_layout(
                                                        template="plotly_white",
                                                        height=max(300, len(nicu_kmc_selected_weeks) * 200),
                                                        margin=dict(l=8, r=8, t=48, b=8),
                                                        font=dict(size=9),
                                                        hoverlabel=dict(font_size=9),
                                                        legend=dict(
                                                            orientation="h",
                                                            yanchor="bottom",
                                                            y=1.02,
                                                            xanchor="left",
                                                            x=0,
                                                            font=dict(size=8),
                                                            title=dict(text="Card", font=dict(size=8)),
                                                        ),
                                                    )
                                                    nicu_kmc_fig2.for_each_annotation(lambda a: a.update(
                                                        text=a.text.split("=")[-1].strip(),
                                                        font=dict(size=12, color="black"),
                                                    ))
                                                    nicu_kmc_fig2.update_xaxes(
                                                        matches=None,
                                                        showticklabels=True,
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        tickangle=45,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    nicu_kmc_fig2.update_yaxes(
                                                        range=[0, 110],
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    st.plotly_chart(nicu_kmc_fig2, use_container_width=True, key="mentorship_nicu_kmc_care_delivered")

                                                with st.expander("How this is computed", expanded=False):
                                                    st.markdown(
                                                        f"""
**Percentage of care delivered per patient** = (Bundle items = Yes ÷ 6) × 100

For each patient (card), all 6 bundle items are evaluated. Missing values are treated as **No (0)** for consistency.

**KMC Bundle Variables – NICU**

| Variable | Description |
|---|---|
| kmc-kmc1 | Eligible baby has received skin-to-skin contact on same day of birth at NICU/iKMC/KMC ward |
| kmc-kmc2 | The baby is skin to skin more than 8hours with in 24hrs |
| kmc-kmc3 | Baby is secured firmly to the mother's or care giver's chest with a binder that ensures a patent airway |
| kmc-kmc4 | Baby is being breast fed every 2-3hours and a record of this maintained |
| kmc-kmc5 | Baby vitals are monitored and recorded at least four times per day |
| kmc-kmc6 | Baby is weighed daily, and weight gain progress recorded and analyzed |
                                                        """
                                                    )

                                    with nicu_nutrition_tab:
                                        nicu_nutrition_tab_cols_present = [c for c in ["Referral-referral1", "Referral-referral2", "Referral-referral3", "Referral-referral4", "Referral-referral5"] if c in nicu_work_df.columns]
                                        nicu_nutrition_all_or_none_tab, nicu_nutrition_care_delivered_tab = st.tabs([
                                            "Percentage of Nutrition All or None Bundle of care provided",
                                            "Percentage of care delivered",
                                        ])

                                        with nicu_nutrition_all_or_none_tab:
                                            if len(nicu_nutrition_tab_cols_present) < 5:
                                                st.info("Nutrition columns not fully available in NICU data.")
                                            else:
                                                nicu_nutrition_round_info = nicu_scoped_work_df[["week", "card", "region_label", "hospital", "round"]].drop_duplicates(
                                                    subset=["week", "card", "region_label", "hospital"]
                                                )
                                                nicu_nutrition_all_or_none = compute_all_or_none(nicu_scoped_work_df, bundle_cols=nicu_nutrition_tab_cols_present)
                                                nicu_nutrition_all_or_none = nicu_nutrition_all_or_none.merge(
                                                    nicu_nutrition_round_info, on=["week", "card", "region_label", "hospital"], how="left"
                                                )

                                                nicu_nutrition_filtered = nicu_nutrition_all_or_none.copy()
                                                if nicu_selected_round != "All Rounds":
                                                    nicu_nutrition_filtered = nicu_nutrition_filtered[nicu_nutrition_filtered["round"] == nicu_selected_round]
                                                if nicu_selected_entities:
                                                    nicu_nutrition_filtered = nicu_nutrition_filtered[nicu_nutrition_filtered[nicu_entity_col].isin(nicu_selected_entities)]
                                                else:
                                                    nicu_nutrition_filtered = nicu_nutrition_filtered.iloc[0:0]

                                                if nicu_nutrition_filtered.empty:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_nutrition_show_card_details = (nicu_group_mode == "Regional" and len(nicu_selected_entities) == 1 and nicu_entity_col == "hospital")

                                                    if nicu_nutrition_show_card_details:
                                                        nicu_nutrition_card_hover = nicu_nutrition_filtered.groupby("week").apply(
                                                            lambda g: "<br>".join(
                                                                f"Card {int(row['card'])}: {int(row['all_or_none'])}"
                                                                for _, row in g.iterrows()
                                                            )
                                                        ).reset_index(name="card_details")

                                                    nicu_nutrition_weekly_stats = nicu_nutrition_filtered.groupby("week").agg(
                                                        total_cards=("all_or_none", "count"),
                                                        compliant=("all_or_none", "sum"),
                                                    ).reset_index()
                                                    nicu_nutrition_weekly_stats["percentage"] = np.where(
                                                        nicu_nutrition_weekly_stats["total_cards"] > 0,
                                                        (nicu_nutrition_weekly_stats["compliant"] / nicu_nutrition_weekly_stats["total_cards"]) * 100,
                                                        0.0,
                                                    )
                                                    if nicu_nutrition_show_card_details:
                                                        nicu_nutrition_weekly_stats = nicu_nutrition_weekly_stats.merge(nicu_nutrition_card_hover, on="week", how="left")
                                                    nicu_nutrition_weekly_stats = nicu_nutrition_weekly_stats.sort_values("week")

                                                    nicu_nutrition_week_display_map = {w: f"Week {int(w)}" for w in sorted(nicu_nutrition_weekly_stats["week"].unique())}
                                                    nicu_nutrition_weekly_stats["week_label"] = nicu_nutrition_weekly_stats["week"].map(nicu_nutrition_week_display_map)

                                                    nicu_nutrition_median_val = np.median(nicu_nutrition_weekly_stats["percentage"]) if len(nicu_nutrition_weekly_stats) > 0 else 0

                                                    nicu_nutrition_ci_total = 0
                                                    nicu_nutrition_ci_compliant = 1
                                                    nicu_nutrition_customdata_cols = ["total_cards", "compliant"]
                                                    if nicu_nutrition_show_card_details:
                                                        nicu_nutrition_customdata_cols = ["card_details", "total_cards", "compliant"]
                                                        nicu_nutrition_ci_total = 1
                                                        nicu_nutrition_ci_compliant = 2

                                                    nicu_nutrition_hover_parts = [
                                                        "<b>%{x}</b><br>",
                                                        f"Patients with full bundle: %{{customdata[{nicu_nutrition_ci_compliant}]:.0f}}<br>",
                                                        f"Total patients observed: %{{customdata[{nicu_nutrition_ci_total}]:.0f}}<br>",
                                                        "% of Nutrition All or None: %{y:.1f}%<br>",
                                                        f"Median: {nicu_nutrition_median_val:.1f}%",
                                                    ]
                                                    if nicu_nutrition_show_card_details:
                                                        nicu_nutrition_hover_parts.append("<br><br><b>All-or-None</b><br>%{customdata[0]}")
                                                    nicu_nutrition_hover_parts.append("<extra></extra>")

                                                    nicu_nutrition_fig = go.Figure()
                                                    nicu_nutrition_fig.add_trace(go.Scatter(
                                                        x=nicu_nutrition_weekly_stats["week_label"],
                                                        y=nicu_nutrition_weekly_stats["percentage"],
                                                        mode="lines+markers",
                                                        name="Nutrition All-or-None Bundle",
                                                        marker=dict(size=8, symbol="circle", color="#2563eb", line=dict(color="#2563eb", width=1)),
                                                        line=dict(color="#2563eb", width=2.5),
                                                        customdata=nicu_nutrition_weekly_stats[nicu_nutrition_customdata_cols].fillna("").values,
                                                        hovertemplate="".join(nicu_nutrition_hover_parts),
                                                    ))
                                                    nicu_nutrition_fig.add_hline(
                                                        y=nicu_nutrition_median_val,
                                                        line=dict(color="#e91e9e", width=2.5, dash="solid"),
                                                        annotation_text=f"Median: {nicu_nutrition_median_val:.1f}%",
                                                        annotation_font_size=10,
                                                        annotation_position="right",
                                                    )
                                                    nicu_nutrition_fig.add_trace(go.Scatter(
                                                        x=[None], y=[None],
                                                        mode="lines",
                                                        name=f"Median: {nicu_nutrition_median_val:.1f}%",
                                                        line=dict(color="#e91e9e", width=2.5),
                                                        showlegend=True,
                                                    ))

                                                    nicu_nutrition_fig.update_layout(
                                                        template="plotly_white",
                                                        height=290,
                                                        margin=dict(l=8, r=8, t=20, b=8),
                                                        xaxis_title=dict(text="Week", font=dict(size=11)),
                                                        yaxis_title=dict(text="Percentage (%)", font=dict(size=11)),
                                                        font=dict(size=10),
                                                        hoverlabel=dict(font_size=10, namelength=-1),
                                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=9)),
                                                        yaxis=dict(range=[0, 110], tickformat=".0f", dtick=20),
                                                    )
                                                    nicu_nutrition_sorted_week_labels = [f"Week {int(w)}" for w in sorted(nicu_nutrition_weekly_stats["week"].unique())]
                                                    nicu_nutrition_fig.update_xaxes(tickfont=dict(size=9), automargin=True, categoryorder="array", categoryarray=nicu_nutrition_sorted_week_labels)
                                                    nicu_nutrition_fig.update_yaxes(tickfont=dict(size=9), automargin=True, gridcolor="#f0f0f0")

                                                    st.plotly_chart(nicu_nutrition_fig, use_container_width=True, key="mentorship_nicu_nutrition_run_chart")

                                                    with st.expander("How this is computed", expanded=False):
                                                        st.markdown(
                                                            """
**Percentage** = (Patients with full bundle ÷ Total patients observed) × 100

**Nutrition Bundle of Care – NICU**

1. Feeding initiated as early as clinically appropriate (preferably within the first hour for stable newborns).
2. Appropriate Feeding type and methods used based on baby age and condition (Exclusive or Expressed breast milk, cup feeding, gavage/NG feeding).
3. The Baby is breast fed every 2-3hours
4. Feeding plan is prescribed and administered accordingly
5. Feeding volume, frequency and tolerance monitored and recorded after every feeding
                                                            """
                                                        )

                                        with nicu_nutrition_care_delivered_tab:
                                            nicu_nutrition_selected_weeks = st.multiselect(
                                                "Select Weeks",
                                                options=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                default=sorted([int(w) for w in nicu_scoped_work_df["week"].dropna().unique() if str(w).strip()]),
                                                key="mentorship_nicu_nutrition_care_weeks",
                                            )

                                            if not nicu_nutrition_selected_weeks:
                                                st.info("Please select at least one week.")
                                            else:
                                                nicu_nutrition_bundle_labels = {
                                                    "Referral-referral1": "Feeding within 1hr",
                                                    "Referral-referral2": "Appropriate type & method",
                                                    "Referral-referral3": "Breast fed q2-3h",
                                                    "Referral-referral4": "Feeding plan prescribed",
                                                    "Referral-referral5": "Volume, freq, tolerance rec",
                                                }
                                                nicu_nutrition_care_entity_col = "hospital"

                                                nicu_nutrition_all_weeks_data = []
                                                for week_val in nicu_nutrition_selected_weeks:
                                                    week_raw = nicu_scoped_work_df[nicu_scoped_work_df["week"] == week_val].copy()
                                                    if nicu_selected_round != "All Rounds":
                                                        week_raw = week_raw[week_raw["round"] == nicu_selected_round]
                                                    if nicu_selected_entities:
                                                        week_raw = week_raw[week_raw[nicu_entity_col].isin(nicu_selected_entities)]
                                                    if week_raw.empty:
                                                        continue
                                                    week_raw["_yes_count"] = week_raw[nicu_nutrition_tab_cols_present].sum(axis=1)
                                                    _idx = week_raw.groupby(["card", nicu_nutrition_care_entity_col])["_yes_count"].idxmax()
                                                    card_data = week_raw.loc[_idx, ["card", nicu_nutrition_care_entity_col] + nicu_nutrition_tab_cols_present].reset_index(drop=True)
                                                    card_data["care_pct_actual"] = card_data[nicu_nutrition_tab_cols_present].sum(axis=1) / 5 * 100
                                                    card_data["care_pct"] = card_data["care_pct_actual"].clip(lower=0.5)
                                                    card_data["card_label"] = card_data.apply(
                                                        lambda r: f"Card {int(r['card'])}", axis=1
                                                    )
                                                    card_data["week"] = f"Week {week_val}"

                                                    bundle_vals = []
                                                    for _, r in card_data.iterrows():
                                                        lines = [f"<b>Card {int(r['card'])}</b>"]
                                                        for col in nicu_nutrition_tab_cols_present:
                                                            val = int(r[col])
                                                            lines.append(f"{nicu_nutrition_bundle_labels[col]}: {'Yes' if val else 'No'}")
                                                        bundle_vals.append("<br>".join(lines))
                                                    card_data["bundle_detail"] = bundle_vals

                                                    nicu_nutrition_all_weeks_data.append(card_data)

                                                if not nicu_nutrition_all_weeks_data:
                                                    st.info("No data for the selected filter combination.")
                                                else:
                                                    nicu_nutrition_plot_df = pd.concat(nicu_nutrition_all_weeks_data, ignore_index=True)
                                                    nicu_nutrition_fig2 = px.bar(
                                                        nicu_nutrition_plot_df,
                                                        x=nicu_nutrition_care_entity_col,
                                                        y="care_pct",
                                                        color="card_label",
                                                        barmode="group",
                                                        facet_col="week",
                                                        facet_col_wrap=2,
                                                        color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3,
                                                        custom_data=["bundle_detail", "care_pct_actual"],
                                                        labels={nicu_nutrition_care_entity_col: "Facility", "care_pct": "%"},
                                                    )
                                                    nicu_nutrition_fig2.update_traces(
                                                        hovertemplate=(
                                                            "<b>%{x}</b><br>"
                                                            + "%{customdata[0]}<br>"
                                                            + "Care delivered: %{customdata[1]:.1f}%"
                                                            + "<extra></extra>"
                                                        ),
                                                        marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
                                                    )
                                                    nicu_nutrition_fig2.update_layout(
                                                        template="plotly_white",
                                                        height=max(300, len(nicu_nutrition_selected_weeks) * 200),
                                                        margin=dict(l=8, r=8, t=48, b=8),
                                                        font=dict(size=9),
                                                        hoverlabel=dict(font_size=9),
                                                        legend=dict(
                                                            orientation="h",
                                                            yanchor="bottom",
                                                            y=1.02,
                                                            xanchor="left",
                                                            x=0,
                                                            font=dict(size=8),
                                                            title=dict(text="Card", font=dict(size=8)),
                                                        ),
                                                    )
                                                    nicu_nutrition_fig2.for_each_annotation(lambda a: a.update(
                                                        text=a.text.split("=")[-1].strip(),
                                                        font=dict(size=12, color="black"),
                                                    ))
                                                    nicu_nutrition_fig2.update_xaxes(
                                                        matches=None,
                                                        showticklabels=True,
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        tickangle=45,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    nicu_nutrition_fig2.update_yaxes(
                                                        range=[0, 110],
                                                        tickfont=dict(size=8),
                                                        automargin=True,
                                                        showline=True,
                                                        linewidth=1,
                                                        linecolor="lightgray",
                                                        mirror=True,
                                                    )
                                                    st.plotly_chart(nicu_nutrition_fig2, use_container_width=True, key="mentorship_nicu_nutrition_care_delivered")

                                                with st.expander("How this is computed", expanded=False):
                                                    st.markdown(
                                                        f"""
**Percentage of care delivered per patient** = (Bundle items = Yes ÷ 5) × 100

For each patient (card), all 5 bundle items are evaluated. Missing values are treated as **No (0)** for consistency.

**Nutrition Bundle Variables – NICU**

| Variable | Description |
|---|---|
| Referral-referral1 | Feeding initiated as early as clinically appropriate (preferably within the first hour for stable newborns) |
| Referral-referral2 | Appropriate Feeding type and methods used based on baby age and condition (Exclusive or Expressed breast milk, cup feeding, gavage/NG feeding) |
| Referral-referral3 | The Baby is breast fed every 2-3hours |
| Referral-referral4 | Feeding plan is prescribed and administered accordingly |
| Referral-referral5 | Feeding volume, frequency and tolerance monitored and recorded after every feeding |
                                                        """
                                                    )


def display_odk_dashboard(user: dict = None):
    """
    Display simplified forms dashboard with downloadable CSV files.
    Automatically loads data on first render.
    """
    st.markdown(
        """
    <style>
    .form-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .form-card h3 {
        color: white;
        margin: 0 0 10px 0;
    }
    .form-id {
        font-size: 0.8em;
        opacity: 0.9;
        font-family: monospace;
        background: rgba(255,255,255,0.2);
        padding: 2px 8px;
        border-radius: 10px;
        display: inline-block;
        margin-top: 5px;
    }
    .stats-badge {
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 0.8em;
        margin-right: 8px;
    }
    .refresh-info {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .auto-load-info {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .action-buttons-container {
        display: flex;
        gap: 10px;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Get current user info directly from session state
    current_user = st.session_state.get("user", {})
    if not current_user:
        st.warning("Please log in to access data")
        return

    current_user_id = current_user.get("id", "anonymous")
    current_username = current_user.get("username", "anonymous")
    current_region_id = current_user.get("region_id")
    current_role = current_user.get("role", "anonymous")

    st.markdown(
        """
        <style>
        .st-key-mentorship_section_selector [data-testid="stRadio"] {
            padding: 0.55rem 0.7rem;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            background: #f8fafc;
            margin-bottom: 0 !important;
        }
        .st-key-mentorship_section_selector [data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] {
            gap: 0.45rem;
        }
        .st-key-mentorship_section_selector [data-testid="stRadio"] label {
            margin: 0 !important;
            padding: 0.55rem 1rem;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            background: #ffffff;
            font-size: 1.0rem !important;
            font-weight: 800 !important;
        }
        .st-key-mentorship_section_selector [data-testid="stRadio"] label:has(input:checked) {
            border-color: #1d4ed8;
            color: #ffffff !important;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.25);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-mentorship_section_selector) + div[data-testid="stVerticalBlockBorderWrapper"] {
            margin-top: -0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    section_choice = st.radio(
        "Mentorship Section",
        options=["Mentorship Data Analysis", "Mentorship Data"],
        horizontal=True,
        key="mentorship_section_selector",
        label_visibility="collapsed",
    )
    if section_choice == "Mentorship Data Analysis":
        render_mentorship_analysis_dashboard()
        return

    mentorship_data_loaded_key = f"mentorship_data_loaded_{current_user_id}"
    if mentorship_data_loaded_key not in st.session_state:
        st.session_state[mentorship_data_loaded_key] = False

    if not st.session_state[mentorship_data_loaded_key]:
        st.markdown(
            """
            <div style="text-align: center; padding: 2.2rem 1rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                 border-radius: 12px; border: 2px dashed #dee2e6; margin: 0.8rem 0 1rem 0;">
                <h3 style="color: #495057; margin-bottom: 0.6rem;">Mentorship Data</h3>
                <p style="color: #6c757d; font-size: 1rem; max-width: 700px; margin: 0 auto;">
                    ODK mentorship forms are loaded only when needed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "Load Mentorship Data",
                use_container_width=True,
                type="primary",
                key=f"load_mentorship_data_{current_user_id}",
            ):
                st.session_state[mentorship_data_loaded_key] = True
                st.rerun()
        return

    is_afar_user = False
    if current_role == "regional":
        try:
            is_afar_user = int(current_region_id) == AFAR_REGION_ID
        except (TypeError, ValueError):
            is_afar_user = False

    # Create session state keys
    odk_data_key = f"odk_forms_data_{current_user_id}"
    afar_odk_data_key = f"odk_forms_data_afar_{current_user_id}"
    last_refresh_key = f"last_odk_refresh_{current_user_id}"
    user_tracker_key = "current_odk_user"

    # Check if user has changed
    current_user_info = f"{current_user_id}_{current_region_id}_{current_role}"

    if user_tracker_key not in st.session_state:
        st.session_state[user_tracker_key] = current_user_info
    else:
        previous_user_info = st.session_state[user_tracker_key]
        if previous_user_info != current_user_info:
            # User changed - clear old data
            st.info(f"Loading data for {current_username}...")

            # Clear ALL session data
            for key in list(st.session_state.keys()):
                if key.startswith("odk_forms_data_") or key.startswith(
                    "last_odk_refresh_"
                ):
                    del st.session_state[key]

            st.session_state[user_tracker_key] = current_user_info
            st.rerun()

    # Initialize session state
    if last_refresh_key not in st.session_state:
        st.session_state[last_refresh_key] = None

    if odk_data_key not in st.session_state:
        st.session_state[odk_data_key] = {}

    if afar_odk_data_key not in st.session_state:
        st.session_state[afar_odk_data_key] = {}

    # Header with action buttons
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<div class="main-header">Integrated Mentorship Data</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**All available forms are automatically loaded below**")

    with col2:
        st.markdown('<div class="action-buttons-container">', unsafe_allow_html=True)

        if st.button("Refresh Data", use_container_width=True, type="primary"):
            st.session_state[odk_data_key] = {}
            st.session_state[afar_odk_data_key] = {}
            st.session_state[last_refresh_key] = pd.Timestamp.now()
            st.rerun()

        # Download All button ALWAYS visible - no conditions
        if st.button("Download All", use_container_width=True, type="secondary"):
            has_data = (
                st.session_state.get(odk_data_key)
                and len(st.session_state[odk_data_key]) > 0
            )
            if has_data:
                download_all_forms(st.session_state[odk_data_key])
            else:
                st.warning("No data available to download. Please refresh data first.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Load data only if needed
    if (not st.session_state[odk_data_key]) or (
        is_afar_user and not st.session_state[afar_odk_data_key]
    ):
        with st.spinner("Loading forms data..."):
            try:
                # Fetch data once
                odk_data = fetch_odk_data_for_user(current_user)
                forms_data = odk_data.get("odk_forms", {})
                afar_forms_data = odk_data.get(AFAR_MENTORSHIP_SECTION_LABEL, {})

                st.session_state[odk_data_key] = forms_data
                st.session_state[afar_odk_data_key] = afar_forms_data
                st.session_state[last_refresh_key] = pd.Timestamp.now()

                st.success(f"Loaded {len(forms_data)} forms")

            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")

    # Show refresh info
    if st.session_state[last_refresh_key]:
        refresh_time = st.session_state[last_refresh_key].strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f'<div class="auto-load-info">Last refresh: {refresh_time}</div>',
            unsafe_allow_html=True,
        )

    # Display forms
    top_forms_data = dict(st.session_state.get(odk_data_key, {}))
    if is_afar_user and top_forms_data:
        for form_id in AFAR_MENTORSHIP_PROJECT14_FORM_IDS:
            top_forms_data.pop(form_id, None)

    if top_forms_data and len(top_forms_data) > 0:
        display_forms_grid(top_forms_data, key_prefix="mentorship")
    else:
        st.info("No forms data available. Click 'Refresh Data' to try again.")


    # Afar mentorship section (Project 17) - STRICTLY for Afar regional users only
    if is_afar_user:
        st.markdown("## IMNID Blended Mentorship Afar HC")

        afar_forms = st.session_state.get(afar_odk_data_key, {})
        if afar_forms and len(afar_forms) > 0:
            display_forms_grid(
                afar_forms,
                odk_project_id=AFAR_MENTORSHIP_ODK_PROJECT_ID,
                key_prefix="afar_mentorship",
            )
        else:
            st.info("No Afar mentorship forms data available.")


def display_forms_grid(
    forms_data: Dict[str, pd.DataFrame],
    *,
    odk_project_id: str | int | None = None,
    key_prefix: str = "odk",
):
    """Display all loaded forms in an attractive grid layout"""
    st.markdown(f"### Available Forms ({len(forms_data)})")

    consistent_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    cols = st.columns(2)

    for i, (form_id, df) in enumerate(forms_data.items()):
        # Get display name from cached forms
        forms = list_forms_cached(odk_project_id)
        form_metadata = next((f for f in forms if f.get("xmlFormId") == form_id), {})

        # Fallback: if form is from another project (e.g., selected Project 14 forms in Afar section),
        # resolve its display name from default project metadata.
        if not form_metadata and odk_project_id is not None:
            default_forms = list_forms_cached()
            form_metadata = next(
                (f for f in default_forms if f.get("xmlFormId") == form_id), {}
            )

        display_name = form_metadata.get("name", form_id)

        col = cols[i % 2]

        with col:
            display_form_card(
                form_id,
                df,
                consistent_color,
                i,
                display_name,
                key_prefix=key_prefix,
            )


def display_form_card(
    form_id: str,
    df: pd.DataFrame,
    color: str,
    index: int,
    display_name: str,
    *,
    key_prefix: str = "odk",
):
    """Display an individual form card with download button"""

    with st.container():
        st.markdown(
            f"""
        <div class="form-card" style="background: {color};">
            <h3>{display_name}</h3>
            <div class="form-id">({form_id})</div>
            <div style="display: flex; gap: 8px; margin-bottom: 15px; margin-top: 10px;">
                <span class="stats-badge">{len(df):,} records</span>
                <span class="stats-badge">{len(df.columns)} columns</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{form_id}.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_{index}",
            use_container_width=True,
            type="primary",
        )


def download_all_forms(forms_data: Dict[str, pd.DataFrame]):
    """Create a zip file with all forms for download"""
    import zipfile
    import io

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for form_id, df in forms_data.items():
            csv_data = convert_df_to_csv(df)
            zip_file.writestr(f"{form_id}.csv", csv_data)

    zip_buffer.seek(0)

    st.download_button(
        label="Download All as ZIP",
        data=zip_buffer,
        file_name="forms.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_zip",
    )


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string for download"""
    return df.to_csv(index=False, encoding="utf-8")
