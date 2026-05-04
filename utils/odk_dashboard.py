# utils/odk_dashboard.py
import streamlit as st
import pandas as pd
import logging
import time
import os
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

CIS_BUNDLE_INDICATORS = [
    (
        "cpap_nicu-cpap1nicu",
        "% CPAP within 1 hour",
        "bCPAP initiated at L&D or Operation theatre for inborn baby or within one hour of admission for outborn baby",
        "Started CPAP within 1 hour (Yes)",
        "Assessed for CPAP within 1 hour (Answered)",
    ),
    (
        "kmc-kmc2",
        "% KMC within 24 hours",
        "The baby is skin to skin more than 8hours within 24hrs",
        "Started KMC within 24 hours (Yes)",
        "Assessed for KMC within 24 hours (Answered)",
    ),
    (
        "NICU-nicu5",
        "% Normothermic at admission (36.5–37.5C)",
        "Baby's body temperature is between 36.5 and 37.5 degree Celsius at admission",
        "Normal temperature at admission (Yes)",
        "Admission temperature recorded (Answered)",
    ),
]


# 🔥 OPTIMIZATION: Cache forms listing for 1 hour
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


def render_mentorship_analysis_dashboard():
    """Render mentorship analysis for mentorship datasets."""
    st.markdown(
        """
    <style>
    .mentorship-analysis-shell {
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        background: linear-gradient(135deg, #eff6ff, #e0e7ff);
        padding: 0.7rem 0.9rem;
        margin: 0.3rem 0 0.8rem 0;
    }
    .mentorship-analysis-shell h4 {
        color: #1e3a8a;
        font-size: 1.02rem;
        margin: 0;
        font-weight: 800;
    }
    .mentorship-filter-box {
        margin-bottom: 10px;
    }
    .mentorship-filter-title {
        font-size: 1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .mentorship-filter-subtitle {
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 10px;
    }
    .mentorship-filter-divider {
        height: 1px;
        background: #bfdbfe;
        margin: 8px 0 12px 0;
    }
    .st-key-mentorship_filters_card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 3px solid #1d4ed8;
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
        box-shadow: 0 8px 18px rgba(2, 6, 23, 0.12);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mentorship-analysis-shell">
            <h4>Mentorship Data Analysis</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_cis_indicator = None
    selected_cis_indicator_col = None
    selected_cis_indicator_question = None
    selected_cis_numerator_label = None
    selected_cis_denominator_label = None

    left_col, right_col = st.columns([3, 2])

    with right_col:
        with st.container(key="mentorship_filters_card"):
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

            data_choice = st.selectbox(
                "Data to Analyze",
                options=[
                    "BMET Data",
                    "Skill Assessment Data",
                    "Quality of Care Data",
                    "QI Coaching Data",
                    "Data Mentorship Data",
                    "CIS Bundle of Care",
                ],
                key="mentorship_analysis_data_choice",
            )
            is_skill_analysis = data_choice == "Skill Assessment Data"
            is_qoc_analysis = data_choice == "Quality of Care Data"
            is_qi_coaching_analysis = data_choice == "QI Coaching Data"
            is_data_mentorship_analysis = data_choice == "Data Mentorship Data"
            is_bmet_analysis = data_choice == "BMET Data"
            is_cis_analysis = data_choice == "CIS Bundle of Care"

            if is_skill_analysis:
                try:
                    df = load_merged_skill_data()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:
                    st.error(f"Unable to load merged skill data: {exc}")
                    return

                score_cols = [
                    "POC1-POC1_perc",
                    "POC2-POC2_perc",
                    "POC3-POC3_perc",
                    "POC4-POC4_perc",
                    "POC5-POC5_perc",
                    "POC6-POC6_perc",
                    "POC7-POC7_perc",
                    "POC8-POC8_perc",
                    "POC_perc",
                ]
                required_columns = ["reg-region", "reg-hospital", "reg-round"] + score_cols
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns in merged_skill.csv: {missing_cols}")
                    return

                work_df = df.copy()
                work_df["region_code"] = work_df["reg-region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["reg-hospital"].apply(_normalize_entity_text)
                work_df["round"] = work_df["reg-round"].astype(str).str.strip()
                value_label = "Average %"
                value_title = "Average Score (%)"
            elif is_qoc_analysis:
                try:
                    df = load_merged_qoc_data()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:
                    st.error(f"Unable to load merged QoC data: {exc}")
                    return

                score_cols = [
                    "QOC1_b-QOC1_perce",
                    "QOC2_b-QOC2-QOC2_perce",
                    "QOC3_b-QOC3_perce",
                    "QOC4_b-QOC4_perce",
                    "QOC5_b-QOC5_gr5-QOC5_perce",
                    "QOC_perce",
                ]

                region_col = "region" if "region" in df.columns else "reg-region"
                hospital_col = "hospital" if "hospital" in df.columns else "reg-hospital"
                round_col = "round" if "round" in df.columns else "reg-round"
                unit_col = "unit"

                required_columns = [region_col, hospital_col, round_col, unit_col] + score_cols
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns in merged_qoc.csv: {missing_cols}")
                    return

                work_df = df.copy()
                work_df["region_code"] = work_df[region_col].apply(_normalize_region_code)
                work_df["hospital"] = work_df[hospital_col].apply(_normalize_entity_text)
                work_df["round"] = work_df[round_col].astype(str).str.strip()
                work_df["unit"] = (
                    work_df[unit_col]
                    .astype(str)
                    .str.strip()
                    .replace({"1.0": "1", "2.0": "2"})
                )
                # Unit semantics:
                # - "1" = Labor and Delivery
                # - "2" = NICU/KMC
                # - "1 2", "2 1", or "21" mean both (include in both filters)
                work_df["unit_has_1"] = work_df["unit"].astype(str).str.contains("1", regex=False)
                work_df["unit_has_2"] = work_df["unit"].astype(str).str.contains("2", regex=False)
                for score_col in score_cols:
                    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")
                value_label = "Average %"
                value_title = "Average Score (%)"
            elif is_qi_coaching_analysis:
                try:
                    df = load_merged_qi_coaching_data()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:
                    st.error(f"Unable to load merged QI coaching data: {exc}")
                    return

                indicator_source_cols = [spec[0] for spec in QI_COACHING_INDICATORS]
                score_cols = [spec[1] for spec in QI_COACHING_INDICATORS]
                required_columns = ["reg-region", "reg-hospital", "reg-round"] + indicator_source_cols
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    st.error(
                        "Missing required columns in merged_qi_coaching.csv: "
                        f"{missing_cols}"
                    )
                    return

                work_df = df.copy()
                work_df["region_code"] = work_df["reg-region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["reg-hospital"].apply(_normalize_entity_text)
                work_df["round"] = work_df["reg-round"].astype(str).str.strip()

                has_any_indicator_answer = ~work_df[indicator_source_cols].apply(_blank_like_mask)
                work_df = work_df[has_any_indicator_answer.any(axis=1)].copy()

                if work_df.empty:
                    st.warning(
                        "No QI Coaching submissions contain answered indicator questions."
                    )
                    return

                for source_col, metric_col, _ in QI_COACHING_INDICATORS:
                    if source_col == "part5_project-q502":
                        work_df[metric_col] = _to_positive_binary_indicator_value(
                            work_df[source_col]
                        )
                    else:
                        work_df[metric_col] = _to_numeric_indicator_value(work_df[source_col])

                value_label = "Percentage"
                value_title = "Percentage (%)"
            elif is_data_mentorship_analysis:
                try:
                    df = load_data_mentorship_form_data()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:
                    st.error(f"Unable to load merged Data Mentorship form data: {exc}")
                    return

                system_access_cols = ["q21", "q22", "q23", "q24"]
                tracker_data_cols = [
                    "q31",
                    "q32",
                    "q33",
                    "q34",
                    "q35",
                    "q36",
                    "q37",
                    "q38",
                    "q39",
                    "q391",
                ]
                tracker_feature_cols = ["q41", "q42", "q43", "q44"]
                analysis_cols = ["q51", "q52", "q53", "q54", "q55", "q56", "q57"]
                all_indicator_cols = (
                    system_access_cols
                    + tracker_data_cols
                    + tracker_feature_cols
                    + analysis_cols
                )

                required_columns = (
                    ["region", "hospital", "round"]
                    + system_access_cols
                    + tracker_data_cols
                    + tracker_feature_cols
                    + analysis_cols
                )
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    st.error(
                        "Missing required columns in merged_data.csv: "
                        f"{missing_cols}"
                    )
                    return

                work_df = df.copy()
                work_df["region_code"] = work_df["region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["hospital"].apply(_normalize_entity_text)
                work_df["round"] = work_df["round"].astype(str).str.strip()

                # Exclude submissions where all mentorship indicator questions are empty.
                cleaned_answers = work_df[all_indicator_cols].apply(
                    lambda col: col.astype(str).str.strip().str.lower()
                )
                has_any_indicator_answer = ~cleaned_answers.isin(
                    {"", "nan", "none", "null", "na", "n/a"}
                )
                work_df = work_df[has_any_indicator_answer.any(axis=1)].copy()

                if work_df.empty:
                    st.warning(
                        "No Data Mentorship submissions contain answered indicator questions."
                    )
                    return

                for col in all_indicator_cols:
                    work_df[col] = _yes_no_to_binary(work_df[col])

                work_df["system_access_avg_score"] = work_df[system_access_cols].sum(axis=1)
                work_df["tracker_data_entry_avg_score"] = work_df[tracker_data_cols].sum(
                    axis=1
                )
                work_df["tracker_features_avg_score"] = work_df[
                    tracker_feature_cols
                ].sum(axis=1)
                work_df["analysis_visualization_avg_score"] = work_df[
                    analysis_cols
                ].sum(axis=1)

                score_cols = [
                    "system_access_avg_score",
                    "tracker_data_entry_avg_score",
                    "tracker_features_avg_score",
                    "analysis_visualization_avg_score",
                ]
                value_label = "Average Score"
                value_title = "Average Score per Submission"
            elif is_cis_analysis:
                try:
                    df = load_cis_bundle_data()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:
                    st.error(f"Unable to load CIS Bundle of Care data: {exc}")
                    return

                indicator_source_cols = [spec[0] for spec in CIS_BUNDLE_INDICATORS]
                required_columns = (
                    ["reg-region", "reg-hospital", "reg-round", "week", "unit"]
                    + indicator_source_cols
                )
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    st.error(
                        "Missing required columns in CIS_Bundle_of_Care.csv: "
                        f"{missing_cols}"
                    )
                    return

                work_df = df.copy()
                work_df["region_code"] = work_df["reg-region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["reg-hospital"].apply(_normalize_entity_text)
                work_df["round"] = work_df["reg-round"].astype(str).str.strip()
                work_df["week"] = work_df["week"].apply(_normalize_entity_text)
                work_df["unit"] = (
                    work_df["unit"]
                    .astype(str)
                    .str.strip()
                    .replace({"1.0": "1", "2.0": "2"})
                )
                value_label = "Percentage"
                value_title = "Percentage (%)"
            else:
                try:
                    df = load_merged_bmet_data()
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    return
                except Exception as exc:
                    st.error(f"Unable to load merged mentorship data: {exc}")
                    return

                score_cols = ["competency_assessment-Score", "facility_assessment-score_fac"]
                required_columns = ["region", "hospital", "round"] + score_cols
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    st.error(f"Missing required columns in merged_bmet.csv: {missing_cols}")
                    return

                work_df = df.copy()
                work_df["region_code"] = work_df["region"].apply(_normalize_region_code)
                work_df["hospital"] = work_df["hospital"].apply(_normalize_entity_text)
                work_df["round"] = work_df["round"].astype(str).str.strip()
                for score_col in score_cols:
                    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")
                value_label = "Average Score"
                value_title = "Average Score"

            if is_skill_analysis:
                for score_col in score_cols:
                    work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce")

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

            if is_cis_analysis:
                indicator_labels = [spec[1] for spec in CIS_BUNDLE_INDICATORS]
                selected_cis_indicator = st.selectbox(
                    "Indicator",
                    options=indicator_labels,
                    key="mentorship_cis_indicator",
                )
                selected_spec = next(
                    spec for spec in CIS_BUNDLE_INDICATORS if spec[1] == selected_cis_indicator
                )
                selected_cis_indicator_col = selected_spec[0]
                selected_cis_indicator_question = selected_spec[2]
                selected_cis_numerator_label = selected_spec[3]
                selected_cis_denominator_label = selected_spec[4]

            all_rounds_label = "All Rounds"
            round_values = sorted([r for r in work_df["round"].dropna().unique() if r != ""])
            if not round_values:
                if is_skill_analysis:
                    data_file = "merged_skill.csv"
                elif is_qoc_analysis:
                    data_file = "merged_qoc.csv"
                elif is_qi_coaching_analysis:
                    data_file = "merged_qi_coaching.csv"
                elif is_data_mentorship_analysis:
                    data_file = "merged_data.csv"
                elif is_cis_analysis:
                    data_file = "CIS_Bundle_of_Care.csv"
                else:
                    data_file = "merged_bmet.csv"
                st.warning(f"No round values found in {data_file}.")
                return
            round_options = [all_rounds_label] + round_values

            group_mode_options = ["Regional"] if is_regional_user else ["Multi Regional", "Regional"]
            group_mode = st.radio(
                "Group By",
                options=group_mode_options,
                key="mentorship_analysis_group_mode",
            )

            selected_region_for_facility = None
            if group_mode == "Multi Regional":
                entity_col = "region_label"
                entity_options = sorted(
                    [
                        v
                        for v in work_df[entity_col].dropna().unique().tolist()
                        if str(v).strip()
                    ]
                )
                all_token = "All Regions"
                selector_options = [all_token] + entity_options
                entity_key = "mentorship_entities_multi_regional_selection"
                if entity_key not in st.session_state or not st.session_state.get(entity_key):
                    st.session_state[entity_key] = [all_token]

                st.multiselect(
                    "Select Regions",
                    options=selector_options,
                    key=entity_key,
                )

                current_selected = st.session_state.get(entity_key, [all_token])
                effective_selected = (
                    [all_token]
                    if (all_token in current_selected or len(current_selected) == 0)
                    else current_selected
                )
                selected_entities = (
                    entity_options if all_token in effective_selected else effective_selected
                )
            else:
                # Regional mode: choose one region, then compare facilities only within that region.
                if is_regional_user:
                    selected_region_for_facility = regional_locked_label
                    st.caption(f"Regional scope: {selected_region_for_facility}")
                else:
                    region_options = sorted(
                        [
                            v
                            for v in work_df["region_label"].dropna().unique().tolist()
                            if str(v).strip()
                        ]
                    )
                    selected_region_for_facility = st.selectbox(
                        "Select Region",
                        options=region_options,
                        key="mentorship_regional_selected_region",
                    )
                facilities_in_region = sorted(
                    [
                        v
                        for v in work_df.loc[
                            work_df["region_label"] == selected_region_for_facility, "hospital"
                        ]
                        .dropna()
                        .unique()
                        .tolist()
                        if str(v).strip()
                    ]
                )
                entity_col = "hospital"
                all_token = "All Facilities in Region"
                selector_options = [all_token] + facilities_in_region
                entity_key = "mentorship_entities_regional_facilities_selection"
                if entity_key not in st.session_state or not st.session_state.get(entity_key):
                    st.session_state[entity_key] = [all_token]

                st.multiselect(
                    "Select Facilities",
                    options=selector_options,
                    key=entity_key,
                )
                current_selected = st.session_state.get(entity_key, [all_token])
                effective_selected = (
                    [all_token]
                    if (all_token in current_selected or len(current_selected) == 0)
                    else current_selected
                )
                selected_entities = (
                    facilities_in_region
                    if all_token in effective_selected
                    else effective_selected
                )

            selected_round = st.selectbox(
                "Round",
                options=round_options,
                index=0,
                key="mentorship_analysis_round",
            )

            selected_unit = None
            if is_qoc_analysis:
                if work_df.empty:
                    st.warning("No unit values found in merged_qoc.csv.")
                    return
                unit_display_options = ["1 - Labor and Delivery", "2 - NICU/KMC"]
                selected_unit_display = st.selectbox(
                    "Unit",
                    options=unit_display_options,
                    key="mentorship_analysis_qoc_unit",
                )
                reverse_map = {
                    "1 - Labor and Delivery": "1",
                    "2 - NICU/KMC": "2",
                }
                selected_unit = reverse_map.get(selected_unit_display, selected_unit_display)

            st.markdown("</div>", unsafe_allow_html=True)

    with left_col:
        qoc_records_df = None
        qoc_sums_df = None
        qoc_non_null_counts_df = None
        filtered_df = work_df.copy()
        if selected_round != "All Rounds":
            filtered_df = filtered_df[filtered_df["round"] == selected_round]
        if is_qoc_analysis and selected_unit is not None:
            if selected_unit == "1":
                filtered_df = filtered_df[filtered_df["unit_has_1"]]
            elif selected_unit == "2":
                filtered_df = filtered_df[filtered_df["unit_has_2"]]
        if is_cis_analysis:
            if selected_cis_indicator_col is None:
                st.warning("Select an indicator to continue.")
                return
            filtered_df = filtered_df[filtered_df["unit"] == "2"]
            filtered_df["cis_indicator_value"] = _to_positive_binary_indicator_value(
                filtered_df[selected_cis_indicator_col]
            )
            filtered_df = filtered_df[filtered_df["cis_indicator_value"].notna()].copy()
            filtered_df["week_value"] = filtered_df["week"].apply(_normalize_entity_text)
            filtered_df = filtered_df[filtered_df["week_value"] != ""].copy()
        if group_mode == "Regional" and selected_region_for_facility:
            filtered_df = filtered_df[
                filtered_df["region_label"] == selected_region_for_facility
            ]

        if selected_entities:
            filtered_df = filtered_df[filtered_df[entity_col].isin(selected_entities)]
        else:
            filtered_df = filtered_df.iloc[0:0]

        if filtered_df.empty:
            st.info("No data for the selected filter combination.")
            return

        if is_skill_analysis:
            agg_df = (
                filtered_df.groupby(entity_col, as_index=False)[score_cols]
                .mean()
                .sort_values(entity_col)
            )
        elif is_qoc_analysis:
            grouped = filtered_df.groupby(entity_col, as_index=False)
            counts_df = grouped.size().rename(columns={"size": "_record_count"})
            sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
            non_null_counts_df = (
                grouped[score_cols]
                .count()
                .rename(columns={col: f"{col}__non_null_count" for col in score_cols})
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
                columns=["_record_count"]
                + [f"{col}__non_null_count" for col in score_cols]
            ).sort_values(entity_col)
        elif is_qi_coaching_analysis:
            grouped = filtered_df.groupby(entity_col, as_index=False)
            sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
            non_null_counts_df = (
                grouped[score_cols]
                .count()
                .rename(columns={col: f"{col}__non_null_count" for col in score_cols})
            )
            agg_df = sums_df.merge(non_null_counts_df, on=entity_col, how="left")
            for col in score_cols:
                denom_col = f"{col}__non_null_count"
                agg_df[col] = (
                    (
                        pd.to_numeric(agg_df[col], errors="coerce").fillna(0)
                        / pd.to_numeric(agg_df[denom_col], errors="coerce").replace(
                            0, pd.NA
                        )
                    )
                    * 100
                ).fillna(0)
            agg_df = agg_df.drop(
                columns=[f"{col}__non_null_count" for col in score_cols]
            ).sort_values(entity_col)
        elif is_data_mentorship_analysis:
            grouped = filtered_df.groupby(entity_col, as_index=False)
            counts_df = grouped.size().rename(columns={"size": "_record_count"})
            sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
            agg_df = counts_df.merge(sums_df, on=entity_col, how="left")
            for col in score_cols:
                agg_df[col] = (
                    pd.to_numeric(agg_df[col], errors="coerce").fillna(0)
                    / pd.to_numeric(agg_df["_record_count"], errors="coerce").replace(
                        0, pd.NA
                    )
                ).fillna(0)
            agg_df = agg_df.sort_values(entity_col)
        elif is_cis_analysis:
            grouped = filtered_df.groupby([entity_col, "week_value"], as_index=False)
            sums_df = (
                grouped["cis_indicator_value"]
                .sum(numeric_only=True)
                .fillna(0)
                .rename(columns={"cis_indicator_value": "cis_indicator_numerator"})
            )
            non_null_counts_df = grouped["cis_indicator_value"].count().rename(
                columns={"cis_indicator_value": "cis_indicator_denominator"}
            )
            agg_df = sums_df.merge(
                non_null_counts_df, on=[entity_col, "week_value"], how="left"
            )
            agg_df["cis_indicator_value"] = (
                pd.to_numeric(agg_df["cis_indicator_numerator"], errors="coerce").fillna(0)
                / pd.to_numeric(
                    agg_df["cis_indicator_denominator"], errors="coerce"
                ).replace(0, pd.NA)
            ).fillna(0) * 100
            agg_df["week_sort"] = pd.to_numeric(agg_df["week_value"], errors="coerce")
            agg_df["week_sort"] = agg_df["week_sort"].fillna(10**9)
            agg_df["week_display"] = agg_df["week_value"].apply(lambda value: f"Week {value}")
            agg_df = agg_df.sort_values([entity_col, "week_sort", "week_value"])
        elif is_bmet_analysis:
            grouped = filtered_df.groupby(entity_col, as_index=False)
            sums_df = grouped[score_cols].sum(numeric_only=True).fillna(0)
            non_null_counts_df = (
                grouped[score_cols]
                .count()
                .rename(columns={col: f"{col}__non_null_count" for col in score_cols})
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
        else:
            agg_df = (
                filtered_df.groupby(entity_col, as_index=False)[score_cols]
                .sum()
                .sort_values(entity_col)
            )

        hover_entity_label = "Region" if group_mode == "Multi Regional" else "Facility"

        if is_skill_analysis:
            metric_colors = MENTORSHIP_METRIC_COLORS
            subplot_titles = [
                "Point of Care 1",
                "Point of Care 2",
                "Point of Care 3",
                "Point of Care 4",
                "Point of Care 5",
                "Point of Care 6",
                "Point of Care 7",
                "Point of Care 8",
                "Overall Point of Care",
            ]
            hover_metric_labels = subplot_titles
            fig = make_subplots(
                rows=3,
                cols=3,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                horizontal_spacing=0.06,
            )
            for idx, metric in enumerate(score_cols):
                row = idx // 3 + 1
                col = idx % 3 + 1
                fig.add_bar(
                    y=agg_df[entity_col],
                    x=agg_df[metric].fillna(0),
                    orientation="h",
                    marker_color=metric_colors[idx % len(metric_colors)],
                    showlegend=False,
                    customdata=agg_df[[entity_col]].values,
                    hovertemplate=(
                        f"{hover_entity_label}: %{{customdata[0]}}"
                        f"<br>{hover_metric_labels[idx]}: %{{x:.2f}}"
                        "<extra></extra>"
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(
                    title_text=value_title,
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )

            skill_chart_height = max(760, min(1350, 620 + len(agg_df) * 14))
            fig.update_layout(
                template="plotly_white",
                height=skill_chart_height,
                margin=dict(l=8, r=8, t=36, b=8),
                font=dict(size=9),
                hoverlabel=dict(font_size=10),
                bargap=0.24,
                bargroupgap=0.08,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"mentorship_skill_bars_{group_mode}_{selected_round}",
            )

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
            table_df = agg_df.rename(
                columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **skill_table_labels,
                }
            ).round(2)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.caption(
                "Averages are computed per selected group as: "
                "`sum(variable values) / count(records with non-null variable)`."
            )
        elif is_qoc_analysis:
            metric_colors = MENTORSHIP_METRIC_COLORS
            subplot_titles = [
                "Maintaining Warm",
                "KMC",
                "CPAP",
                "Neonatal Jaundice",
                "IPC Measures",
                "Overall QoC",
            ]
            hover_metric_labels = subplot_titles
            fig = make_subplots(
                rows=2,
                cols=3,
                subplot_titles=subplot_titles,
                vertical_spacing=0.10,
                horizontal_spacing=0.07,
            )
            for idx, metric in enumerate(score_cols):
                row = idx // 3 + 1
                col = idx % 3 + 1
                fig.add_bar(
                    y=agg_df[entity_col],
                    x=agg_df[metric].fillna(0),
                    orientation="h",
                    marker_color=metric_colors[idx % len(metric_colors)],
                    showlegend=False,
                    customdata=agg_df[[entity_col]].values,
                    hovertemplate=(
                        f"{hover_entity_label}: %{{customdata[0]}}"
                        f"<br>{hover_metric_labels[idx]}: %{{x:.2f}}"
                        "<extra></extra>"
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(
                    title_text=value_title,
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )

            qoc_chart_height = max(640, min(1200, 520 + len(agg_df) * 14))
            fig.update_layout(
                template="plotly_white",
                height=qoc_chart_height,
                margin=dict(l=8, r=8, t=36, b=8),
                font=dict(size=9),
                hoverlabel=dict(font_size=10),
                bargap=0.24,
                bargroupgap=0.08,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"mentorship_qoc_bars_{group_mode}_{selected_round}_{selected_unit}",
            )

            qoc_table_labels = {
                score_cols[0]: "Maintaining Warm (%)",
                score_cols[1]: "KMC (%)",
                score_cols[2]: "CPAP (%)",
                score_cols[3]: "Neonatal Jaundice (%)",
                score_cols[4]: "IPC Measures (%)",
                score_cols[5]: "Overall QoC (%)",
            }
            table_df = agg_df.rename(
                columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **qoc_table_labels,
                }
            ).round(2)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            st.caption(
                "Averages are computed per selected group as: "
                "`sum(variable values) / count(records with non-null variable)`."
            )
            st.caption(
                "QoC variables: QOC1_b-QOC1_perce, QOC2_b-QOC2-QOC2_perce, "
                "QOC3_b-QOC3_perce, QOC4_b-QOC4_perce, QOC5_b-QOC5_gr5-QOC5_perce, QOC_perce."
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
                    audit_df = audit_df.rename(
                        columns={
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
                        }
                    )
                    st.dataframe(audit_df.round(2), use_container_width=True, hide_index=True)
        elif is_qi_coaching_analysis:
            metric_colors = MENTORSHIP_METRIC_COLORS
            subplot_titles = [spec[2] for spec in QI_COACHING_INDICATORS]
            fig = make_subplots(
                rows=4,
                cols=2,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                horizontal_spacing=0.08,
            )
            for idx, metric in enumerate(score_cols):
                row = idx // 2 + 1
                col = idx % 2 + 1
                fig.add_bar(
                    y=agg_df[entity_col],
                    x=agg_df[metric].fillna(0),
                    orientation="h",
                    marker_color=metric_colors[idx % len(metric_colors)],
                    showlegend=False,
                    customdata=agg_df[[entity_col]].values,
                    hovertemplate=(
                        f"{hover_entity_label}: %{{customdata[0]}}"
                        f"<br>{subplot_titles[idx]}: %{{x:.2f}}%"
                        "<extra></extra>"
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(
                    title_text=value_title,
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )

            qi_chart_height = max(860, min(1500, 720 + len(agg_df) * 16))
            fig.update_layout(
                template="plotly_white",
                height=qi_chart_height,
                margin=dict(l=8, r=8, t=36, b=8),
                font=dict(size=9),
                hoverlabel=dict(font_size=10),
                bargap=0.24,
                bargroupgap=0.08,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"mentorship_qi_coaching_bars_{group_mode}_{selected_round}",
            )

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
            table_df = agg_df.rename(
                columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **qi_table_labels,
                }
            ).round(2)
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            st.caption(
                "For each QI coaching indicator, chart values are computed as "
                "`(sum(response values) / count(non-null submissions for that question)) * 100`."
            )
            st.caption(
                "Yes/no responses are converted to 1/0 before averaging. "
                "For `part5_project-q502`, any value greater than zero is treated as "
                "yes=1 so the indicator stays a percentage instead of exceeding 100."
            )
            st.caption(
                "Indicator source columns: `part8_project-q801`, "
                "`part2_qi_structure-q203`, `part5_project-q502`, "
                "`part6_project-q604`, `part3_project-q301`, `part3_project-q309`, "
                "`part4_project-q408`, `part5_project-q504`."
            )
        elif is_cis_analysis:
            indicator_label = selected_cis_indicator or "CIS Indicator"
            numerator_label = selected_cis_numerator_label or "Numerator (Yes)"
            denominator_label = selected_cis_denominator_label or "Denominator (Answered)"
            week_order = (
                agg_df[["week_display", "week_sort", "week_value"]]
                .drop_duplicates()
                .sort_values(["week_sort", "week_value"])["week_display"]
                .tolist()
            )
            fig = go.Figure()
            
            # Combine multiple plotly qualitative sequences for high distinctiveness
            import plotly.express as px
            distinct_colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
            
            unique_entities = agg_df[entity_col].dropna().unique().tolist()
            for idx, entity_name in enumerate(unique_entities):
                entity_df = agg_df[agg_df[entity_col] == entity_name].copy()
                fig.add_trace(
                    go.Scatter(
                        x=entity_df["week_display"],
                        y=entity_df["cis_indicator_value"].fillna(0),
                        mode="lines+markers",
                        name=str(entity_name),
                        marker=dict(size=8),
                        line=dict(width=3, color=distinct_colors[idx % len(distinct_colors)]),
                        customdata=entity_df[
                            [entity_col, "cis_indicator_numerator", "cis_indicator_denominator"]
                        ].fillna(0).values,
                        hovertemplate=(
                            "Week: %{x}"
                            f"<br>{hover_entity_label}: %{{customdata[0]}}"
                            f"<br>{indicator_label}: %{{y:.2f}}%"
                            f"<br>{numerator_label}: %{{customdata[1]:.0f}}"
                            f"<br>{denominator_label}: %{{customdata[2]:.0f}}"
                            "<extra></extra>"
                        ),
                    )
                )
            chart_height = max(340, min(680, 320 + len(unique_entities) * 30))
            fig.update_layout(
                template="plotly_white",
                height=chart_height,
                margin=dict(l=8, r=8, t=26, b=8),
                xaxis_title="Week",
                yaxis_title=value_title,
                font=dict(size=10),
                hoverlabel=dict(font_size=11),
                legend_title_text="Region" if group_mode == "Multi Regional" else "Facility",
            )
            fig.update_xaxes(
                tickfont=dict(size=9),
                title_font=dict(size=10),
                automargin=True,
                categoryorder="array",
                categoryarray=week_order,
            )
            fig.update_yaxes(
                tickfont=dict(size=9),
                title_font=dict(size=10),
                automargin=True,
                rangemode="tozero",
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"mentorship_cis_lines_{group_mode}_{selected_round}_{selected_cis_indicator_col}",
            )

            table_df = agg_df.rename(
                columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    "week_display": "Week",
                    "cis_indicator_value": indicator_label,
                    "cis_indicator_numerator": numerator_label,
                    "cis_indicator_denominator": denominator_label,
                }
            ).round(2)
            table_columns = [
                "Region" if group_mode == "Multi Regional" else "Facility",
                "Week",
                indicator_label,
                numerator_label,
                denominator_label,
            ]
            st.dataframe(
                table_df[[col for col in table_columns if col in table_df.columns]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Weekly percentage is computed as `(sum(yes=1 values) / count(non-null submissions for the indicator)) * 100`."
            )
        elif is_data_mentorship_analysis:
            metric_colors = MENTORSHIP_METRIC_COLORS
            subplot_titles = [
                "System Access & User Management",
                "Tracker Data Entry (IMNID Program)",
                "Tracker Features & Program Functionality",
                "Analysis & Visualization (Dashboard Use)",
            ]
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=subplot_titles,
                vertical_spacing=0.11,
                horizontal_spacing=0.08,
            )
            for idx, metric in enumerate(score_cols):
                row = idx // 2 + 1
                col = idx % 2 + 1
                fig.add_bar(
                    y=agg_df[entity_col],
                    x=agg_df[metric].fillna(0),
                    orientation="h",
                    marker_color=metric_colors[idx % len(metric_colors)],
                    showlegend=False,
                    customdata=agg_df[[entity_col]].values,
                    hovertemplate=(
                        f"{hover_entity_label}: %{{customdata[0]}}"
                        f"<br>{subplot_titles[idx]}: %{{x:.2f}}"
                        "<extra></extra>"
                    ),
                    row=row,
                    col=col,
                )
                fig.update_xaxes(
                    title_text=value_title,
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    tickfont=dict(size=8),
                    title_font=dict(size=9),
                    automargin=True,
                    row=row,
                    col=col,
                )

            mentorship_chart_height = max(620, min(1200, 520 + len(agg_df) * 14))
            fig.update_layout(
                template="plotly_white",
                height=mentorship_chart_height,
                margin=dict(l=8, r=8, t=36, b=8),
                font=dict(size=9),
                hoverlabel=dict(font_size=10),
                bargap=0.24,
                bargroupgap=0.08,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"mentorship_data_form_bars_{group_mode}_{selected_round}",
            )

            mentorship_table_labels = {
                score_cols[0]: "System Access & User Management (Avg)",
                score_cols[1]: "Tracker Data Entry (Avg)",
                score_cols[2]: "Tracker Features & Program Functionality (Avg)",
                score_cols[3]: "Analysis & Visualization (Avg)",
            }
            table_df = agg_df.rename(
                columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    **mentorship_table_labels,
                }
            ).round(2)
            table_columns = [
                "Region" if group_mode == "Multi Regional" else "Facility",
                "System Access & User Management (Avg)",
                "Tracker Data Entry (Avg)",
                "Tracker Features & Program Functionality (Avg)",
                "Analysis & Visualization (Avg)",
            ]
            table_df = table_df[[c for c in table_columns if c in table_df.columns]]
            st.dataframe(table_df, use_container_width=True, hide_index=True)
            st.caption(
                "For each indicator, yes=1 and no=0 for the specified question set. "
                "Average score is computed as `sum(binary values across filtered non-empty submissions) / total non-empty submissions`."
            )
            st.caption(
                "Indicator groups: "
                "`q21-q24`, `q31-q39 + q391`, `q41-q44`, `q51-q57`."
            )
        else:
            fig = go.Figure()
            fig.add_bar(
                y=agg_df[entity_col],
                x=agg_df["competency_assessment-Score"],
                name="Competency Score",
                marker_color=MENTORSHIP_METRIC_COLORS[0],
                orientation="h",
                customdata=agg_df[[entity_col]].values,
                hovertemplate=f"{hover_entity_label}: %{{customdata[0]}}<br>Competency Score: %{{x:.2f}}<extra></extra>",
            )
            fig.add_bar(
                y=agg_df[entity_col],
                x=agg_df["facility_assessment-score_fac"],
                name="Facility Score",
                marker_color=MENTORSHIP_METRIC_COLORS[1],
                orientation="h",
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
                barmode="group",
                template="plotly_white",
                height=chart_height,
                margin=dict(l=8, r=8, t=14, b=6),
                xaxis_title=value_title,
                yaxis_title="Region" if group_mode == "Multi Regional" else "Facility",
                legend_title_text="Indicators",
                bargap=bar_gap,
                bargroupgap=bar_group_gap,
                font=dict(size=10),
                hoverlabel=dict(font_size=11),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.0,
                    xanchor="left",
                    x=0,
                    font=dict(size=9),
                ),
            )
            fig.update_xaxes(
                tickfont=dict(size=9),
                title_font=dict(size=10),
                automargin=True,
                tickformat=".2f",
            )
            fig.update_yaxes(tickfont=dict(size=9), title_font=dict(size=10), automargin=True)
            st.plotly_chart(fig, use_container_width=True, key=f"mentorship_bar_{group_mode}")

            table_df = agg_df.rename(
                columns={
                    entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                    "competency_assessment-Score": "Competency Score",
                    "facility_assessment-score_fac": "Facility Score",
                }
            ).round(2)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            competency_series = pd.to_numeric(
                filtered_df["competency_assessment-Score"], errors="coerce"
            )
            facility_series = pd.to_numeric(
                filtered_df["facility_assessment-score_fac"], errors="coerce"
            )
            competency_avg = float(competency_series.mean()) if competency_series.notna().any() else 0.0
            facility_avg = float(facility_series.mean()) if facility_series.notna().any() else 0.0
            round_summary_label = (
                "All Rounds"
                if selected_round == "All Rounds"
                else f"Round {selected_round}"
            )
            st.caption(
                f"{round_summary_label} summary: "
                f"Competency Average = {competency_avg:.2f}, "
                f"Facility Average = {facility_avg:.2f}."
            )
            st.caption(
                "Variables: `competency_assessment-Score` and "
                "`facility_assessment-score_fac` are averaged as "
                "`sum(values) / count(non-null submissions)` for each selected filter."
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

    # 🔥 OPTIMIZATION: Get current user info directly from session state
    current_user = st.session_state.get("user", {})
    if not current_user:
        st.warning("🚪 Please log in to access data")
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
            margin-bottom: 0.8rem;
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

    # 🔥 OPTIMIZATION: Create session state keys
    odk_data_key = f"odk_forms_data_{current_user_id}"
    afar_odk_data_key = f"odk_forms_data_afar_{current_user_id}"
    last_refresh_key = f"last_odk_refresh_{current_user_id}"
    user_tracker_key = "current_odk_user"

    # 🔥 CRITICAL FIX: Check if user has changed
    current_user_info = f"{current_user_id}_{current_region_id}_{current_role}"

    if user_tracker_key not in st.session_state:
        st.session_state[user_tracker_key] = current_user_info
    else:
        previous_user_info = st.session_state[user_tracker_key]
        if previous_user_info != current_user_info:
            # 🔥 USER CHANGED - CLEAR ALL OLD DATA
            st.info(f"🔄 Loading data for {current_username}...")

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
            '<div class="main-header">📋 Integrated Mentorship Data</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**All available forms are automatically loaded below**")

    with col2:
        st.markdown('<div class="action-buttons-container">', unsafe_allow_html=True)

        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.session_state[odk_data_key] = {}
            st.session_state[afar_odk_data_key] = {}
            st.session_state[last_refresh_key] = pd.Timestamp.now()
            st.rerun()

        # Download All button ALWAYS visible - no conditions
        if st.button("💾 Download All", use_container_width=True, type="secondary"):
            has_data = (
                st.session_state.get(odk_data_key)
                and len(st.session_state[odk_data_key]) > 0
            )
            if has_data:
                download_all_forms(st.session_state[odk_data_key])
            else:
                st.warning("No data available to download. Please refresh data first.")

        st.markdown("</div>", unsafe_allow_html=True)

    # 🔥 OPTIMIZATION: Load data only if needed
    if (not st.session_state[odk_data_key]) or (
        is_afar_user and not st.session_state[afar_odk_data_key]
    ):
        with st.spinner("🔄 Loading forms data..."):
            try:
                # Fetch data once
                odk_data = fetch_odk_data_for_user(current_user)
                forms_data = odk_data.get("odk_forms", {})
                afar_forms_data = odk_data.get(AFAR_MENTORSHIP_SECTION_LABEL, {})

                st.session_state[odk_data_key] = forms_data
                st.session_state[afar_odk_data_key] = afar_forms_data
                st.session_state[last_refresh_key] = pd.Timestamp.now()

                st.success(f"✅ Loaded {len(forms_data)} forms")

            except Exception as e:
                st.error(f"❌ Failed to load data: {str(e)}")

    # Show refresh info
    if st.session_state[last_refresh_key]:
        refresh_time = st.session_state[last_refresh_key].strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f'<div class="auto-load-info">🕒 Last refresh: {refresh_time}</div>',
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
        st.info("📭 No forms data available. Click 'Refresh Data' to try again.")


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
    st.markdown(f"### 📁 Available Forms ({len(forms_data)})")

    consistent_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    cols = st.columns(2)

    for i, (form_id, df) in enumerate(forms_data.items()):
        # 🔥 OPTIMIZATION: Get display name from cached forms
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
            <h3>📄 {display_name}</h3>
            <div class="form-id">({form_id})</div>
            <div style="display: flex; gap: 8px; margin-bottom: 15px; margin-top: 10px;">
                <span class="stats-badge">📊 {len(df):,} records</span>
                <span class="stats-badge">📋 {len(df.columns)} columns</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="⬇️ Download CSV",
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
        label="💾 Download All as ZIP",
        data=zip_buffer,
        file_name="forms.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_zip",
    )


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string for download"""
    return df.to_csv(index=False, encoding="utf-8")
