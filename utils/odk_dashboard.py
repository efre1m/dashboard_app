# utils/odk_dashboard.py
import streamlit as st
import pandas as pd
import logging
import time
import os
import plotly.graph_objects as go
from typing import Dict
from utils.odk_api import (
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


def _normalize_region_code(value) -> str:
    """Normalize region code values to canonical string form like '1'..'6'."""
    if pd.isna(value):
        return ""
    code = str(value).strip()
    if code.endswith(".0"):
        code = code[:-2]
    return code


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
    """Render analysis for merged_bmet.csv with right-side filters and bar charts."""
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

    try:
        df = load_merged_bmet_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Unable to load merged mentorship data: {exc}")
        return

    required_columns = [
        "region",
        "hospital",
        "round",
        "competency_assessment-Score",
        "facility_assessment-score_fac",
    ]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in merged_bmet.csv: {missing_cols}")
        return

    work_df = df.copy()
    work_df["region_code"] = work_df["region"].apply(_normalize_region_code)
    region_code_map = _get_region_code_to_name_mapping()
    work_df["region_label"] = work_df["region_code"].map(region_code_map).fillna(
        work_df["region_code"].apply(lambda x: f"Unknown ({x})" if x else "Unknown")
    )

    score_cols = ["competency_assessment-Score", "facility_assessment-score_fac"]
    for score_col in score_cols:
        work_df[score_col] = pd.to_numeric(work_df[score_col], errors="coerce").fillna(0)

    work_df["hospital"] = work_df["hospital"].astype(str).str.strip()
    work_df["round"] = work_df["round"].astype(str).str.strip()

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

    round_options = sorted([r for r in work_df["round"].dropna().unique() if r != ""])
    if not round_options:
        st.warning("No round values found in merged_bmet.csv.")
        return

    st.markdown(
        """
        <div class="mentorship-analysis-shell">
            <h4>Mentorship Data Analysis</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([4, 1])

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
                options=["BMET Data", "Skill Assessment Data"],
                key="mentorship_analysis_data_choice",
            )

            # Skill assessment analysis is intentionally gated for now.
            if data_choice == "Skill Assessment Data":
                st.info("Skill assessment analysis will be available soon.")
                st.markdown("</div>", unsafe_allow_html=True)
                with left_col:
                    st.markdown(
                        """
                        <div style="padding: 2rem 1.2rem; border: 1px solid #cbd5e1; border-radius: 12px;
                             background: linear-gradient(135deg, #f8fafc, #eef2ff);">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">Skill Assessment Analysis</h4>
                            <p style="margin: 0; color: #475569;">
                                This section is under preparation and will be enabled soon.
                                Please select <strong>BMET Data</strong> to continue current analysis.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                return

            group_mode_options = ["Regional"] if is_regional_user else ["Multi Regional", "Regional"]
            group_mode = st.radio(
                "Group By",
                options=group_mode_options,
                key="mentorship_analysis_group_mode",
            )

            selected_round = st.selectbox(
                "Round",
                options=round_options,
                key="mentorship_analysis_round",
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

            st.markdown("</div>", unsafe_allow_html=True)

    with left_col:
        filtered_df = work_df[work_df["round"] == selected_round].copy()
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

        agg_df = (
            filtered_df.groupby(entity_col, as_index=False)[score_cols]
            .sum()
            .sort_values(entity_col)
        )

        fig = go.Figure()
        hover_entity_label = "Region" if group_mode == "Multi Regional" else "Facility"
        fig.add_bar(
            y=agg_df[entity_col],
            x=agg_df["competency_assessment-Score"],
            name="competency_assessment-Score",
            marker_color="#2563eb",
            orientation="h",
            customdata=agg_df[[entity_col]].values,
            hovertemplate=f"{hover_entity_label}: %{{customdata[0]}}<br>Competency Score: %{{x}}<extra></extra>",
        )
        fig.add_bar(
            y=agg_df[entity_col],
            x=agg_df["facility_assessment-score_fac"],
            name="facility_assessment-score_fac",
            marker_color="#16a34a",
            orientation="h",
            customdata=agg_df[[entity_col]].values,
            hovertemplate=f"{hover_entity_label}: %{{customdata[0]}}<br>Facility Score: %{{x}}<extra></extra>",
        )
        chart_height = max(560, min(1200, 90 + len(agg_df) * 36))
        fig.update_layout(
            barmode="group",
            template="plotly_white",
            height=chart_height,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Score",
            yaxis_title="Region" if group_mode == "Multi Regional" else "Facility",
            legend_title_text="Indicators",
            bargap=0.14,
            bargroupgap=0.08,
            font=dict(size=14),
            hoverlabel=dict(font_size=16),
        )
        fig.update_xaxes(tickfont=dict(size=13), title_font=dict(size=15))
        fig.update_yaxes(tickfont=dict(size=13), title_font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True, key=f"mentorship_bar_{group_mode}")

        table_df = agg_df.rename(
            columns={
                entity_col: "Region" if group_mode == "Multi Regional" else "Facility",
                "competency_assessment-Score": "Competency Score",
                "facility_assessment-score_fac": "Facility Score",
            }
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        competency_total = float(filtered_df["competency_assessment-Score"].sum())
        facility_total = float(filtered_df["facility_assessment-score_fac"].sum())
        overall_score = competency_total + facility_total
        st.caption(
            f"Round {selected_round} summary: "
            f"Competency Score = {competency_total:.0f}, "
            f"Facility Score = {facility_total:.0f}, "
            f"Overall Score = {overall_score:.0f}."
        )
        st.caption(
            "Variables: `competency_assessment-Score` and "
            "`facility_assessment-score_fac` are aggregated totals for the selected filter."
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
        options=["Mentorship Data", "Mentorship Data Analysis"],
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
    if st.session_state.get(odk_data_key) and len(st.session_state[odk_data_key]) > 0:
        display_forms_grid(st.session_state[odk_data_key], key_prefix="mentorship")
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
