# kpi_newborn.py - UPDATED WITH DATASET COLUMN NAMES, REMOVED CULTURE TABS

import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
import datetime
import logging

from newborns_dashboard.kpi_utils_newborn import (
    auto_text_color,
    extract_period_columns_newborn,
    get_relevant_date_column_for_newborn_kpi,
    get_numerator_denominator_for_newborn_kpi,
    render_newborn_trend_chart,
    render_newborn_facility_comparison_chart,
    render_newborn_region_comparison_chart,
    render_admitted_newborns_trend_chart,
    render_admitted_newborns_facility_comparison_chart,
    render_admitted_newborns_region_comparison_chart,
)

# Import simplified functions with SINGLE TABLE DISPLAY
from newborns_dashboard.kpi_utils_newborn_simplified import (
    render_birth_weight_trend_chart,
    render_birth_weight_facility_comparison,
    render_birth_weight_region_comparison,
    render_kmc_coverage_trend_chart,
    # Individual CPAP chart functions with single tables
    # render_cpap_general_trend_chart,
    render_cpap_rds_trend_chart,
    render_cpap_by_weight_trend_chart,
    # render_cpap_general_facility_comparison,
    render_cpap_rds_facility_comparison,
    # render_cpap_general_region_comparison,
    render_cpap_rds_region_comparison,
    # Rate comparison aliases (call the above functions)
    render_kmc_facility_comparison,
    render_kmc_region_comparison,
    # render_cpap_facility_comparison,

)

# KPI mapping for newborn comparison charts
NEWBORN_KPI_MAPPING = {
    "Inborn Rate (%)": {
        "title": "Inborn Rate (%)",
        "numerator_name": "Inborn Babies",
        "denominator_name": "Total Admitted Newborns",
    },
    "Outborn Rate (%)": {
        "title": "Outborn Rate (%)",
        "numerator_name": "Outborn Babies",
        "denominator_name": "Total Admitted Newborns",
    },
    "Inborn Hypothermia Rate (%)": {
        "title": "Hypothermia in Inborn Babies (%)",
        "numerator_name": "Inborn Hypothermia Cases",
        "denominator_name": "Total Inborn Babies",
    },
    "Outborn Hypothermia Rate (%)": {
        "title": "Hypothermia in Outborn Babies (%)",
        "numerator_name": "Outborn Hypothermia Cases",
        "denominator_name": "Total Outborn Babies",
    },
    "Hypothermia on Admission Rate (%)": {
        "title": "Hypothermia on Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Neonatal Mortality Rate (%)": {
        "title": "Neonatal Mortality Rate (%)",
        "numerator_name": "Dead Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Inborn Hypothermia Rate (%)": {
        "title": "Hypothermia in Inborn Babies (%)",
        "numerator_name": "Inborn Hypothermia Cases",
        "denominator_name": "Total Inborn Babies",
    },
    "Outborn Hypothermia Rate (%)": {
        "title": "Hypothermia in Outborn Babies (%)",
        "numerator_name": "Outborn Hypothermia Cases",
        "denominator_name": "Total Outborn Babies",
    },
    "Admitted Newborns": {
        "title": "Total Admitted Newborns",
        "value_name": "Admitted Newborns",
    },
    # ANTIBIOTICS KPI - UPDATED WITH DATASET COLUMN NAMES
    # "Antibiotics for Clinical Sepsis (%)": {
    #     "title": "Antibiotics for Clinical Sepsis (%)",
    #     "numerator_name": "Newborns with Sepsis Receiving Antibiotics",
    #     "denominator_name": "Newborns with Probable Sepsis",
    # },
    # NEW SIMPLIFIED KPIs WITH SINGLE TABLE DISPLAY
    "Birth Weight Rate": {
        "title": "Birth Weight Rate",
        "value_name": "Percentage of Newborns (%)",
        "type": "simplified",
        "category": "birth_weight",
        "comparison_type": "stacked_cases",  # Uses stacked bars in comparison
    },
    "KMC Coverage by Birth Weight": {
        "title": "KMC Coverage by Birth Weight Category",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total Newborns with Valid Birth Weight",
        "type": "simplified",
        "category": "kmc",
        "comparison_type": "rates",  # Shows rates in comparison charts
    },
    # CPAP KPIs WITH SINGLE TABLE DISPLAY
    # "General CPAP Coverage": {
    #     "title": "General CPAP Coverage",
    #     "numerator_name": "CPAP Cases",
    #     "denominator_name": "Total Admitted Newborns",
    #     "type": "simplified",
    #     "category": "cpap_general",
    #     "comparison_type": "rates",  # Shows rates in comparison charts
    # },
    "CPAP for RDS": {
        "title": "CPAP for Respiratory Distress Syndrome (RDS)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total RDS Cases",
        "type": "simplified",
        "category": "cpap_rds",
        "comparison_type": "rates",  # Shows rates in comparison charts
    },
    "CPAP Coverage by Birth Weight": {
        "title": "CPAP Coverage by Birth Weight Category",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total Newborns with Valid Birth Weight",
        "type": "simplified",
        "category": "cpap_by_weight",
        "comparison_type": "rates",  # Shows rates in comparison charts
    },
    # DATA QUALITY KPIs
    "Missing Temperature (%)": {
        "title": "Missing Temperature at Admission (%)",
        "numerator_name": "Patients with Missing Temperature",
        "denominator_name": "Total Admitted Newborns",
    },
    "Missing Birth Weight (%)": {
        "title": "Missing Birth Weight (%)",
        "numerator_name": "Patients with Missing Birth Weight",
        "denominator_name": "Total Admitted Newborns",
    },
    "Missing Discharge Status (%)": {
        "title": "Missing Newborn Status at Discharge (%)",
        "numerator_name": "Patients with Missing Status",
        "denominator_name": "Total Admitted Newborns",
    },
    "Missing Status of Discharge (%)": {
        "title": "Missing Status of Discharge (%)",
        "numerator_name": "Patients with Missing Status",
        "denominator_name": "Total Admitted Newborns",
    },
    "Missing Discharge Status (%)": {
        "title": "Missing Newborn Status at Discharge (%)",
        "numerator_name": "Patients with Missing Status",
        "denominator_name": "Total Admitted Newborns",
    },
    "Missing Birth Location (%)": {
        "title": "Missing Birth Location (%)",
        "numerator_name": "Patients with Missing Birth Location",
        "denominator_name": "Total Admitted Newborns",
    },
}

# KPI options for newborn dashboard (REMOVED CULTURE KPIs)
NEWBORN_KPI_OPTIONS = [
    "Inborn Rate (%)",
    "Outborn Rate (%)",
    "Hypothermia on Admission Rate (%)",
    "Neonatal Mortality Rate (%)",
    "Inborn Hypothermia Rate (%)",
    "Outborn Hypothermia Rate (%)",
    "Admitted Newborns",
    # "Antibiotics for Clinical Sepsis (%)",
    # NEW SIMPLIFIED KPIs WITH SINGLE TABLE DISPLAY
    "Birth Weight Rate",
    "KMC Coverage by Birth Weight",
    # CPAP KPIs WITH SINGLE TABLE DISPLAY
    # "General CPAP Coverage",
    "CPAP for RDS",
    "CPAP Coverage by Birth Weight",
    # DATA QUALITY
    "Missing Temperature (%)",
    "Missing Birth Weight (%)",
    "Missing Birth Weight (%)",
    "Missing Status of Discharge (%)",
    "Missing Birth Location (%)",
]

# KPI Groups for Tab Navigation (UPDATED - REGROUPED AS REQUESTED)
NEWBORN_KPI_GROUPS = {
    "📝 Enrollment": [
         "Admitted Newborns",
    ],
    "👶 Birth": [
        "Inborn Rate (%)",
        "Outborn Rate (%)",
        "Birth Weight Rate",
    ],
    "⚠️ Complication": [
        "Hypothermia on Admission Rate (%)",
        "Inborn Hypothermia Rate (%)",
        "Outborn Hypothermia Rate (%)",
    ],
    "🏥 Intervention": [
        "KMC Coverage by Birth Weight",
        # "General CPAP Coverage", # Commented out
        "CPAP for RDS",
        "CPAP Coverage by Birth Weight",
    ],
    "📉 Mortality": [
        "Neonatal Mortality Rate (%)",
    ],
    "❓ Data Quality": [
         "Missing Temperature (%)",
         "Missing Birth Weight (%)",
         "Missing Birth Weight (%)",
         "Missing Status of Discharge (%)",
         "Missing Birth Location (%)",
    ],
}

# KPI Column Requirements - UPDATED WITH DATASET COLUMN NAMES
NEWBORN_KPI_COLUMN_REQUIREMENTS = {
    "Inborn Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Outborn Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Hypothermia on Admission Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Neonatal Mortality Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "newborn_status_at_discharge_n_discharge_care_form",
        "event_date_discharge_care_form",
    ],
    "Inborn Hypothermia Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Outborn Hypothermia Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Admitted Newborns": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
    ],
    # "Antibiotics for Clinical Sepsis (%)": [
    #     "orgUnit",
    #     "tei_id",
    #     "enrollment_date",
    #     "sub_categories_of_infection_n_discharge_care_form",
    #     "maternal_medication_during_pregnancy_and_labor_nicu_admission_careform",
    #     "event_date_nicu_admission_careform",
    # ],
    # NEW SIMPLIFIED KPIs WITH SINGLE TABLE DISPLAY
    "Birth Weight Rate": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_weight_n_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "KMC Coverage by Birth Weight": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_weight_n_nicu_admission_careform",
        # Multiple KMC columns
        "kmc_done_nurse_followup_sheet",
        "kmc_done_nurse_followup_sheet_v2",
        "kmc_done_nurse_followup_sheet_v3",
        "kmc_done_nurse_followup_sheet_v4",
        "kmc_done_nurse_followup_sheet_v5",
        "event_date_nurse_followup_sheet",
    ],
    # CPAP REQUIREMENTS - UPDATED WITH DATASET COLUMN NAMES
    # "General CPAP Coverage": [
    #     "orgUnit",
    #     "tei_id",
    #     "enrollment_date",
    #     "baby_placed_on_cpap_neonatal_referral_form",
    #     "event_date_neonatal_referral_form",
    # ],
    "CPAP for RDS": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "baby_placed_on_cpap_neonatal_referral_form",
        "sub_categories_of_prematurity_n_discharge_care_form",  # RDS diagnosis column
        "event_date_neonatal_referral_form",
    ],
    "CPAP Coverage by Birth Weight": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "baby_placed_on_cpap_neonatal_referral_form",
        "birth_weight_n_nicu_admission_careform",
        "event_date_neonatal_referral_form",
    ],
    "Missing Temperature (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "temp_at_admission_nicu_admission_careform",
    ],
    "Missing Birth Weight (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_weight_n_nicu_admission_careform",
    ],
    "Missing Status of Discharge (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "newborn_status_at_discharge_n_discharge_care_form",
    ],
    "Missing Birth Location (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
    ],
}

# SIMPLIFIED KPI DATE COLUMN MAPPING - UPDATED WITH DATASET NAMES
SIMPLIFIED_KPI_DATE_COLUMNS = {
    "Birth Weight Rate": "enrollment_date",
    "KMC Coverage by Birth Weight": "enrollment_date",
    # CPAP DATE COLUMNS - UPDATED WITH DATASET NAMES
    # "General CPAP Coverage": "enrollment_date",
    "CPAP for RDS": "enrollment_date",
    "CPAP Coverage by Birth Weight": "enrollment_date",
}


def get_newborn_kpi_filtered_dataframe(df, kpi_name):
    """
    Filter DataFrame to only include columns needed for the selected newborn KPI
    """
    if df is None or df.empty:
        return df

    if kpi_name not in NEWBORN_KPI_COLUMN_REQUIREMENTS:
        return df.copy()

    required_columns = NEWBORN_KPI_COLUMN_REQUIREMENTS[kpi_name]
    available_columns = []

    for col in required_columns:
        if col in df.columns:
            available_columns.append(col)

    essential_cols = ["orgUnit", "tei_id", "enrollment_date"]
    for col in essential_cols:
        if col in df.columns and col not in available_columns:
            available_columns.append(col)

    if not available_columns:
        return df.copy()

    filtered_df = df[available_columns].copy()

    return filtered_df


def get_text_color(bg_color):
    """Get auto text color for background"""
    return auto_text_color(bg_color)


def get_newborn_kpi_config(kpi_selection):
    """Get newborn KPI configuration"""
    # Migration fallback for renamed KPI
    if kpi_selection == "Birth Weight Distribution":
        kpi_selection = "Birth Weight Rate"
    return NEWBORN_KPI_MAPPING.get(kpi_selection, {})


def is_simplified_kpi(kpi_name):
    """Check if a KPI is a simplified KPI (Birth Weight, KMC, CPAP)"""
    kpi_config = get_newborn_kpi_config(kpi_name)
    return kpi_config.get("type") == "simplified"


def get_relevant_date_column_for_newborn_kpi_with_all(kpi_name):
    """
    Get the relevant event date column for a specific newborn KPI
    Includes simplified KPIs with correct date columns
    """
    # Migration fallback for renamed KPI
    if kpi_name == "Birth Weight Distribution":
        kpi_name = "Birth Weight Rate"

    # Check if it's a simplified KPI
    if kpi_name in SIMPLIFIED_KPI_DATE_COLUMNS:
        return SIMPLIFIED_KPI_DATE_COLUMNS[kpi_name]

    # Use original function for non-simplified KPIs
    return get_relevant_date_column_for_newborn_kpi(kpi_name)


def get_numerator_denominator_for_newborn_kpi_with_all(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for a specific newborn KPI with date range filtering
    Supports V1 and simplified KPIs
    """
    # For simplified KPIs, return 0,0,0 as they have special rendering
    if is_simplified_kpi(kpi_name):
        return (0, 0, 0.0)
    else:
        # Use existing V1 function for non-simplified KPIs
        return get_numerator_denominator_for_newborn_kpi(
            df, kpi_name, facility_uids, date_range_filters
        )


def render_newborn_kpi_tab_navigation():
    """Render professional tab navigation for Neonatal KPI selection - UPDATED WITH 3 TABS"""

    st.markdown(
        """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1a6790 !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #1668a1 !important;
        color: white !important;
        border-color: #145a8c !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border-color: #dee2e6 !important;
        font-weight: 500 !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e9ecef !important;
        color: #495057 !important;
        border-color: #ced4da !important;
    }

    .newborn-tab-scope { display: none; }
    div[data-baseweb="tab-panel"]:has(.newborn-tab-scope) { background-color: #FFE5D0 !important; }

    /* Force peach background at top-most app containers for newborn tab */
    html, body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stAppViewContainer"] .main .block-container {
        background: #FFDAB9 !important;
        background-color: #FFDAB9 !important;
    }

    /* Force peach styling for the "Newborn" (2nd) dashboard section radio option */
    .st-key-facility_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2),
    .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2),
    .st-key-national_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2),
    .st-key-facility_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2) label,
    .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2) label,
    .st-key-national_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2) label {
        background: #fff2e6 !important;
        border-color: #fdba74 !important;
        color: #9a3412 !important;
    }

    .st-key-facility_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2)[aria-checked="true"] label,
    .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2)[aria-checked="true"] label,
    .st-key-national_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] [data-baseweb="radio"]:nth-of-type(2)[aria-checked="true"] label,
    .st-key-facility_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2):has(input:checked),
    .st-key-regional_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2):has(input:checked),
    .st-key-national_active_tab_selector [data-testid="stRadio"] [role="radiogroup"][aria-orientation="horizontal"] label:nth-of-type(2):has(input:checked) {
        background: linear-gradient(135deg, #fed7aa, #fdba74) !important;
        border-color: #f97316 !important;
        color: #7c2d12 !important;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.28) !important;
    }

    /* Target the container for the newborn dashboard content */
    /* Note: This might be broad, targeting the main area if not specific enough. */
    /* Since we want just the newborn dashboard area, and we are in a function in that context */
    /* We can try to target the main block or specific elements. */
    /* User requested "background of the newborn dashboard to be peach color" */
    
    .stApp > header {
        background-color: transparent !important;
    }
    
    .stApp > header {
        background-color: transparent !important;
    }
    </style>
    <div class="newborn-tab-scope"></div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for newborn KPI selection
    if "selected_newborn_kpi" not in st.session_state:
        st.session_state.selected_newborn_kpi = "Admitted Newborns" # Default to first tab item if appropriate

    # Create main KPI group tabs - UPDATED TO 6 TABS & REORDERED
    # Enrollment -> Birth -> Data Quality -> Complication -> Intervention -> Mortality
    tab_enrollment, tab_birth, tab_complication, tab_intervention, tab_mortality, tab_dq = st.tabs(
        [
            "Enrollment",
            "Birth",
            "Complication",
            "Intervention",
            "Mortality",
            "Data Quality",
        ]
    )

    selected_kpi = st.session_state.selected_newborn_kpi

    with tab_enrollment:
        # Enrollment - 1 button
        cols = st.columns(5)
        with cols[0]:
             if st.button("Admitted Newborns", key="admitted_newborns_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Admitted Newborns" else "secondary")):
                selected_kpi = "Admitted Newborns"

    with tab_birth:
        cols = st.columns(5)
        with cols[0]:
            if st.button("Inborn Rate", key="inborn_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Inborn Rate (%)" else "secondary")):
                selected_kpi = "Inborn Rate (%)"
        with cols[1]:
            if st.button("Outborn Rate", key="outborn_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Outborn Rate (%)" else "secondary")):
                selected_kpi = "Outborn Rate (%)"
        with cols[2]:
            if st.button("Birth Weight", key="birth_weight_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Birth Weight Rate" else "secondary")):
                selected_kpi = "Birth Weight Rate"

    with tab_dq:
        # Data Quality - 4 buttons
        # Row 1
        cols = st.columns(3)
        with cols[0]:
            if st.button("Missing Temperature", key="missing_temp_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Missing Temperature (%)" else "secondary")):
                selected_kpi = "Missing Temperature (%)"
        with cols[1]:
            if st.button("Missing Birth Weight", key="missing_bw_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Missing Birth Weight (%)" else "secondary")):
                selected_kpi = "Missing Birth Weight (%)"
        with cols[2]:
            if st.button("Missing Status of Discharge", key="missing_status_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Missing Status of Discharge (%)" else "secondary")):
                selected_kpi = "Missing Status of Discharge (%)"
        
        # Row 2
        cols2 = st.columns(3)
        with cols2[0]:
            if st.button("Missing Birth Location", key="missing_loc_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Missing Birth Location (%)" else "secondary")):
                selected_kpi = "Missing Birth Location (%)"


    with tab_complication:
        # Complication - 3 buttons
        # UPDATED: Use columns(3) instead of (5) to give more space for text
        cols_row1 = st.columns(3)
        with cols_row1[0]:
            # Full name requested: "Hypothermia on Admission"
            if st.button("Hypothermia on Admission", key="hypo_admission_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Hypothermia on Admission Rate (%)" else "secondary")):
                selected_kpi = "Hypothermia on Admission Rate (%)"
        with cols_row1[1]:
            if st.button("Inborn Hypothermia Rate", key="inborn_hypo_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Inborn Hypothermia Rate (%)" else "secondary")):
                selected_kpi = "Inborn Hypothermia Rate (%)"
        with cols_row1[2]:
            if st.button("Outborn Hypothermia Rate", key="outborn_hypo_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Outborn Hypothermia Rate (%)" else "secondary")):
                selected_kpi = "Outborn Hypothermia Rate (%)"

    with tab_intervention:
        # Intervention - 3 buttons (General CPAP removed)
        cols = st.columns(5)
        with cols[0]:
            if st.button("KMC Coverage", key="kmc_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "KMC Coverage by Birth Weight" else "secondary")):
                selected_kpi = "KMC Coverage by Birth Weight"
        # with cols[1]:
        #     if st.button("General CPAP", key="cpap_general_btn", use_container_width=True,
        #                  type=("primary" if selected_kpi == "General CPAP Coverage" else "secondary")):
        #         selected_kpi = "General CPAP Coverage"
        with cols[1]:
            if st.button("CPAP for RDS", key="cpap_rds_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "CPAP for RDS" else "secondary")):
                selected_kpi = "CPAP for RDS"
        with cols[2]:
            if st.button("CPAP by Weight", key="cpap_by_weight_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "CPAP Coverage by Birth Weight" else "secondary")):
                selected_kpi = "CPAP Coverage by Birth Weight"

    with tab_mortality:
        # Mortality - 1 button
        cols = st.columns(5)
        with cols[0]:
            if st.button("Neonatal Mortality", key="nmr_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Neonatal Mortality Rate (%)" else "secondary")):
                selected_kpi = "Neonatal Mortality Rate (%)"

    # Update session state with final selection
    if selected_kpi != st.session_state.selected_newborn_kpi:
        st.session_state.selected_newborn_kpi = selected_kpi
        st.rerun()

    return st.session_state.selected_newborn_kpi


def get_period_columns(df):
    """Get period column names from patient data"""
    period_col = None
    period_display_col = None
    period_sort_col = None

    for col in df.columns:
        if "period_display" in col:
            period_display_col = col
        elif "period_sort" in col:
            period_sort_col = col
        elif col == "period":
            period_col = col

    if not period_col:
        period_col = "period" if "period" in df.columns else df.columns[0]

    return period_col, period_display_col, period_sort_col


def render_newborn_trend_chart_section(
    kpi_selection,
    patient_df,
    facility_uids,
    display_names,
    bg_color,
    text_color,
    comparison_mode="overall",
    facilities_by_region=None,
    region_names=None,
):
    """Render the trend chart for newborn KPIs with KPI-specific program stage dates"""

    # STANDARDIZE COLUMN NAMES
    if "orgUnit" not in patient_df.columns:
        for col in patient_df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                patient_df = patient_df.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in patient_df.columns:
        st.error("❌ 'orgUnit' column not found in data. Cannot filter by UIDs.")
        return

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_newborn_kpi_filtered_dataframe(patient_df, kpi_selection)

    # Get KPI configuration for labels
    kpi_config = get_newborn_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)
    value_name = kpi_config.get("value_name", "Value")

    if kpi_df.empty:
        st.info("⚠️ No data available for trend analysis.")
        return

    # Apply UID filter
    working_df = kpi_df.copy()
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()

    # Get date range filters from session state
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # SPECIAL HANDLING FOR SIMPLIFIED KPIs WITH SINGLE TABLE DISPLAY
    if is_simplified_kpi(kpi_selection):
        _render_simplified_kpi_trend_chart(
            kpi_selection,
            working_df,
            chart_title,
            bg_color,
            text_color,
            facility_uids,
            date_range_filters,
        )
        return

    # FIXED: Use correct date column function
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(kpi_selection)

    # Prepare data using the correct date column
    if date_column not in working_df.columns:
        # Fallback to original function
        date_column = get_relevant_date_column_for_newborn_kpi(kpi_selection)

    # Filter by date column
    if date_column in working_df.columns:
        # Convert to datetime
        working_df["enrollment_date"] = pd.to_datetime(
            working_df[date_column], errors="coerce"
        )

        # Apply date range filtering
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                working_df = working_df[
                    (working_df["enrollment_date"] >= start_dt)
                    & (working_df["enrollment_date"] < end_dt)
                ].copy()

        # Filter out rows without valid dates
        working_df = working_df[working_df["enrollment_date"].notna()].copy()
    else:
        st.warning(
            f"⚠️ Required date column '{date_column}' not found for {kpi_selection}"
        )
        return

    if working_df.empty:
        st.warning(
            f"⚠️ No data available for {kpi_selection} using date column: '{date_column}'"
        )
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "enrollment_date", period_label)
    except Exception as e:
        st.error(f"Error assigning periods: {str(e)}")
        return

    # Get unique periods in order
    unique_periods = working_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Create period data
    period_data = []

    for _, row in unique_periods.iterrows():
        period_display = row["period_display"]
        period_sort = row["period_sort"]

        # Get data for this period
        period_df = working_df[working_df["period_display"] == period_display]

        if not period_df.empty:
            # Compute KPI using date-filtered data WITH ALL SUPPORT
            numerator, denominator, _ = (
                get_numerator_denominator_for_newborn_kpi_with_all(
                    period_df,
                    kpi_selection,
                    facility_uids,
                    date_range_filters,
                )
            )

            # For Admitted Newborns, the value is the count (numerator)
            if kpi_selection == "Admitted Newborns":
                value = float(numerator)
            else:
                value = (numerator / denominator * 100) if denominator > 0 else 0

            period_data.append(
                {
                    "period": period_display,
                    "period_display": period_display,
                    "period_sort": period_sort,
                    "value": value,
                    "numerator": int(numerator),
                    "denominator": int(denominator),
                }
            )

    if not period_data:
        st.info("⚠️ No period data available for chart.")
        return

    # Create DataFrame
    group = pd.DataFrame(period_data)
    group = group.sort_values("period_sort")

    # Render the chart WITH TABLE
    try:
        # SPECIAL HANDLING FOR ADMITTED NEWBORNS
        if kpi_selection == "Admitted Newborns":
            render_admitted_newborns_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                value_name,
                facility_uids,
            )
        else:
            # Standard trend chart for all other KPIs
            render_newborn_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
            )
    except Exception as e:
        st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")


def render_newborn_comparison_chart(
    kpi_selection,
    patient_df=None,
    comparison_mode="facility",
    display_names=None,
    facility_uids=None,
    facilities_by_region=None,
    region_names=None,
    bg_color="#FFFFFF",
    text_color=None,
    is_national=False,
    filtered_patients=None,
    show_chart=True,  # New argument
):
    """Render comparison charts for both national and regional views WITH SINGLE TABLES"""

    df_to_use = filtered_patients if filtered_patients is not None else patient_df

    if df_to_use is None or df_to_use.empty:
        st.info("⚠️ No data available for comparison.")
        return

    # SPECIAL HANDLING FOR SIMPLIFIED KPIs WITH SINGLE TABLE DISPLAY
    if is_simplified_kpi(kpi_selection):
        _render_simplified_kpi_comparison_chart(
            kpi_selection,
            df_to_use,
            comparison_mode,
            display_names,
            facility_uids,
            facilities_by_region,
            region_names,
            bg_color,
            text_color,
            is_national,
            show_chart=show_chart,
        )
        return

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_newborn_kpi_filtered_dataframe(df_to_use, kpi_selection)
    df_to_use = kpi_df

    kpi_config = get_newborn_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)
    value_name = kpi_config.get("value_name", "Value")

    # STANDARDIZE COLUMN NAMES
    if "orgUnit" not in df_to_use.columns:
        for col in df_to_use.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                df_to_use = df_to_use.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in df_to_use.columns:
        st.error("❌ Facility identifier column not found. Cannot perform comparison.")
        return

    # Get date range filters
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    if comparison_mode == "facility":
        comparison_data = []

        for facility_uid, facility_name in zip(facility_uids, display_names):
            # Filter data for this specific facility
            facility_df = df_to_use[df_to_use["orgUnit"] == facility_uid].copy()

            if facility_df.empty:
                comparison_data.append(
                    {
                        "period_display": "All Periods",
                        "orgUnit": facility_uid,
                        "orgUnit_name": facility_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Get correct date column for this KPI
            date_column = get_relevant_date_column_for_newborn_kpi_with_all(
                kpi_selection
            )

            # Prepare data for this facility
            if date_column in facility_df.columns:
                facility_df["event_date"] = pd.to_datetime(
                    facility_df[date_column], errors="coerce"
                )

                # Apply date filtering
                if date_range_filters:
                    start_date = date_range_filters.get("start_date")
                    end_date = date_range_filters.get("end_date")

                    if start_date and end_date:
                        start_dt = pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                        facility_df = facility_df[
                            (facility_df["event_date"] >= start_dt)
                            & (facility_df["event_date"] < end_dt)
                        ].copy()

                # Filter valid dates
                facility_df = facility_df[facility_df["event_date"].notna()].copy()
            else:
                # Skip if date column not found
                continue

            if facility_df.empty:
                continue

            # Assign periods
            period_label = st.session_state.get("period_label", "Monthly")
            try:
                facility_df = assign_period(facility_df, "event_date", period_label)
            except:
                continue

            # Group by period for this facility
            for period_display, period_group in facility_df.groupby("period_display"):
                if not period_group.empty:
                    # Compute KPI WITH ALL SUPPORT - PASS ONLY THIS FACILITY'S UID
                    numerator, denominator, _ = (
                        get_numerator_denominator_for_newborn_kpi_with_all(
                            period_group,
                            kpi_selection,
                            [facility_uid],  # CRITICAL FIX: Pass only this facility UID
                            date_range_filters,
                        )
                    )

                    # For Admitted Newborns, value is the count (numerator)
                    if kpi_selection == "Admitted Newborns":
                        value = float(numerator)
                    else:
                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                    comparison_data.append(
                        {
                            "period_display": period_display,
                            "orgUnit": facility_uid,
                            "orgUnit_name": facility_name,
                            "value": value,
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        if not comparison_data:
            st.info("⚠️ No comparison data available.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        if "orgUnit_name" in comparison_df.columns:
            comparison_df = comparison_df.rename(columns={"orgUnit_name": "Facility"})

        # RENDER TABLE IF CHART IS HIDDEN
        if not show_chart:
            st.markdown(f"### {chart_title} - Comparison Table")
            
            # Simplified dataframe for display
            cols_to_show = ["period_display", "Facility", "value", "numerator", "denominator"]
            
            # Ensure columns exist
            cols_to_use = [c for c in cols_to_show if c in comparison_df.columns]
            
            display_df = comparison_df[cols_to_use].copy()
            
            # Rename for display
            display_df = display_df.rename(columns={
                "period_display": "Period",
                "value": value_name if kpi_selection == "Admitted Newborns" else (
                    "Value (%)" if "Rate" in kpi_selection or "%" in kpi_selection else "Value"
                ),
                "numerator": numerator_label,
                "denominator": denominator_label
            })
            
            st.dataframe(display_df, use_container_width=True)
            return

        # Call the appropriate chart function
        if kpi_selection == "Admitted Newborns":
            render_admitted_newborns_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                value_name=value_name,
            )
        else:
            # Standard comparison for non-simplified KPIs
            render_newborn_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=f"{chart_title} - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )

    elif comparison_mode == "region" and is_national:
        region_data = []

        region_facility_mapping = {}
        for region_name in region_names:
            region_facility_mapping[region_name] = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

        for region_name in region_names:
            region_facility_uids = region_facility_mapping.get(region_name, [])
            if not region_facility_uids:
                continue

            # Filter data for THIS REGION ONLY
            region_df = df_to_use[
                df_to_use["orgUnit"].isin(region_facility_uids)
            ].copy()

            if region_df.empty:
                # Add zero values for empty regions
                region_data.append(
                    {
                        "period_display": "All Periods",
                        "Region": region_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Get correct date column
            date_column = get_relevant_date_column_for_newborn_kpi_with_all(
                kpi_selection
            )

            if date_column in region_df.columns:
                region_df["event_date"] = pd.to_datetime(
                    region_df[date_column], errors="coerce"
                )

                # Apply date filtering
                if date_range_filters:
                    start_date = date_range_filters.get("start_date")
                    end_date = date_range_filters.get("end_date")

                    if start_date and end_date:
                        start_dt = pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                        region_df = region_df[
                            (region_df["event_date"] >= start_dt)
                            & (region_df["event_date"] < end_dt)
                        ].copy()

                # Filter valid dates
                region_df = region_df[region_df["event_date"].notna()].copy()
            else:
                # Skip if date column not found
                continue

            if region_df.empty:
                region_data.append(
                    {
                        "period_display": "All Periods",
                        "Region": region_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Assign periods
            period_label = st.session_state.get("period_label", "Monthly")
            try:
                region_df = assign_period(region_df, "event_date", period_label)
            except:
                continue

            # Group by period for this region
            for period_display, period_group in region_df.groupby("period_display"):
                if not period_group.empty:
                    # Compute KPI WITH ALL SUPPORT - PASS ONLY THIS REGION'S FACILITY UIDS
                    numerator, denominator, _ = (
                        get_numerator_denominator_for_newborn_kpi_with_all(
                            period_group,  # This should already be filtered to this region
                            kpi_selection,
                            region_facility_uids,  # Pass region's facility UIDs
                            date_range_filters,
                        )
                    )

                    if kpi_selection == "Admitted Newborns":
                        value = float(numerator)
                    else:
                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                    region_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": value,
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        if not region_data:
            st.info("⚠️ No comparison data available for regions.")
            return

        region_df = pd.DataFrame(region_data)

        # Call the appropriate region comparison function
        if kpi_selection == "Admitted Newborns":
            render_admitted_newborns_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                value_name=value_name,
            )
        else:
            # For non-simplified KPIs, use the original function
            render_newborn_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=f"{chart_title} - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
    else:
        if comparison_mode == "region":
            st.info(
                "⚠️ Region comparison is only available in national view when region data is provided."
            )
        else:
            st.info("⚠️ Invalid comparison mode selected.")


def _render_simplified_kpi_trend_chart(
    kpi_selection,
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """Render trend chart for simplified KPIs (Birth Weight, KMC, CPAP) WITH SINGLE TABLE DISPLAY"""
    # Get KPI configuration
    kpi_config = get_newborn_kpi_config(kpi_selection)
    category = kpi_config.get("category", "")

    # Get date column for this simplified KPI
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(kpi_selection)

    if date_column not in working_df.columns:
        st.warning(
            f"⚠️ Required date column '{date_column}' not found for {kpi_selection}"
        )
        return

    # Convert to datetime and filter by date range
    working_df["event_date"] = pd.to_datetime(working_df[date_column], errors="coerce")

    # Apply date range filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            working_df = working_df[
                (working_df["event_date"] >= start_dt)
                & (working_df["event_date"] < end_dt)
            ].copy()

    # Filter out rows without valid dates
    working_df = working_df[working_df["event_date"].notna()].copy()

    if working_df.empty:
        st.warning(f"⚠️ No data available for {kpi_selection}")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "event_date", period_label)
    except:
        st.error("Error assigning periods")
        return

    # Render based on category USING UPDATED FUNCTIONS WITH GROUP BARS
    if category == "birth_weight":
        render_birth_weight_trend_chart(
            working_df,
            "period_display",
            chart_title,
            bg_color,
            text_color,
            facility_uids,
        )
    elif category == "kmc":
        render_kmc_coverage_trend_chart(
            working_df,
            "period_display",
            chart_title,
            bg_color,
            text_color,
            facility_uids,
            numerator_name=kpi_config.get("numerator_name", "KMC Cases"),
            denominator_name=kpi_config.get("denominator_name", "Total Newborns"),
        )
    elif category == "cpap_general":
        # "General CPAP Coverage" is retired; route to by-weight visualization if encountered.
        render_cpap_by_weight_trend_chart(
            working_df,
            "period_display",
            chart_title,
            bg_color,
            text_color,
            facility_uids,
            numerator_name=kpi_config.get("numerator_name", "CPAP Cases"),
            denominator_name=kpi_config.get("denominator_name", "Total Newborns"),
        )
    elif category == "cpap_rds":
        render_cpap_rds_trend_chart(
            working_df,
            "period_display",
            chart_title,
            bg_color,
            text_color,
            facility_uids,
            numerator_name=kpi_config.get("numerator_name", "CPAP Cases"),
            denominator_name=kpi_config.get("denominator_name", "Total RDS Cases"),
        )
    elif category == "cpap_by_weight":
        render_cpap_by_weight_trend_chart(
            working_df,
            "period_display",
            chart_title,
            bg_color,
            text_color,
            facility_uids,
            numerator_name=kpi_config.get("numerator_name", "CPAP Cases"),
            denominator_name=kpi_config.get("denominator_name", "Total Newborns"),
        )
    else:
        st.warning(f"⚠️ Unsupported simplified KPI category: {category}")


def _render_simplified_kpi_comparison_chart(
    kpi_selection,
    df_to_use,
    comparison_mode,
    display_names,
    facility_uids,
    facilities_by_region,
    region_names,
    bg_color,
    text_color,
    is_national,
    show_chart=True,  # New argument
):
    """Render comparison chart for simplified KPIs WITH SINGLE TABLE DISPLAY - UPDATED WITH GROUP BARS"""
    kpi_config = get_newborn_kpi_config(kpi_selection)
    chart_title = kpi_config.get("title", kpi_selection)
    category = kpi_config.get("category", "")
    comparison_type = kpi_config.get("comparison_type", "rates")

    # Get date column
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(kpi_selection)

    if date_column not in df_to_use.columns:
        st.warning(
            f"⚠️ Required date column '{date_column}' not found for {kpi_selection}"
        )
        return

    # Convert to datetime
    df_to_use["event_date"] = pd.to_datetime(df_to_use[date_column], errors="coerce")
    df_to_use = df_to_use[df_to_use["event_date"].notna()].copy()

    # Get date range filters
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Apply date filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            df_to_use = df_to_use[
                (df_to_use["event_date"] >= start_dt)
                & (df_to_use["event_date"] < end_dt)
            ].copy()

    if df_to_use.empty:
        st.info("⚠️ No data available for comparison.")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        df_to_use = assign_period(df_to_use, "event_date", period_label)
    except:
        st.error("Error assigning periods")
        return

    if comparison_type == "stacked_cases":
        # RENDER TABLE IF CHART IS HIDDEN
        if not show_chart:
            st.markdown(f"### {chart_title} - Comparison Table")
            if comparison_mode == "facility":
                # Create simple table for facility
                st.dataframe(df_to_use[["period_display", "orgUnit", "value"]], use_container_width=True)
            elif comparison_mode == "region":
                 st.dataframe(df_to_use[["period_display", "Region", "value"]], use_container_width=True)
            return

        # Use case counts for birth weight rate (with stacked bars)
        # Use case counts for birth weight rate (with stacked bars)
        if comparison_mode == "facility" and category == "birth_weight":
            render_birth_weight_facility_comparison(
                df_to_use,
                "period_display",
                f"{chart_title} - Facility Comparison",
                bg_color,
                text_color,
                display_names,
                facility_uids,
            )
        elif comparison_mode == "region" and is_national and category == "birth_weight":
            render_birth_weight_region_comparison(
                df_to_use,
                "period_display",
                f"{chart_title} - Region Comparison",
                bg_color,
                text_color,
                region_names,
                facilities_by_region,
                facilities_by_region,
            )
    elif comparison_type == "rates":
        # RENDER TABLE IF CHART IS HIDDEN
        if not show_chart:
            st.markdown(f"### {chart_title} - Comparison Table")
             # Simplified dataframe for display - depends on structure, but generally just show relevant cols
            cols_to_show = ["period_display", "orgUnit", "Facility", "Region"] + [c for c in df_to_use.columns if "rate" in c or "value" in c]
            cols_to_use = [c for c in cols_to_show if c in df_to_use.columns]
            st.dataframe(df_to_use[cols_to_use], use_container_width=True)
            return

        # Use rate comparison functions for KMC and CPAP
        if comparison_mode == "facility":
            if category == "kmc":
                # UPDATED: Call the NEW 3x2 grid comparison function
                from newborns_dashboard.kpi_utils_newborn_simplified import render_kmc_coverage_comparison_chart
                render_kmc_coverage_comparison_chart(
                    df_to_use,
                    comparison_mode="facility",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Facility Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
            elif category == "cpap_general":
                # UPDATED: Call the NEW 3x2 grid comparison function for CPAP
                from newborns_dashboard.kpi_utils_newborn_simplified import render_cpap_by_weight_comparison_chart
                render_cpap_by_weight_comparison_chart(
                    df_to_use,
                    comparison_mode="facility",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Facility Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
            elif category == "cpap_rds":
                # UPDATED: Call the NEW line chart comparison function for CPAP RDS
                from newborns_dashboard.kpi_utils_newborn_simplified import render_cpap_rds_comparison_line_chart
                render_cpap_rds_comparison_line_chart(
                    df_to_use,
                    comparison_mode="facility",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Facility Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
            elif category == "cpap_by_weight":
                # UPDATED: Call the NEW 3x2 grid comparison function for CPAP
                from newborns_dashboard.kpi_utils_newborn_simplified import render_cpap_by_weight_comparison_chart
                render_cpap_by_weight_comparison_chart(
                    df_to_use,
                    comparison_mode="facility",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Facility Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
        elif comparison_mode == "region" and is_national:
            if category == "kmc":
                # UPDATED: Call the NEW 3x2 grid comparison function for regions
                from newborns_dashboard.kpi_utils_newborn_simplified import render_kmc_coverage_comparison_chart
                render_kmc_coverage_comparison_chart(
                    df_to_use,
                    comparison_mode="region",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Region Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
            elif category == "cpap_general":
                # UPDATED: Call the NEW 3x2 grid comparison function for CPAP regions
                from newborns_dashboard.kpi_utils_newborn_simplified import render_cpap_by_weight_comparison_chart
                render_cpap_by_weight_comparison_chart(
                    df_to_use,
                    comparison_mode="region",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Region Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
            elif category == "cpap_rds":
                # UPDATED: Call the NEW line chart comparison function for CPAP RDS regions
                from newborns_dashboard.kpi_utils_newborn_simplified import render_cpap_rds_comparison_line_chart
                render_cpap_rds_comparison_line_chart(
                    df_to_use,
                    comparison_mode="region",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Region Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
            elif category == "cpap_by_weight":
                # UPDATED: Call the NEW 3x2 grid comparison function for CPAP regions
                from newborns_dashboard.kpi_utils_newborn_simplified import render_cpap_by_weight_comparison_chart
                render_cpap_by_weight_comparison_chart(
                    df_to_use,
                    comparison_mode="region",
                    display_names=display_names,
                    facility_uids=facility_uids,
                    facilities_by_region=facilities_by_region,
                    region_names=region_names,
                    period_col="period_display",
                    title=f"{chart_title} - Region Comparison",
                    bg_color=bg_color,
                    text_color=text_color,
                )
    else:
        st.warning(f"⚠️ Unsupported comparison type: {comparison_type}")


def render_newborn_additional_analytics(
    kpi_selection, patient_df, facility_uids, bg_color, text_color
):
    """Render additional analytics charts for newborn KPIs"""
    # For now, no additional analytics
    pass


def normalize_newborn_patient_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single datetime column 'enrollment_date' exists for newborn patient data"""
    if df.empty:
        if "enrollment_date" not in df.columns:
            df["enrollment_date"] = pd.Series(dtype="datetime64[ns]")
        return df

    df = df.copy()

    # Get current KPI to use the right date column
    current_kpi = st.session_state.get("selected_newborn_kpi", "Inborn Rate (%)")

    # Try enrollment_date first if it exists
    if "enrollment_date" in df.columns:
        df["enrollment_date"] = pd.to_datetime(df["enrollment_date"], errors="coerce")
    else:
        # Get the SPECIFIC date column for this KPI
        kpi_date_column = get_relevant_date_column_for_newborn_kpi_with_all(current_kpi)

        # Try KPI-specific date column
        if kpi_date_column and kpi_date_column in df.columns:
            df["enrollment_date"] = pd.to_datetime(df[kpi_date_column], errors="coerce")
        elif "combined_date" in df.columns:
            df["enrollment_date"] = pd.to_datetime(df["combined_date"], errors="coerce")
        else:
            # Look for program stage event dates
            program_stage_date_columns = [
                col
                for col in df.columns
                if col.startswith("event_date_") or col == "event_date"
            ]

            for col in program_stage_date_columns:
                try:
                    df["enrollment_date"] = pd.to_datetime(df[col], errors="coerce")
                    if not df["enrollment_date"].isna().all():
                        break
                except:
                    continue

    # If still no date found, use current date
    if "enrollment_date" not in df.columns or df["enrollment_date"].isna().all():
        df["enrollment_date"] = pd.Timestamp.now().normalize()

    return df


def render_newborn_patient_filter_controls(
    patient_df, container=None, context="newborn"
):
    """Simple filter controls for newborn patient data"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix
    key_suffix = f"_{context}"

    # Time Period options
    time_options = [
        "All Time",
        "Custom Range",
        "Today",
        "This Week",
        "Last Week",
        "This Month",
        "Last Month",
        "This Year",
        "Last Year",
    ]

    # Get current selection
    current_selection = st.session_state.get(f"quick_range{key_suffix}", "All Time")

    # Ensure the value is valid
    if current_selection not in time_options:
        current_selection = "All Time"

    filters["quick_range"] = container.selectbox(
        "📅 Time Period",
        time_options,
        index=time_options.index(current_selection),
        key=f"quick_range{key_suffix}",
    )

    # Get REAL VALID dates from patient dataframe
    min_date, max_date = _get_newborn_patient_date_range(patient_df)

    # Ensure min_date is not earlier than 2000
    if min_date < datetime.date(2000, 1, 1):
        min_date = datetime.date(2000, 1, 1)

    # Ensure max_date is not later than 2035
    if max_date > datetime.date(2035, 12, 31):
        max_date = datetime.date(2035, 12, 31)

    # Set dates based on selection
    if filters["quick_range"] == "All Time":
        filters["start_date"] = min_date
        filters["end_date"] = max_date

    elif filters["quick_range"] == "Custom Range":
        # For Custom Range, use CURRENT YEAR as default
        today = datetime.date.today()
        default_start = datetime.date(today.year, 1, 1)
        default_end = datetime.date(today.year, 12, 31)

        # Ensure defaults are within min/max bounds
        if default_start < min_date:
            default_start = min_date
        if default_end > max_date:
            default_end = max_date

        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=default_start,
                min_value=datetime.date(2000, 1, 1),
                max_value=max_date,
                key=f"start_date{key_suffix}",
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=default_end,
                min_value=min_date,
                max_value=datetime.date(2035, 12, 31),
                key=f"end_date{key_suffix}",
            )
    else:
        # Use the FULL patient dataframe directly
        start_date, end_date = get_date_range(patient_df, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date

    # Aggregation Level
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )

    default_index = (
        available_aggregations.index("Monthly")
        if "Monthly" in available_aggregations
        else 0
    )

    # Get current value from session state if exists
    current_period_label = st.session_state.get("period_label", "Monthly")

    if current_period_label in available_aggregations:
        default_index = available_aggregations.index(current_period_label)

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",
    )

    # Save to session state
    st.session_state.period_label = filters["period_label"]

    # Also save to filters dictionary in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters["period_label"] = filters["period_label"]
    st.session_state.filters["start_date"] = filters["start_date"]
    st.session_state.filters["end_date"] = filters["end_date"]

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # Add placeholder for kpi_selection
    filters["kpi_selection"] = st.session_state.get(
        "selected_newborn_kpi", "Inborn Rate (%)"
    )

    return filters


def _get_newborn_patient_date_range(patient_df):
    """Get REAL min/max dates from newborn patient dataframe - HARDCODED ALL TIME"""
    import datetime

    # ALWAYS return the full 2000-2035 range
    all_time_start = datetime.date(2000, 1, 1)
    all_time_end = datetime.date(2035, 12, 31)

    return all_time_start, all_time_end


def apply_newborn_patient_filters(patient_df, filters, facility_uids=None):
    """Apply filters to newborn patient dataframe - FIXED DATE FILTERING"""
    if patient_df.empty:
        return patient_df

    df = patient_df.copy()

    # Get current KPI selection from session state
    current_kpi = st.session_state.get("selected_newborn_kpi", "Inborn Rate (%)")

    # Get the SPECIFIC date column for this KPI WITH ALL SUPPORT
    kpi_date_column = get_relevant_date_column_for_newborn_kpi_with_all(current_kpi)

    # STANDARDIZE COLUMN NAMES - Ensure we have orgUnit
    if "orgUnit" not in df.columns:
        # Try to find the facility ID column
        for col in df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                df = df.rename(columns={col: "orgUnit"})
                break

    # CRITICAL FIX: Apply facility filter FIRST
    if facility_uids and "orgUnit" in df.columns:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]

        # Check if this is "All Facilities" or actual facility UIDs
        is_all_facilities = (
            len(facility_uids) == 0
            or facility_uids == ["All Facilities"]
            or (len(facility_uids) == 1 and facility_uids[0] == "All Facilities")
        )

        if not is_all_facilities:
            facility_mask = df["orgUnit"].isin(facility_uids)
            df = df[facility_mask].copy()

    # STEP 1: Create a list of possible date columns to check
    date_columns_to_try = []

    # 1. Try KPI-specific date column first
    if kpi_date_column and kpi_date_column in df.columns:
        date_columns_to_try.append(kpi_date_column)

    # 2. Try enrollment_date as fallback (ALWAYS exists for newborn data)
    if "enrollment_date" in df.columns and "enrollment_date" not in date_columns_to_try:
        date_columns_to_try.append("enrollment_date")

    # 3. Try any other date columns
    date_cols = [
        col
        for col in df.columns
        if "date" in col.lower() and col not in date_columns_to_try
    ]
    date_columns_to_try.extend(date_cols[:3])  # Try up to 3 additional date columns

    # 4. Last resort: event_date if it exists
    if "event_date" in df.columns and "event_date" not in date_columns_to_try:
        date_columns_to_try.append("event_date")

    # STEP 2: Apply date filtering if we have date columns
    should_filter_by_date = (
        filters.get("quick_range") != "All Time"
        and filters.get("start_date")
        and filters.get("end_date")
        and date_columns_to_try  # We have date columns to check
    )

    # Always create event_date column, but only filter if not "All Time"
    event_date_created = False

    for date_col in date_columns_to_try:
        if date_col in df.columns:
            try:
                # Convert date column to datetime - CORRECT METHOD
                df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")

                # Check if we have valid dates
                valid_dates = df["event_date"].notna().sum()
                if valid_dates > 0:
                    event_date_created = True

                    # Apply date filtering if needed
                    if should_filter_by_date:
                        start_date = pd.Timestamp(filters["start_date"])
                        end_date = pd.Timestamp(filters["end_date"]) + pd.Timedelta(
                            days=1
                        )

                        # Filter by date range
                        date_mask = (df["event_date"] >= start_date) & (
                            df["event_date"] < end_date
                        )
                        df = df[date_mask].copy()

                    # Break after first successful date column
                    break

            except Exception as e:
                # Try next date column if this one fails
                continue

    # If no date column worked, create a default event_date
    if not event_date_created:
        df["event_date"] = pd.Timestamp.now().normalize()

    # STEP 3: Assign periods for trend analysis
    if "period_label" not in filters:
        filters["period_label"] = st.session_state.get("period_label", "Monthly")

    try:
        valid_date_rows = df["event_date"].notna().sum()
        if valid_date_rows > 0:
            df = assign_period(df, "event_date", filters["period_label"])
        else:
            # Create empty period columns
            df["period_display"] = ""
            df["period_sort"] = 0
    except Exception as e:
        # Create empty period columns as fallback
        df["period_display"] = ""
        df["period_sort"] = 0

    # Save to session state
    st.session_state.period_label = filters["period_label"]
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters.update(filters)

    # OPTIMIZATION: Filter for current KPI
    if current_kpi in NEWBORN_KPI_COLUMN_REQUIREMENTS:
        df = get_newborn_kpi_filtered_dataframe(df, current_kpi)

    return df


def get_newborn_period_data_for_kpi(kpi_selection, patient_df, facility_uids):
    """Get period-based data for a specific newborn KPI from patient data"""
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(kpi_selection)

    # Extract period columns using program stage specific date
    df_with_periods = extract_period_columns_newborn(patient_df, date_column)

    if df_with_periods.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df_with_periods = df_with_periods[
            df_with_periods["orgUnit"].isin(facility_uids)
        ]

    # Sort by period
    if "period_sort" in df_with_periods.columns:
        df_with_periods = df_with_periods.sort_values("period_sort")

    return df_with_periods


def compute_newborn_kpi_for_period(kpi_selection, period_df, facility_uids):
    """Compute newborn KPI for a specific period using patient data"""
    if period_df.empty:
        return _get_newborn_default_kpi_data(kpi_selection)

    # For ALL KPIs, use get_numerator_denominator_for_newborn_kpi WITH ALL SUPPORT
    numerator, denominator, value = get_numerator_denominator_for_newborn_kpi_with_all(
        period_df, kpi_selection, facility_uids
    )

    return {
        "value": float(value),
        "numerator": int(numerator),
        "denominator": int(denominator),
        "kpi_name": kpi_selection,
    }


def _get_newborn_default_kpi_data(kpi_selection):
    """Get default empty data for newborn KPIs"""
    return {"value": 0.0, "numerator": 0, "denominator": 0}


def get_date_column_from_newborn_df(df, kpi_selection):
    """Get the appropriate date column from newborn patient-level data based on KPI"""
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(kpi_selection)

    if date_column in df.columns:
        return date_column

    st.warning(f"⚠️ Required date column '{date_column}' not found for {kpi_selection}")
    return None


# ---------------- Export all functions ----------------
__all__ = [
    # KPI configurations
    "NEWBORN_KPI_MAPPING",
    "NEWBORN_KPI_OPTIONS",
    "NEWBORN_KPI_GROUPS",
    "NEWBORN_KPI_COLUMN_REQUIREMENTS",
    "SIMPLIFIED_KPI_DATE_COLUMNS",
    # Main functions
    "get_newborn_kpi_filtered_dataframe",
    "get_text_color",
    "get_newborn_kpi_config",
    "is_simplified_kpi",
    "get_relevant_date_column_for_newborn_kpi_with_all",
    "get_numerator_denominator_for_newborn_kpi_with_all",
    "render_newborn_kpi_tab_navigation",
    "get_period_columns",
    "render_newborn_trend_chart_section",
    "render_newborn_comparison_chart",
    "render_newborn_additional_analytics",
    # Simplified KPI helper functions (WITH SINGLE TABLE DISPLAY)
    "_render_simplified_kpi_trend_chart",
    "_render_simplified_kpi_comparison_chart",
    # Date handling functions
    "normalize_newborn_patient_dates",
    "render_newborn_patient_filter_controls",
    "_get_newborn_patient_date_range",
    "apply_newborn_patient_filters",
    # Data processing functions
    "get_newborn_period_data_for_kpi",
    "compute_newborn_kpi_for_period",
    "_get_newborn_default_kpi_data",
    "get_date_column_from_newborn_df",
    # CPAP KPI names
    "CPAP for RDS",
    "CPAP Coverage by Birth Weight",
    # NEW CPAP comparison functions for import
    "render_cpap_rds_facility_comparison",
    "render_cpap_rds_region_comparison",
    "render_kmc_facility_comparison",
    "render_kmc_region_comparison",
]
