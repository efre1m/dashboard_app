# kpi_newborn.py - UPDATED WITH DATASET COLUMN NAMES, REMOVED CULTURE TABS

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
import datetime
import logging
from utils.dash_co import (
    SOURCE_DEFAULT_SELECTION,
    SOURCE_FILTER_ALL,
    _available_source_options,
    _normalize_source_value,
    parse_dashboard_dates,
)
from utils.ethiopian_periods import (
    build_period_definitions_from_denominator,
    map_gregorian_dates_to_ethiopian_yearmonths,
)

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
    # CPAP timing computation
    compute_cpap_timing_data,
    CPAP_TIMING_BIRTH_COL,
    BIRTH_WEIGHT_CATEGORIES,
    compute_kmc_timing_data,
    get_kmc_status_for_tei,
    KMC_COLUMNS,
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
    "Inborn Hypothermia at Admission Rate (%)": {
        "title": "Hypothermia at admission in Inborn Babies (%)",
        "numerator_name": "Inborn Hypothermia Cases",
        "denominator_name": "Total Inborn Babies",
    },
    "Outborn Hypothermia at Admission Rate (%)": {
        "title": "Hypothermia at admission in Outborn Babies (%)",
        "numerator_name": "Outborn Hypothermia Cases",
        "denominator_name": "Total Outborn Babies",
    },
    "Hypothermia on Admission Rate (%)": {
        "title": "Hypothermia on Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Not hypothermic at admission (%)": {
        "title": "Not hypothermic at admission (%)",
        "numerator_name": "Not Hypothermic Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Not hypothermic at admission inborn (%)": {
        "title": "Not hypothermic at admission inborn (%)",
        "numerator_name": "Inborn Not Hypothermic Cases",
        "denominator_name": "Total Inborn Babies",
    },
    "Not hypothermic at admission outborn (%)": {
        "title": "Not hypothermic at admission outborn (%)",
        "numerator_name": "Outborn Not Hypothermic Cases",
        "denominator_name": "Total Outborn Babies",
    },
    "Neonatal Mortality Rate (%)": {
        "title": "Neonatal Mortality Rate (%)",
        "numerator_name": "Dead Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Inborn Hypothermia at Admission Rate (%)": {
        "title": "Hypothermia at admission in Inborn Babies (%)",
        "numerator_name": "Inborn Hypothermia Cases",
        "denominator_name": "Total Inborn Babies",
    },
    "Outborn Hypothermia at Admission Rate (%)": {
        "title": "Hypothermia at admission in Outborn Babies (%)",
        "numerator_name": "Outborn Hypothermia Cases",
        "denominator_name": "Total Outborn Babies",
    },
    "Admitted Newborns": {
        "title": "Total Admitted Newborns",
        "value_name": "Admitted Newborns",
    },
    "Newborn Coverage Rate": {
        "title": "Newborn Coverage Rate",
        "numerator_name": "Admitted Newborns",
        "denominator_name": "Aggregated Admissions",
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
        "title": "CPAP Coverage (Dual Panel)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Eligible Babies / RDS Cases",
        "type": "simplified",
        "category": "cpap_rds",
        "comparison_type": "rates",
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
    # VITAL MONITORING KPIs
    "Temperature Taken at Admission (%)": {
        "title": "Temperature Taken at Admission (%)",
        "numerator_name": "Temp. Taken",
        "denominator_name": "Total Admitted Newborns",
    },
    "Birth Weight Taken (%)": {
        "title": "Birth Weight Taken (%)",
        "numerator_name": "BWeight. Taken",
        "denominator_name": "Total Admitted Newborns",
    },
    "Weight Taken at Admission (%)": {
        "title": "Weight Taken at Admission (%)",
        "numerator_name": "Weight. Taken",
        "denominator_name": "Total Admitted Newborns",
    },
}

# KPI options for newborn dashboard (REMOVED CULTURE KPIs)
NEWBORN_KPI_OPTIONS = [
    "Inborn Rate (%)",
    "Outborn Rate (%)",
    "Hypothermia on Admission Rate (%)",
    "Neonatal Mortality Rate (%)",
    "Inborn Hypothermia at Admission Rate (%)",
    "Outborn Hypothermia at Admission Rate (%)",
    "Not hypothermic at admission (%)",
    "Not hypothermic at admission inborn (%)",
    "Not hypothermic at admission outborn (%)",
    "Admitted Newborns",
    "Newborn Coverage Rate",
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
         "Newborn Coverage Rate",
    ],
    "👶 Birth": [
        "Inborn Rate (%)",
        "Outborn Rate (%)",
        "Birth Weight Rate",
    ],
    "🌡️ Hypothermia": [
        "Hypothermia on Admission Rate (%)",
        "Inborn Hypothermia at Admission Rate (%)",
        "Outborn Hypothermia at Admission Rate (%)",
        "Not hypothermic at admission (%)",
        "Not hypothermic at admission inborn (%)",
        "Not hypothermic at admission outborn (%)",
    ],
    "🏥 CPAP": [
        "CPAP for RDS",
        "CPAP Coverage by Birth Weight",
    ],
    "👶 KMC": [
        "KMC Coverage by Birth Weight",
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
    "Inborn Hypothermia at Admission Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Outborn Hypothermia at Admission Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Not hypothermic at admission (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Not hypothermic at admission inborn (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "place_of_delivery_nicu_admission_careform",
        "temp_at_admission_nicu_admission_careform",
        "event_date_nicu_admission_careform",
    ],
    "Not hypothermic at admission outborn (%)": [
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
        "birth_weight_n_nicu_admission_careform",
        "baby_placed_on_cpap_neonatal_referral_form",
        "sub_categories_of_prematurity_n_discharge_care_form",
        "lowest_recorded_oxygen_saturation_pct_observations_and_nursing_care_2",
        "lowest_recorded_oxygen_saturation",
        "lowest_recorded_oxygen_saturation_pct",
        "lowest_recorded_oxygen_saturation_observations_and_nursing_care_2",
        "Lowest recorded oxygen saturation (%)",
        "Lowest recorded oxygen saturation",
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
    "Newborn Coverage Rate": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
    ],
    # VITAL MONITORING
    "Temperature Taken at Admission (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "temp_at_admission_nicu_admission_careform",
    ],
    "Birth Weight Taken (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_weight_n_nicu_admission_careform",
    ],
    "Weight Taken at Admission (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "weight_at_admission_n_nicu_admission_careform",
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

    if kpi_name == "Newborn Coverage Rate":
        return "enrollment_date"

    # Use original function for non-simplified KPIs
    return get_relevant_date_column_for_newborn_kpi(kpi_name)


def get_numerator_denominator_for_newborn_kpi_with_all(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for a specific newborn KPI with date range filtering
    Supports V1 and simplified KPIs
    """
    if kpi_name == "Newborn Coverage Rate":
        from newborns_dashboard.kpi_newborn_coverage_rate import (
            get_numerator_denominator_for_newborn_coverage_rate,
        )

        return get_numerator_denominator_for_newborn_coverage_rate(
            df, facility_uids, date_range_filters
        )

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

    # Create main KPI group tabs - UPDATED TO 7 TABS & REORDERED
    # Enrollment -> Birth -> Hypothermia -> Vital Monitoring -> Intervention -> Mortality -> Data Quality
    tab_enrollment, tab_birth, tab_thermal, tab_vital, tab_cpap, tab_kmc, tab_mortality, tab_dq = st.tabs(
        [
            "Enrollment",
            "Birth",
            "Hypothermia",
            "Vital Monitoring",
            "CPAP",
            "KMC",
            "Mortality",
            "Data Quality",
        ]
    )

    selected_kpi = st.session_state.selected_newborn_kpi

    # Migration: replace individual hypothermia KPIs with combined marker
    hypo_kpis = set(NEWBORN_KPI_GROUPS.get("🌡️ Hypothermia", []))
    if selected_kpi in hypo_kpis:
        st.session_state.selected_newborn_kpi = HYPO_COMBINED_MARKER
        selected_kpi = HYPO_COMBINED_MARKER

    with tab_enrollment:
        # Enrollment - 2 buttons
        cols = st.columns(5)
        with cols[0]:
             if st.button("Admitted Newborns", key="admitted_newborns_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Admitted Newborns" else "secondary")):
                selected_kpi = "Admitted Newborns"
        with cols[1]:
             if st.button("Coverage Rate", key="newborn_coverage_rate_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "Newborn Coverage Rate" else "secondary")):
                selected_kpi = "Newborn Coverage Rate"

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
        st.markdown(
            """
            <div style="
                background-color:#FFF4E5;
                border-left:6px solid #E67E22;
                padding:14px 16px;
                border-radius:8px;
                margin: 0 0 14px 0;
                font-size:16px;
                line-height:1.45;
            ">
                <strong>To see patients with missing variables, please visit:</strong>
                <a href="http://143.198.108.241:8082/" target="_blank" style="font-weight:700; color:#B45309; text-decoration:underline;">
                    Open Missing Variables Patient List
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
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


    with tab_thermal:
        cols = st.columns(3)
        with cols[0]:
            if st.button("Indicator Coverage Run Charts", key="hypo_combined_btn", use_container_width=True,
                         type=("primary" if selected_kpi == HYPO_COMBINED_MARKER else "secondary")):
                selected_kpi = HYPO_COMBINED_MARKER
        with cols[1]:
            if st.button("Quality of Care", key="hypo_qoc_btn", use_container_width=True,
                         type=("primary" if selected_kpi == HYPO_QOC_MARKER else "secondary")):
                selected_kpi = HYPO_QOC_MARKER

    with tab_vital:
        cols = st.columns(3)
        with cols[0]:
            if st.button("Indicator Coverage Run Charts", key="vital_monitoring_btn", use_container_width=True,
                         type=("primary" if selected_kpi == VITAL_MONITORING_MARKER else "secondary")):
                selected_kpi = VITAL_MONITORING_MARKER

    with tab_cpap:
        # CPAP - 3 buttons (General CPAP removed)
        cols = st.columns(5)
        with cols[0]:
            if st.button("CPAP Coverage", key="cpap_rds_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "CPAP for RDS" else "secondary")):
                selected_kpi = "CPAP for RDS"
        with cols[1]:
            if st.button("CPAP by Weight", key="cpap_by_weight_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "CPAP Coverage by Birth Weight" else "secondary")):
                selected_kpi = "CPAP Coverage by Birth Weight"
        with cols[2]:
            if st.button("Quality of Care", key="cpap_timing_qoc_btn", use_container_width=True,
                         type=("primary" if selected_kpi == CPAP_TIMING_QOC_MARKER else "secondary")):
                selected_kpi = CPAP_TIMING_QOC_MARKER

    with tab_kmc:
        # KMC - 2 buttons
        cols = st.columns(5)
        with cols[0]:
            if st.button("KMC Coverage", key="kmc_btn", use_container_width=True,
                         type=("primary" if selected_kpi == "KMC Coverage by Birth Weight" else "secondary")):
                selected_kpi = "KMC Coverage by Birth Weight"
        with cols[1]:
            if st.button("Quality of Care", key="kmc_timing_qoc_btn", use_container_width=True,
                         type=("primary" if selected_kpi == KMC_TIMING_QOC_MARKER else "secondary")):
                selected_kpi = KMC_TIMING_QOC_MARKER

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

    # SPECIAL HANDLING: Hypothermia combined view
    if kpi_selection == HYPO_COMBINED_MARKER:
        _render_hypothermia_combined_trend_chart(
            working_df,
            "Indicator Coverage Run Charts",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters,
        )
        return

    # SPECIAL HANDLING: Hypothermia Quality of Care stacked chart
    if kpi_selection == HYPO_QOC_MARKER:
        _render_hypothermia_qoc_trend_chart(
            working_df,
            "Thermal Status at Admission",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters,
        )
        return

    # SPECIAL HANDLING: CPAP Timing Quality of Care stacked chart
    if kpi_selection == CPAP_TIMING_QOC_MARKER:
        _render_cpap_timing_qoc_trend_chart(
            working_df,
            "CPAP Quality of Care",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters,
        )
        return

    # SPECIAL HANDLING: KMC Timing Quality of Care stacked chart
    if kpi_selection == KMC_TIMING_QOC_MARKER:
        _render_kmc_timing_qoc_trend_chart(
            working_df,
            "KMC Quality of Care",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters,
        )
        return

    # SPECIAL HANDLING: Vital Monitoring combined chart
    if kpi_selection == VITAL_MONITORING_MARKER:
        _render_vital_monitoring_trend_chart(
            working_df,
            "Indicator Coverage Run Charts",
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

    # SPECIAL HANDLING: Newborn Coverage Rate should display periods even when numerator is zero.
    # The denominator comes from an external aggregated admissions file, so months/years with no
    # patient rows still need to appear as 0% when the denominator exists (>0).
    if kpi_selection == "Newborn Coverage Rate":
        try:
            from newborns_dashboard.kpi_newborn_coverage_rate import (
                load_newborn_coverage_denominator,
                _sum_denominator_for_regions,
                _sum_denominator_for_facilities,
                _resolve_facility_names,
            )
        except Exception as e:
            st.error(f"Failed to load Newborn Coverage Rate denominator utilities: {e}")
            return

        den_long = load_newborn_coverage_denominator()
        if den_long is None or den_long.empty:
            st.warning(
                "⚠️ Newborn Coverage Rate denominator data is missing or empty. "
                "Please check `utils/aggregated_admission_newborn.xlsx`."
            )
            return

        period_label = st.session_state.get("period_label", "Monthly")
        if "filters" in st.session_state and "period_label" in st.session_state.filters:
            period_label = st.session_state.filters["period_label"]
        if period_label not in ["Monthly", "Yearly"]:
            period_label = "Monthly"

        # Determine the time window from filters; fall back to available denominator range.
        start_date = date_range_filters.get("start_date") if date_range_filters else None
        end_date = date_range_filters.get("end_date") if date_range_filters else None

        period_defs = build_period_definitions_from_denominator(
            den_long,
            period_label,
            start_date=start_date,
            end_date=end_date,
        )
        if not period_defs:
            st.info("Warning: No denominator periods available for Newborn Coverage Rate.")
            return

        # Precompute patient numerator keys (df may be empty; that's OK).
        numerator_df = working_df.copy()
        if not numerator_df.empty:
            numerator_df["enrollment_date"] = pd.to_datetime(
                numerator_df["enrollment_date"], errors="coerce"
            )
            numerator_df = numerator_df[numerator_df["enrollment_date"].notna()].copy()
            numerator_df["_ethiopian_yearmonth"] = map_gregorian_dates_to_ethiopian_yearmonths(
                numerator_df["enrollment_date"],
                den_long["yearmonth"].dropna().astype(int).unique().tolist(),
            )
            numerator_df = numerator_df[numerator_df["_ethiopian_yearmonth"].notna()].copy()

        # Determine denominator scope (regions vs facilities) based on the current dashboard filter mode.
        region_scope = None
        facility_scope_names = None
        user = st.session_state.get("user", {}) or {}
        role = str(user.get("role") or "").lower()

        if comparison_mode == "region" and region_names:
            region_scope = list(region_names)
        elif facilities_by_region and display_names == ["All Facilities"]:
            if role == "dq_officer":
                facility_scope_names = [
                    facility_name
                    for facilities in facilities_by_region.values()
                    for facility_name, _ in facilities
                    if facility_name
                ]
            else:
                region_scope = list(facilities_by_region.keys())
        elif (
            comparison_mode == "facility"
            and display_names
            and facility_uids
            and display_names != ["All Facilities"]
            and len(display_names) == len(facility_uids)
        ):
            facility_scope_names = list(display_names)
        else:
            facility_scope_names = _resolve_facility_names(facility_uids, df=numerator_df)
            if not facility_scope_names and facilities_by_region and not facility_uids:
                region_scope = list(facilities_by_region.keys())

        period_rows = []
        for pdef in period_defs:
            if numerator_df.empty:
                numerator = 0
            else:
                period_df = numerator_df[
                    numerator_df["_ethiopian_yearmonth"].isin(pdef["yearmonths"])
                ]
                numerator = (
                    int(period_df["tei_id"].dropna().nunique())
                    if "tei_id" in period_df.columns
                    else int(len(period_df))
                )

            if region_scope:
                denominator = _sum_denominator_for_regions(
                    den_long, region_scope, yearmonths=pdef["yearmonths"]
                )
            else:
                denominator = _sum_denominator_for_facilities(
                    den_long, facility_scope_names or [], yearmonths=pdef["yearmonths"]
                )

            value = (numerator / denominator * 100) if denominator > 0 else 0.0
            period_rows.append(
                {
                    "period": pdef["period_display"],
                    "period_display": pdef["period_display"],
                    "period_sort": pdef["period_sort"],
                    "value": float(value),
                    "numerator": int(numerator),
                    "denominator": int(denominator),
                }
            )

        if not period_rows:
            st.info("⚠️ No period data available for Newborn Coverage Rate.")
            return

        group = pd.DataFrame(period_rows).sort_values("period_sort")
        try:
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
                forecast_enabled=False,
                forecast_min_points=4,
                forecast_bounds=(0.0, 100.0),
                show_markers=False,
                forecast_show_markers=False,
            )
        except Exception as e:
            st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")
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

    enable_next_period_forecast = False
    # Keep newborn trend lines smooth (no points) for all standard line KPIs.
    use_markers_for_trend = False

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
                forecast_enabled=enable_next_period_forecast,
                forecast_min_points=4,
                forecast_bounds=(0.0, 100.0),
                show_markers=use_markers_for_trend,
                forecast_show_markers=use_markers_for_trend,
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

    # SPECIAL HANDLING: Hypothermia combined view
    if kpi_selection == HYPO_COMBINED_MARKER:
        _render_hypothermia_combined_comparison_chart(
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

    # SPECIAL HANDLING: Hypothermia Quality of Care
    if kpi_selection == HYPO_QOC_MARKER:
        _render_hypothermia_qoc_trend_chart(
            df_to_use,
            "Thermal Status at Admission",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters={},
        )
        return

    # SPECIAL HANDLING: CPAP Timing QOC
    if kpi_selection == CPAP_TIMING_QOC_MARKER:
        _render_cpap_timing_qoc_trend_chart(
            df_to_use,
            "CPAP Quality of Care",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters={},
        )
        return

    # SPECIAL HANDLING: KMC Timing QOC
    if kpi_selection == KMC_TIMING_QOC_MARKER:
        _render_kmc_timing_qoc_trend_chart(
            df_to_use,
            "KMC Quality of Care",
            bg_color,
            text_color,
            facility_uids,
            date_range_filters={},
        )
        return

    # SPECIAL HANDLING: Vital Monitoring
    if kpi_selection == VITAL_MONITORING_MARKER:
        _render_vital_monitoring_comparison_chart(
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

    date_column = get_relevant_date_column_for_newborn_kpi_with_all(kpi_selection)
    if date_column not in df_to_use.columns:
        fallback_date_column = get_relevant_date_column_for_newborn_kpi(kpi_selection)
        if fallback_date_column in df_to_use.columns:
            date_column = fallback_date_column

    if comparison_mode == "facility":
        comparison_data = []

        if kpi_selection == "Newborn Coverage Rate":
            try:
                from newborns_dashboard.kpi_newborn_coverage_rate import (
                    load_newborn_coverage_denominator,
                    _sum_denominator_for_facilities,
                )
            except Exception as e:
                st.error(f"Failed to load Newborn Coverage Rate denominator utilities: {e}")
                return

            den_long = load_newborn_coverage_denominator()
            if den_long is None or den_long.empty:
                st.warning(
                    "⚠️ Newborn Coverage Rate denominator data is missing or empty. "
                    "Please check `utils/aggregated_admission_newborn.xlsx`."
                )
                return

            period_label = st.session_state.get("period_label", "Monthly")
            if "filters" in st.session_state and "period_label" in st.session_state.filters:
                period_label = st.session_state.filters["period_label"]
            if period_label not in ["Monthly", "Yearly"]:
                period_label = "Monthly"

            start_date = date_range_filters.get("start_date") if date_range_filters else None
            end_date = date_range_filters.get("end_date") if date_range_filters else None

            period_defs = build_period_definitions_from_denominator(
                den_long,
                period_label,
                start_date=start_date,
                end_date=end_date,
            )
            if not period_defs:
                st.info("Warning: No denominator periods available for Newborn Coverage Rate.")
                return

            for facility_uid, facility_name in zip(facility_uids, display_names):
                facility_df = df_to_use[df_to_use["orgUnit"] == facility_uid].copy()

                if facility_df.empty or date_column not in facility_df.columns:
                    facility_df = facility_df.iloc[0:0].copy()
                else:
                    facility_df["event_date"] = pd.to_datetime(
                        facility_df[date_column], errors="coerce"
                    )
                    if start_date and end_date:
                        start_dt = pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                        facility_df = facility_df[
                            (facility_df["event_date"] >= start_dt)
                            & (facility_df["event_date"] < end_dt)
                        ].copy()
                    facility_df = facility_df[facility_df["event_date"].notna()].copy()
                    facility_df["_ethiopian_yearmonth"] = map_gregorian_dates_to_ethiopian_yearmonths(
                        facility_df["event_date"],
                        den_long["yearmonth"].dropna().astype(int).unique().tolist(),
                    )
                    facility_df = facility_df[facility_df["_ethiopian_yearmonth"].notna()].copy()

                for pdef in period_defs:
                    if facility_df.empty:
                        numerator = 0
                    else:
                        period_df = facility_df[
                            facility_df["_ethiopian_yearmonth"].isin(pdef["yearmonths"])
                        ]
                        numerator = (
                            int(period_df["tei_id"].dropna().nunique())
                            if "tei_id" in period_df.columns
                            else int(len(period_df))
                        )

                    denominator = _sum_denominator_for_facilities(
                        den_long, [facility_name], yearmonths=pdef["yearmonths"]
                    )
                    value = (numerator / denominator * 100) if denominator > 0 else 0.0

                    comparison_data.append(
                        {
                            "period_display": pdef["period_display"],
                            "period_sort": pdef["period_sort"],
                            "orgUnit": facility_uid,
                            "orgUnit_name": facility_name,
                            "value": float(value),
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        else:
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
            if "denominator" in comparison_df.columns:
                comparison_df = comparison_df[
                    pd.to_numeric(comparison_df["denominator"], errors="coerce").fillna(0)
                    > 0
                ].copy()
            if comparison_df.empty:
                st.info("⚠️ No valid comparison data available (denominator is zero for all periods).")
                return

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
                title=chart_title,
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

        # SPECIAL HANDLING: Newborn Coverage Rate should include periods even when numerator is zero.
        if kpi_selection == "Newborn Coverage Rate":
            try:
                from newborns_dashboard.kpi_newborn_coverage_rate import (
                    load_newborn_coverage_denominator,
                    _sum_denominator_for_regions,
                )
            except Exception as e:
                st.error(f"Failed to load Newborn Coverage Rate denominator utilities: {e}")
                return

            den_long = load_newborn_coverage_denominator()
            if den_long is None or den_long.empty:
                st.warning(
                    "⚠️ Newborn Coverage Rate denominator data is missing or empty. "
                    "Please check `utils/aggregated_admission_newborn.xlsx`."
                )
                return

            period_label = st.session_state.get("period_label", "Monthly")
            if "filters" in st.session_state and "period_label" in st.session_state.filters:
                period_label = st.session_state.filters["period_label"]
            if period_label not in ["Monthly", "Yearly"]:
                period_label = "Monthly"

            start_date = date_range_filters.get("start_date") if date_range_filters else None
            end_date = date_range_filters.get("end_date") if date_range_filters else None

            period_defs = build_period_definitions_from_denominator(
                den_long,
                period_label,
                start_date=start_date,
                end_date=end_date,
            )
            if not period_defs:
                st.info("Warning: No denominator periods available for Newborn Coverage Rate.")
                return

            for region_name in region_names:
                region_facility_uids = region_facility_mapping.get(region_name, [])
                if not region_facility_uids:
                    continue

                region_scope_df = df_to_use[
                    df_to_use["orgUnit"].isin(region_facility_uids)
                ].copy()

                if region_scope_df.empty or date_column not in region_scope_df.columns:
                    region_scope_df = region_scope_df.iloc[0:0].copy()
                else:
                    region_scope_df["event_date"] = pd.to_datetime(
                        region_scope_df[date_column], errors="coerce"
                    )
                    if start_date and end_date:
                        start_dt = pd.Timestamp(start_date)
                        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                        region_scope_df = region_scope_df[
                            (region_scope_df["event_date"] >= start_dt)
                            & (region_scope_df["event_date"] < end_dt)
                        ].copy()
                    region_scope_df = region_scope_df[
                        region_scope_df["event_date"].notna()
                    ].copy()
                    region_scope_df["_ethiopian_yearmonth"] = map_gregorian_dates_to_ethiopian_yearmonths(
                        region_scope_df["event_date"],
                        den_long["yearmonth"].dropna().astype(int).unique().tolist(),
                    )
                    region_scope_df = region_scope_df[
                        region_scope_df["_ethiopian_yearmonth"].notna()
                    ].copy()

                for pdef in period_defs:
                    if region_scope_df.empty:
                        numerator = 0
                    else:
                        period_df = region_scope_df[
                            region_scope_df["_ethiopian_yearmonth"].isin(pdef["yearmonths"])
                        ]
                        numerator = (
                            int(period_df["tei_id"].dropna().nunique())
                            if "tei_id" in period_df.columns
                            else int(len(period_df))
                        )

                    denominator = _sum_denominator_for_regions(
                        den_long, [region_name], yearmonths=pdef["yearmonths"]
                    )
                    value = (numerator / denominator * 100) if denominator > 0 else 0.0

                    region_data.append(
                        {
                            "period_display": pdef["period_display"],
                            "period_sort": pdef["period_sort"],
                            "Region": region_name,
                            "value": float(value),
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

            if not region_data:
                st.info("⚠️ No comparison data available for regions.")
                return

            region_df = pd.DataFrame(region_data)
            render_newborn_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
            return

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
                title=chart_title,
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


# Hypothermia Combined View - Indicator keys and display names
HYPO_COMBINED_INDICATORS = [
    {
        "kpi_name": "Hypothermia on Admission Rate (%)",
        "display_name": "Hypothermia on Admission Rate (%)",
        "short_name": "Hypothermia on Admission",
        "sort_order": 1,
    },
    {
        "kpi_name": "Inborn Hypothermia at Admission Rate (%)",
        "display_name": "Inborn Hypothermia at Admission Rate (%)",
        "short_name": "Inborn Hypothermia",
        "sort_order": 2,
    },
    {
        "kpi_name": "Outborn Hypothermia at Admission Rate (%)",
        "display_name": "Outborn Hypothermia at Admission Rate (%)",
        "short_name": "Outborn Hypothermia",
        "sort_order": 3,
    },
    {
        "kpi_name": "Not hypothermic at admission (%)",
        "display_name": "Not hypothermic at admission (%)",
        "short_name": "Not Hypothermic",
        "sort_order": 4,
    },
    {
        "kpi_name": "Not hypothermic at admission inborn (%)",
        "display_name": "Not hypothermic at admission inborn (%)",
        "short_name": "Not Hypo Inborn",
        "sort_order": 5,
    },
    {
        "kpi_name": "Not hypothermic at admission outborn (%)",
        "display_name": "Not hypothermic at admission outborn (%)",
        "short_name": "Not Hypo Outborn",
        "sort_order": 6,
    },
]

HYPO_COMBINED_MARKER = "__hypothermia_combined__"
HYPO_QOC_MARKER = "__hypothermia_qoc__"

# CPAP Timing - Time from Admission/ Birth to CPAP initiation for 1000-1999g babies
CPAP_TIMING_QOC_MARKER = "__cpap_timing_qoc__"

# CPAP Timing categories (for QOC stacked bar) - green→yellow→red gradient
CPAP_TIMING_CATEGORIES = [
    ("CPAP within 1h", "#27AE60"),
    ("CPAP 1-4h", "#2ECC71"),
    ("CPAP 4-12h", "#F1C40F"),
    ("CPAP 12-24h", "#E67E22"),
    ("CPAP after 24h", "#E74C3C"),
    ("Missing CPAP Timing", "#7F8C8D"),
]

# CPAP Machine Type categories (for QOC stacked bar)
CPAP_MACHINE_TYPE_COL = "type_of_cpap_machine_used_interventions"
CPAP_MACHINE_TYPE_MAP = {
    1: "Improvised bubble CPAP with 100% O2",
    2: "CPAP blends O2 without humidification",
    3: "CPAP blends O2 and humidifies",
    4: "CPAP blends O2, heats and humidifies",
    5: "Mechanical ventilator nasal CPAP mode",
}
CPAP_MACHINE_TYPE_COLORS = ["#8E44AD", "#3498DB", "#2ECC71", "#F39C12", "#E74C3C"]
CPAP_MACHINE_TYPE_CATEGORIES = list(zip(CPAP_MACHINE_TYPE_MAP.values(), CPAP_MACHINE_TYPE_COLORS))

KMC_TIMING_QOC_MARKER = "__kmc_timing_qoc__"

KMC_TIMING_CATEGORIES = [
    ("Same day KMC", "#27AE60"),
    ("Early KMC (1-3 days)", "#2ECC71"),
    ("Delayed KMC (3-7 days)", "#F1C40F"),
    ("Late KMC (>7 days)", "#E74C3C"),
    ("Missing KMC Timing", "#7F8C8D"),
]

VITAL_MONITORING_MARKER = "__vital_monitoring__"

VITAL_MONITORING_INDICATORS = [
    {
        "kpi_name": "Temperature Taken at Admission (%)",
        "display_name": "Temperature Taken at Admission (%)",
        "short_name": "Temp. taken",
        "target": 100,
        "sort_order": 1,
    },
    {
        "kpi_name": "Birth Weight Taken (%)",
        "display_name": "Birth Weight Taken (%)",
        "short_name": "BWeight. taken",
        "target": 100,
        "sort_order": 2,
    },
    {
        "kpi_name": "Weight Taken at Admission (%)",
        "display_name": "Weight Taken at Admission (%)",
        "short_name": "Weight. taken",
        "target": 100,
        "sort_order": 3,
    },
    {
        "kpi_name": "Glucose Monitored at Admission (%)",
        "display_name": "Glucose Monitored at Admission (%)",
        "short_name": "Glucose monit.",
        "target": 75,
        "sort_order": 4,
    },
    {
        "kpi_name": "Pulse Oximeter Used at Admission (%)",
        "display_name": "Pulse Oximeter Used at Admission (%)",
        "short_name": "Pulse ox. used",
        "target": 75,
        "sort_order": 5,
    },
]

# Thermal status categories and colors for Quality of Care chart
THERMAL_CATEGORIES = [
    ("Fever >37.5\u00b0C", "#9B59B6"),
    ("Normal 36.5\u201337.4\u00b0C", "#2ECC71"),
    ("Mild Hypothermia 36.0\u201336.4\u00b0C", "#F1C40F"),
    ("Moderate Hypothermia 32.0\u201335.9\u00b0C", "#E67E22"),
    ("Severe Hypothermia <32.0\u00b0C", "#B30000"),
    ("Missing Temperature", "#7F8C8D"),
]


def _style_coverage_subplot_titles(fig, font_size=12):
    """Keep subplot titles above the plot frames in compact coverage grids."""
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=font_size)
        ann["yshift"] = 16
        ann["bgcolor"] = "rgba(255,255,255,0.85)"
        ann["borderpad"] = 2


def _render_hypothermia_combined_trend_chart(
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """Render combined Hypothermia indicators in a 3x2 multi-panel chart"""
    indicators = HYPO_COMBINED_INDICATORS
    all_kpi_names = [ind["kpi_name"] for ind in indicators]

    # Multiselect filter for showing/hiding panels
    with st.expander("Filter Hypothermia Indicators", expanded=False):
        selected_display_names = st.multiselect(
            "Select indicators to display:",
            options=[ind["display_name"] for ind in indicators],
            default=[ind["display_name"] for ind in indicators],
            key="hypo_combined_indicator_filter",
        )

    filtered_indicators = [
        ind for ind in indicators if ind["display_name"] in selected_display_names
    ]
    if not filtered_indicators:
        st.warning("No indicators selected.")
        return

    # Use enrollment_date as date column (same as all hypothermia KPIs)
    date_column = "enrollment_date"

    if date_column not in working_df.columns:
        st.warning(f"⚠️ Required date column '{date_column}' not found.")
        return

    working_df = working_df.copy()
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

    working_df = working_df[working_df["event_date"].notna()].copy()
    if working_df.empty:
        st.warning("⚠️ No data available for hypothermia indicators.")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "event_date", period_label)
    except Exception:
        st.error("Error assigning periods")
        return

    # Get unique periods in order
    unique_periods = working_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Compute all indicators for each period
    trend_data = []
    for _, row_data in unique_periods.iterrows():
        period_display = row_data["period_display"]
        period_sort = row_data["period_sort"]
        period_df = working_df[working_df["period_display"] == period_display]

        if period_df.empty:
            continue

        period_row = {
            "period_display": period_display,
            "period_sort": period_sort,
        }

        for ind in indicators:
            kpi_name = ind["kpi_name"]
            numerator, denominator, _ = get_numerator_denominator_for_newborn_kpi_with_all(
                period_df, kpi_name, facility_uids, date_range_filters,
            )
            value = (numerator / denominator * 100) if denominator > 0 else None
            period_row[f"{ind['kpi_name']}_value"] = value
            period_row[f"{ind['kpi_name']}_num"] = int(numerator)
            period_row[f"{ind['kpi_name']}_den"] = int(denominator)

        trend_data.append(period_row)

    if not trend_data:
        st.info("⚠️ No period data available for chart.")
        return

    trend_df = pd.DataFrame(trend_data)
    trend_df = trend_df.sort_values("period_sort")

    periods = trend_df["period_display"].tolist()

    # Build 2x3 subplot grid
    rows, cols = 2, 3
    sorted_indicators = sorted(filtered_indicators, key=lambda x: x["sort_order"])

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[ind["display_name"] for ind in sorted_indicators],
        vertical_spacing=0.28,
        horizontal_spacing=0.10,
    )

    for idx, ind in enumerate(sorted_indicators):
        current_row = (idx // cols) + 1
        current_col = (idx % cols) + 1
        value_col = f"{ind['kpi_name']}_value"
        num_col = f"{ind['kpi_name']}_num"
        den_col = f"{ind['kpi_name']}_den"

        fig.add_trace(
            go.Scatter(
                x=trend_df["period_display"],
                y=trend_df[value_col],
                name=ind["short_name"],
                mode="lines",
                line=dict(color="#1f77b4", width=3, shape="spline", smoothing=0.35),
                connectgaps=False,
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{ind['display_name']}</b><br>"
                    "Period: %{x}<br>"
                    "Rate: %{y:.1f}%<br>"
                    "Numerator: %{customdata[0]}<br>"
                    "Denominator: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                customdata=np.column_stack(
                    (trend_df[num_col].values, trend_df[den_col].values)
                ),
            ),
            row=current_row,
            col=current_col,
        )

        fig.update_xaxes(
            row=current_row, col=current_col,
            type="category",
            categoryorder="array",
            categoryarray=periods,
            tickangle=-45,
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
            showticklabels=(current_row == rows),
        )
        fig.update_yaxes(
            row=current_row, col=current_col,
            range=[-2, 102],
            dtick=25,
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
            ticksuffix="%",
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
            title_text=None,
        )

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=16)),
        height=700,
        showlegend=False,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        title_y=0.95,
        margin=dict(l=50, r=50, t=130, b=80),
    )
    _style_coverage_subplot_titles(fig, font_size=12)

    st.plotly_chart(fig, use_container_width=True)

    # Combined table
    st.subheader("📊 Hypothermia Indicators Table")
    st.caption("Values shown as: Rate% (numerator / denominator)")

    table_data = []
    for period in periods:
        period_row_data = trend_df[trend_df["period_display"] == period].iloc[0]
        row = {"Period": period}
        for ind in sorted_indicators:
            num = int(period_row_data[f"{ind['kpi_name']}_num"])
            den = int(period_row_data[f"{ind['kpi_name']}_den"])
            val = period_row_data[f"{ind['kpi_name']}_value"]
            if den > 0:
                row[ind["short_name"]] = f"{val:.1f}% ({num}/{den})"
            else:
                row[ind["short_name"]] = "-"
        table_data.append(row)

    # Overall row
    overall_row = {"Period": "Overall"}
    for ind in sorted_indicators:
        total_num = int(trend_df[f"{ind['kpi_name']}_num"].sum())
        total_den = int(trend_df[f"{ind['kpi_name']}_den"].sum())
        overall_val = (total_num / total_den * 100) if total_den > 0 else 0.0
        if total_den > 0:
            overall_row[ind["short_name"]] = f"{overall_val:.1f}% ({total_num}/{total_den})"
        else:
            overall_row[ind["short_name"]] = "-"
    table_data.append(overall_row)

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, height=300)

    with st.expander("ℹ️ How each indicator is computed"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Hypothermia on Admission</b></td>
                <td style="padding:8px;">Newborns with temp &lt; 36.5°C</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Inborn Hypothermia</b></td>
                <td style="padding:8px;">Inborn newborns with temp &lt; 36.5°C</td>
                <td style="padding:8px;">Total inborn newborns</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Outborn Hypothermia</b></td>
                <td style="padding:8px;">Outborn newborns with temp &lt; 36.5°C</td>
                <td style="padding:8px;">Total outborn newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Not Hypothermic</b></td>
                <td style="padding:8px;">Newborns with temp ≥ 36.5°C</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Not Hypo Inborn</b></td>
                <td style="padding:8px;">Inborn newborns with temp ≥ 36.5°C</td>
                <td style="padding:8px;">Total inborn newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Not Hypo Outborn</b></td>
                <td style="padding:8px;">Outborn newborns with temp ≥ 36.5°C</td>
                <td style="padding:8px;">Total outborn newborns</td>
            </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_hypothermia_combined_comparison_chart(
    df_to_use,
    comparison_mode,
    display_names,
    facility_uids,
    facilities_by_region,
    region_names,
    bg_color,
    text_color,
    is_national,
    show_chart=True,
):
    """Render combined hypothermia comparison with multi-line charts (facility/region comparison)"""
    indicators = HYPO_COMBINED_INDICATORS

    # Multiselect filter
    with st.expander("Filter Hypothermia Indicators", expanded=False):
        selected_display_names = st.multiselect(
            "Select indicators to display:",
            options=[ind["display_name"] for ind in indicators],
            default=[ind["display_name"] for ind in indicators],
            key="hypo_combined_comp_indicator_filter",
        )

    filtered_indicators = [
        ind for ind in indicators if ind["display_name"] in selected_display_names
    ]
    if not filtered_indicators:
        st.warning("No indicators selected.")
        return

    # Get date range filters
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Determine entities
    if comparison_mode == "facility" and display_names and facility_uids:
        entities = list(zip(facility_uids, display_names))
        entity_label_col = "Facility"
    elif comparison_mode == "region" and is_national and region_names:
        entities = [(r, r) for r in region_names]
        entity_label_col = "Region"
    else:
        entities = []
        entity_label_col = "Entity"

    if not entities:
        st.info("No entities available for comparison.")
        return

    # Prepare date column
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(filtered_indicators[0]["kpi_name"])
    if date_column not in df_to_use.columns:
        fallback = get_relevant_date_column_for_newborn_kpi(filtered_indicators[0]["kpi_name"])
        if fallback in df_to_use.columns:
            date_column = fallback

    df = df_to_use.copy()
    df["event_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df[df["event_date"].notna()].copy()

    if df.empty:
        st.info("No data available for comparison.")
        return

    # Apply date range filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")
        if start_date and end_date:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            df = df[(df["event_date"] >= start_dt) & (df["event_date"] < end_dt)].copy()

    if df.empty:
        st.info("No data available for comparison.")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        df = assign_period(df, "event_date", period_label)
    except Exception:
        st.error("Error assigning periods")
        return

    # Get unique periods in order
    unique_periods = df[["period_display", "period_sort"]].drop_duplicates().sort_values("period_sort")
    period_sort_map = dict(zip(unique_periods["period_display"], unique_periods["period_sort"]))
    periods = unique_periods["period_display"].tolist()
    if not periods:
        st.info("No period data available.")
        return

    # Build per-entity per-period per-indicator data
    comparison_rows = []
    for uid, name in entities:
        if comparison_mode == "facility" and uid != "all":
            entity_df = df[df["orgUnit"] == uid].copy()
            entity_facility_uids = [uid]
        elif comparison_mode == "region" and is_national and uid != "all":
            region_facility_uids = [
                f[1] for f in facilities_by_region.get(uid, [])
            ]
            entity_df = df[df["orgUnit"].isin(region_facility_uids)].copy()
            entity_facility_uids = region_facility_uids
        else:
            entity_df = df.copy()
            entity_facility_uids = facility_uids

        if entity_df.empty:
            continue

        for period_display in periods:
            period_df = entity_df[entity_df["period_display"] == period_display].copy()
            if period_df.empty:
                continue
            row = {
                entity_label_col: name,
                "period_display": period_display,
                "period_sort": period_sort_map.get(period_display, 0),
            }
            for ind in filtered_indicators:
                numerator, denominator, _ = get_numerator_denominator_for_newborn_kpi_with_all(
                    period_df, ind["kpi_name"], entity_facility_uids, {},
                )
                rate = (numerator / denominator * 100) if denominator > 0 else None
                row[f"{ind['short_name']}_rate"] = rate
                row[f"{ind['short_name']}_num"] = int(numerator)
                row[f"{ind['short_name']}_den"] = int(denominator)
            comparison_rows.append(row)

    if not comparison_rows:
        st.info("No comparison data available.")
        return

    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values(["period_sort", entity_label_col])

    # Build 2x3 subplot grid - only for selected indicators
    n_indicators = len(filtered_indicators)
    if n_indicators > 0:
        n_cols = 3
        n_rows = 2
        subplot_titles = [ind["display_name"] for ind in filtered_indicators]
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.10,
            vertical_spacing=0.28,
        )

        entity_names = comp_df[entity_label_col].unique()
        entity_colors = {}
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for idx, en in enumerate(entity_names):
            entity_colors[en] = palette[idx % len(palette)]

        for idx, ind in enumerate(filtered_indicators):
            row_idx = (idx // n_cols) + 1
            col_idx = (idx % n_cols) + 1
            rate_col = f"{ind['short_name']}_rate"
            num_col = f"{ind['short_name']}_num"
            den_col = f"{ind['short_name']}_den"

            ind_data = comp_df[[entity_label_col, "period_display", "period_sort", rate_col, num_col, den_col]].dropna(subset=[rate_col]).copy()

            for en in entity_names:
                en_data = ind_data[ind_data[entity_label_col] == en].sort_values("period_sort")
                if en_data.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        name=en,
                        x=en_data["period_display"],
                        y=en_data[rate_col],
                        mode="lines",
                        line=dict(color=entity_colors.get(en, "#333333"), width=2),
                        connectgaps=False,
                        hovertemplate=(
                            f"<b>{en}</b><br>"
                            "Period: %{x}<br>"
                            f"{ind['display_name']}: %{{y:.1f}}%<br>"
                            "Numerator: %{customdata[0]}<br>"
                            "Denominator: %{customdata[1]}<br>"
                            "<extra></extra>"
                        ),
                        customdata=en_data[[num_col, den_col]].values,
                        showlegend=(idx == 0),
                    ),
                    row=row_idx, col=col_idx,
                )

            fig.update_xaxes(
                row=row_idx, col=col_idx,
                type="category",
                categoryorder="array",
                categoryarray=periods,
                tickangle=-45,
                gridcolor="rgba(128,128,128,0.2)",
                showline=True,
                linewidth=1,
                linecolor="rgba(128,128,128,0.5)",
                mirror=True,
                showticklabels=(row_idx == n_rows),
            )
            fig.update_yaxes(
                row=row_idx, col=col_idx,
                range=[0, 100],
                dtick=25,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinecolor="rgba(128,128,128,0.3)",
                showline=True,
                linewidth=1,
                linecolor="rgba(128,128,128,0.5)",
                mirror=True,
                ticksuffix="%",
                title_text=None,
            )

        chart_title = "Hypothermia Indicators - Facility Comparison" if comparison_mode == "facility" else "Hypothermia Indicators - Region Comparison"
        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=16)),
            height=700,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color,
            title_font_color=text_color,
            title_y=0.95,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            margin=dict(l=50, r=50, t=130, b=80),
            hovermode="x unified",
        )
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=12)

        st.plotly_chart(fig, use_container_width=True)

    # Aggregated summary table
    st.subheader("Aggregated Comparison Table")
    st.caption("Values shown as: Rate% (numerator / denominator)")
    summary_rows = []
    for uid, name in entities:
        row = {entity_label_col: name}
        entity_comp = comp_df[comp_df[entity_label_col] == name]
        if entity_comp.empty:
            for ind in filtered_indicators:
                row[ind["short_name"]] = "-"
            summary_rows.append(row)
            continue
        for ind in filtered_indicators:
            total_num = int(entity_comp[f"{ind['short_name']}_num"].sum())
            total_den = int(entity_comp[f"{ind['short_name']}_den"].sum())
            overall_pct = (total_num / total_den * 100) if total_den > 0 else 0.0
            row[ind["short_name"]] = f"{overall_pct:.1f}% ({total_num}/{total_den})" if total_den > 0 else "-"
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("ℹ️ How each indicator is computed"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Hypothermia on Admission</b></td>
                <td style="padding:8px;">Newborns with temp &lt; 36.5°C</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Inborn Hypothermia</b></td>
                <td style="padding:8px;">Inborn newborns with temp &lt; 36.5°C</td>
                <td style="padding:8px;">Total inborn newborns</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Outborn Hypothermia</b></td>
                <td style="padding:8px;">Outborn newborns with temp &lt; 36.5°C</td>
                <td style="padding:8px;">Total outborn newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Not Hypothermic</b></td>
                <td style="padding:8px;">Newborns with temp ≥ 36.5°C</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Not Hypo Inborn</b></td>
                <td style="padding:8px;">Inborn newborns with temp ≥ 36.5°C</td>
                <td style="padding:8px;">Total inborn newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Not Hypo Outborn</b></td>
                <td style="padding:8px;">Outborn newborns with temp ≥ 36.5°C</td>
                <td style="padding:8px;">Total outborn newborns</td>
            </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


TEMP_COL_QOC = "temp_at_admission_nicu_admission_careform"


def _get_temp_category(temp_val):
    if pd.isna(temp_val):
        return "Missing Temperature"
    if temp_val > 37.5:
        return "Fever >37.5\u00b0C"
    if temp_val >= 36.5:
        return "Normal 36.5\u201337.4\u00b0C"
    if temp_val >= 36.0:
        return "Mild Hypothermia 36.0\u201336.4\u00b0C"
    if temp_val >= 32.0:
        return "Moderate Hypothermia 32.0\u201335.9\u00b0C"
    return "Severe Hypothermia <32.0\u00b0C"


def _render_hypothermia_qoc_trend_chart(
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """Render stacked bar chart for Thermal Status at Admission"""
    categories = THERMAL_CATEGORIES
    cat_names = [c[0] for c in categories]
    cat_colors = [c[1] for c in categories]

    if TEMP_COL_QOC not in working_df.columns:
        st.warning("\u26a0\ufe0f Required temperature column not found.")
        return

    working_df = working_df.copy()
    working_df["event_date"] = pd.to_datetime(
        working_df["enrollment_date"], errors="coerce"
    )

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

    working_df = working_df[working_df["event_date"].notna()].copy()
    if working_df.empty:
        st.warning("\u26a0\ufe0f No data available for thermal status chart.")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "event_date", period_label)
    except Exception:
        st.error("Error assigning periods")
        return

    # Categorize temperature
    working_df["temp_numeric"] = pd.to_numeric(
        working_df[TEMP_COL_QOC], errors="coerce"
    )
    working_df["temp_category"] = working_df["temp_numeric"].apply(_get_temp_category)

    # Get unique periods in order
    unique_periods = working_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")
    periods = unique_periods["period_display"].tolist()

    # Compute counts per period per category
    all_data = []
    for _, row_data in unique_periods.iterrows():
        period_display = row_data["period_display"]
        period_df = working_df[working_df["period_display"] == period_display]
        total = len(period_df)
        row = {"period_display": period_display, "period_sort": row_data["period_sort"], "total": total}
        for cat_name in cat_names:
            count = int((period_df["temp_category"] == cat_name).sum())
            pct = (count / total * 100) if total > 0 else 0.0
            row[f"{cat_name}_count"] = count
            row[f"{cat_name}_pct"] = pct
        all_data.append(row)

    if not all_data:
        st.info("\u26a0\ufe0f No period data available.")
        return

    agg_df = pd.DataFrame(all_data)
    agg_df = agg_df.sort_values("period_sort")

    # Build 100% stacked bar chart
    fig = go.Figure()
    for cat_name, cat_color in categories:
        pct_col = f"{cat_name}_pct"
        count_col = f"{cat_name}_count"
        fig.add_trace(go.Bar(
            name=cat_name,
            x=agg_df["period_display"],
            y=agg_df[pct_col],
            marker_color=cat_color,
            text=[f"{v:.1f}%" for v in agg_df[pct_col]],
            textposition="inside",
            textfont=dict(color="white", size=10),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{cat_name}<br>"
                "Percentage: %{y:.1f}%%<br>"
                "Count: %{customdata[0]}<br>"
                "Total Admissions: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=agg_df[[count_col, "total"]].values,
            cliponaxis=False,
        ))

    fig.update_layout(
        title=chart_title,
        barmode="stack",
        barnorm="percent",
        height=500,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        yaxis=dict(
            title="Percentage (%)",
            range=[0, 100],
            dtick=20,
            ticksuffix="%",
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
        ),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,
            tickangle=-45,
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("\U0001f4ca Thermal Status at Admission Table")
    st.caption("Values shown as: Rate% (numerator / denominator)")

    table_data = []
    for _, r in agg_df.iterrows():
        row = {"Period": r["period_display"]}
        for cat_name in cat_names:
            cnt = int(r[f"{cat_name}_count"])
            den = int(r["total"])
            pct = r[f"{cat_name}_pct"]
            row[cat_name] = f"{pct:.1f}% ({cnt}/{den})" if den > 0 else "-"
        table_data.append(row)

    overall_row = {"Period": "Overall"}
    for cat_name in cat_names:
        total_cnt = int(agg_df[f"{cat_name}_count"].sum())
        total_den = int(agg_df["total"].sum())
        overall_pct = (total_cnt / total_den * 100) if total_den > 0 else 0.0
        overall_row[cat_name] = f"{overall_pct:.1f}% ({total_cnt}/{total_den})" if total_den > 0 else "-"
    table_data.append(overall_row)

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, height=300)

    with st.expander("\u2139\ufe0f How each thermal category is defined"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Category</th>
                <th style="padding:8px; text-align:left;">Temperature Range</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><span style="color:#9B59B6;">\u25cf</span> <b>Fever</b></td>
                <td style="padding:8px;">> 37.5\u00b0C</td>
                <td style="padding:8px;" rowspan="6">Total admitted newborns in period</td>
            </tr>
            <tr>
                <td style="padding:8px;"><span style="color:#2ECC71;">\u25cf</span> <b>Normal</b></td>
                <td style="padding:8px;">36.5 \u2013 37.4\u00b0C</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><span style="color:#F1C40F;">\u25cf</span> <b>Mild Hypothermia</b></td>
                <td style="padding:8px;">36.0 \u2013 36.4\u00b0C</td>
            </tr>
            <tr>
                <td style="padding:8px;"><span style="color:#E67E22;">\u25cf</span> <b>Moderate Hypothermia</b></td>
                <td style="padding:8px;">32.0 \u2013 35.9\u00b0C</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><span style="color:#B30000;">\u25cf</span> <b>Severe Hypothermia</b></td>
                <td style="padding:8px;">< 32.0\u00b0C</td>
            </tr>
            <tr>
                <td style="padding:8px;"><span style="color:#7F8C8D;">\u25cf</span> <b>Missing Temperature</b></td>
                <td style="padding:8px;">No temperature recorded</td>
            </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================== CPAP TIMING RENDER FUNCTIONS ====================

# CPAP Timing column constants (must match kpi_utils_newborn_simplified.py)
_CPAP_TIMING_BW_COL = "birth_weight_n_nicu_admission_careform"
_CPAP_TIMING_ADM_DATE_COL = "date_of_admission_n_nicu_admission_careform"
_CPAP_TIMING_ADM_TIME_COL = "time_of_admission_admission_information"
_CPAP_TIMING_CPAP_DATE_COL = "cpap_1_start_date_interventions"
_CPAP_TIMING_CPAP_TIME_COL = "cpap_1_start_time_interventions"
_CPAP_TIMING_CPAP_COL = "baby_placed_on_cpap_neonatal_referral_form"
_CPAP_TIMING_BIRTH_COL = "time_of_birth_admission_information"


def _render_cpap_timing_combined_trend_chart(
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """(Removed) Combined run-chart has been replaced by QOC stacked bar charts"""
    st.info("CPAP Timing Coverage has been replaced by the stacked bar charts above.")
    return


def _render_cpap_timing_qoc_trend_chart(
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """Render Quality of Care section with 3 sub-tabs: Admission→CPAP, Birth→CPAP, CPAPs Done & Mortality"""
    categories = CPAP_TIMING_CATEGORIES
    cat_names = [c[0] for c in categories]
    cat_colors = [c[1] for c in categories]

    # Compute both timing types
    admission_timing = compute_cpap_timing_data(working_df, timing_type="admission")
    birth_timing = compute_cpap_timing_data(working_df, timing_type="birth")

    if admission_timing.empty and birth_timing.empty:
        st.warning("No CPAP timing data available for 1000-1999g babies who received CPAP.")

    def _build_agg_df(timing_df):
        """Build per-period aggregated dataframe for one timing type (admission or birth).

        Numerator: 1000-1999g CPAP babies in each timing bucket.
        Denominator: ALL babies who received CPAP (any birth weight).
        """
        if timing_df.empty:
            return None

        # --- Numerator: 1000-1999g CPAP babies with timing info ---
        df = working_df[["tei_id", "enrollment_date", "orgUnit"]].drop_duplicates(subset=["tei_id"]).copy()
        df = df.merge(timing_df, on="tei_id", how="inner")
        df["event_date"] = pd.to_datetime(df["enrollment_date"], errors="coerce")
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                df = df[(df["event_date"] >= start_dt) & (df["event_date"] < end_dt)].copy()
        df = df[df["event_date"].notna()].copy()
        if df.empty:
            return None

        period_label = st.session_state.get("period_label", "Monthly")
        try:
            df = assign_period(df, "event_date", period_label)
        except Exception:
            return None

        # --- Denominator: ALL babies who received CPAP (any birth weight) ---
        cpap_col = _CPAP_TIMING_CPAP_COL
        denom_raw = working_df[["tei_id", "enrollment_date", cpap_col]].drop_duplicates(subset=["tei_id"]).copy()
        denom_raw["has_cpap"] = (
            pd.to_numeric(
                denom_raw[cpap_col].astype(str).str.split(".").str[0],
                errors="coerce",
            )
            == 1.0
        )
        denom_df = denom_raw[denom_raw["has_cpap"]].copy()
        if denom_df.empty:
            return None
        denom_df["event_date"] = pd.to_datetime(denom_df["enrollment_date"], errors="coerce")
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                denom_df = denom_df[(denom_df["event_date"] >= start_dt) & (denom_df["event_date"] < end_dt)].copy()
        denom_df = denom_df[denom_df["event_date"].notna()].copy()
        if denom_df.empty:
            return None
        try:
            denom_df = assign_period(denom_df, "event_date", period_label)
        except Exception:
            return None
        denom_counts = denom_df.groupby("period_display").size()

        # --- Aggregate: numerator buckets / denominator total ---
        unique_periods = df[["period_display", "period_sort"]].drop_duplicates().sort_values("period_sort")
        all_data = []
        for _, row_data in unique_periods.iterrows():
            period_display = row_data["period_display"]
            period_df = df[df["period_display"] == period_display]
            total = int(denom_counts.get(period_display, 0))
            if total == 0:
                continue
            row = {"period_display": period_display, "period_sort": row_data["period_sort"], "total": total}
            for cat_name in cat_names:
                count = int((period_df["cpap_timing_category"] == cat_name).sum())
                pct = (count / total * 100) if total > 0 else 0.0
                row[f"{cat_name}_count"] = count
                row[f"{cat_name}_pct"] = pct
            all_data.append(row)

        if not all_data:
            return None
        agg_df = pd.DataFrame(all_data)
        agg_df = agg_df.sort_values("period_sort")
        return agg_df

    admission_agg = _build_agg_df(admission_timing)
    birth_agg = _build_agg_df(birth_timing)

    def _build_stacked_bar(agg_df, title_suffix, periods):
        """Build a 100% stacked bar figure."""
        fig = go.Figure()
        for cat_name, cat_color in categories:
            pct_col = f"{cat_name}_pct"
            count_col = f"{cat_name}_count"
            fig.add_trace(go.Bar(
                name=cat_name,
                x=agg_df["period_display"],
                y=agg_df[pct_col],
                marker_color=cat_color,
                text=[f"{v:.1f}%" for v in agg_df[pct_col]],
                textposition="inside",
                textfont=dict(color="white", size=13),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{cat_name}<br>"
                    "Numerator: %{customdata[0]}<br>"
                    "Denominator: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                customdata=agg_df[[count_col, "total"]].values,
                cliponaxis=False,
            ))

        fig.update_layout(
            title=dict(
                text=f"{title_suffix}",
                font=dict(size=13),
            ),
            barmode="stack",
            height=450,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color,
            title_font_color=text_color,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                font=dict(size=9),
            ),
            margin=dict(l=60, r=120, t=80, b=60),
            yaxis=dict(
                title="Percentage (%)",
                range=[0, 100],
                dtick=20,
                ticksuffix="%",
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinecolor="rgba(128,128,128,0.5)",
                showline=True,
                linewidth=2,
                linecolor="rgba(128,128,128,0.8)",
                mirror=True,
            ),
            xaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=periods,
                tickangle=-45,
                gridcolor="rgba(128,128,128,0.2)",
                showgrid=True,
                showline=True,
                linewidth=2,
                linecolor="rgba(128,128,128,0.8)",
                mirror=True,
            ),
        )
        return fig

    def _build_table(agg_df):
        """Build per-period table for one timing type."""
        table_data = []
        for _, r in agg_df.iterrows():
            row = {"Period": r["period_display"]}
            for cat_name in cat_names:
                cnt = int(r[f"{cat_name}_count"])
                den = int(r["total"])
                pct = r[f"{cat_name}_pct"]
                row[cat_name] = f"{pct:.1f}% ({cnt}/{den})" if den > 0 else "-"
            table_data.append(row)
        overall_row = {"Period": "Overall"}
        for cat_name in cat_names:
            total_cnt = int(agg_df[f"{cat_name}_count"].sum())
            total_den = int(agg_df["total"].sum())
            overall_pct = (total_cnt / total_den * 100) if total_den > 0 else 0.0
            overall_row[cat_name] = f"{overall_pct:.1f}% ({total_cnt}/{total_den})" if total_den > 0 else "-"
        table_data.append(overall_row)
        return pd.DataFrame(table_data)

    # ---- CPAP Mortality: per-period per-BW-category ----
    def _build_cpap_mortality_data():
        """
        Compute CPAP mortality rate per birth weight category per period.

        Numerator:   CPAP babies who died (newborn_status_at_discharge == 0)
        Denominator: Total CPAP babies in the same birth weight category
        """
        df = working_df.copy()
        if df.empty or "tei_id" not in df.columns:
            return None
        df = df.drop_duplicates(subset=["tei_id"])

        cpap_col = _CPAP_TIMING_CPAP_COL
        if cpap_col not in df.columns:
            return None
        df["has_cpap"] = (
            pd.to_numeric(df[cpap_col].astype(str).str.split(".").str[0], errors="coerce") == 1.0
        )
        cpap_df = df[df["has_cpap"]].copy()
        if cpap_df.empty:
            return None

        status_col = "newborn_status_at_discharge_n_discharge_care_form"
        if status_col in cpap_df.columns:
            cpap_df["status_num"] = pd.to_numeric(
                cpap_df[status_col].astype(str).str.split(".").str[0], errors="coerce"
            )
            cpap_df["died"] = cpap_df["status_num"] == 0
        else:
            cpap_df["died"] = False

        bw_col = _CPAP_TIMING_BW_COL
        if bw_col not in cpap_df.columns:
            return None
        cpap_df["bw_num"] = pd.to_numeric(cpap_df[bw_col], errors="coerce")

        def _bw_category(val):
            if pd.isna(val):
                return None
            for key, info in BIRTH_WEIGHT_CATEGORIES.items():
                if info["min"] <= val <= info["max"]:
                    return key
            return None

        cpap_df["bw_category"] = cpap_df["bw_num"].apply(_bw_category)
        cpap_df = cpap_df[cpap_df["bw_category"].notna()].copy()
        if cpap_df.empty:
            return None

        cpap_df["event_date"] = pd.to_datetime(cpap_df["enrollment_date"], errors="coerce")
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                cpap_df = cpap_df[(cpap_df["event_date"] >= start_dt) & (cpap_df["event_date"] < end_dt)].copy()
        cpap_df = cpap_df[cpap_df["event_date"].notna()].copy()
        if cpap_df.empty:
            return None

        period_label = st.session_state.get("period_label", "Monthly")
        try:
            cpap_df = assign_period(cpap_df, "event_date", period_label)
        except Exception:
            return None

        period_sort_map = cpap_df[["period_display", "period_sort"]].drop_duplicates().set_index("period_display")["period_sort"]
        grouped = cpap_df.groupby(["period_display", "bw_category"])
        agg_list = []
        for (period, cat), group in grouped:
            total = len(group)
            died = int(group["died"].sum())
            rate = (died / total * 100) if total > 0 else 0.0
            sort_val = period_sort_map.get(period, pd.NaT)
            agg_list.append({
                "period_display": period,
                "period_sort": sort_val,
                "bw_category": cat,
                "died_count": died,
                "total_cpap": total,
                "mortality_rate": rate,
            })
        if not agg_list:
            return None
        return pd.DataFrame(agg_list)

    # ---- Create 4 sub-tabs ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Admission → CPAP",
        "Birth → CPAP",
        "CPAPs Done & Mortality",
        "CPAP Type / Machine Used",
    ])

    with tab1:
        if admission_agg is not None:
            periods = admission_agg["period_display"].tolist()
            fig = _build_stacked_bar(
                admission_agg,
                "Time between Admission<br>and CPAP initiation",
                periods,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📊 Quality of Care Table")
            st.caption("Values shown as: Rate% (numerator / denominator)")
            st.dataframe(_build_table(admission_agg), use_container_width=True, height=300)
        else:
            st.info("No data available for Time between Admission and CPAP initiation.")

    with tab2:
        if birth_agg is not None:
            periods = birth_agg["period_display"].tolist()
            fig = _build_stacked_bar(
                birth_agg,
                "Time between Birth<br>and CPAP initiation",
                periods,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📊 Quality of Care Table")
            st.caption("Values shown as: Rate% (numerator / denominator)")
            st.dataframe(_build_table(birth_agg), use_container_width=True, height=300)
        else:
            st.info("No data available for Time between Birth and CPAP initiation.")

    with tab3:
        mortality_df = _build_cpap_mortality_data()
        if mortality_df is None or mortality_df.empty:
            st.info("No CPAP mortality data available.")
        else:
            period_sort_map = mortality_df[["period_display", "period_sort"]].drop_duplicates().dropna(subset=["period_sort"])
            periods_mort = period_sort_map.sort_values("period_sort")["period_display"].tolist()
            cats_with_data = mortality_df["bw_category"].unique()
            all_cats_with_data = {
                k: v for k, v in BIRTH_WEIGHT_CATEGORIES.items()
                if k in cats_with_data
            }
            if not all_cats_with_data:
                st.info("No CPAP mortality data by weight category.")
            else:
                with st.expander("Filter Birth Weight Categories", expanded=False):
                    selected_cat_names = st.multiselect(
                        "Select Birth Weight Categories:",
                        options=[cat["name"] for cat in BIRTH_WEIGHT_CATEGORIES.values()],
                        default=[cat["name"] for cat in all_cats_with_data.values()],
                        key="cpap_mortality_cat_filter",
                    )
                filtered_cats = {
                    k: v for k, v in all_cats_with_data.items()
                    if v["name"] in selected_cat_names
                }
                if not filtered_cats:
                    st.warning("No categories selected.")
                else:
                    rows, cols = 3, 2
                    fig = make_subplots(
                        rows=rows, cols=cols,
                        subplot_titles=[
                            cat["name"] for cat in sorted(
                                filtered_cats.values(), key=lambda x: x["sort_order"]
                            )
                        ],
                        vertical_spacing=0.10,
                        horizontal_spacing=0.08,
                    )
                    axis_periods = list(periods_mort)

                    for idx, (cat_key, cat_info) in enumerate(sorted(
                        filtered_cats.items(), key=lambda x: x[1]["sort_order"]
                    )):
                        cat_df = (
                            mortality_df[mortality_df["bw_category"] == cat_key]
                            .copy()
                            .set_index("period_display")
                            .reindex(periods_mort)
                            .reset_index()
                        )
                        row_pos = (idx // cols) + 1
                        col_pos = (idx % cols) + 1

                        fig.add_trace(
                            go.Scatter(
                                x=cat_df["period_display"],
                                y=cat_df["mortality_rate"],
                                name=cat_info["name"],
                                mode="lines",
                                line=dict(
                                    color="#1f77b4", width=3,
                                    shape="spline", smoothing=0.35,
                                ),
                                connectgaps=True,
                                cliponaxis=False,
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    f"{cat_info['name']}<br>"
                                    "Numerator: %{customdata[0]}<br>"
                                    "Denominator: %{customdata[1]}<br>"
                                    "<extra></extra>"
                                ),
                                customdata=np.column_stack((
                                    cat_df["died_count"].fillna(0).astype(int),
                                    cat_df["total_cpap"].fillna(0).astype(int),
                                )),
                                text=[
                                    f"{v:.1f}%" if pd.notna(v) else ""
                                    for v in cat_df["mortality_rate"]
                                ],
                                textposition="top center",
                            ),
                            row=row_pos, col=col_pos,
                        )

                    fig.update_layout(
                        title="CPAPs Done & Mortality Rate by Birth Weight Category",
                        height=1000,
                        showlegend=False,
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        font_color=text_color,
                        title_font_color=text_color,
                        margin=dict(l=60, r=60, t=80, b=60),
                    )
                    fig.update_xaxes(
                        type="category",
                        categoryorder="array",
                        categoryarray=axis_periods,
                        tickangle=-45,
                        gridcolor="rgba(128,128,128,0.2)",
                        showgrid=True,
                        showline=True,
                        linewidth=2,
                        linecolor="rgba(128,128,128,0.8)",
                        mirror=True,
                    )
                    fig.update_yaxes(
                        range=[-2, 102],
                        dtick=25,
                        ticksuffix="%",
                        gridcolor="rgba(128,128,128,0.2)",
                        showgrid=True,
                        zeroline=True,
                        zerolinecolor="rgba(128,128,128,0.5)",
                        showline=True,
                        linewidth=2,
                        linecolor="rgba(128,128,128,0.8)",
                        mirror=True,
                    )
                    fig.update_layout(yaxis_tickformat=".1f")
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📊 CPAPs Done & Mortality Table")
                    table_rows = []
                    for period in periods_mort:
                        row = {"Period": period}
                        for cat_key, cat_info in sorted(
                            filtered_cats.items(), key=lambda x: x[1]["sort_order"]
                        ):
                            sub = mortality_df[
                                (mortality_df["period_display"] == period) &
                                (mortality_df["bw_category"] == cat_key)
                            ]
                            if not sub.empty:
                                r = sub.iloc[0]
                                row[cat_info["short_name"]] = (
                                    f"{r['mortality_rate']:.1f}% "
                                    f"({int(r['died_count'])}/{int(r['total_cpap'])})"
                                )
                            else:
                                row[cat_info["short_name"]] = "-"
                        table_rows.append(row)
                    overall = {"Period": "Overall"}
                    for cat_key, cat_info in sorted(
                        filtered_cats.items(), key=lambda x: x[1]["sort_order"]
                    ):
                        cat_df = mortality_df[mortality_df["bw_category"] == cat_key]
                        total_died = int(cat_df["died_count"].sum())
                        total_cpap = int(cat_df["total_cpap"].sum())
                        rate = (total_died / total_cpap * 100) if total_cpap > 0 else 0.0
                        overall[cat_info["short_name"]] = f"{rate:.1f}% ({total_died}/{total_cpap})"
                    table_rows.append(overall)
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=300)

    # ---- CPAP Machine Type helper ----
    def _build_cpap_machine_type_data():
        """Compute CPAP machine type distribution per period.
        Denominator: all babies who received CPAP.
        Grouping: valid machine types 1-5. Excludes -1 and -3.
        """
        df = working_df.copy()
        if df.empty or "tei_id" not in df.columns:
            return None
        df = df.drop_duplicates(subset=["tei_id"])

        cpap_col = _CPAP_TIMING_CPAP_COL
        if cpap_col not in df.columns:
            return None
        df["has_cpap"] = (
            pd.to_numeric(df[cpap_col].astype(str).str.split(".").str[0], errors="coerce") == 1.0
        )
        cpap_df = df[df["has_cpap"]].copy()
        if cpap_df.empty:
            return None

        machine_col = CPAP_MACHINE_TYPE_COL
        if machine_col not in cpap_df.columns:
            return None
        cpap_df["machine_code"] = pd.to_numeric(
            cpap_df[machine_col].astype(str).str.split(".").str[0], errors="coerce"
        )

        cpap_df["event_date"] = pd.to_datetime(cpap_df["enrollment_date"], errors="coerce")
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                cpap_df = cpap_df[(cpap_df["event_date"] >= start_dt) & (cpap_df["event_date"] < end_dt)].copy()
        cpap_df = cpap_df[cpap_df["event_date"].notna()].copy()
        if cpap_df.empty:
            return None

        period_label = st.session_state.get("period_label", "Monthly")
        try:
            cpap_df = assign_period(cpap_df, "event_date", period_label)
        except Exception:
            return None

        # Per period: count total CPAP and each machine type
        result = []
        for period_display in cpap_df["period_display"].unique():
            period_df = cpap_df[cpap_df["period_display"] == period_display]
            total_cpap = len(period_df)
            if total_cpap == 0:
                continue
            sort_val = period_df["period_sort"].iloc[0] if "period_sort" in period_df.columns else pd.NaT
            row = {"period_display": period_display, "period_sort": sort_val, "total": total_cpap}
            for code, label in CPAP_MACHINE_TYPE_MAP.items():
                count = int((period_df["machine_code"] == code).sum())
                pct = (count / total_cpap * 100) if total_cpap > 0 else 0.0
                row[f"{label}_count"] = count
                row[f"{label}_pct"] = pct
            result.append(row)

        if not result:
            return None
        return pd.DataFrame(result).sort_values("period_sort")

    with tab4:
        machine_df = _build_cpap_machine_type_data()
        if machine_df is None or machine_df.empty:
            st.info("No CPAP machine type data available.")
        else:
            periods = machine_df["period_display"].tolist()
            fig = go.Figure()
            for label, color in CPAP_MACHINE_TYPE_CATEGORIES:
                pct_col = f"{label}_pct"
                count_col = f"{label}_count"
                fig.add_trace(go.Bar(
                    name=label,
                    x=machine_df["period_display"],
                    y=machine_df[pct_col],
                    marker_color=color,
                    text=[f"{v:.1f}%" if v > 0 else "" for v in machine_df[pct_col]],
                    textposition="inside",
                    textfont=dict(color="white", size=11),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        f"{label}<br>"
                        "Numerator: %{customdata[0]}<br>"
                        "Denominator: %{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=machine_df[[count_col, "total"]].values,
                    cliponaxis=False,
                ))

            fig.update_layout(
                title=dict(
                    text="Distribution of CPAP Machine Types<br>among babies who received CPAP",
                    font=dict(size=13),
                ),
                barmode="stack",
                height=450,
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font_color=text_color,
                title_font_color=text_color,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9),
                ),
                margin=dict(l=60, r=140, t=80, b=60),
                yaxis=dict(
                    title="Percentage (%)",
                    range=[0, 100],
                    dtick=20,
                    ticksuffix="%",
                    gridcolor="rgba(128,128,128,0.2)",
                    zeroline=True,
                    zerolinecolor="rgba(128,128,128,0.5)",
                    showline=True,
                    linewidth=2,
                    linecolor="rgba(128,128,128,0.8)",
                    mirror=True,
                ),
                xaxis=dict(
                    type="category",
                    categoryorder="array",
                    categoryarray=periods,
                    tickangle=-45,
                    gridcolor="rgba(128,128,128,0.2)",
                    showgrid=True,
                    showline=True,
                    linewidth=2,
                    linecolor="rgba(128,128,128,0.8)",
                    mirror=True,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.subheader("📊 CPAP Machine Type Table")
            st.caption("Values shown as: Rate% (numerator / denominator)")
            table_data = []
            for _, r in machine_df.iterrows():
                row = {"Period": r["period_display"]}
                for label in CPAP_MACHINE_TYPE_MAP.values():
                    cnt = int(r[f"{label}_count"])
                    den = int(r["total"])
                    pct = r[f"{label}_pct"]
                    row[label] = f"{pct:.1f}% ({cnt}/{den})" if den > 0 else "-"
                table_data.append(row)
            overall = {"Period": "Overall"}
            for label in CPAP_MACHINE_TYPE_MAP.values():
                total_cnt = int(machine_df[f"{label}_count"].sum())
                total_den = int(machine_df["total"].sum())
                pct = (total_cnt / total_den * 100) if total_den > 0 else 0.0
                overall[label] = f"{pct:.1f}% ({total_cnt}/{total_den})" if total_den > 0 else "-"
            table_data.append(overall)
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=300)

    # Info expander at bottom
    with st.expander("ℹ️ Numerator & Denominator Definitions"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">

            <h4 style="margin-top:0;">1. Time between Admission and CPAP initiation</h4>
            <ul>
              <li><b>Numerator:</b> 1000–1999g babies who received CPAP, in each timing bucket (≤1h, 1-4h, 4-12h, 12-24h, &gt;24h).</li>
              <li><b>Denominator:</b> ALL babies who received CPAP (any birth weight) in the period.</li>
              <li><b>How it is computed:</b> Time from admission to CPAP start is calculated and assigned to a timing bucket.</li>
            </ul>

            <h4>2. Time between Birth and CPAP initiation</h4>
            <ul>
              <li><b>Numerator:</b> 1000–1999g babies who received CPAP, in each timing bucket.</li>
              <li><b>Denominator:</b> ALL babies who received CPAP (any birth weight) in the period.</li>
              <li><b>How it is computed:</b> Time from birth (admission date used as proxy) to CPAP start is calculated and assigned to a timing bucket.</li>
            </ul>

            <h4>3. CPAPs Done &amp; Mortality</h4>
            <ul>
              <li><b>Numerator:</b> Babies of the specified weight category who received CPAP <b>and</b> died.</li>
              <li><b>Denominator:</b> Total babies of the specified weight category who received CPAP.</li>
            </ul>

            <h4>4. CPAP Type / Machine Used</h4>
            <ul>
              <li><b>Numerator:</b> Babies who received CPAP, grouped by CPAP machine type (codes 1–5). Codes −1 and −3 are excluded from the breakdown.</li>
              <li><b>Denominator:</b> All babies who received CPAP in the period.</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_cpap_timing_combined_comparison_chart(
    df_to_use,
    comparison_mode,
    display_names,
    facility_uids,
    facilities_by_region,
    region_names,
    bg_color,
    text_color,
    is_national,
    show_chart=True,
):
    """(Removed) Combined run-chart has been replaced by QOC stacked bar charts"""
    st.info("CPAP Timing Comparison has been replaced by the stacked bar charts in the trend view.")
    return


def _render_kmc_timing_qoc_trend_chart(
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """Render KMC Time to Initiation stacked bar chart (Quality of Care)"""
    categories = KMC_TIMING_CATEGORIES
    cat_names = [c[0] for c in categories]
    cat_colors = [c[1] for c in categories]

    kmc_timing = compute_kmc_timing_data(working_df)

    if kmc_timing.empty:
        st.warning("No KMC timing data available for babies who received KMC.")
        return

    def _build_agg_df(timing_df):
        if timing_df.empty:
            return None

        df = working_df[["tei_id", "enrollment_date", "orgUnit"]].drop_duplicates(subset=["tei_id"]).copy()
        df = df.merge(timing_df, on="tei_id", how="inner")
        df["event_date"] = pd.to_datetime(df["enrollment_date"], errors="coerce")
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                df = df[(df["event_date"] >= start_dt) & (df["event_date"] < end_dt)].copy()
        df = df[df["event_date"].notna()].copy()
        if df.empty:
            return None

        period_label = st.session_state.get("period_label", "Monthly")
        try:
            df = assign_period(df, "event_date", period_label)
        except Exception:
            return None

        denom_raw = working_df.drop_duplicates(subset=["tei_id"]).copy()
        denom_raw["has_kmc"] = denom_raw.apply(
            lambda r: get_kmc_status_for_tei(r), axis=1
        )
        denom_df = denom_raw[denom_raw["has_kmc"]].copy()
        if denom_df.empty:
            return None
        denom_df["event_date"] = pd.to_datetime(denom_df["enrollment_date"], errors="coerce")
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")
            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                denom_df = denom_df[(denom_df["event_date"] >= start_dt) & (denom_df["event_date"] < end_dt)].copy()
        denom_df = denom_df[denom_df["event_date"].notna()].copy()
        if denom_df.empty:
            return None
        try:
            denom_df = assign_period(denom_df, "event_date", period_label)
        except Exception:
            return None
        denom_counts = denom_df.groupby("period_display").size()

        unique_periods = df[["period_display", "period_sort"]].drop_duplicates().sort_values("period_sort")
        all_data = []
        for _, row_data in unique_periods.iterrows():
            period_display = row_data["period_display"]
            period_df = df[df["period_display"] == period_display]
            total = int(denom_counts.get(period_display, 0))
            if total == 0:
                continue
            row = {"period_display": period_display, "period_sort": row_data["period_sort"], "total": total}
            for cat_name in cat_names:
                count = int((period_df["kmc_timing_category"] == cat_name).sum())
                pct = (count / total * 100) if total > 0 else 0.0
                row[f"{cat_name}_count"] = count
                row[f"{cat_name}_pct"] = pct
            all_data.append(row)

        if not all_data:
            return None
        agg_df = pd.DataFrame(all_data)
        agg_df = agg_df.sort_values("period_sort")
        return agg_df

    agg_df = _build_agg_df(kmc_timing)

    if agg_df is None or agg_df.empty:
        st.warning("No aggregated KMC timing data available.")
        return

    periods = agg_df["period_display"].tolist()

    fig = go.Figure()
    for cat_name, cat_color in categories:
        pct_col = f"{cat_name}_pct"
        count_col = f"{cat_name}_count"
        fig.add_trace(go.Bar(
            name=cat_name,
            x=agg_df["period_display"],
            y=agg_df[pct_col],
            marker_color=cat_color,
            text=[f"{v:.1f}%" for v in agg_df[pct_col]],
            textposition="inside",
            textfont=dict(color="white", size=13),
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{cat_name}<br>"
                "Numerator: %{customdata[0]}<br>"
                "Denominator: %{customdata[1]}<br>"
                "<extra></extra>"
            ),
            customdata=agg_df[[count_col, "total"]].values,
            cliponaxis=False,
        ))

    fig.update_layout(
        title=dict(text="Time from Birth to KMC Initiation", font=dict(size=13)),
        barmode="stack",
        height=450,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=9),
        ),
        margin=dict(l=60, r=120, t=80, b=60),
        yaxis=dict(
            title="Percentage (%)",
            range=[0, 100],
            dtick=20,
            ticksuffix="%",
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
        ),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,
            tickangle=-45,
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("📊 KMC Timing Distribution Table")
    st.caption("Values shown as: Percentage% (numerator / denominator)")

    table_data = []
    for _, r in agg_df.iterrows():
        row = {"Period": r["period_display"]}
        for cat_name in cat_names:
            cnt = int(r[f"{cat_name}_count"])
            den = int(r["total"])
            pct = r[f"{cat_name}_pct"]
            row[cat_name] = f"{pct:.1f}% ({cnt}/{den})" if den > 0 else "-"
        table_data.append(row)
    overall_row = {"Period": "Overall"}
    for cat_name in cat_names:
        total_cnt = int(agg_df[f"{cat_name}_count"].sum())
        total_den = int(agg_df["total"].sum())
        overall_pct = (total_cnt / total_den * 100) if total_den > 0 else 0.0
        overall_row[cat_name] = f"{overall_pct:.1f}% ({total_cnt}/{total_den})" if total_den > 0 else "-"
    table_data.append(overall_row)
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, height=200)

    with st.expander("ℹ️ How KMC timing is computed"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <p><b>Time from Birth to KMC Initiation</b></p>
            <ul>
              <li><b>Numerator:</b> Babies who received KMC, grouped by time from birth (admission date as proxy) to KMC start date.</li>
              <li><b>Denominator:</b> All babies who received KMC.</li>
            </ul>
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Category</th>
                <th style="padding:8px; text-align:left;">Definition</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Same day KMC</b></td>
                <td style="padding:8px;">KMC started on the same day as admission (&lt;1 day)</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Early KMC</b></td>
                <td style="padding:8px;">KMC started 1-3 days after admission</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Delayed KMC</b></td>
                <td style="padding:8px;">KMC started 3-7 days after admission</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Late KMC</b></td>
                <td style="padding:8px;">KMC started &gt;7 days after admission</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Missing KMC Timing</b></td>
                <td style="padding:8px;">KMC start date is not available</td>
            </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_vital_monitoring_trend_chart(
    working_df,
    chart_title,
    bg_color,
    text_color,
    facility_uids,
    date_range_filters,
):
    """Render combined Vital Monitoring indicators in a 2x3 panel chart"""
    indicators = VITAL_MONITORING_INDICATORS

    if not indicators:
        return

    # Multiselect filter for showing/hiding panels
    with st.expander("Filter Vital Monitoring Indicators", expanded=False):
        selected_display_names = st.multiselect(
            "Select indicators to display:",
            options=[ind["display_name"] for ind in indicators],
            default=[ind["display_name"] for ind in indicators],
            key="vital_monitoring_indicator_filter",
        )

    filtered_indicators = [
        ind for ind in indicators if ind["display_name"] in selected_display_names
    ]
    if not filtered_indicators:
        st.warning("No indicators selected.")
        return
    indicators = filtered_indicators

    date_column = "enrollment_date"

    if date_column not in working_df.columns:
        st.warning(f"⚠️ Required date column '{date_column}' not found.")
        return

    working_df = working_df.copy()
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

    working_df = working_df[working_df["event_date"].notna()].copy()
    if working_df.empty:
        st.warning("⚠️ No data available for vital monitoring indicators.")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "event_date", period_label)
    except Exception:
        st.error("Error assigning periods")
        return

    # Get unique periods in order
    unique_periods = working_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Compute all indicators for each period
    trend_data = []
    for _, row_data in unique_periods.iterrows():
        period_display = row_data["period_display"]
        period_sort = row_data["period_sort"]
        period_df = working_df[working_df["period_display"] == period_display]

        if period_df.empty:
            continue

        period_row = {
            "period_display": period_display,
            "period_sort": period_sort,
        }

        for ind in indicators:
            kpi_name = ind["kpi_name"]
            numerator, denominator, _ = get_numerator_denominator_for_newborn_kpi_with_all(
                period_df, kpi_name, facility_uids, date_range_filters,
            )
            value = (numerator / denominator * 100) if denominator > 0 else None
            period_row[f"{ind['kpi_name']}_value"] = value
            period_row[f"{ind['kpi_name']}_num"] = int(numerator)
            period_row[f"{ind['kpi_name']}_den"] = int(denominator)

        trend_data.append(period_row)

    if not trend_data:
        st.info("⚠️ No period data available for chart.")
        return

    trend_df = pd.DataFrame(trend_data)
    trend_df = trend_df.sort_values("period_sort")

    periods = trend_df["period_display"].tolist()

    # Build dynamic subplot grid based on indicator count
    n = len(indicators)
    if n <= 3:
        rows, cols = 1, n
    else:
        rows, cols = 2, 3

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[ind["display_name"] for ind in indicators],
        horizontal_spacing=0.12,
        vertical_spacing=0.20,
    )

    for idx, ind in enumerate(indicators):
        current_row = 1 if idx < 3 else 2
        current_col = (idx % 3) + 1
        value_col = f"{ind['kpi_name']}_value"
        num_col = f"{ind['kpi_name']}_num"
        den_col = f"{ind['kpi_name']}_den"

        target_y = ind["target"]

        fig.add_trace(
            go.Scatter(
                x=trend_df["period_display"],
                y=trend_df[value_col],
                name=ind["short_name"],
                mode="lines",
                line=dict(color="#1f77b4", width=3, shape="spline", smoothing=0.35),
                connectgaps=False,
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{ind['display_name']}</b><br>"
                    "Period: %{x}<br>"
                    "Rate: %{y:.1f}%<br>"
                    "Numerator: %{customdata[0]}<br>"
                    "Denominator: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
                customdata=np.column_stack(
                    (trend_df[num_col].values, trend_df[den_col].values)
                ),
            ),
            row=current_row,
            col=current_col,
        )

        fig.add_hline(
            y=target_y, line_dash="dash", line_color="green", line_width=1.5,
            row=current_row, col=current_col,
        )

        fig.update_xaxes(
            row=current_row, col=current_col,
            type="category",
            categoryorder="array",
            categoryarray=periods,
            tickangle=-45,
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
        )
        fig.update_yaxes(
            row=current_row, col=current_col,
            range=[0, 105],
            dtick=25,
            gridcolor="rgba(128,128,128,0.2)",
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
            ticksuffix="%",
            showline=True,
            linewidth=2,
            linecolor="rgba(128,128,128,0.8)",
            mirror=True,
        )

    # Hide empty subplot (row 2, col 3) when using 2x3 grid with <=5 indicators
    if rows == 2 and n < 6:
        fig.update_xaxes(row=2, col=3, visible=False)
        fig.update_yaxes(row=2, col=3, visible=False)

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=16)),
        height=700,
        showlegend=False,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        title_y=0.95,
        margin=dict(l=50, r=50, t=120, b=80),
    )
    _style_coverage_subplot_titles(fig, font_size=11)

    st.plotly_chart(fig, use_container_width=True)


def _render_vital_monitoring_comparison_chart(
    df_to_use,
    comparison_mode,
    display_names,
    facility_uids,
    facilities_by_region,
    region_names,
    bg_color,
    text_color,
    is_national,
    show_chart=True,
):
    """Render combined Vital Monitoring comparison with multi-line charts (facility/region comparison)"""
    indicators = VITAL_MONITORING_INDICATORS

    # Get date range filters
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Determine entities
    if comparison_mode == "facility" and display_names and facility_uids:
        entities = list(zip(facility_uids, display_names))
        entity_label_col = "Facility"
    elif comparison_mode == "region" and is_national and region_names:
        entities = [(r, r) for r in region_names]
        entity_label_col = "Region"
    else:
        entities = []
        entity_label_col = "Entity"

    if not entities:
        st.info("No entities available for comparison.")
        return

    # Prepare date column
    date_column = get_relevant_date_column_for_newborn_kpi_with_all(indicators[0]["kpi_name"])
    if date_column not in df_to_use.columns:
        fallback = get_relevant_date_column_for_newborn_kpi(indicators[0]["kpi_name"])
        if fallback in df_to_use.columns:
            date_column = fallback

    df = df_to_use.copy()
    df["event_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df[df["event_date"].notna()].copy()

    if df.empty:
        st.info("No data available for comparison.")
        return

    # Apply date range filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")
        if start_date and end_date:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            df = df[(df["event_date"] >= start_dt) & (df["event_date"] < end_dt)].copy()

    if df.empty:
        st.info("No data available for comparison.")
        return

    # Assign periods
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        df = assign_period(df, "event_date", period_label)
    except Exception:
        st.error("Error assigning periods")
        return

    # Get unique periods in order
    unique_periods = df[["period_display", "period_sort"]].drop_duplicates().sort_values("period_sort")
    period_sort_map = dict(zip(unique_periods["period_display"], unique_periods["period_sort"]))
    periods = unique_periods["period_display"].tolist()
    if not periods:
        st.info("No period data available.")
        return

    # Build per-entity per-period per-indicator data
    comparison_rows = []
    for uid, name in entities:
        if comparison_mode == "facility" and uid != "all":
            entity_df = df[df["orgUnit"] == uid].copy()
            entity_facility_uids = [uid]
        elif comparison_mode == "region" and is_national and uid != "all":
            region_facility_uids = [
                f[1] for f in facilities_by_region.get(uid, [])
            ]
            entity_df = df[df["orgUnit"].isin(region_facility_uids)].copy()
            entity_facility_uids = region_facility_uids
        else:
            entity_df = df.copy()
            entity_facility_uids = facility_uids

        if entity_df.empty:
            continue

        for period_display in periods:
            period_df = entity_df[entity_df["period_display"] == period_display].copy()
            if period_df.empty:
                continue
            row = {
                entity_label_col: name,
                "period_display": period_display,
                "period_sort": period_sort_map.get(period_display, 0),
            }
            for ind in indicators:
                numerator, denominator, _ = get_numerator_denominator_for_newborn_kpi_with_all(
                    period_df, ind["kpi_name"], entity_facility_uids, {},
                )
                rate = (numerator / denominator * 100) if denominator > 0 else None
                row[f"{ind['short_name']}_rate"] = rate
                row[f"{ind['short_name']}_num"] = int(numerator)
                row[f"{ind['short_name']}_den"] = int(denominator)
            comparison_rows.append(row)

    if not comparison_rows:
        st.info("No comparison data available.")
        return

    comp_df = pd.DataFrame(comparison_rows)
    comp_df = comp_df.sort_values(["period_sort", entity_label_col])

    # Build 1xN subplot grid
    n_indicators = len(indicators)
    if n_indicators > 0:
        n_cols = n_indicators
        n_rows = 1
        subplot_titles = [ind["display_name"] for ind in indicators]
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.12,
        )

        entity_names = comp_df[entity_label_col].unique()
        entity_colors = {}
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for idx, en in enumerate(entity_names):
            entity_colors[en] = palette[idx % len(palette)]

        for idx, ind in enumerate(indicators):
            col_idx = idx + 1
            rate_col = f"{ind['short_name']}_rate"
            num_col = f"{ind['short_name']}_num"
            den_col = f"{ind['short_name']}_den"

            ind_data = comp_df[[entity_label_col, "period_display", "period_sort", rate_col, num_col, den_col]].dropna(subset=[rate_col]).copy()

            for en in entity_names:
                en_data = ind_data[ind_data[entity_label_col] == en].sort_values("period_sort")
                if en_data.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        name=en,
                        x=en_data["period_display"],
                        y=en_data[rate_col],
                        mode="lines",
                        line=dict(color=entity_colors.get(en, "#333333"), width=2),
                        connectgaps=False,
                        hovertemplate=(
                            f"<b>{en}</b><br>"
                            "Period: %{x}<br>"
                            f"{ind['display_name']}: %{{y:.1f}}%<br>"
                            "Numerator: %{customdata[0]}<br>"
                            "Denominator: %{customdata[1]}<br>"
                            "<extra></extra>"
                        ),
                        customdata=en_data[[num_col, den_col]].values,
                        showlegend=(idx == 0),
                    ),
                    row=1, col=col_idx,
                )

            fig.add_hline(
                y=100, line_dash="dash", line_color="green", line_width=1.5,
                row=1, col=col_idx,
            )

            fig.update_xaxes(
                row=1, col=col_idx,
                type="category",
                categoryorder="array",
                categoryarray=periods,
                tickangle=-45,
                gridcolor="rgba(128,128,128,0.2)",
                showline=True,
                linewidth=1,
                linecolor="rgba(128,128,128,0.5)",
                mirror=True,
            )
            fig.update_yaxes(
                row=1, col=col_idx,
                range=[0, 105],
                dtick=25,
                gridcolor="rgba(128,128,128,0.2)",
                zeroline=True,
                zerolinecolor="rgba(128,128,128,0.3)",
                showline=True,
                linewidth=1,
                linecolor="rgba(128,128,128,0.5)",
                mirror=True,
                ticksuffix="%",
            )

        chart_title = "Vital Monitoring - Facility Comparison" if comparison_mode == "facility" else "Vital Monitoring - Region Comparison"
        fig.update_layout(
            title=dict(text=chart_title, font=dict(size=16)),
            height=500,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color,
            title_font_color=text_color,
            title_y=0.95,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            margin=dict(l=50, r=50, t=100, b=80),
            hovermode="x unified",
        )
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=12)

        st.plotly_chart(fig, use_container_width=True)

    # Aggregated comparison table
    st.subheader("Aggregated Comparison Table")
    st.caption("Values shown as: Rate% (numerator / denominator)")
    summary_rows = []
    for uid, name in entities:
        row = {entity_label_col: name}
        entity_comp = comp_df[comp_df[entity_label_col] == name]
        if entity_comp.empty:
            for ind in indicators:
                row[ind["short_name"]] = "-"
            summary_rows.append(row)
            continue
        for ind in indicators:
            total_num = int(entity_comp[f"{ind['short_name']}_num"].sum())
            total_den = int(entity_comp[f"{ind['short_name']}_den"].sum())
            overall_pct = (total_num / total_den * 100) if total_den > 0 else 0.0
            row[ind["short_name"]] = f"{overall_pct:.1f}% ({total_num}/{total_den})" if total_den > 0 else "-"
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("ℹ️ How each indicator is computed"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Temperature Taken at Admission</b></td>
                <td style="padding:8px;">Newborns with temperature recorded at admission</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Birth Weight Taken</b></td>
                <td style="padding:8px;">Newborns with birth weight recorded</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Weight Taken at Admission</b></td>
                <td style="padding:8px;">Newborns with weight recorded at NICU admission</td>
                <td style="padding:8px;">Total admitted newborns</td>
            </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )


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

    source_series = df["source"] if "source" in df.columns else None

    # Try enrollment_date first if it exists
    if "enrollment_date" in df.columns:
        df["enrollment_date"] = parse_dashboard_dates(
            df["enrollment_date"], source_series
        )
    else:
        # Get the SPECIFIC date column for this KPI
        kpi_date_column = get_relevant_date_column_for_newborn_kpi_with_all(current_kpi)

        # Try KPI-specific date column
        if kpi_date_column and kpi_date_column in df.columns:
            df["enrollment_date"] = parse_dashboard_dates(
                df[kpi_date_column], source_series
            )
        elif "combined_date" in df.columns:
            df["enrollment_date"] = parse_dashboard_dates(
                df["combined_date"], source_series
            )
        else:
            # Look for program stage event dates
            program_stage_date_columns = [
                col
                for col in df.columns
                if col.startswith("event_date_") or col == "event_date"
            ]

            for col in program_stage_date_columns:
                try:
                    df["enrollment_date"] = parse_dashboard_dates(
                        df[col], source_series
                    )
                    if not df["enrollment_date"].isna().all():
                        break
                except:
                    continue

    # If still no date found, use current date
    if "enrollment_date" not in df.columns or df["enrollment_date"].isna().all():
        df["enrollment_date"] = pd.Timestamp.now().normalize()

    return df


def render_newborn_patient_filter_controls(
    patient_df, container=None, context="newborn", facility_uids=None
):
    """Simple filter controls for newborn patient data"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix
    key_suffix = f"_{context}"

    current_kpi = st.session_state.get("selected_newborn_kpi", "Inborn Rate (%)")
    is_coverage_rate = current_kpi == "Newborn Coverage Rate"

    # Time Period options
    time_options = [
        "All Time",
        "Custom Range",
        "This Month",
        "Last Month",
        "This Year",
        "Last Year",
    ] if is_coverage_rate else [
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

    source_options = _available_source_options(patient_df, facility_uids)
    if source_options:
        default_sources = [
            source for source in SOURCE_DEFAULT_SELECTION if source in source_options
        ] or source_options
        current_source = st.session_state.get(
            f"source_filter{key_suffix}", default_sources
        )
        if isinstance(current_source, str):
            current_source = default_sources if current_source == SOURCE_FILTER_ALL else [current_source]
        current_source = [
            source for source in current_source if source in source_options
        ] or default_sources

        filters["source"] = container.multiselect(
            "Source",
            source_options,
            default=current_source,
            key=f"source_filter{key_suffix}",
        )
        filters["source_options"] = source_options
    else:
        filters["source"] = source_options or [SOURCE_FILTER_ALL]
        filters["source_options"] = source_options

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

    if is_coverage_rate:
        available_aggregations = [
            a for a in available_aggregations if a in ["Monthly", "Yearly"]
        ]
        if not available_aggregations:
            available_aggregations = ["Monthly"]

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
    st.session_state.filters["source"] = filters["source"]

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

    selected_sources = filters.get("source", [SOURCE_FILTER_ALL])
    if isinstance(selected_sources, str):
        selected_sources = [selected_sources]

    if (
        "source" in df.columns
        and filters.get("source_options")
        and not selected_sources
    ):
        df = df.iloc[0:0].copy()

    normalized_sources = {
        _normalize_source_value(source)
        for source in selected_sources
        if source and source != SOURCE_FILTER_ALL
    }
    if normalized_sources and "source" in df.columns:
        source_mask = df["source"].map(_normalize_source_value).isin(normalized_sources)
        df = df[source_mask].copy()

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
                source_series = df["source"] if "source" in df.columns else None
                df["event_date"] = parse_dashboard_dates(df[date_col], source_series)

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
    # Hypothermia combined view
    "HYPO_COMBINED_MARKER",
    "HYPO_COMBINED_INDICATORS",
    "_render_hypothermia_combined_trend_chart",
    "_render_hypothermia_combined_comparison_chart",
    # Hypothermia Quality of Care
    "HYPO_QOC_MARKER",
    "THERMAL_CATEGORIES",
    "_render_hypothermia_qoc_trend_chart",
    # CPAP Timing
    "CPAP_TIMING_QOC_MARKER",
    "CPAP_TIMING_CATEGORIES",
    "_render_cpap_timing_qoc_trend_chart",
    # Vital Monitoring
    "VITAL_MONITORING_MARKER",
    "VITAL_MONITORING_INDICATORS",
    "_render_vital_monitoring_trend_chart",
    "_render_vital_monitoring_comparison_chart",
    # KMC Timing QoC
    "KMC_TIMING_QOC_MARKER",
    "KMC_TIMING_CATEGORIES",
    "_render_kmc_timing_qoc_trend_chart",
]
