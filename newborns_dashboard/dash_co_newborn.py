# dash_co_newborn.py
import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.kpi_utils import auto_text_color
from newborns_dashboard.kmc_coverage import (
    compute_kmc_kpi,
    render_kmc_trend_chart,
    render_kmc_facility_comparison_chart,
    render_kmc_region_comparison_chart,
)
from newborns_dashboard.kpi_cpap import (
    compute_cpap_kpi,
    compute_cpap_general_kpi,
    compute_cpap_prophylactic_kpi,
    render_cpap_trend_chart,
    render_cpap_general_trend_chart,
    render_cpap_prophylactic_trend_chart,
    render_cpap_facility_comparison_chart,
    render_cpap_general_facility_comparison_chart,
    render_cpap_prophylactic_facility_comparison_chart,
    render_cpap_region_comparison_chart,
    render_cpap_general_region_comparison_chart,
    render_cpap_prophylactic_region_comparison_chart,
)
from newborns_dashboard.kpi_hypothermia import (
    compute_hypothermia_kpi,
    render_hypothermia_trend_chart,
    render_hypothermia_facility_comparison_chart,
    render_hypothermia_region_comparison_chart,
)

# ✅ IMPORT NEW HYPOTHERMIA AFTER ADMISSION KPI
from newborns_dashboard.kpi_hypo_after_adm import (
    compute_hypothermia_after_admission_kpi,
    render_hypothermia_after_admission_trend_chart,
    render_hypothermia_after_admission_facility_comparison_chart,
    render_hypothermia_after_admission_region_comparison_chart,
)

# ✅ IMPORT CORRECTED KMC BOTH RANGES (1000-1999g AND 2000-2499g) KPI WITH FIXED TABLES
from newborns_dashboard.kpi_kmc_1000_2499 import (
    compute_kmc_1000_1999_kpi,
    compute_kmc_2000_2499_kpi,
    render_kmc_both_ranges_trend_chart,  # ✅ CORRECTED FUNCTION WITH FIXED TABLES
    render_kmc_both_ranges_facility_comparison_chart,  # ✅ CORRECTED FUNCTION WITH FIXED TABLES
    render_kmc_both_ranges_region_comparison_chart,  # ✅ CORRECTED FUNCTION WITH FIXED TABLES
)

from newborns_dashboard.kpi_inborn import (
    compute_inborn_kpi,
    compute_inborn_numerator,  # ✅ IMPORT THE NUMERATOR FUNCTION
    render_inborn_trend_chart,
    render_inborn_facility_comparison_chart,
    render_inborn_region_comparison_chart,
)

from newborns_dashboard.kpi_antibiotic import (  # ✅ IMPORT ANTIBIOTICS KPI
    compute_antibiotics_kpi,
    compute_antibiotics_numerator,
    render_antibiotics_trend_chart,
    render_antibiotics_facility_comparison_chart,
    render_antibiotics_region_comparison_chart,
)

# ✅ IMPORT NEW CULTURE DONE KPI
from newborns_dashboard.kpi_culture_done import (
    compute_culture_done_kpi,
    compute_culture_done_numerator,
    compute_antibiotics_denominator,
    render_culture_done_trend_chart,
    render_culture_done_facility_comparison_chart,
    render_culture_done_region_comparison_chart,
    render_culture_done_comprehensive_summary,
)

# ✅ IMPORT NEW CULTURE DONE FOR SEPSIS KPI
from newborns_dashboard.kpi_culture_sepsis import (
    compute_culture_done_sepsis_kpi,
    compute_sepsis_culture_numerator,
    compute_sepsis_cases,
    render_culture_done_sepsis_trend_chart,
    render_culture_done_sepsis_facility_comparison_chart,
    render_culture_done_sepsis_region_comparison_chart,
    render_culture_done_sepsis_comprehensive_summary,
)

from newborns_dashboard.kpi_nmr import (
    compute_nmr_kpi,
    compute_nmr_numerator,  # ✅ IMPORT THE NUMERATOR FUNCTION
    render_nmr_trend_chart,
    render_nmr_facility_comparison_chart,
    render_nmr_region_comparison_chart,
)
from newborns_dashboard.kpi_newborn_bw import (  # ✅ IMPORT NEWBORN BIRTH WEIGHT KPI
    compute_newborn_bw_kpi,
    render_newborn_bw_trend_chart,
    render_newborn_bw_facility_comparison_chart,
    render_newborn_bw_region_comparison_chart,
)

# ✅ IMPORT NEW HYPOTHERMIA INBORN/OUTBORN KPI
from newborns_dashboard.kpi_in_out_hypo import (
    compute_inborn_outborn_hypothermia_kpi,
    render_inborn_outborn_hypothermia_trend_chart,
    render_inborn_outborn_hypothermia_comparison_chart,
    render_inborn_outborn_hypothermia_summary,
)

# KPI mapping for KMC coverage, CPAP coverage, Hypothermia, Inborn, Antibiotics, Culture Done, Culture Done for Sepsis, NMR, and Birth Weight
KPI_MAPPING = {
    "LBW KMC Coverage (%)": {
        "title": "LBW KMC Coverage (%)",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total LBW Newborns",
    },
    "KMC Coverage by Birth Weight Range": {
        "title": "KMC Coverage by Birth Weight Range",
        "numerator_name": "KMC Cases",
        "denominator_name": "Total Newborns by Weight Range",
    },
    "CPAP Coverage for RDS (%)": {
        "title": "CPAP Coverage for RDS (%)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total RDS Newborns",
    },
    "General CPAP Coverage (%)": {
        "title": "General CPAP Coverage (%)",
        "numerator_name": "CPAP Cases",
        "denominator_name": "Total Admissions",
    },
    "Prophylactic CPAP Coverage (%)": {
        "title": "Prophylactic CPAP Coverage (%)",
        "numerator_name": "Prophylactic CPAP Cases",
        "denominator_name": "Total Newborns (1000-2499g)",
    },
    "Hypothermia on Admission (%)": {
        "title": "Hypothermia on Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admissions",
    },
    "Hypothermia After Admission (%)": {
        "title": "Hypothermia After Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admissions",
    },
    "Hypothermia Inborn/Outborn": {
        "title": "Hypothermia Inborn/Outborn",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Babies by Birth Location",
    },
    "Inborn Babies (%)": {
        "title": "Inborn Babies (%)",
        "numerator_name": "Inborn Cases",
        "denominator_name": "Total Admissions",
    },
    "Antibiotics for Clinical Sepsis (%)": {
        "title": "Antibiotics for Clinical Sepsis (%)",
        "numerator_name": "Antibiotics Cases",
        "denominator_name": "Probable Sepsis Cases",
    },
    "Culture Done for Babies on Antibiotics (%)": {  # ✅ EXISTING CULTURE DONE KPI
        "title": "Culture Done for Babies on Antibiotics (%)",
        "numerator_name": "Culture Done Cases",
        "denominator_name": "Total Babies on Antibiotics",
    },
    "Culture Done for Babies with Clinical Sepsis (%)": {  # ✅ NEW CULTURE DONE FOR SEPSIS KPI
        "title": "Culture Done for Babies with Clinical Sepsis (%)",
        "numerator_name": "Culture Done Cases",
        "denominator_name": "Probable Sepsis Cases",
    },
    "Neonatal Mortality Rate (%)": {
        "title": "Neonatal Mortality Rate (%)",
        "numerator_name": "Dead Cases",
        "denominator_name": "Total Admissions",
    },
    "Newborn Birth Weight Distribution": {
        "title": "Newborn Birth Weight Distribution",
        "numerator_name": "Birth Weight Cases",
        "denominator_name": "Total Admissions",
    },
}

# All KPI options including Birth Weight
KPI_OPTIONS = [
    "LBW KMC Coverage (%)",
    "KMC Coverage by Birth Weight Range",  # ✅ NEW COMBINED KPI FOR BOTH RANGES
    "CPAP Coverage for RDS (%)",
    "General CPAP Coverage (%)",
    "Prophylactic CPAP Coverage (%)",
    "Antibiotics for Clinical Sepsis (%)",  # ✅ ADDED ANTIBIOTICS KPI TO NEWBORN CARE
    "Culture Done for Babies on Antibiotics (%)",  # ✅ ADDED EXISTING CULTURE DONE KPI
    "Culture Done for Babies with Clinical Sepsis (%)",  # ✅ ADDED NEW CULTURE DONE FOR SEPSIS KPI
    "Hypothermia on Admission (%)",
    "Hypothermia After Admission (%)",  # ✅ NEW KPI
    "Hypothermia Inborn/Outborn",  # ✅ NEW KPI - RIGHT AFTER HYPOTHERMIA
    "Inborn Babies (%)",
    "Newborn Birth Weight Distribution",
    "Neonatal Mortality Rate (%)",
]

# KPI Grouping for Tab Navigation - Antibiotics and Culture Done added to Newborn Care group
KPI_GROUPS = {
    "Newborn Care": [
        "LBW KMC Coverage (%)",
        "KMC Coverage by Birth Weight Range",  # ✅ ADDED NEW COMBINED KPI
        "CPAP Coverage for RDS (%)",
        "General CPAP Coverage (%)",
        "Prophylactic CPAP Coverage (%)",
        "Antibiotics for Clinical Sepsis (%)",  # ✅ ADDED ANTIBIOTICS TO NEWBORN CARE
        "Culture Done for Babies on Antibiotics (%)",  # ✅ ADDED EXISTING CULTURE DONE KPI
        "Culture Done for Babies with Clinical Sepsis (%)",  # ✅ ADDED NEW CULTURE DONE FOR SEPSIS KPI
    ],
    "Admission Assessment": [
        "Hypothermia on Admission (%)",
        "Hypothermia After Admission (%)",  # ✅ NEW KPI
        "Hypothermia Inborn/Outborn",  # ✅ PLACED RIGHT AFTER HYPOTHERMIA
        "Inborn Babies (%)",
        "Newborn Birth Weight Distribution",
    ],
    "Newborn Outcomes": [
        "Neonatal Mortality Rate (%)",
    ],
}


def get_text_color(bg_color):
    """Get auto text color for background"""
    return auto_text_color(bg_color)


def get_kpi_config(kpi_selection):
    """Get KPI configuration"""
    return KPI_MAPPING.get(kpi_selection, {})


def render_kpi_tab_navigation():
    """Render professional tab navigation for Neonatal KPI selection with smaller buttons"""

    # Custom CSS for smaller, more compact button styling
    st.markdown(
        """
    <style>
    /* Make KPI buttons much smaller like normal buttons */
    div.stButton > button {
        padding: 0.2rem 0.6rem !important;
        font-size: 0.8rem !important;
        height: auto !important;
        min-height: 1.8rem !important;
        margin: 0.05rem !important;
    }
    
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
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ✅ FIX: UNIQUE session state key for newborn dashboard only
    if "selected_kpi_NEWBORN_DASHBOARD" not in st.session_state:
        st.session_state.selected_kpi_NEWBORN_DASHBOARD = "LBW KMC Coverage (%)"

    # Create tabs for all groups - KEEP ORIGINAL TAB NAMES
    tab1, tab2, tab3 = st.tabs(
        [
            "👶 **Newborn Care**",
            "🩺 **Admission Assessment**",
            "📊 **Newborn Outcomes**",
        ]
    )

    selected_kpi = st.session_state.selected_kpi_NEWBORN_DASHBOARD

    with tab1:
        # Newborn Care KPIs - 8 buttons layout (3 rows)
        # First row: 5 buttons
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button(
                "LBW KMC Coverage",
                key="kmc_lbw_btn_newborn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "LBW KMC Coverage (%)" else "secondary"
                ),
            ):
                selected_kpi = "LBW KMC Coverage (%)"

        with col2:
            if st.button(
                "KMC by BW Range",
                key="kmc_both_ranges_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "KMC Coverage by Birth Weight Range"
                    else "secondary"
                ),
            ):
                selected_kpi = "KMC Coverage by Birth Weight Range"

        with col3:
            if st.button(
                "CPAP for RDS",
                key="cpap_rds_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "CPAP Coverage for RDS (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "CPAP Coverage for RDS (%)"

        with col4:
            if st.button(
                "General CPAP",
                key="cpap_general_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "General CPAP Coverage (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "General CPAP Coverage (%)"

        with col5:
            if st.button(
                "Prophylactic CPAP",
                key="cpap_prophylactic_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Prophylactic CPAP Coverage (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Prophylactic CPAP Coverage (%)"

        # Second row: 3 buttons (Antibiotics and Culture Dones)
        col6, col7, col8, col9, col10 = st.columns(5)
        with col6:
            if st.button(
                "Antibiotics for Probable Sepsis",
                key="antibiotics_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Antibiotics for Clinical Sepsis (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Antibiotics for Clinical Sepsis (%)"

        with col7:
            if st.button(
                "Culture Done for Antibiotics",  # ✅ EXISTING CULTURE DONE BUTTON
                key="culture_done_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Culture Done for Babies on Antibiotics (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Culture Done for Babies on Antibiotics (%)"

        with col8:
            if st.button(
                "Culture Done for Sepsis",  # ✅ NEW CULTURE DONE FOR SEPSIS BUTTON
                key="culture_done_sepsis_newborn_care_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi
                    == "Culture Done for Babies with Clinical Sepsis (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Culture Done for Babies with Clinical Sepsis (%)"

    with tab2:
        # Admission Assessment KPIs - FOUR buttons layout (Hypothermia Inborn/Outborn right after Hypothermia)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "Hypothermia on Admission",
                key="hypothermia_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia on Admission (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia on Admission (%)"

        with col2:
            if st.button(
                "Hypothermia After Admission",
                key="hypothermia_after_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia After Admission (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia After Admission (%)"

        with col3:
            if st.button(
                "Hypothermia Inborn/Outborn",
                key="hypo_in_out_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia Inborn/Outborn"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia Inborn/Outborn"

        with col4:
            if st.button(
                "Inborn Babies",
                key="inborn_btn_newborn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "Inborn Babies (%)" else "secondary"
                ),
            ):
                selected_kpi = "Inborn Babies (%)"

        # Birth Weight button on a new row
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            if st.button(
                "Birth Weight",
                key="bw_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Newborn Birth Weight Distribution"
                    else "secondary"
                ),
            ):
                selected_kpi = "Newborn Birth Weight Distribution"

    with tab3:
        # Newborn Outcomes KPIs - single button layout
        (col1,) = st.columns(1)

        with col1:
            if st.button(
                "Neonatal Mortality",
                key="nmr_btn_newborn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Neonatal Mortality Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Neonatal Mortality Rate (%)"

    # ✅ FIX: Use the UNIQUE session state key
    if selected_kpi != st.session_state.selected_kpi_NEWBORN_DASHBOARD:
        st.session_state.selected_kpi_NEWBORN_DASHBOARD = selected_kpi
        st.rerun()

    return st.session_state.selected_kpi_NEWBORN_DASHBOARD


def compute_kmc_both_ranges_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ Compute both KMC ranges for dashboard summary
    """
    if df is None or df.empty:
        return {
            "kmc_1000_1999_rate": 0.0,
            "kmc_1000_1999_count": 0,
            "kmc_1000_1999_total": 0,
            "kmc_2000_2499_rate": 0.0,
            "kmc_2000_2499_count": 0,
            "kmc_2000_2499_total": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported KMC functions directly
    kmc_1000_1999_data = compute_kmc_1000_1999_kpi(df, facility_uids)
    kmc_2000_2499_data = compute_kmc_2000_2499_kpi(df, facility_uids)

    return {
        "kmc_1000_1999_rate": kmc_1000_1999_data["kmc_rate"],
        "kmc_1000_1999_count": kmc_1000_1999_data["kmc_count"],
        "kmc_1000_1999_total": kmc_1000_1999_data["total_newborns"],
        "kmc_2000_2499_rate": kmc_2000_2499_data["kmc_rate"],
        "kmc_2000_2499_count": kmc_2000_2499_data["kmc_count"],
        "kmc_2000_2499_total": kmc_2000_2499_data["total_newborns"],
    }


def compute_inborn_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of inborn KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "inborn_rate": 0.0,
            "inborn_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    inborn_count = compute_inborn_numerator(df, facility_uids)

    # Count unique TEIs in the filtered dataset
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate inborn rate
    inborn_rate = (
        (inborn_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "inborn_rate": float(inborn_rate),
        "inborn_count": int(inborn_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_antibiotics_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of antibiotics KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "antibiotics_rate": 0.0,
            "antibiotics_count": 0,
            "probable_sepsis_count": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    antibiotics_count = compute_antibiotics_numerator(df, facility_uids)

    # Count babies with Probable Sepsis (denominator)
    from newborns_dashboard.kpi_antibiotic import (
        SUBCATEGORIES_INFECTION_UID,
        PROBABLE_SEPSIS_CODE,
    )

    sepsis_mask = (df["dataElement_uid"] == SUBCATEGORIES_INFECTION_UID) & (
        df["value"] == PROBABLE_SEPSIS_CODE
    )

    sepsis_events = df[sepsis_mask]
    probable_sepsis_count = sepsis_events["tei_id"].nunique()

    # Calculate antibiotics rate
    antibiotics_rate = (
        (antibiotics_count / probable_sepsis_count * 100)
        if probable_sepsis_count > 0
        else 0.0
    )

    return {
        "antibiotics_rate": float(antibiotics_rate),
        "antibiotics_count": int(antibiotics_count),
        "probable_sepsis_count": int(probable_sepsis_count),
    }


def compute_culture_done_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of Culture Done KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "culture_rate": 0.0,
            "culture_count": 0,
            "antibiotics_count": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    culture_count = compute_culture_done_numerator(df, facility_uids)
    antibiotics_count = compute_antibiotics_denominator(df, facility_uids)

    # Calculate culture rate
    culture_rate = (
        (culture_count / antibiotics_count * 100) if antibiotics_count > 0 else 0.0
    )

    return {
        "culture_rate": float(culture_rate),
        "culture_count": int(culture_count),
        "antibiotics_count": int(antibiotics_count),
    }


def compute_culture_done_sepsis_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of Culture Done for Sepsis KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "culture_sepsis_rate": 0.0,
            "culture_count": 0,
            "sepsis_count": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    culture_count = compute_sepsis_culture_numerator(df, facility_uids)
    sepsis_count = compute_sepsis_cases(df, facility_uids)

    # Calculate culture rate for sepsis
    culture_sepsis_rate = (
        (culture_count / sepsis_count * 100) if sepsis_count > 0 else 0.0
    )

    return {
        "culture_sepsis_rate": float(culture_sepsis_rate),
        "culture_count": int(culture_count),
        "sepsis_count": int(sepsis_count),
    }


def compute_nmr_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of NMR KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "nmr_rate": 0.0,
            "dead_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported numerator function directly
    dead_count = compute_nmr_numerator(df, facility_uids)

    # Count unique TEIs in the filtered dataset
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate NMR rate
    nmr_rate = (
        (dead_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "nmr_rate": float(nmr_rate),
        "dead_count": int(dead_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_bw_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of Birth Weight KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "total_admissions": 0,
            "bw_categories": {},
            "category_rates": {},
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported birth weight function directly
    bw_data = compute_newborn_bw_kpi(df, facility_uids)

    return {
        "total_admissions": bw_data["total_admissions"],
        "bw_categories": bw_data["bw_categories"],
        "category_rates": bw_data["category_rates"],
    }


def compute_cpap_general_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of General CPAP KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "cpap_general_rate": 0.0,
            "cpap_general_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported general CPAP function directly
    cpap_data = compute_cpap_general_kpi(df, facility_uids, tei_df)

    return {
        "cpap_general_rate": cpap_data["cpap_general_rate"],
        "cpap_general_count": cpap_data["cpap_general_count"],
        "total_admitted_newborns": cpap_data["total_admitted_newborns"],
    }


def compute_cpap_prophylactic_for_dashboard(df, facility_uids=None, tei_df=None):
    """
    ✅ FIX: Independent computation of Prophylactic CPAP KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "cpap_prophylactic_rate": 0.0,
            "cpap_prophylactic_count": 0,
            "total_target_weight": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported prophylactic CPAP function directly
    cpap_data = compute_cpap_prophylactic_kpi(df, facility_uids)

    return {
        "cpap_prophylactic_rate": cpap_data["cpap_prophylactic_rate"],
        "cpap_prophylactic_count": cpap_data["cpap_prophylactic_count"],
        "total_target_weight": cpap_data["total_target_weight"],
    }


def compute_hypothermia_after_admission_for_dashboard(
    df, facility_uids=None, tei_df=None
):
    """
    ✅ Independent computation of Hypothermia After Admission KPI for dashboard
    This ensures the counting is done correctly without relying on trend functions
    """
    if df is None or df.empty:
        return {
            "hypothermia_rate": 0.0,
            "hypothermia_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # ✅ Use the imported hypothermia after admission function directly
    hypothermia_data = compute_hypothermia_after_admission_kpi(
        df, facility_uids, tei_df
    )

    return {
        "hypothermia_rate": hypothermia_data["hypothermia_rate"],
        "hypothermia_count": hypothermia_data["hypothermia_count"],
        "total_admitted_newborns": hypothermia_data["total_admitted_newborns"],
    }


def render_trend_chart_section(
    kpi_selection,
    filtered_events,
    facility_uids,
    display_names,
    bg_color,
    text_color,
    tei_df=None,
):
    """Render the trend chart based on KPI selection - ONLY LINE CHART"""

    if kpi_selection == "LBW KMC Coverage (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            kmc_data = compute_kmc_kpi(period_df, facility_uids)

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": kmc_data["kmc_rate"],
                    "KMC Cases": kmc_data["kmc_count"],
                    "Total LBW Newborns": kmc_data["total_lbw"],
                }
            )

        group = pd.DataFrame(period_data)
        render_kmc_trend_chart(
            group,
            "period_display",
            "value",
            "LBW KMC Coverage (%)",
            bg_color,
            text_color,
            display_names,
            "KMC Cases",
            "Total LBW Newborns",
            facility_uids,
        )

    elif kpi_selection == "KMC Coverage by Birth Weight Range":
        # ✅ NEW: Render BOTH KMC ranges (1000-1999g and 2000-2499g) in one chart WITH CORRECTED TABLES
        render_kmc_both_ranges_trend_chart(
            filtered_events,
            "period_display",
            "KMC Coverage by Birth Weight Range",
            bg_color,
            text_color,
            display_names,
            facility_uids=facility_uids,
        )

    elif kpi_selection == "CPAP Coverage for RDS (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            cpap_data = compute_cpap_kpi(period_df, facility_uids)

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": cpap_data["cpap_rate"],
                    "CPAP Cases": cpap_data["cpap_count"],
                    "Total RDS Newborns": cpap_data["total_rds"],
                }
            )

        group = pd.DataFrame(period_data)
        render_cpap_trend_chart(
            group,
            "period_display",
            "value",
            "CPAP Coverage for RDS (%)",
            bg_color,
            text_color,
            display_names,
            "CPAP Cases",
            "Total RDS Newborns",
            facility_uids,
        )

    elif kpi_selection == "General CPAP Coverage (%)":
        # ✅ NEW: Render general CPAP trend chart
        render_cpap_general_trend_chart(
            filtered_events,
            "period_display",
            "General CPAP Coverage Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Prophylactic CPAP Coverage (%)":
        period_data = []
        for period in filtered_events["period"].unique():
            period_df = filtered_events[filtered_events["period"] == period]
            period_display = (
                period_df["period_display"].iloc[0] if not period_df.empty else period
            )
            cpap_data = compute_cpap_prophylactic_kpi(period_df, facility_uids)

            period_data.append(
                {
                    "period": period,
                    "period_display": period_display,
                    "value": cpap_data["cpap_prophylactic_rate"],
                    "Prophylactic CPAP Cases": cpap_data["cpap_prophylactic_count"],
                    "Total Newborns (1000-2499g)": cpap_data["total_target_weight"],
                }
            )

        group = pd.DataFrame(period_data)
        render_cpap_prophylactic_trend_chart(
            group,
            "period_display",
            "value",
            "Prophylactic CPAP Coverage Trend",
            bg_color,
            text_color,
            display_names,
            "Prophylactic CPAP Cases",
            "Total Newborns (1000-2499g)",
            facility_uids,
        )

    elif kpi_selection == "Antibiotics for Clinical Sepsis (%)":
        # ✅ NEW: Render antibiotics trend chart
        render_antibiotics_trend_chart(
            filtered_events,
            "period_display",
            "Antibiotics for Clinical Sepsis Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Culture Done for Babies on Antibiotics (%)":
        # ✅ NEW: Render culture done trend chart
        render_culture_done_trend_chart(
            filtered_events,
            "period_display",
            "Culture Done for Babies on Antibiotics Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Culture Done for Babies with Clinical Sepsis (%)":
        # ✅ NEW: Render culture done for sepsis trend chart
        render_culture_done_sepsis_trend_chart(
            filtered_events,
            "period_display",
            "Culture Done for Babies with Clinical Sepsis Trend",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Hypothermia on Admission (%)":
        # ✅ FIX: Pass TEI dataframe to prevent overcounting
        render_hypothermia_trend_chart(
            filtered_events,
            "period_display",
            "Hypothermia on Admission (%)",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,  # ✅ Pass TEI dataframe
        )

    elif kpi_selection == "Hypothermia After Admission (%)":
        # ✅ NEW: Render hypothermia after admission trend chart
        render_hypothermia_after_admission_trend_chart(
            filtered_events,
            "period_display",
            "Hypothermia After Admission (%)",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,  # ✅ Pass TEI dataframe
        )

    elif kpi_selection == "Hypothermia Inborn/Outborn":
        # ✅ NEW: Render hypothermia inborn/outborn trend chart
        render_inborn_outborn_hypothermia_trend_chart(
            filtered_events,
            "period_display",
            "Hypothermia at Admission by Birth Location",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,  # ✅ Pass TEI dataframe
        )

    elif kpi_selection == "Inborn Babies (%)":
        # Then render the trend chart
        render_inborn_trend_chart(
            filtered_events,
            "period_display",
            "Inborn Babies Trend Over Time",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )

    elif kpi_selection == "Newborn Birth Weight Distribution":
        # ✅ NEW: Render birth weight trend chart
        render_newborn_bw_trend_chart(
            filtered_events,
            "period_display",
            "Newborn Birth Weight Distribution Trend",
            bg_color,
            text_color,
            facility_names=display_names,
            facility_uids=facility_uids,
        )

    elif kpi_selection == "Neonatal Mortality Rate (%)":
        # Then render the trend chart
        render_nmr_trend_chart(
            filtered_events,
            "period_display",
            "Neonatal Mortality Rate Trend Over Time",
            bg_color,
            text_color,
            facility_uids=facility_uids,
            tei_df=tei_df,
        )


def render_comparison_chart(
    kpi_selection,
    filtered_events,
    comparison_mode,
    display_names,
    facility_uids,
    facilities_by_region,
    bg_color,
    text_color,
    is_national=False,
    tei_df=None,
):
    """Render comparison charts for both national and regional views"""

    if comparison_mode == "facility":
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="LBW KMC Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )
        elif kpi_selection == "KMC Coverage by Birth Weight Range":
            # ✅ NEW: Render BOTH KMC ranges facility comparison chart WITH CORRECTED TABLES
            render_kmc_both_ranges_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="KMC Coverage by Birth Weight Range - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
            )
        elif kpi_selection == "CPAP Coverage for RDS (%)":
            render_cpap_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="CPAP Coverage for RDS (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="CPAP Cases",
                denominator_name="Total RDS Newborns",
            )
        elif kpi_selection == "General CPAP Coverage (%)":
            # ✅ NEW: Render general CPAP facility comparison chart
            render_cpap_general_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="General CPAP Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Prophylactic CPAP Coverage (%)":
            # ✅ NEW: Render prophylactic CPAP facility comparison chart
            render_cpap_prophylactic_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="Prophylactic CPAP Coverage (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name="Prophylactic CPAP Cases",
                denominator_name="Total Newborns (1000-2499g)",
            )
        elif kpi_selection == "Antibiotics for Clinical Sepsis (%)":
            # ✅ NEW: Render antibiotics facility comparison chart
            render_antibiotics_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Antibiotics for Clinical Sepsis (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Done for Babies on Antibiotics (%)":
            # ✅ NEW: Render culture done facility comparison chart
            render_culture_done_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Culture Done for Babies on Antibiotics (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Done for Babies with Clinical Sepsis (%)":
            # ✅ NEW: Render culture done for sepsis facility comparison chart
            render_culture_done_sepsis_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Culture Done for Babies with Clinical Sepsis (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia on Admission (%)":
            # ✅ FIX: Pass TEI dataframe to prevent overcounting
            render_hypothermia_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia on Admission (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia After Admission (%)":
            # ✅ NEW: Render hypothermia after admission facility comparison chart
            render_hypothermia_after_admission_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia After Admission (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia Inborn/Outborn":
            # ✅ NEW: Render hypothermia inborn/outborn facility comparison chart
            render_inborn_outborn_hypothermia_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia Inborn/Outborn - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                comparison_type="facility",
                tei_df=tei_df,  # ✅ Pass TEI dataframe
            )
        elif kpi_selection == "Inborn Babies (%)":
            # Then render the comparison chart
            render_inborn_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Inborn Babies (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )
        elif kpi_selection == "Newborn Birth Weight Distribution":
            # ✅ NEW: Render birth weight facility comparison chart
            render_newborn_bw_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Newborn Birth Weight Distribution - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
            )
        elif kpi_selection == "Neonatal Mortality Rate (%)":
            # Then render the comparison chart
            render_nmr_facility_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Neonatal Mortality Rate (%) - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                tei_df=tei_df,
            )

    else:  # region comparison (only for national)
        if kpi_selection == "LBW KMC Coverage (%)":
            render_kmc_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="LBW KMC Coverage (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="KMC Cases",
                denominator_name="Total LBW Newborns",
            )
        elif kpi_selection == "KMC Coverage by Birth Weight Range":
            # ✅ NEW: Render BOTH KMC ranges region comparison chart WITH CORRECTED TABLES
            st.subheader("📊 KMC Coverage by Birth Weight Range - Region Comparison")

            # Compute overall data for context
            overall_kmc_data = compute_kmc_both_ranges_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="KMC 1000-1999g Rate",
                    value=f"{overall_kmc_data['kmc_1000_1999_rate']:.1f}%",
                )
            with col2:
                st.metric(
                    label="KMC 2000-2499g Rate",
                    value=f"{overall_kmc_data['kmc_2000_2499_rate']:.1f}%",
                )
            with col3:
                # Calculate combined rate
                total_kmc_cases = (
                    overall_kmc_data["kmc_1000_1999_count"]
                    + overall_kmc_data["kmc_2000_2499_count"]
                )
                total_newborns = (
                    overall_kmc_data["kmc_1000_1999_total"]
                    + overall_kmc_data["kmc_2000_2499_total"]
                )
                combined_rate = (
                    (total_kmc_cases / total_newborns * 100)
                    if total_newborns > 0
                    else 0
                )
                st.metric(
                    label="Combined Rate (1000-2499g)",
                    value=f"{combined_rate:.1f}%",
                )

            # Then render the comparison chart WITH CORRECTED TABLES
            render_kmc_both_ranges_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="KMC Coverage by Birth Weight Range - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
            )
        elif kpi_selection == "CPAP Coverage for RDS (%)":
            render_cpap_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="CPAP Coverage for RDS (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="CPAP Cases",
                denominator_name="Total RDS Newborns",
            )
        elif kpi_selection == "General CPAP Coverage (%)":
            # ✅ NEW: Render general CPAP region comparison chart
            st.subheader("📊 General CPAP Coverage - Region Comparison")

            # Compute overall data for context
            overall_cpap_data = compute_cpap_general_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total CPAP Cases",
                    value=f"{overall_cpap_data['cpap_general_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Admitted Newborns",
                    value=f"{overall_cpap_data['total_admitted_newborns']:,}",
                )
            with col3:
                st.metric(
                    label="Overall General CPAP Rate",
                    value=f"{overall_cpap_data['cpap_general_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_cpap_general_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="General CPAP Coverage (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Prophylactic CPAP Coverage (%)":
            # ✅ NEW: Render prophylactic CPAP region comparison chart
            st.subheader("📊 Prophylactic CPAP Coverage - Region Comparison")

            # Compute overall data for context
            overall_cpap_data = compute_cpap_prophylactic_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Prophylactic CPAP Cases",
                    value=f"{overall_cpap_data['cpap_prophylactic_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Newborns (1000-2499g)",
                    value=f"{overall_cpap_data['total_target_weight']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Prophylactic CPAP Rate",
                    value=f"{overall_cpap_data['cpap_prophylactic_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_cpap_prophylactic_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                value_col="value",
                title="Prophylactic CPAP Coverage (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                numerator_name="Prophylactic CPAP Cases",
                denominator_name="Total Newborns (1000-2499g)",
            )
        elif kpi_selection == "Antibiotics for Clinical Sepsis (%)":
            # ✅ NEW: Show independent computation first
            st.subheader("📊 Antibiotics for Clinical Sepsis - Region Comparison")

            # Compute overall data for context
            overall_antibiotics_data = compute_antibiotics_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Antibiotics Cases",
                    value=f"{overall_antibiotics_data['antibiotics_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Probable Sepsis Cases",
                    value=f"{overall_antibiotics_data['probable_sepsis_count']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Antibiotics Rate",
                    value=f"{overall_antibiotics_data['antibiotics_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_antibiotics_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Antibiotics for Clinical Sepsis (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Done for Babies on Antibiotics (%)":
            # ✅ NEW: Show independent computation first
            st.subheader(
                "📊 Culture Done for Babies on Antibiotics - Region Comparison"
            )

            # Compute overall data for context
            overall_culture_data = compute_culture_done_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Culture Done Cases",
                    value=f"{overall_culture_data['culture_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Babies on Antibiotics",
                    value=f"{overall_culture_data['antibiotics_count']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Culture Done Rate",
                    value=f"{overall_culture_data['culture_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_culture_done_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Culture Done for Babies on Antibiotics (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                region_mapping={},
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Culture Done for Babies with Clinical Sepsis (%)":
            # ✅ NEW: Show independent computation first
            st.subheader(
                "📊 Culture Done for Babies with Clinical Sepsis - Region Comparison"
            )

            # Compute overall data for context
            overall_culture_sepsis_data = compute_culture_done_sepsis_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Culture Done Cases",
                    value=f"{overall_culture_sepsis_data['culture_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Probable Sepsis Cases",
                    value=f"{overall_culture_sepsis_data['sepsis_count']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Culture Done for Sepsis Rate",
                    value=f"{overall_culture_sepsis_data['culture_sepsis_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_culture_done_sepsis_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Culture Done for Babies with Clinical Sepsis (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia on Admission (%)":
            # ✅ FIX: Pass TEI dataframe to prevent overcounting
            render_hypothermia_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia on Admission (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia After Admission (%)":
            # ✅ NEW: Render hypothermia after admission region comparison chart
            st.subheader("📊 Hypothermia After Admission - Region Comparison")

            # Compute overall data for context
            overall_hypothermia_data = (
                compute_hypothermia_after_admission_for_dashboard(
                    filtered_events, None, tei_df  # No facility filter for regional
                )
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Hypothermia Cases",
                    value=f"{overall_hypothermia_data['hypothermia_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Admissions",
                    value=f"{overall_hypothermia_data['total_admitted_newborns']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Hypothermia Rate",
                    value=f"{overall_hypothermia_data['hypothermia_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_hypothermia_after_admission_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia After Admission (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Hypothermia Inborn/Outborn":
            # ✅ NEW: Render hypothermia inborn/outborn region comparison chart
            st.subheader("📊 Hypothermia Inborn/Outborn - Region Comparison")

            # Compute overall data for context - PASS tei_df
            overall_data = compute_inborn_outborn_hypothermia_kpi(
                filtered_events, tei_df=tei_df
            )

            # Then render the comparison chart
            render_inborn_outborn_hypothermia_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Hypothermia Inborn/Outborn - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=None,  # For region comparison, we don't filter by facility
                comparison_type="region",
                tei_df=tei_df,  # ✅ Pass TEI dataframe
            )
        elif kpi_selection == "Inborn Babies (%)":
            # ✅ FIX: Show independent computation first
            st.subheader("📊 Inborn Babies - Region Comparison")

            # Compute overall data for context
            overall_inborn_data = compute_inborn_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Inborn Cases",
                    value=f"{overall_inborn_data['inborn_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Admitted Newborns",
                    value=f"{overall_inborn_data['total_admitted_newborns']:,}",
                )
            with col3:
                st.metric(
                    label="Overall Inborn Rate",
                    value=f"{overall_inborn_data['inborn_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_inborn_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Inborn Babies (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )
        elif kpi_selection == "Newborn Birth Weight Distribution":
            # ✅ NEW: Render birth weight region comparison chart
            st.subheader("📊 Newborn Birth Weight Distribution - Region Comparison")

            # Compute overall data for context
            overall_bw_data = compute_bw_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Total Admissions",
                    value=f"{overall_bw_data['total_admissions']:,}",
                )
            with col2:
                # Show normal birth weight rate (2500-4000g) as key indicator
                normal_bw_rate = overall_bw_data["category_rates"].get("2500_4000", 0)
                st.metric(
                    label="Normal Birth Weight Rate (2500-4000g)",
                    value=f"{normal_bw_rate:.1f}%",
                )

            # Then render the comparison chart
            render_newborn_bw_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Newborn Birth Weight Distribution - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
            )
        elif kpi_selection == "Neonatal Mortality Rate (%)":
            # ✅ FIX: Show independent computation first
            st.subheader("📊 Neonatal Mortality Rate - Region Comparison")

            # Compute overall data for context
            overall_nmr_data = compute_nmr_for_dashboard(
                filtered_events, None, tei_df  # No facility filter for regional
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Total Dead Cases",
                    value=f"{overall_nmr_data['dead_count']:,}",
                )
            with col2:
                st.metric(
                    label="Total Admitted Newborns",
                    value=f"{overall_nmr_data['total_admitted_newborns']:,}",
                )
            with col3:
                st.metric(
                    label="Overall NMR Rate",
                    value=f"{overall_nmr_data['nmr_rate']:.1f}%",
                )

            # Then render the comparison chart
            render_nmr_region_comparison_chart(
                df=filtered_events,
                period_col="period_display",
                title="Neonatal Mortality Rate (%) - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=display_names,
                facilities_by_region=facilities_by_region,
                tei_df=tei_df,
            )


def render_additional_analytics(
    kpi_selection, filtered_events, facility_uids, bg_color, text_color
):
    """NO ADDITIONAL ANALYTICS - as requested"""
    pass


def normalize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a single datetime column 'event_date' exists and is timezone-naive.
    Handles:
      - eventDate like '2025-07-25T00:00:00.000'
      - event_date like '7/25/2025'
    """
    if df.empty:
        return df

    df = df.copy()

    # Parse ISO 'eventDate' if present
    if "eventDate" in df.columns:
        # pandas can parse ISO 8601 with milliseconds without explicit format
        iso_parsed = pd.to_datetime(df["eventDate"], errors="coerce")
    else:
        iso_parsed = pd.Series(pd.NaT, index=df.index)

    # Parse US 'event_date' (m/d/Y) if present
    if "event_date" in df.columns:
        us_parsed = pd.to_datetime(df["event_date"], format="%m/%d/%Y", errors="coerce")
    else:
        us_parsed = pd.Series(pd.NaT, index=df.index)

    # Prefer ISO if available, else fallback to US
    df["event_date"] = iso_parsed.where(iso_parsed.notna(), us_parsed)

    # Final safety: coerce any str leftovers
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    return df


def normalize_enrollment_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure enrollmentDate is datetime from '7/25/2025' format."""
    if df.empty or "enrollmentDate" not in df.columns:
        return df
    df = df.copy()
    df["enrollmentDate"] = pd.to_datetime(
        df["enrollmentDate"], format="%m/%d/%Y", errors="coerce"
    )
    return df


def render_simple_filter_controls(events_df, container=None, context="neonatal"):
    """Simple filter controls without KPI selection (KPI selection moved to tabs)"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix based on context
    key_suffix = f"_{context}"

    # NOTE: KPI Selection removed - now handled by tab navigation

    # Time Period
    filters["quick_range"] = container.selectbox(
        "📅 Time Period",
        [
            "Custom Range",
            "Today",
            "This Week",
            "Last Week",
            "This Month",
            "Last Month",
            "This Year",
            "Last Year",
        ],
        index=0,
        key=f"quick_range{key_suffix}",  # Unique key
    )

    # Get dates from dataframe
    min_date, max_date = _get_simple_date_range(events_df)

    # Handle Custom Range vs Predefined Ranges
    if filters["quick_range"] == "Custom Range":
        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"start_date{key_suffix}",  # Unique key
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"end_date{key_suffix}",  # Unique key
            )
    else:
        # For predefined ranges
        _df_for_dates = (
            events_df[["event_date"]].copy()
            if not events_df.empty and "event_date" in events_df.columns
            else pd.DataFrame()
        )
        start_date, end_date = get_date_range(_df_for_dates, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date

    # Aggregation Level
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )
    # Force default to "Monthly" if present, otherwise fallback to first option
    if "Monthly" in available_aggregations:
        default_index = available_aggregations.index("Monthly")
    else:
        default_index = 0

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",  # Unique key
    )

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"  # Unique key
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # ✅ FIX: Use the correct newborn session state key
    filters["kpi_selection"] = st.session_state.get(
        "selected_kpi_NEWBORN_DASHBOARD", "LBW KMC Coverage (%)"
    )

    return filters


def _get_simple_date_range(events_df):
    """Get min/max dates from dataframe"""
    import datetime

    if not events_df.empty and "event_date" in events_df.columns:
        valid_dates = events_df["event_date"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            if hasattr(min_date, "date"):
                min_date = min_date.date()
            if hasattr(max_date, "date"):
                max_date = max_date.date()
            return min_date, max_date

    # Fallback to current date
    today = datetime.date.today()
    return today, today


def apply_simple_filters(events_df, filters, facility_uids=None):
    """Apply simple filters to events dataframe"""
    if events_df.empty:
        return events_df

    df = events_df.copy()

    # Apply date filters
    start_datetime = pd.to_datetime(filters["start_date"])
    end_datetime = pd.to_datetime(filters["end_date"])

    df = df[
        (df["event_date"] >= start_datetime) & (df["event_date"] <= end_datetime)
    ].copy()

    # Apply facility filter if provided
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Assign period
    df = assign_period(df, "event_date", filters["period_label"])

    return df
