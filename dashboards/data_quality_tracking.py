# dashboards/data_quality_tracking.py
import streamlit as st
import pandas as pd
import logging
from utils.queries import get_facilities_grouped_by_region

# Import the data quality functions
try:
    from dashboards.maternal_dq import render_maternal_data_quality
    from dashboards.newborn_dq import render_newborn_data_quality
except ImportError as e:
    logging.error(f"Failed to import DQ modules: {e}")

    # Create fallback functions
    def render_maternal_data_quality():
        st.error("Maternal DQ module not available")

    def render_newborn_data_quality():
        st.error("Newborn DQ module not available")


def check_data_availability():
    """Check if data is available and log details"""
    maternal_events_available = (
        "maternal_events_df" in st.session_state
        and st.session_state.maternal_events_df is not None
        and not st.session_state.maternal_events_df.empty
    )

    maternal_tei_available = (
        "maternal_tei_df" in st.session_state
        and st.session_state.maternal_tei_df is not None
        and not st.session_state.maternal_tei_df.empty
    )

    newborn_events_available = (
        "newborn_events_df" in st.session_state
        and st.session_state.newborn_events_df is not None
        and not st.session_state.newborn_events_df.empty
    )

    newborn_tei_available = (
        "newborn_tei_df" in st.session_state
        and st.session_state.newborn_tei_df is not None
        and not st.session_state.newborn_tei_df.empty
    )

    # Log detailed information
    if maternal_events_available:
        maternal_events = st.session_state.maternal_events_df
        logging.info(f"📊 Maternal events data: {len(maternal_events)} rows")
        if "has_actual_event" in maternal_events.columns:
            event_counts = maternal_events["has_actual_event"].value_counts()
            logging.info(
                f"   - has_actual_event distribution: True={event_counts.get(True, 0)}, False={event_counts.get(False, 0)}"
            )
        else:
            logging.warning("   - ❌ No 'has_actual_event' column in maternal events!")
    else:
        logging.warning("❌ No maternal events data available")

    if newborn_events_available:
        newborn_events = st.session_state.newborn_events_df
        logging.info(f"📊 Newborn events data: {len(newborn_events)} rows")
        if "has_actual_event" in newborn_events.columns:
            event_counts = newborn_events["has_actual_event"].value_counts()
            logging.info(
                f"   - has_actual_event distribution: True={event_counts.get(True, 0)}, False={event_counts.get(False, 0)}"
            )
        else:
            logging.warning("   - ❌ No 'has_actual_event' column in newborn events!")
    else:
        logging.warning("❌ No newborn events data available")

    return {
        "maternal": maternal_events_available and maternal_tei_available,
        "newborn": newborn_events_available and newborn_tei_available,
    }


def render_data_quality_tracking(user):
    """Main function to render Data Quality Tracking dashboard"""

    st.markdown(
        '<div class="main-header">Data Quality Tracking</div>', unsafe_allow_html=True
    )

    # ✅ FIX: Check data availability with detailed logging
    data_availability = check_data_availability()
    maternal_data_available = data_availability["maternal"]
    newborn_data_available = data_availability["newborn"]

    if not maternal_data_available and not newborn_data_available:
        st.warning(
            """
        ⚠️ **No data available for Data Quality Analysis**
        
        To use Data Quality Tracking:
        1. First visit either the **Maternal Dashboard** or **Newborn Dashboard** tab
        2. Wait for the data to load completely (you'll see KPI cards and charts)
        3. Then return to this tab to analyze data quality
        
        The data quality dashboard needs the processed data from the main dashboards.
        """
        )
        return

    # Create facility to region mapping
    try:
        facilities_by_region = get_facilities_grouped_by_region(user)
        facility_to_region_map = {}
        for region_name, facilities in facilities_by_region.items():
            for facility_name, dhis2_uid in facilities:
                facility_to_region_map[facility_name] = region_name
        st.session_state.facility_to_region_map = facility_to_region_map
        logging.info("✅ Created facility to region mapping")
    except Exception as e:
        logging.error(f"❌ Failed to create facility mapping: {e}")
        st.session_state.facility_to_region_map = {}

    # Create tabs
    tab1, tab2 = st.tabs(["Maternal Data Quality", "Newborn Data Quality"])

    with tab1:
        if maternal_data_available:
            # ✅ DEBUG: Show data info
            maternal_events = st.session_state.maternal_events_df
            st.info(f"📊 Analyzing {len(maternal_events)} maternal events")
            if "has_actual_event" in maternal_events.columns:
                false_count = (maternal_events["has_actual_event"] == False).sum()
                true_count = (maternal_events["has_actual_event"] == True).sum()
                st.write(
                    f"🔍 Events with actual data: {true_count}, Placeholders: {false_count}"
                )

            render_maternal_data_quality()
        else:
            st.warning(
                """
            ⚠️ **Maternal data not available**
            
            To analyze maternal data quality:
            1. Visit the **Maternal Dashboard** tab first
            2. Wait for data to load completely (KPI cards will appear)
            3. Return to this tab
            """
            )

    with tab2:
        if newborn_data_available:
            # ✅ DEBUG: Show data info
            newborn_events = st.session_state.newborn_events_df
            st.info(f"📊 Analyzing {len(newborn_events)} newborn events")
            if "has_actual_event" in newborn_events.columns:
                false_count = (newborn_events["has_actual_event"] == False).sum()
                true_count = (newborn_events["has_actual_event"] == True).sum()
                st.write(
                    f"🔍 Events with actual data: {true_count}, Placeholders: {false_count}"
                )

            render_newborn_data_quality()
        else:
            st.warning(
                """
            ⚠️ **Newborn data not available**
            
            To analyze newborn data quality:
            1. Visit the **Newborn Dashboard** tab first
            2. Wait for data to load completely (KPI cards will appear)
            3. Return to this tab
            """
            )

    # Add refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Data Quality Analysis", use_container_width=True):
        st.rerun()
