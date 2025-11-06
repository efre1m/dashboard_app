# dashboards/data_quality_tracking.py
import streamlit as st
from dashboards.maternal_dq import render_maternal_data_quality
from dashboards.newborn_dq import render_newborn_data_quality
from utils.queries import get_facilities_grouped_by_region


def render_data_quality_tracking(user):
    """Main function to render Data Quality Tracking dashboard with both tabs"""

    st.markdown(
        f'<div class="main-header">🔍 Data Quality Tracking</div>',
        unsafe_allow_html=True,
    )

    # Check if data is available
    maternal_events_available = (
        "maternal_events_df" in st.session_state
        and not st.session_state.maternal_events_df.empty
    )
    maternal_tei_available = (
        "maternal_tei_df" in st.session_state
        and not st.session_state.maternal_tei_df.empty
    )
    newborn_events_available = (
        "newborn_events_df" in st.session_state
        and not st.session_state.newborn_events_df.empty
    )
    newborn_tei_available = (
        "newborn_tei_df" in st.session_state
        and not st.session_state.newborn_tei_df.empty
    )

    maternal_data_available = maternal_events_available and maternal_tei_available
    newborn_data_available = newborn_events_available and newborn_tei_available

    if not maternal_data_available and not newborn_data_available:
        st.warning(
            "⚠️ No data available. Please visit the Maternal or Newborn Dashboard first to load data."
        )
        return

    # Create facility to region mapping for the current user
    facilities_by_region = get_facilities_grouped_by_region(user)
    facility_to_region_map = {}
    for region_name, facilities in facilities_by_region.items():
        for facility_name, dhis2_uid in facilities:
            facility_to_region_map[facility_name] = region_name

    # Store the mapping in session state for use in child components
    st.session_state.facility_to_region_map = facility_to_region_map

    # Create tabs for maternal and newborn data quality
    tab1, tab2 = st.tabs(
        ["🤰 **Maternal Data Quality**", "👶 **Newborn Data Quality**"]
    )

    with tab1:
        if maternal_data_available:
            render_maternal_data_quality()
        else:
            st.warning("⚠️ No maternal data available. Visit Maternal Dashboard first.")

    with tab2:
        if newborn_data_available:
            render_newborn_data_quality()
        else:
            st.warning("⚠️ No newborn data available. Visit Newborn Dashboard first.")
