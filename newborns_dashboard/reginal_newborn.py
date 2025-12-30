# newborns_dashboard/regional_newborn.py
import streamlit as st
import pandas as pd
import logging
import time

# Import from newborn-specific dashboard components
from newborns_dashboard.dash_co_newborn import (
    normalize_newborn_patient_dates,
    render_newborn_trend_chart_section,
    render_newborn_comparison_chart,
    render_newborn_additional_analytics,
    get_text_color,
    apply_newborn_patient_filters,
    render_newborn_patient_filter_controls,
    render_newborn_kpi_tab_navigation,
)

logging.basicConfig(level=logging.INFO)


def render_newborn_dashboard_shared(
    user,
    newborn_data,
    region_name,
    facilities_by_region,
    facility_mapping,
    view_mode="Normal Trend",
):
    """Optimized Newborn Dashboard rendering using patient-level data with UID filtering - REGIONAL VERSION"""

    # Only run if this is the active tab
    if st.session_state.active_tab != "newborn":
        return

    logging.info("Newborn dashboard rendering with patient-level data - REGIONAL")

    if not newborn_data:
        st.error("No newborn data available")
        return

    # GET PATIENT-LEVEL DATA
    patients_df = newborn_data.get("patients", pd.DataFrame())

    if patients_df.empty:
        st.error("No newborn patient data available")
        return

    # Ensure orgUnit column exists
    if "orgUnit" not in patients_df.columns:
        st.error(
            "❌ Missing 'orgUnit' column in newborn data. Cannot filter by facility UIDs."
        )
        return

    # Get selected facilities from session state (REGIONAL SIMPLIFIED VERSION)
    selected_facilities = st.session_state.get(
        "selected_facilities", ["All Facilities"]
    )

    # Update facility selection (REGIONAL SIMPLIFIED VERSION)
    facility_uids, display_names, comparison_mode = update_facility_selection(
        selected_facilities, facility_mapping
    )

    # Use patient data directly
    working_df = patients_df.copy()

    # Filter by UID EARLY
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()
        logging.info(
            f"✅ NEWBORN REGIONAL: Filtered by {len(facility_uids)} facility UIDs: {len(working_df)} patients remain"
        )
    else:
        logging.info(
            f"⚠️ NEWBORN REGIONAL: No facility UIDs or orgUnit column. Keeping all {len(working_df)} patients"
        )

    # =========== CRITICAL: Use normalize_newborn_patient_dates ===========
    working_df = normalize_newborn_patient_dates(working_df)

    # Log date statistics
    valid_dates = working_df["event_date"].notna().sum()
    total_patients = len(working_df)
    logging.info(
        f"📅 NEWBORN REGIONAL: event_date - {valid_dates}/{total_patients} valid dates"
    )

    if valid_dates > 0:
        sample_dates = working_df["event_date"].dropna().head(3).tolist()
        logging.info(f"📅 NEWBORN REGIONAL: Sample dates: {sample_dates}")
    # =========== END OF CRITICAL ADDITION ===========

    # Store the original df for KPI calculations
    st.session_state.newborn_patients_df = working_df.copy()

    # Update session state
    st.session_state.current_facility_uids = facility_uids
    st.session_state.current_display_names = display_names
    st.session_state.current_comparison_mode = comparison_mode

    # Optimized header rendering - REGIONAL SIMPLIFIED VERSION
    if selected_facilities == ["All Facilities"]:
        header_title = f"👶 Newborn Care Form - {region_name}"
        header_subtitle = f"all {len(facility_mapping)} facilities"
    elif len(selected_facilities) == 1:
        header_title = f"👶 Newborn Care Form - {selected_facilities[0]}"
        header_subtitle = "1 facility"
    else:
        header_title = "👶 Newborn Care Form - Multiple Facilities"
        header_subtitle = f"{len(selected_facilities)} facilities"

    st.markdown(
        f'<div class="main-header" style="margin-bottom: 0.3rem;">{header_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**Displaying data from {header_subtitle}**")

    # Progress container
    progress_container = st.empty()
    with progress_container.container():
        st.markdown("---")
        st.markdown("### 📈 Preparing Newborn Dashboard...")
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            st.markdown(
                """
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <h4 style="margin: 0 0 10px 0; color: #1f77b4;">Processing Data</h4>
            <p style="margin: 5px 0; font-size: 14px;">• Computing newborn KPIs and indicators...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Generating charts and visualizations...</p>
            <p style="margin: 5px 0; font-size: 14px;">• Preparing data tables...</p>
            <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">This may take 2-4 minutes depending on data size</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with progress_col2:
            st.markdown(
                """
            <div style="text-align: center; padding: 10px;">
            <div style="font-size: 24px;">⏳</div>
            <div style="font-size: 12px; margin-top: 5px;">Processing</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Create containers for better performance
    kpi_container = st.container()

    # Optimized filter layout
    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_newborn_patient_filter_controls(
            working_df, container=col_ctrl, context="regional_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply date filters FIRST to get the correct time period
    logging.info(
        f"🔍 NEWBORN REGIONAL: Calling apply_newborn_patient_filters with {len(working_df)} patients"
    )
    logging.info(f"   - Quick range: {filters.get('quick_range', 'N/A')}")
    if "start_date" in filters and "end_date" in filters:
        logging.info(
            f"   - Date range: {filters['start_date']} to {filters['end_date']}"
        )
    logging.info(f"   - Period label: {filters.get('period_label', 'Monthly')}")
    logging.info(f"   - Facility UIDs: {len(facility_uids)}")

    filtered_for_all = apply_newborn_patient_filters(working_df, filters, facility_uids)

    logging.info(
        f"🔍 NEWBORN REGIONAL: After apply_newborn_patient_filters: {len(filtered_for_all)} patients"
    )

    # Store BOTH versions
    st.session_state["newborn_filtered_patients"] = filtered_for_all.copy()
    st.session_state["newborn_all_patients_for_kpi"] = filtered_for_all.copy()

    # KPI Cards with FILTERED data
    with kpi_container:
        location_name, location_type = get_location_display_name(
            selected_facilities, region_name
        )

        user_id = str(user.get("id", user.get("username", "default_user")))

        # Log before KPI computation
        logging.info(
            f"📊 NEWBORN REGIONAL: Computing KPIs for {len(filtered_for_all):,} filtered patients"
        )

    # CLEAR THE PROGRESS INDICATOR ONCE KPI CARDS ARE DONE
    progress_container.empty()

    # Charts section
    bg_color = filters["bg_color"]
    text_color = get_text_color(bg_color)

    with col_chart:
        selected_kpi = render_newborn_kpi_tab_navigation()

        # Use the view_mode parameter - REGIONAL SIMPLIFIED VERSION
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            render_newborn_comparison_chart(
                kpi_selection=selected_kpi,
                patient_df=filtered_for_all,
                comparison_mode="facility",
                display_names=display_names,
                facility_uids=facility_uids,
                facilities_by_region=facilities_by_region,
                region_names=display_names if comparison_mode == "region" else None,
                bg_color=bg_color,
                text_color=text_color,
                is_national=False,
                filtered_patients=filtered_for_all,
            )
        else:
            render_newborn_trend_chart_section(
                selected_kpi,
                filtered_for_all,
                facility_uids,
                display_names,
                bg_color,
                text_color,
                comparison_mode=comparison_mode,
                facilities_by_region=facilities_by_region,
                region_names=display_names if comparison_mode == "region" else None,
            )

        render_newborn_additional_analytics(
            selected_kpi,
            filtered_for_all,
            facility_uids,
            bg_color,
            text_color,
        )


def update_facility_selection(selected_facilities, facility_mapping):
    """Simplified facility selection update - REGIONAL ONLY"""
    if selected_facilities == ["All Facilities"]:
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
    elif selected_facilities:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    else:
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    return facility_uids, display_names, comparison_mode


def get_location_display_name(selected_facilities, region_name):
    """Simplified location display name generation - REGIONAL ONLY"""
    if selected_facilities == ["All Facilities"]:
        return region_name, "Region"
    elif len(selected_facilities) == 1:
        return selected_facilities[0], "Facility"
    else:
        return f"{len(selected_facilities)} Facilities", "Facilities"
