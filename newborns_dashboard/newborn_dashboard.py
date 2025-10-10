# dashboards/newborn_dashboard.py
import streamlit as st
import pandas as pd
import concurrent.futures
import requests
from utils.data_service import fetch_program_data_for_user
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    apply_simple_filters,
    render_simple_filter_controls,
)
from utils.status import update_last_sync_time

CACHE_TTL = 600  # 10 minutes


# -------------------- Data Fetch --------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_newborn_data(user, program_uid):
    """Fetch Newborn Care Form data from data_service"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user, program_uid)
        return future.result(timeout=180)


# -------------------- Counting Functions --------------------
def count_unique_newborn_teis(newborn_tei_df):
    """Count unique TEIs from newborn_tei_df using correct column name"""
    if newborn_tei_df.empty:
        return 0

    # Use tei_id column (confirmed from your data)
    if "tei_id" in newborn_tei_df.columns:
        unique_count = newborn_tei_df["tei_id"].nunique()
        return unique_count
    # Fallback to other possible column names
    elif "trackedEntityInstance" in newborn_tei_df.columns:
        unique_count = newborn_tei_df["trackedEntityInstance"].nunique()
        return unique_count
    else:
        return 0


# -------------------- Render Dashboard --------------------
def render_newborn_dashboard(
    user,
    program_uid,
    region_name=None,
    selected_facilities=None,
    facility_uids=None,
    facility_mapping=None,
    facility_names=None,
    view_mode=None,
    country_name=None,
    facilities_by_region=None,
):
    """Render Newborn Care Form dashboard content for all user levels"""

    # ==================== NATIONAL LEVEL USER ====================
    if country_name and facilities_by_region is not None:
        return _render_national_newborn_dashboard(
            user,
            program_uid,
            country_name,
            facilities_by_region,
            facility_mapping,
        )

    # ==================== REGIONAL LEVEL USER ====================
    elif region_name and facility_mapping is not None:
        return _render_regional_newborn_dashboard(
            user,
            program_uid,
            region_name,
            facility_uids,
            facility_mapping,
        )

    # ==================== FACILITY LEVEL USER ====================
    else:
        return _render_facility_newborn_dashboard(user, program_uid)


# ==================== NATIONAL LEVEL IMPLEMENTATION ====================
def _render_national_newborn_dashboard(
    user,
    program_uid,
    country_name,
    facilities_by_region,
    facility_mapping,
):
    """Render Newborn Care Form for national level users"""

    # Get the current selection from session state
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    # Update facility selection
    facility_uids, display_names, comparison_mode = _update_newborn_facility_selection(
        filter_mode,
        selected_regions,
        selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Determine header
    total_facilities = len(facility_mapping)
    selected_facilities_count = len(facility_uids)

    if comparison_mode == "facility" and "All Facilities" in display_names:
        header_text = f"👶 Newborn Care Dashboard - {country_name}"
        selection_text = (
            f"**📊 Displaying data from all {total_facilities} facilities**"
        )
    elif comparison_mode == "facility" and len(display_names) == 1:
        header_text = f"👶 Newborn Care Dashboard - {display_names[0]}"
        selection_text = f"**📊 Displaying data from 1 facility**"
    elif comparison_mode == "facility":
        header_text = f"👶 Newborn Care Dashboard - Multiple Facilities"
        selection_text = (
            f"**📊 Displaying data from {selected_facilities_count} facilities**"
        )
    elif comparison_mode == "region" and len(display_names) == 1:
        header_text = f"👶 Newborn Care Dashboard - {display_names[0]} Region"
        region_facilities_count = 0
        for region in display_names:
            if region in facilities_by_region:
                region_facilities_count += len(facilities_by_region[region])
        selection_text = f"**📊 Displaying data from {region_facilities_count} facilities in 1 region**"
    else:
        header_text = f"👶 Newborn Care Dashboard - Multiple Regions"
        selection_text = f"**📊 Displaying data from {selected_facilities_count} facilities across {len(display_names)} regions**"

    st.markdown(f'<div class="main-header">{header_text}</div>', unsafe_allow_html=True)
    st.markdown(selection_text)

    # Fetch Data
    with st.spinner("Fetching National Newborn Care Data..."):
        try:
            newborn_dfs = fetch_newborn_data(user, program_uid)
            update_last_sync_time()
        except concurrent.futures.TimeoutError:
            st.error("⚠️ Newborn Care data could not be fetched within 3 minutes.")
            return
        except requests.RequestException as e:
            st.error(f"⚠️ Newborn Care data request failed: {e}")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error during Newborn data fetch: {e}")
            return

    # Extract and process DataFrames
    newborn_tei_df = newborn_dfs.get("tei", pd.DataFrame())
    newborn_enrollments_df = newborn_dfs.get("enrollments", pd.DataFrame())
    newborn_events_df = newborn_dfs.get("events", pd.DataFrame())

    # Tag dataset type
    newborn_tei_df["_dataset_type"] = "newborn"

    # Normalize dates
    newborn_enrollments_df = normalize_enrollment_dates(newborn_enrollments_df)
    newborn_events_df = normalize_event_dates(newborn_events_df)

    # Apply facility filtering based on selection - FIXED FOR NATIONAL LEVEL
    filtered_newborn_tei, filtered_newborn_enrollments, filtered_newborn_events = (
        _apply_newborn_facility_filtering(
            newborn_tei_df, newborn_enrollments_df, newborn_events_df, facility_uids
        )
    )

    # Calculate UNIQUE counts using FIXED counting function
    unique_newborn_tei_count = count_unique_newborn_teis(filtered_newborn_tei)
    unique_newborn_enrollments_count = (
        filtered_newborn_enrollments["enrollment"].nunique()
        if not filtered_newborn_enrollments.empty
        and "enrollment" in filtered_newborn_enrollments.columns
        else 0
    )
    unique_newborn_events_count = (
        filtered_newborn_events["event"].nunique()
        if not filtered_newborn_events.empty
        and "event" in filtered_newborn_events.columns
        else 0
    )

    # Display summary metrics - show UNIQUE counts
    st.success("✅ Successfully fetched Newborn Care Data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Entities (Newborn)", unique_newborn_tei_count)
    col2.metric("Unique Enrollments", unique_newborn_enrollments_count)
    col3.metric("Unique Events", unique_newborn_events_count)

    # National level controls and filtering
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            filtered_newborn_events, container=col_ctrl, context="national_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply time-based filters
    time_filtered_newborn_events = apply_simple_filters(
        filtered_newborn_events, filters, facility_uids
    )
    st.session_state["filtered_newborn_events"] = time_filtered_newborn_events.copy()

    # Display charts coming soon message
    with col_chart:
        st.markdown("### 📊 Analytics & Charts")
        st.info("🚧 **Charts & KPIs Coming Soon!**")

        # Show basic data preview
        if not time_filtered_newborn_events.empty:
            st.markdown("#### 📋 Filtered Data Preview")
            st.dataframe(
                time_filtered_newborn_events.head(10), use_container_width=True
            )
        else:
            st.warning("No data available for the selected facilities and time period.")


# ==================== REGIONAL LEVEL IMPLEMENTATION ====================
def _render_regional_newborn_dashboard(
    user,
    program_uid,
    region_name,
    facility_uids,
    facility_mapping,
):
    """Render Newborn Care Form for regional level users"""

    selected_facilities = st.session_state.get("selected_facilities", [])

    # Determine header name
    if selected_facilities == ["All Facilities"]:
        header_name = region_name
        selection_text = f"**📊 Displaying data from all {len(facility_mapping)} facilities in {region_name}**"
    elif len(selected_facilities) == 1:
        header_name = selected_facilities[0]
        selection_text = f"**📊 Displaying data from 1 facility**"
    else:
        header_name = f"Multiple Facilities ({len(selected_facilities)})"
        selection_text = (
            f"**📊 Displaying data from {len(selected_facilities)} facilities**"
        )

    st.markdown(
        f'<div class="main-header">👶 Newborn Care Dashboard - {header_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(selection_text)

    # Fetch Data
    with st.spinner("Fetching Regional Newborn Care Data..."):
        try:
            newborn_dfs = fetch_newborn_data(user, program_uid)
            update_last_sync_time()
        except concurrent.futures.TimeoutError:
            st.error("⚠️ Newborn Care data could not be fetched within 3 minutes.")
            return
        except requests.RequestException as e:
            st.error(f"⚠️ Newborn Care data request failed: {e}")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error during Newborn data fetch: {e}")
            return

    # Extract and process DataFrames
    newborn_tei_df = newborn_dfs.get("tei", pd.DataFrame())
    newborn_enrollments_df = newborn_dfs.get("enrollments", pd.DataFrame())
    newborn_events_df = newborn_dfs.get("events", pd.DataFrame())

    # Tag dataset type
    newborn_tei_df["_dataset_type"] = "newborn"

    # Apply facility filtering based on selection
    filtered_newborn_tei, filtered_newborn_enrollments, filtered_newborn_events = (
        _apply_newborn_facility_filtering(
            newborn_tei_df, newborn_enrollments_df, newborn_events_df, facility_uids
        )
    )

    # Normalize dates
    filtered_newborn_enrollments = normalize_enrollment_dates(
        filtered_newborn_enrollments
    )
    filtered_newborn_events = normalize_event_dates(filtered_newborn_events)

    # Calculate UNIQUE counts using FIXED counting function
    unique_newborn_tei_count = count_unique_newborn_teis(filtered_newborn_tei)
    unique_newborn_enrollments_count = (
        filtered_newborn_enrollments["enrollment"].nunique()
        if not filtered_newborn_enrollments.empty
        and "enrollment" in filtered_newborn_enrollments.columns
        else 0
    )
    unique_newborn_events_count = (
        filtered_newborn_events["event"].nunique()
        if not filtered_newborn_events.empty
        and "event" in filtered_newborn_events.columns
        else 0
    )

    # Display summary metrics - show UNIQUE counts
    st.success("✅ Successfully fetched Newborn Care Data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Entities (Newborn)", unique_newborn_tei_count)
    col2.metric("Unique Enrollments", unique_newborn_enrollments_count)
    col3.metric("Unique Events", unique_newborn_events_count)

    # Regional controls and filtering
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            filtered_newborn_events, container=col_ctrl, context="regional_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply time-based filters
    time_filtered_newborn_events = apply_simple_filters(
        filtered_newborn_events, filters, facility_uids
    )
    st.session_state["filtered_newborn_events"] = time_filtered_newborn_events.copy()

    # Display charts coming soon message
    with col_chart:
        st.markdown("### 📊 Analytics & Charts")
        st.info("🚧 **Charts & KPIs Coming Soon!**")

        # Show basic data preview
        if not time_filtered_newborn_events.empty:
            st.markdown("#### 📋 Filtered Data Preview")
            st.dataframe(
                time_filtered_newborn_events.head(10), use_container_width=True
            )
        else:
            st.warning("No data available for the selected facilities and time period.")


# ==================== FACILITY LEVEL IMPLEMENTATION ====================
def _render_facility_newborn_dashboard(user, program_uid):
    """Render Newborn Care Form for facility level users"""
    facility_name = user.get("facility_name", "Unknown Facility")
    facility_uid = user.get("facility_uid")

    st.markdown(
        f'<div class="main-header">👶 Newborn Care Dashboard - {facility_name}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**📊 Displaying data from 1 facility**")

    # Fetch Data
    with st.spinner("Fetching Facility Newborn Care Data..."):
        try:
            newborn_dfs = fetch_newborn_data(user, program_uid)
            update_last_sync_time()
        except concurrent.futures.TimeoutError:
            st.error("⚠️ Newborn Care data could not be fetched within 3 minutes.")
            return
        except requests.RequestException as e:
            st.error(f"⚠️ Newborn Care data request failed: {e}")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error during Newborn data fetch: {e}")
            return

    # Extract and process DataFrames
    newborn_tei_df = newborn_dfs.get("tei", pd.DataFrame())
    newborn_enrollments_df = newborn_dfs.get("enrollments", pd.DataFrame())
    newborn_events_df = newborn_dfs.get("events", pd.DataFrame())

    # Tag dataset type
    newborn_tei_df["_dataset_type"] = "newborn"

    # Apply facility filtering for single facility
    filtered_newborn_tei, filtered_newborn_enrollments, filtered_newborn_events = (
        _apply_newborn_facility_filtering(
            newborn_tei_df,
            newborn_enrollments_df,
            newborn_events_df,
            [facility_uid] if facility_uid else [],
        )
    )

    # Normalize dates
    filtered_newborn_enrollments = normalize_enrollment_dates(
        filtered_newborn_enrollments
    )
    filtered_newborn_events = normalize_event_dates(filtered_newborn_events)

    # Calculate UNIQUE counts using FIXED counting function
    unique_newborn_tei_count = count_unique_newborn_teis(filtered_newborn_tei)
    unique_newborn_enrollments_count = (
        filtered_newborn_enrollments["enrollment"].nunique()
        if not filtered_newborn_enrollments.empty
        and "enrollment" in filtered_newborn_enrollments.columns
        else 0
    )
    unique_newborn_events_count = (
        filtered_newborn_events["event"].nunique()
        if not filtered_newborn_events.empty
        and "event" in filtered_newborn_events.columns
        else 0
    )

    # Display summary metrics - show UNIQUE counts
    st.success("✅ Successfully fetched Newborn Care Data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Entities (Newborn)", unique_newborn_tei_count)
    col2.metric("Unique Enrollments", unique_newborn_enrollments_count)
    col3.metric("Unique Events", unique_newborn_events_count)

    # Facility controls and filtering
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            filtered_newborn_events, container=col_ctrl, context="facility_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply time-based filters
    time_filtered_newborn_events = apply_simple_filters(
        filtered_newborn_events, filters, facility_uid
    )
    st.session_state["filtered_newborn_events"] = time_filtered_newborn_events.copy()

    # Display charts coming soon message
    with col_chart:
        st.markdown("### 📊 Analytics & Charts")
        st.info("🚧 **Charts & KPIs Coming Soon!**")

        # Show basic data preview
        if not time_filtered_newborn_events.empty:
            st.markdown("#### 📋 Filtered Data Preview")
            st.dataframe(
                time_filtered_newborn_events.head(10), use_container_width=True
            )
        else:
            st.warning("No data available for the selected time period.")


# ==================== COMMON UTILITY FUNCTIONS ====================


def _update_newborn_facility_selection(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
):
    """Update facility selection based on current mode and selections"""
    if filter_mode == "All Facilities":
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"
    elif filter_mode == "By Region" and selected_regions:
        facility_uids, display_names = [], selected_regions
        for region in selected_regions:
            if region in facilities_by_region:
                for fac_name, fac_uid in facilities_by_region[region]:
                    facility_uids.append(fac_uid)
        comparison_mode = "region"
    elif filter_mode == "By Facility" and selected_facilities:
        facility_uids = [
            facility_mapping[f] for f in selected_facilities if f in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    else:
        # Default fallback - all facilities
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    return facility_uids, display_names, comparison_mode


def _apply_newborn_facility_filtering(
    newborn_tei_df, newborn_enrollments_df, newborn_events_df, facility_uids
):
    """Apply facility filtering to all newborn dataframes based on selection - FIXED FOR NATIONAL LEVEL"""
    if not facility_uids:
        return newborn_tei_df, newborn_enrollments_df, newborn_events_df

    # Start with empty dataframes
    filtered_newborn_tei = pd.DataFrame()
    filtered_newborn_enrollments = pd.DataFrame()
    filtered_newborn_events = pd.DataFrame()

    # Filter TEI based on tei_orgUnit (FIXED FOR NATIONAL LEVEL)
    if not newborn_tei_df.empty:
        if "tei_orgUnit" in newborn_tei_df.columns:
            filtered_newborn_tei = newborn_tei_df[
                newborn_tei_df["tei_orgUnit"].isin(facility_uids)
            ].copy()
        elif "orgUnit" in newborn_tei_df.columns:
            filtered_newborn_tei = newborn_tei_df[
                newborn_tei_df["orgUnit"].isin(facility_uids)
            ].copy()
        else:
            # If no orgUnit column, return all TEI data
            filtered_newborn_tei = newborn_tei_df.copy()

    # Filter events by facility UIDs
    if not newborn_events_df.empty and "orgUnit" in newborn_events_df.columns:
        filtered_newborn_events = newborn_events_df[
            newborn_events_df["orgUnit"].isin(facility_uids)
        ].copy()

    # Filter enrollments based on orgUnit
    if not newborn_enrollments_df.empty and "orgUnit" in newborn_enrollments_df.columns:
        filtered_newborn_enrollments = newborn_enrollments_df[
            newborn_enrollments_df["orgUnit"].isin(facility_uids)
        ].copy()

    return filtered_newborn_tei, filtered_newborn_enrollments, filtered_newborn_events
