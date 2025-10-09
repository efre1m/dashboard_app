# dashboards/newborn_dashboard.py
import streamlit as st
import pandas as pd
import concurrent.futures
import requests
import json
from io import BytesIO
import zipfile
from utils.data_service import fetch_program_data_for_user
from utils.dash_co import (
    normalize_event_dates,
    normalize_enrollment_dates,
    apply_simple_filters,
    render_simple_filter_controls,
)
from utils.status import update_last_sync_time
from utils.kpi_utils import clear_cache

CACHE_TTL = 600  # 10 minutes


# -------------------- Data Fetch --------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_newborn_data(user, program_uid):
    """Fetch Newborn Care Form data from data_service"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user, program_uid)
        return future.result(timeout=180)


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
            selected_facilities,
            facility_uids,
            facility_names,
            view_mode,
        )

    # ==================== REGIONAL LEVEL USER ====================
    elif region_name and facility_mapping is not None:
        return _render_regional_newborn_dashboard(
            user,
            program_uid,
            region_name,
            selected_facilities,
            facility_uids,
            facility_mapping,
            facility_names,
            view_mode,
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
    selected_facilities,
    facility_uids,
    facility_names,
    view_mode,
):
    """Render Newborn Care Form for national level users"""

    # Get the current selection from session state (EXACTLY like maternal)
    filter_mode = st.session_state.get("filter_mode", "All Facilities")
    selected_regions = st.session_state.get("selected_regions", [])
    selected_facilities = st.session_state.get("selected_facilities", [])

    # Update facility selection (EXACTLY like maternal)
    facility_uids, display_names, comparison_mode = _update_facility_selection(
        filter_mode,
        selected_regions,
        selected_facilities,
        facilities_by_region,
        facility_mapping,
    )

    # Determine header (EXACTLY like maternal)
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
            dfs = fetch_newborn_data(user, program_uid)
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
    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])

    # Tag dataset type
    tei_df["_dataset_type"] = "newborn"

    # Normalize dates
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    # Apply facility filtering based on selection
    filtered_tei, filtered_enrollments, filtered_events = (
        _apply_facility_filtering_fixed(
            tei_df, enrollments_df, events_df, facility_uids
        )
    )

    # Calculate UNIQUE counts (not total rows)
    unique_tei_count = (
        filtered_tei["trackedEntityInstance"].nunique()
        if "trackedEntityInstance" in filtered_tei.columns
        else 0
    )
    unique_enrollments_count = (
        filtered_enrollments["enrollment"].nunique()
        if "enrollment" in filtered_enrollments.columns
        else 0
    )
    unique_events_count = (
        filtered_events["event"].nunique() if "event" in filtered_events.columns else 0
    )

    # Display summary metrics - show UNIQUE counts
    st.success("✅ Successfully fetched and filtered Newborn Care Data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Entities (Newborn)", unique_tei_count)
    col2.metric("Unique Enrollments", unique_enrollments_count)
    col3.metric("Unique Events", unique_events_count)

    # Export buttons - ALWAYS VISIBLE like maternal dashboard
    st.markdown("---")
    st.markdown("### 📤 Export Data")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.markdown("**Raw JSON Export**")
        if raw_json:
            st.download_button(
                "📥 Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{country_name}_newborn_raw.json",
                mime="application/json",
            )
        else:
            st.warning("No raw JSON data available")

    with col_exp2:
        st.markdown("**All Data Export**")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            if not tei_df.empty:
                zf.writestr("tei.csv", tei_df.to_csv(index=False).encode("utf-8"))
            if not enrollments_df.empty:
                zf.writestr(
                    "enrollments.csv",
                    enrollments_df.to_csv(index=False).encode("utf-8"),
                )
            if not events_df.empty:
                zf.writestr("events.csv", events_df.to_csv(index=False).encode("utf-8"))
        buffer.seek(0)
        st.download_button(
            "📊 Download All DataFrames (ZIP)",
            data=buffer,
            file_name=f"{country_name}_newborn_dataframes.zip",
            mime="application/zip",
        )

    with col_exp3:
        st.markdown("**Filtered Data Export**")
        if not filtered_events.empty:
            # Create individual CSV download buttons for filtered data
            tei_csv = filtered_tei.to_csv(index=False) if not filtered_tei.empty else ""
            enroll_csv = (
                filtered_enrollments.to_csv(index=False)
                if not filtered_enrollments.empty
                else ""
            )
            events_csv = (
                filtered_events.to_csv(index=False) if not filtered_events.empty else ""
            )

            if tei_csv:
                st.download_button(
                    "📥 Filtered TEI Data",
                    data=tei_csv,
                    file_name="filtered_newborn_tei.csv",
                    mime="text/csv",
                )
            if enroll_csv:
                st.download_button(
                    "📥 Filtered Enrollments",
                    data=enroll_csv,
                    file_name="filtered_newborn_enrollments.csv",
                    mime="text/csv",
                )
            if events_csv:
                st.download_button(
                    "📥 Filtered Events",
                    data=events_csv,
                    file_name="filtered_newborn_events.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No filtered data available")

    # National level controls and filtering
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            filtered_events, container=col_ctrl, context="national_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply time-based filters
    time_filtered_events = apply_simple_filters(filtered_events, filters, facility_uids)
    st.session_state["filtered_events"] = time_filtered_events.copy()

    # Display charts coming soon message
    with col_chart:
        st.markdown("### 📊 Analytics & Charts")
        st.info("🚧 **Charts & KPIs Coming Soon!**")

        # Show view mode information (like maternal)
        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f"**Comparison Mode:** {comparison_mode.title()} Comparison across {len(display_names)} {comparison_mode}s"
            )
        else:
            st.markdown(f"**View Mode:** Normal Trend")

        # Show basic data preview
        if not time_filtered_events.empty:
            st.markdown("#### 📋 Filtered Data Preview")
            st.dataframe(time_filtered_events.head(10), use_container_width=True)

            # Show facility distribution in filtered data
            if "orgUnit" in time_filtered_events.columns:
                facility_counts = time_filtered_events["orgUnit"].value_counts()
                st.markdown("#### 🏥 Events per Facility in Selection")
                st.dataframe(facility_counts, use_container_width=True)
        else:
            st.warning("No data available for the selected facilities and time period.")


# ==================== REGIONAL LEVEL IMPLEMENTATION ====================
def _render_regional_newborn_dashboard(
    user,
    program_uid,
    region_name,
    selected_facilities,
    facility_uids,
    facility_mapping,
    facility_names,
    view_mode,
):
    """Render Newborn Care Form for regional level users"""
    # Determine header name (EXACTLY like maternal)
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
            dfs = fetch_newborn_data(user, program_uid)
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
    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])

    # Tag dataset type
    tei_df["_dataset_type"] = "newborn"

    # Apply facility filtering based on selection
    filtered_tei, filtered_enrollments, filtered_events = (
        _apply_facility_filtering_fixed(
            tei_df, enrollments_df, events_df, facility_uids
        )
    )

    # Normalize dates
    filtered_enrollments = normalize_enrollment_dates(filtered_enrollments)
    filtered_events = normalize_event_dates(filtered_events)

    # Calculate UNIQUE counts (not total rows)
    unique_tei_count = (
        filtered_tei["trackedEntityInstance"].nunique()
        if "trackedEntityInstance" in filtered_tei.columns
        else 0
    )
    unique_enrollments_count = (
        filtered_enrollments["enrollment"].nunique()
        if "enrollment" in filtered_enrollments.columns
        else 0
    )
    unique_events_count = (
        filtered_events["event"].nunique() if "event" in filtered_events.columns else 0
    )

    # Display summary metrics - show UNIQUE counts
    st.success("✅ Successfully fetched and filtered Newborn Care Data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Entities (Newborn)", unique_tei_count)
    col2.metric("Unique Enrollments", unique_enrollments_count)
    col3.metric("Unique Events", unique_events_count)

    # Export buttons - ALWAYS VISIBLE like maternal dashboard
    st.markdown("---")
    st.markdown("### 📤 Export Data")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.markdown("**Raw JSON Export**")
        if raw_json:
            st.download_button(
                "📥 Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{region_name}_newborn_raw.json",
                mime="application/json",
            )
        else:
            st.warning("No raw JSON data available")

    with col_exp2:
        st.markdown("**All Data Export**")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            if not tei_df.empty:
                zf.writestr("tei.csv", tei_df.to_csv(index=False).encode("utf-8"))
            if not enrollments_df.empty:
                zf.writestr(
                    "enrollments.csv",
                    enrollments_df.to_csv(index=False).encode("utf-8"),
                )
            if not events_df.empty:
                zf.writestr("events.csv", events_df.to_csv(index=False).encode("utf-8"))
        buffer.seek(0)
        st.download_button(
            "📊 Download All DataFrames (ZIP)",
            data=buffer,
            file_name=f"{region_name}_newborn_dataframes.zip",
            mime="application/zip",
        )

    with col_exp3:
        st.markdown("**Filtered Data Export**")
        if not filtered_events.empty:
            # Create individual CSV download buttons for filtered data
            tei_csv = filtered_tei.to_csv(index=False) if not filtered_tei.empty else ""
            enroll_csv = (
                filtered_enrollments.to_csv(index=False)
                if not filtered_enrollments.empty
                else ""
            )
            events_csv = (
                filtered_events.to_csv(index=False) if not filtered_events.empty else ""
            )

            if tei_csv:
                st.download_button(
                    "📥 Filtered TEI Data",
                    data=tei_csv,
                    file_name="filtered_newborn_tei.csv",
                    mime="text/csv",
                )
            if enroll_csv:
                st.download_button(
                    "📥 Filtered Enrollments",
                    data=enroll_csv,
                    file_name="filtered_newborn_enrollments.csv",
                    mime="text/csv",
                )
            if events_csv:
                st.download_button(
                    "📥 Filtered Events",
                    data=events_csv,
                    file_name="filtered_newborn_events.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No filtered data available")

    # Regional controls and filtering
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            filtered_events, container=col_ctrl, context="regional_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply time-based filters
    time_filtered_events = apply_simple_filters(filtered_events, filters, facility_uids)
    st.session_state["filtered_events"] = time_filtered_events.copy()

    # Display charts coming soon message
    with col_chart:
        st.markdown("### 📊 Analytics & Charts")
        st.info("🚧 **Charts & KPIs Coming Soon!**")

        # Show view mode information (like maternal)
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(
                f"**Comparison Mode:** Facility Comparison across {len(selected_facilities)} facilities"
            )
        else:
            st.markdown(f"**View Mode:** Normal Trend for {header_name}")

        # Show basic data preview
        if not time_filtered_events.empty:
            st.markdown("#### 📋 Filtered Data Preview")
            st.dataframe(time_filtered_events.head(10), use_container_width=True)

            # Show facility distribution in filtered data
            if "orgUnit" in time_filtered_events.columns:
                facility_counts = time_filtered_events["orgUnit"].value_counts()
                st.markdown("#### 🏥 Events per Facility in Selection")
                st.dataframe(facility_counts, use_container_width=True)
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
            dfs = fetch_newborn_data(user, program_uid)
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
    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])

    # Tag dataset type
    tei_df["_dataset_type"] = "newborn"

    # Apply facility filtering for single facility
    filtered_tei, filtered_enrollments, filtered_events = (
        _apply_facility_filtering_fixed(
            tei_df, enrollments_df, events_df, [facility_uid] if facility_uid else []
        )
    )

    # Normalize dates
    filtered_enrollments = normalize_enrollment_dates(filtered_enrollments)
    filtered_events = normalize_event_dates(filtered_events)

    # Calculate UNIQUE counts (not total rows)
    unique_tei_count = (
        filtered_tei["trackedEntityInstance"].nunique()
        if "trackedEntityInstance" in filtered_tei.columns
        else 0
    )
    unique_enrollments_count = (
        filtered_enrollments["enrollment"].nunique()
        if "enrollment" in filtered_enrollments.columns
        else 0
    )
    unique_events_count = (
        filtered_events["event"].nunique() if "event" in filtered_events.columns else 0
    )

    # Display summary metrics - show UNIQUE counts
    st.success("✅ Successfully fetched and filtered Newborn Care Data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Tracked Entities (Newborn)", unique_tei_count)
    col2.metric("Unique Enrollments", unique_enrollments_count)
    col3.metric("Unique Events", unique_events_count)

    # Export buttons - ALWAYS VISIBLE like maternal dashboard
    st.markdown("---")
    st.markdown("### 📤 Export Data")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.markdown("**Raw JSON Export**")
        if raw_json:
            st.download_button(
                "📥 Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{facility_name}_newborn_raw.json",
                mime="application/json",
            )
        else:
            st.warning("No raw JSON data available")

    with col_exp2:
        st.markdown("**All Data Export**")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            if not tei_df.empty:
                zf.writestr("tei.csv", tei_df.to_csv(index=False).encode("utf-8"))
            if not enrollments_df.empty:
                zf.writestr(
                    "enrollments.csv",
                    enrollments_df.to_csv(index=False).encode("utf-8"),
                )
            if not events_df.empty:
                zf.writestr("events.csv", events_df.to_csv(index=False).encode("utf-8"))
        buffer.seek(0)
        st.download_button(
            "📊 Download All DataFrames (ZIP)",
            data=buffer,
            file_name=f"{facility_name}_newborn_dataframes.zip",
            mime="application/zip",
        )

    with col_exp3:
        st.markdown("**Filtered Data Export**")
        if not filtered_events.empty:
            # Create individual CSV download buttons for filtered data
            tei_csv = filtered_tei.to_csv(index=False) if not filtered_tei.empty else ""
            enroll_csv = (
                filtered_enrollments.to_csv(index=False)
                if not filtered_enrollments.empty
                else ""
            )
            events_csv = (
                filtered_events.to_csv(index=False) if not filtered_events.empty else ""
            )

            if tei_csv:
                st.download_button(
                    "📥 Filtered TEI Data",
                    data=tei_csv,
                    file_name="filtered_newborn_tei.csv",
                    mime="text/csv",
                )
            if enroll_csv:
                st.download_button(
                    "📥 Filtered Enrollments",
                    data=enroll_csv,
                    file_name="filtered_newborn_enrollments.csv",
                    mime="text/csv",
                )
            if events_csv:
                st.download_button(
                    "📥 Filtered Events",
                    data=events_csv,
                    file_name="filtered_newborn_events.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No filtered data available")

    # Facility controls and filtering
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        filters = render_simple_filter_controls(
            filtered_events, container=col_ctrl, context="facility_newborn"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Apply time-based filters
    time_filtered_events = apply_simple_filters(filtered_events, filters, facility_uid)
    st.session_state["filtered_events"] = time_filtered_events.copy()

    # Display charts coming soon message
    with col_chart:
        st.markdown("### 📊 Analytics & Charts")
        st.info("🚧 **Charts & KPIs Coming Soon!**")

        # Show basic data preview
        if not time_filtered_events.empty:
            st.markdown("#### 📋 Filtered Data Preview")
            st.dataframe(time_filtered_events.head(10), use_container_width=True)
        else:
            st.warning("No data available for the selected time period.")


# ==================== COMMON UTILITY FUNCTIONS ====================


def _update_facility_selection(
    filter_mode,
    selected_regions,
    selected_facilities,
    facilities_by_region,
    facility_mapping,
):
    """Update facility selection based on current mode and selections (EXACTLY like maternal)"""
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


def _apply_facility_filtering_fixed(tei_df, enrollments_df, events_df, facility_uids):
    """FIXED VERSION: Apply facility filtering to all dataframes based on selection"""
    if not facility_uids:
        return tei_df, enrollments_df, events_df

    # Start with empty dataframes
    filtered_tei = pd.DataFrame()
    filtered_enrollments = pd.DataFrame()
    filtered_events = pd.DataFrame()

    # Filter events by facility UIDs
    if not events_df.empty and "orgUnit" in events_df.columns:
        filtered_events = events_df[events_df["orgUnit"].isin(facility_uids)].copy()

    # Filter TEI based on orgUnit (NOT based on events)
    if not tei_df.empty and "orgUnit" in tei_df.columns:
        filtered_tei = tei_df[tei_df["orgUnit"].isin(facility_uids)].copy()

    # Filter enrollments based on orgUnit
    if not enrollments_df.empty and "orgUnit" in enrollments_df.columns:
        filtered_enrollments = enrollments_df[
            enrollments_df["orgUnit"].isin(facility_uids)
        ].copy()

    # If no direct orgUnit filtering worked, try alternative approach
    if (
        filtered_tei.empty
        and "trackedEntityInstance" in tei_df.columns
        and not filtered_events.empty
    ):
        # Get TEI IDs from filtered events and filter TEI
        tei_ids_from_events = filtered_events["trackedEntityInstance"].unique()
        filtered_tei = tei_df[
            tei_df["trackedEntityInstance"].isin(tei_ids_from_events)
        ].copy()

    if (
        filtered_enrollments.empty
        and "trackedEntityInstance" in enrollments_df.columns
        and not filtered_tei.empty
    ):
        # Get TEI IDs from filtered TEI and filter enrollments
        tei_ids_from_tei = filtered_tei["trackedEntityInstance"].unique()
        filtered_enrollments = enrollments_df[
            enrollments_df["trackedEntityInstance"].isin(tei_ids_from_tei)
        ].copy()

    return filtered_tei, filtered_enrollments, filtered_events
