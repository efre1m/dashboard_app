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
):
    """Render Newborn Care Form dashboard content"""
    # Handle None values for selected_facilities (facility-level users)
    if selected_facilities is None:
        # For facility-level users, use facility_name from user object
        facility_name = user.get("facility_name", "Unknown Facility")
        header_name = facility_name
        selected_facilities = [facility_name]  # Convert to list for consistency
    elif selected_facilities == ["All Facilities"]:
        header_name = region_name
    elif len(selected_facilities) == 1:
        header_name = selected_facilities[0]
    else:
        header_name = f"Multiple Facilities ({len(selected_facilities)})"

    st.markdown(
        f'<div class="main-header">👶 Newborn Care Dashboard - {header_name}</div>',
        unsafe_allow_html=True,
    )

    # Fetch Data
    with st.spinner("Fetching Newborn Care Form Data..."):
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

    # Extract DataFrames
    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())

    # 🔹 Tag TEI DataFrame to indicate newborn dataset
    tei_df["_dataset_type"] = "newborn"

    # Filter for selected facilities if applicable
    # For facility-level users, facility_uids might be a single string instead of list
    if facility_uids and not events_df.empty:
        # Ensure facility_uids is a list
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        events_df = events_df[events_df["orgUnit"].isin(facility_uids)]

    # Normalize dates
    enrollments_df = normalize_enrollment_dates(enrollments_df)
    events_df = normalize_event_dates(events_df)

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        # Use simple filter controls (reuse maternal dashboard filters)
        filters = render_simple_filter_controls(
            events_df, container=col_ctrl, context="newborn_standalone"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Apply simple filters
    filtered_events = apply_simple_filters(events_df, filters, facility_uids)

    # Store filtered events in session state
    st.session_state["filtered_events"] = filtered_events.copy()

    # Get variables from filters for future use
    bg_color = filters["bg_color"]
    text_color = filters["text_color"]

    # ---------------- Display Summary ----------------
    st.success("✅ Successfully fetched Newborn Care Form data!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tracked Entities (Newborn)", len(tei_df))
    col2.metric("Enrollments", len(enrollments_df))
    col3.metric("Events", len(events_df))

    st.info("📊 Data loaded. KPIs and charts will be added here in the future.")
