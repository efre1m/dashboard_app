# # newborns_dashboard/facility_newborn.py
# import streamlit as st
# import pandas as pd
# import logging
# import concurrent.futures
# import requests
# from newborns_dashboard.kmc_coverage import compute_kmc_kpi
# from utils.data_service import fetch_program_data_for_user

# # IMPORT FROM NEONATAL DASH CO, NOT MATERNAL DASH_CO
# from newborns_dashboard.dash_co_newborn import (
#     normalize_event_dates,
#     normalize_enrollment_dates,
#     render_trend_chart_section,
#     get_text_color,
#     apply_simple_filters,
#     render_simple_filter_controls,
#     render_kpi_tab_navigation,
# )

# from utils.kpi_utils import clear_cache
# from utils.status import (
#     render_connection_status,
#     update_last_sync_time,
#     initialize_status_system,
# )

# # Initialize status system
# initialize_status_system()

# logging.basicConfig(level=logging.INFO)
# CACHE_TTL = 1800  # 30 minutes


# @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
# def fetch_cached_data(user, program_uid):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(fetch_program_data_for_user, user, program_uid)
#         return future.result(timeout=180)


# def count_unique_teis_filtered(tei_df, facility_uid, org_unit_column="tei_orgUnit"):
#     """Count unique TEIs from tei_df - SIMPLE VERSION FOR FACILITY LEVEL"""
#     if tei_df.empty:
#         return 0

#     # JUST COUNT WHATEVER TEIs ARE IN THE DATAFRAME - NO FILTERING
#     # The data should already be filtered for this facility when fetched

#     # Try common TEI identifier columns
#     if "tei" in tei_df.columns:
#         return tei_df["tei"].nunique()
#     elif "tei_id" in tei_df.columns:
#         return tei_df["tei_id"].nunique()
#     elif "trackedEntityInstance" in tei_df.columns:
#         return tei_df["trackedEntityInstance"].nunique()
#     elif "tracked_entity_instance" in tei_df.columns:
#         return tei_df["tracked_entity_instance"].nunique()
#     else:
#         # If no TEI column found, count rows
#         return len(tei_df)


# def get_earliest_date(df, date_column):
#     """Get the earliest date from a dataframe column"""
#     if df.empty or date_column not in df.columns:
#         return "N/A"

#     try:
#         earliest_date = df[date_column].min()
#         if pd.isna(earliest_date):
#             return "N/A"
#         return earliest_date.strftime("%Y-%m-%d")
#     except:
#         return "N/A"


# def calculate_newborn_indicators(newborn_events_df, facility_uid):
#     """Calculate newborn indicators using appropriate KPI functions"""
#     if newborn_events_df.empty:
#         return {
#             "total_admitted": 0,
#             "kmc_coverage_rate": 0.0,
#             "kmc_cases": 0,
#             "total_lbw": 0,
#         }

#     # Use compute_kmc_kpi for KMC coverage indicators
#     kmc_data = compute_kmc_kpi(newborn_events_df, [facility_uid])

#     kmc_coverage_rate = kmc_data.get("kmc_rate", 0.0)
#     kmc_cases = kmc_data.get("kmc_count", 0)
#     total_lbw = kmc_data.get("total_lbw", 0)

#     return {
#         "total_admitted": 0,  # Will be set from filtered TEI count
#         "kmc_coverage_rate": round(kmc_coverage_rate, 2),
#         "kmc_cases": kmc_cases,
#         "total_lbw": total_lbw,
#     }


# def get_location_display_name(facility_name):
#     """Get the display name for location - SIMPLIFIED FOR FACILITY LEVEL"""
#     return facility_name, "Facility"


# def render_newborn_dashboard(
#     user,
#     program_uid,
#     facility_name,
#     facility_uid,
# ):
#     """Render Newborn Care Form dashboard content for facility level"""

#     # Fetch DHIS2 data for Newborn program
#     with st.spinner(f"Fetching Newborn Care Data..."):
#         try:
#             dfs = fetch_cached_data(user, program_uid)
#             update_last_sync_time()
#         except concurrent.futures.TimeoutError:
#             st.error("⚠️ DHIS2 data could not be fetched within 3 minutes.")
#             return
#         except requests.RequestException as e:
#             st.error(f"⚠️ DHIS2 request failed: {e}")
#             return
#         except Exception as e:
#             st.error(f"⚠️ Unexpected error: {e}")
#             return

#     tei_df = dfs.get("tei", pd.DataFrame())
#     enrollments_df = dfs.get("enrollments", pd.DataFrame())
#     events_df = dfs.get("events", pd.DataFrame())
#     raw_json = dfs.get("raw_json", [])
#     program_info = dfs.get("program_info", {})

#     # Normalize dates using common functions
#     enrollments_df = normalize_enrollment_dates(enrollments_df)
#     events_df = normalize_event_dates(events_df)

#     # Filter data to only show this facility's data
#     if facility_uid and not events_df.empty:
#         events_df = events_df[events_df["orgUnit"] == facility_uid]

#     render_connection_status(events_df, user=user)

#     # MAIN HEADING for Newborn program
#     st.markdown(
#         f'<div class="main-header">👶 Newborn Care Form - {facility_name}</div>',
#         unsafe_allow_html=True,
#     )

#     # ---------------- Controls & Time Filter ----------------
#     col_chart, col_ctrl = st.columns([3, 1])
#     with col_ctrl:
#         st.markdown('<div class="filter-box">', unsafe_allow_html=True)

#         # Use simple filter controls
#         filters = render_simple_filter_controls(
#             events_df, container=col_ctrl, context="facility_newborn"
#         )

#         st.markdown("</div>", unsafe_allow_html=True)

#     # Apply simple filters with single facility UID
#     filtered_events = apply_simple_filters(events_df, filters, facility_uid)

#     # Store for gauge charts
#     st.session_state["filtered_events"] = filtered_events.copy()

#     # Get variables from filters for later use
#     bg_color = filters["bg_color"]
#     text_color = filters["text_color"]

#     # ---------------- KPI Trend Charts ----------------
#     if filtered_events.empty:
#         st.markdown(
#             f'<div class="no-data-warning">⚠️ No Newborn Care Data available for the selected period. Charts are hidden.</div>',
#             unsafe_allow_html=True,
#         )
#         return

#     text_color = get_text_color(bg_color)

#     with col_chart:
#         # Use KPI tab navigation FROM NEONATAL
#         selected_kpi = render_kpi_tab_navigation()

#         st.markdown(
#             f'<div class="section-header">📈 {selected_kpi} Trend - Newborn Care Form</div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown('<div class="chart-container">', unsafe_allow_html=True)

#         # Use render_trend_chart_section FROM NEONATAL with single facility
#         render_trend_chart_section(
#             selected_kpi,
#             filtered_events,
#             facility_uid,  # Single facility UID
#             facility_name,  # Single facility name
#             bg_color,
#             text_color,
#         )

#         st.markdown("</div>", unsafe_allow_html=True)
