import streamlit as st
import pandas as pd
import json
import logging
from io import BytesIO
import zipfile
import concurrent.futures
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.time_filter import get_date_range, assign_period
from utils.kpi_utils import compute_kpis, render_trend_chart, auto_text_color, render_facility_comparison_chart
from utils.queries import get_facilities_for_user, get_facility_mapping_for_user

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes


# ---------------- Cache Wrapper ----------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user)
        return future.result(timeout=180)


def _normalize_event_dates(df: pd.DataFrame) -> pd.DataFrame:
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


def _normalize_enrollment_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure enrollmentDate is datetime from '7/25/2025' format."""
    if df.empty or "enrollmentDate" not in df.columns:
        return df
    df = df.copy()
    df["enrollmentDate"] = pd.to_datetime(df["enrollmentDate"], format="%m/%d/%Y", errors="coerce")
    return df


# ---------------- Page Rendering ----------------
def render():
    st.set_page_config(
        page_title="Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "refresh_trigger" not in st.session_state:
        st.session_state["refresh_trigger"] = False

    # Load CSS if available
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Sidebar user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    region_name = user.get("region_name", "Unknown Region")

    st.sidebar.markdown(f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🗺️ Region: {region_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # Fetch DHIS2 data
    with st.spinner("Fetching maternal data..."):
        try:
            dfs = fetch_cached_data(user)
        except concurrent.futures.TimeoutError:
            st.error("⚠️ DHIS2 data could not be fetched within 3 minutes.")
            return
        except requests.RequestException as e:
            st.error(f"⚠️ DHIS2 request failed: {e}")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            return

    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])

    # Normalize dates
    enrollments_df = _normalize_enrollment_dates(enrollments_df)
    copied_events_df = _normalize_event_dates(events_df)

    # ---------------- Facility Filter ----------------
    # Get facilities from database using queries.py
    db_facilities = get_facilities_for_user(user)
    facilities = [facility[0] for facility in db_facilities]  # Extract facility names
    
    # Create facility mapping for UID lookup (from database)
    facility_mapping = get_facility_mapping_for_user(user)
    
    # Multi-select facility selector in sidebar
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Select Facilities</p>', 
        unsafe_allow_html=True
    )
    
    # Default to all facilities
    default_facilities = ["All Facilities"]
    
    selected_facilities = st.sidebar.multiselect(
        " ",
        ["All Facilities"] + facilities,
        default=default_facilities,
        key="facility_selector",
        label_visibility="collapsed"
    )
    
    # 👇 Dynamic count below the dropdown
    total_facilities = len(facilities)
    if selected_facilities == ["All Facilities"]:
        display_text = f"Selected: All ({total_facilities})"
    else:
        display_text = f"Selected: {len(selected_facilities)} / {total_facilities}"

    st.sidebar.markdown(
        f"<p style='color: white; font-size: 13px; margin-top: -10px;'>{display_text}</p>",
        unsafe_allow_html=True
    )
    
    # Handle "All Facilities" selection logic
    if "All Facilities" in selected_facilities:
        if len(selected_facilities) > 1:
            # If "All Facilities" is selected with others, remove "All Facilities"
            selected_facilities = [f for f in selected_facilities if f != "All Facilities"]
        else:
            # Only "All Facilities" is selected
            selected_facilities = ["All Facilities"]
    
    # Get the facility UIDs for selected facilities (from database mapping)
    facility_uids = None
    facility_names = None
    if selected_facilities != ["All Facilities"]:
        facility_uids = [facility_mapping[facility] for facility in selected_facilities if facility in facility_mapping]
        facility_names = selected_facilities

    # ---------------- View Mode Selection ----------------
    view_mode = "Normal Trend"
    if selected_facilities != ["All Facilities"] and len(selected_facilities) > 1:
        view_mode = st.sidebar.radio(
            "📊 View Mode",
            ["Normal Trend", "Facility Comparison"],
            index=0,
            help="Compare trends across multiple facilities"
        )

    # ---------------- Export Buttons ----------------
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)

    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        if st.button("📥 Raw JSON"):
            st.download_button(
                "Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{region_name}_raw.json",
                mime="application/json"
            )
    with col_exp2:
        if st.button("📊 Export CSV"):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("tei.csv", tei_df.to_csv(index=False).encode("utf-8"))
                zf.writestr("enrollments.csv", enrollments_df.to_csv(index=False).encode("utf-8"))
                zf.writestr("events.csv", copied_events_df.to_csv(index=False).encode("utf-8"))
            buffer.seek(0)
            st.download_button(
                "Download All DataFrames (ZIP)",
                data=buffer,
                file_name=f"{region_name}_dataframes.zip",
                mime="application/zip"
            )

    # MAIN HEADING
    if selected_facilities == ["All Facilities"]:
        st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - {region_name}</div>', unsafe_allow_html=True)
    elif len(selected_facilities) == 1:
        st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - {selected_facilities[0]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - Multiple Facilities ({len(selected_facilities)})</div>', unsafe_allow_html=True)
    
    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown('<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>', unsafe_allow_html=True)
        return

    # Determine display name based on selection
    if selected_facilities == ["All Facilities"]:
        display_name = region_name
    elif len(selected_facilities) == 1:
        display_name = selected_facilities[0]
    else:
        display_name = f"Multiple Facilities ({len(selected_facilities)})"

    # Use the KPI card component (handles KPIs + trends internally)
    render_kpi_cards(copied_events_df, facility_uids, display_name)

    # ---------------- Controls & Time Filter ----------------
    col_chart, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        kpi_selection = st.selectbox(
            "📊 Select KPI to Visualize",
            [
                "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
                "Stillbirth Rate (per 1000 births)",
                "Early Postnatal Care (PNC) Coverage (%)",
                "Institutional Maternal Death Rate (per 100,000 births)",
                "C-Section Rate (%)"
            ]
        )

        # Build a minimal df for date range defaults
        _df_for_dates = copied_events_df[["event_date"]] if "event_date" in copied_events_df.columns else pd.DataFrame()

        quick_range = st.selectbox(
            "📅 Time Period",
            ["Custom Range", "Today", "This Week", "Last Week", "This Month",
             "Last Month", "This Year", "Last Year"]
        )

        # Use your existing helper (returns Python date objects)
        start_date, end_date = get_date_range(_df_for_dates, quick_range)

        KPI_AGGREGATION = {
            "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": ["Monthly", "Quarterly"],
            "Stillbirth Rate (per 1000 births)": ["Monthly"],
            "Early Postnatal Care (PNC) Coverage (%)": ["Monthly", "Quarterly"],
            "Institutional Maternal Death Rate (per 100,000 births)": ["Monthly", "Quarterly"],
            "C-Section Rate (%)": ["Monthly"]
        }
        allowed_periods = KPI_AGGREGATION.get(kpi_selection, ["Monthly"])
        period_label = st.selectbox("⏰ Aggregation Level", allowed_periods)

        bg_color = st.color_picker("🎨 Chart Background", "#FFFFFF")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- APPLY FILTER ----------------
    # Convert date objects to datetimes for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    # Filter events and enrollments by selected range
    filtered_events = copied_events_df[
        (copied_events_df["event_date"] >= start_datetime) &
        (copied_events_df["event_date"] <= end_datetime)
    ].copy()

    filtered_enrollments = enrollments_df.copy()
    if not filtered_enrollments.empty and "enrollmentDate" in filtered_enrollments.columns:
        filtered_enrollments = filtered_enrollments[
            (filtered_enrollments["enrollmentDate"] >= start_datetime) &
            (filtered_enrollments["enrollmentDate"] <= end_datetime)
        ]

    # Apply facility filter if selected (using dhis2_uid from database)
    if facility_uids:
        filtered_events = filtered_events[filtered_events["orgUnit"].isin(facility_uids)]

    # Assign period AFTER filtering (so period aligns with the time window)
    filtered_events = assign_period(filtered_events, "event_date", period_label)

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        st.markdown('<div class="no-data-warning">⚠️ No data available for the selected period. Charts are hidden.</div>', unsafe_allow_html=True)
        return

    text_color = auto_text_color(bg_color)

    with col_chart:
        if view_mode == "Facility Comparison" and len(selected_facilities) > 1:
            st.markdown(f'<div class="section-header">📈 {kpi_selection} - Facility Comparison</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Map KPI selection to the appropriate parameters for render_facility_comparison_chart
            kpi_mapping = {
                "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
                    "title": "IPPCAR (%)",
                    "numerator_name": "FP Acceptances",
                    "denominator_name": "Total Deliveries"
                },
                "Stillbirth Rate (per 1000 births)": {
                    "title": "Stillbirth Rate (per 1000 births)",
                    "numerator_name": "Stillbirths",
                    "denominator_name": "Total Births"
                },
                "Early Postnatal Care (PNC) Coverage (%)": {
                    "title": "Early PNC Coverage (%)",
                    "numerator_name": "Early PNC (≤48 hrs)",
                    "denominator_name": "Total Deliveries"
                },
                "Institutional Maternal Death Rate (per 100,000 births)": {
                    "title": "Maternal Death Rate (per 100,000 births)",
                    "numerator_name": "Maternal Deaths",
                    "denominator_name": "Live Births"
                },
                "C-Section Rate (%)": {
                    "title": "C-Section Rate (%)",
                    "numerator_name": "C-Sections",
                    "denominator_name": "Total Deliveries"
                }
            }
            
            kpi_config = kpi_mapping.get(kpi_selection, {})
            
            # Use the imported render_facility_comparison_chart function
            render_facility_comparison_chart(
                df=filtered_events,
                period_col="period",
                value_col="value",
                title=kpi_config.get("title", kpi_selection),
                bg_color=bg_color,
                text_color=text_color,
                facility_names=facility_names,
                facility_uids=facility_uids,
                numerator_name=kpi_config.get("numerator_name", "Numerator"),
                denominator_name=kpi_config.get("denominator_name", "Denominator")
            )
            
        else:
            st.markdown(f'<div class="section-header">📈 {kpi_selection} Trend</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Build aggregated trend data using standardized compute_kpis function
            if kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, facility_uids)["ippcar"],
                        "FP Acceptances": compute_kpis(x, facility_uids)["fp_acceptance"],
                        "Total Deliveries": compute_kpis(x, facility_uids)["total_deliveries"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "IPPCAR (%)", bg_color, text_color, 
                                  facility_names, "FP Acceptances", "Total Deliveries", facility_uids)

            elif kpi_selection == "Stillbirth Rate (per 1000 births)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, facility_uids)["stillbirth_rate"],
                        "Stillbirths": compute_kpis(x, facility_uids)["stillbirths"],
                        "Total Births": compute_kpis(x, facility_uids)["total_births"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "Stillbirth Rate (per 1000 births)", bg_color, text_color,
                                  facility_names, "Stillbirths", "Total Births", facility_uids)

            elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, facility_uids)["pnc_coverage"],
                        "Early PNC (≤48 hrs)": compute_kpis(x, facility_uids)["early_pnc"],
                        "Total Deliveries": compute_kpis(x, facility_uids)["total_deliveries_pnc"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "Early PNC Coverage (%)", bg_color, text_color,
                                  facility_names, "Early PNC (≤48 hrs)", "Total Deliveries", facility_uids)

            elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, facility_uids)["maternal_death_rate"],
                        "Maternal Deaths": compute_kpis(x, facility_uids)["maternal_deaths"],
                        "Live Births": compute_kpis(x, facility_uids)["live_births"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "Maternal Death Rate (per 100,000 births)", bg_color, text_color,
                                  facility_names, "Maternal Deaths", "Live Births", facility_uids)

            elif kpi_selection == "C-Section Rate (%)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, facility_uids)["csection_rate"],
                        "C-Sections": compute_kpis(x, facility_uids)["csection_deliveries"],
                        "Total Deliveries": compute_kpis(x, facility_uids)["total_deliveries_cs"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "C-Section Rate (%)", bg_color, text_color,
                                  facility_names, "C-Sections", "Total Deliveries", facility_uids)

        st.markdown('</div>', unsafe_allow_html=True)