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
        page_title="National Maternal Health Dashboard",
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
    country_name = user.get("country_name", "National Level")

    st.sidebar.markdown(f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🌍 Country: {country_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # Fetch DHIS2 data
    with st.spinner("Fetching national maternal data..."):
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

    # ---------------- Region Filter ----------------
    # Get unique regions from the events data
    regions = copied_events_df["orgUnit_name"].unique().tolist()
    regions.sort()
    
    # Create region mapping for UID lookup
    region_mapping = {}
    if not copied_events_df.empty:
        for _, row in copied_events_df.iterrows():
            region_mapping[row["orgUnit_name"]] = row["orgUnit"]
    
    # Multi-select region selector in sidebar
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🗺️ Select Regions</p>', 
        unsafe_allow_html=True
    )
    
    # Default to all regions
    default_regions = ["All Regions"]
    
    selected_regions = st.sidebar.multiselect(
        " ",
        ["All Regions"] + regions,
        default=default_regions,
        key="region_selector",
        label_visibility="collapsed"
    )
    
    # Handle "All Regions" selection logic
    if "All Regions" in selected_regions:
        if len(selected_regions) > 1:
            # If "All Regions" is selected with others, remove "All Regions"
            selected_regions = [f for f in selected_regions if f != "All Regions"]
        else:
            # Only "All Regions" is selected
            selected_regions = ["All Regions"]
    
    # Get the region UIDs for selected regions
    region_uids = None
    region_names = None
    if selected_regions != ["All Regions"]:
        region_uids = [region_mapping[region] for region in selected_regions if region in region_mapping]
        region_names = selected_regions

    # ---------------- View Mode Selection ----------------
    view_mode = "Normal Trend"
    if selected_regions != ["All Regions"] and len(selected_regions) > 1:
        view_mode = st.sidebar.radio(
            "📊 View Mode",
            ["Normal Trend", "Region Comparison"],
            index=0,
            help="Compare trends across multiple regions"
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
                file_name=f"{country_name}_raw.json",
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
                file_name=f"{country_name}_dataframes.zip",
                mime="application/zip"
            )

    # MAIN HEADING
    if selected_regions == ["All Regions"]:
        st.markdown(f'<div class="main-header">🌍 National Maternal Health Dashboard - {country_name}</div>', unsafe_allow_html=True)
    elif len(selected_regions) == 1:
        st.markdown(f'<div class="main-header">🌍 National Maternal Health Dashboard - {selected_regions[0]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Regions ({len(selected_regions)})</div>', unsafe_allow_html=True)
    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown('<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>', unsafe_allow_html=True)
        return

    # Decide the display name
    if selected_regions == ["All Regions"]:
        display_name = country_name
    elif len(selected_regions) == 1:
        display_name = selected_regions[0]
    else:
        display_name = f"Multiple Regions ({len(selected_regions)})"

    # Render KPI cards (trends are handled inside the component)
    render_kpi_cards(copied_events_df, region_uids, display_name)

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

    # Apply region filter if selected
    if region_uids:
        filtered_events = filtered_events[filtered_events["orgUnit"].isin(region_uids)]

    # Assign period AFTER filtering (so period aligns with the time window)
    filtered_events = assign_period(filtered_events, "event_date", period_label)

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        st.markdown('<div class="no-data-warning">⚠️ No data available for the selected period. Charts are hidden.</div>', unsafe_allow_html=True)
        return

    text_color = auto_text_color(bg_color)

    with col_chart:
        if view_mode == "Region Comparison" and len(selected_regions) > 1:
            st.markdown(f'<div class="section-header">📈 {kpi_selection} - Region Comparison</div>', unsafe_allow_html=True)
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
                facility_names=region_names,
                facility_uids=region_uids,
                numerator_name=kpi_config.get("numerator_name", "Numerator"),
                denominator_name=kpi_config.get("denominator_name", "Denominator")
            )
            
        else:
            st.markdown(f'<div class="section-header">📈 {kpi_selection} Trend</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Build aggregated trend data
            if kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": (
                            x[(x["dataElement_uid"]=="Q1p7CxWGUoi") &
                              (x["value"].isin(["sn2MGial4TT","aB5By4ATx8M","TAxj9iLvWQ0",
                                                "FyCtuLALNpY","ejFYFZlmlwT"]))]["tei_id"].nunique()
                            / max(1, x[(x["dataElement_uid"]=="lphtwP2ViZU") & (x["value"].notna())]["tei_id"].nunique())
                        ) * 100,
                        "FP Acceptances": x[(x["dataElement_uid"]=="Q1p7CxWGUoi") &
                                          (x["value"].isin(["sn2MGial4TT","aB5By4ATx8M","TAxj9iLvWQ0",
                                                            "FyCtuLALNpY","ejFYFZlmlwT"]))]["tei_id"].nunique(),
                        "Total Deliveries": x[(x["dataElement_uid"]=="lphtwP2ViZU") & (x["value"].notna())]["tei_id"].nunique()
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "IPPCAR (%)", bg_color, text_color, 
                                  region_names, "FP Acceptances", "Total Deliveries", region_uids)

            elif kpi_selection == "Stillbirth Rate (per 1000 births)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x)["stillbirth_rate"],
                        "Stillbirths": compute_kpis(x)["stillbirths"],
                        "Total Births": compute_kpis(x)["total_births"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "Stillbirth Rate (per 1000 births)", bg_color, text_color,
                                  region_names, "Stillbirths", "Total Births", region_uids)

            elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x)["pnc_coverage"],
                        "Early PNC (≤48 hrs)": compute_kpis(x)["early_pnc"],
                        "Total Deliveries": compute_kpis(x)["total_deliveries_pnc"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "Early PNC Coverage (%)", bg_color, text_color,
                                  region_names, "Early PNC (≤48 hrs)", "Total Deliveries", region_uids)

            elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x)["maternal_death_rate"],
                        "Maternal Deaths": compute_kpis(x)["maternal_deaths"],
                        "Live Births": compute_kpis(x)["live_births"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "Maternal Death Rate (per 100,000 births)", bg_color, text_color,
                                  region_names, "Maternal Deaths", "Live Births", region_uids)

            elif kpi_selection == "C-Section Rate (%)":
                group = filtered_events.groupby("period", as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x)["csection_rate"],
                        "C-Sections": compute_kpis(x)["csection_deliveries"],
                        "Total Deliveries": compute_kpis(x)["total_deliveries_cs"]
                    })
                ).reset_index(drop=True)
                render_trend_chart(group, "period", "value", "C-Section Rate (%)", bg_color, text_color,
                                  region_names, "C-Sections", "Total Deliveries", region_uids)

        st.markdown('</div>', unsafe_allow_html=True)