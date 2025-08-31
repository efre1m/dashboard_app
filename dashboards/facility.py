import streamlit as st
import pandas as pd
import json
import logging
from io import BytesIO
import zipfile

from utils.data_service import fetch_program_data_for_user
from utils.time_filter import get_date_range, assign_period
from utils.kpi_utils import (
    compute_kpis,
    render_trend_chart,
    auto_text_color,
    render_maternal_complications_chart
)

logging.basicConfig(level=logging.INFO)

def render():
    st.set_page_config(
        page_title="Maternal Health Dashboard", 
        page_icon="🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---------------- Import CSS ----------------
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # ---------------- User Info ----------------
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    facility_name = user.get("facility_name", "Unknown Facility")

    st.sidebar.markdown(f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🏢 Facility: {facility_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """, unsafe_allow_html=True)

    # ---------------- Fetch Data ----------------
    with st.spinner("fetching maternal data..."):
        dfs = fetch_program_data_for_user(user)

    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])

    # ---- SAFETY PATCHES ----
    if not enrollments_df.empty and "enrollmentDate" in enrollments_df.columns:
        enrollments_df["enrollmentDate"] = pd.to_datetime(enrollments_df["enrollmentDate"], errors="coerce")

    if not enrollments_df.empty and "status" not in enrollments_df.columns:
        st.warning("Enrollment data has no 'status' column. Treating all as 'UNKNOWN'.")
        enrollments_df = enrollments_df.copy()
        enrollments_df["status"] = "UNKNOWN"

    delivery_df = events_df.copy()
    if not delivery_df.empty:
        if "event_date" not in delivery_df.columns and "eventDate" in delivery_df.columns:
            delivery_df["event_date"] = pd.to_datetime(delivery_df["eventDate"], errors="coerce")
        elif "event_date" not in delivery_df.columns:
            delivery_df["event_date"] = pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

    # ---------------- Export Buttons ----------------
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
    
    col_exp1, col_exp2 = st.sidebar.columns(2)

    with col_exp1:
        if st.button("📥 Raw JSON", help="Download raw JSON data"):
            json_str = json.dumps(raw_json, indent=2)
            st.download_button(
                label="Download Raw JSON",
                data=json_str,
                file_name=f"{facility_name}_raw.json",
                mime="application/json"
            )

    with col_exp2:
        if st.button("📊 Export CSV", help="Export all data as CSV files"):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("tei.csv", tei_df.to_csv(index=False).encode("utf-8"))
                zf.writestr("enrollments.csv", enrollments_df.to_csv(index=False).encode("utf-8"))
                zf.writestr("events.csv", events_df.to_csv(index=False).encode("utf-8"))
            buffer.seek(0)
            st.download_button(
                label="Download All DataFrames (ZIP)",
                data=buffer,
                file_name=f"{facility_name}_dataframes.zip",
                mime="application/zip"
            )

    # ---------------- Main Content ----------------
    st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - {facility_name}</div>', unsafe_allow_html=True)
    
     # ---------------- GLOBAL EMPTY CHECK ----------------
    # If both dataframes are empty after filtering, show single warning and stop (no KPIs/charts)
    if (enrollments_df.empty or len(enrollments_df) == 0) and (delivery_df.empty or len(delivery_df) == 0):
        st.warning("⚠️ No data available for the selected period. KPIs, charts and tables are hidden.")
        return  # don't render KPIs or charts
    
    col1, col2 = st.columns([3, 1])

    # ---------------- Filters ----------------
    with col2:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-selector">⚙️ Dashboard Controls</div>', unsafe_allow_html=True)

        kpi_selection = st.selectbox(
            "📊 Select KPI to Visualize",
            ["Maternal Admissions", "Instrumental Delivery Rate", "Cesarean Section Rate", 
             "Maternal Complications", "Maternal Deaths"],
            help="Choose which metric to display in the trend chart"
        )

        _df_for_dates = enrollments_df if ("enrollmentDate" in enrollments_df.columns) else pd.DataFrame()

        quick_range = st.selectbox(
            "📅 Time Period",
            ["Custom Range", "Today", "This Week", "Last Week", "This Month", 
             "Last Month", "This Year", "Last Year"],
            help="Select a predefined time range or choose custom for specific dates"
        )
        
        start_date, end_date = get_date_range(_df_for_dates, quick_range)

        period_label = st.selectbox(
            "⏰ Aggregation Level", 
            ["Daily", "Monthly", "Quarterly", "Annual"],
            help="Choose how to group the data in time series charts"
        )
        
        bg_color = st.color_picker(
            "🎨 Chart Background", 
            "#FFFFFF",
            help="Select background color for the charts"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    text_color = auto_text_color(bg_color)

    # ---------------- Apply Time Filters ----------------
    if not enrollments_df.empty and "enrollmentDate" in enrollments_df.columns:
        enrollments_df = enrollments_df[
            (enrollments_df["enrollmentDate"].dt.date >= start_date) &
            (enrollments_df["enrollmentDate"].dt.date <= end_date)
        ]
        enrollments_df = assign_period(enrollments_df, "enrollmentDate", period_label)

    if not delivery_df.empty:
        delivery_df = delivery_df[
            (delivery_df["event_date"].dt.date >= start_date) &
            (delivery_df["event_date"].dt.date <= end_date)
        ]
        delivery_df = assign_period(delivery_df, "event_date", period_label)

    # ---------------- Compute KPIs ----------------
    kpis = compute_kpis(enrollments_df, delivery_df)

    # ensure kpis is a dict (defensive)
    if not isinstance(kpis, dict):
        st.error("Error computing KPIs. Please check data.")
        return

    # Use kpis value for maternal deaths (always numeric)
    maternal_deaths = kpis.get("maternal_deaths", 0)

    # ---------------- Compute Trend Symbol ----------------
    def compute_trend(current, previous):
        try:
            current = float(current)
        except Exception:
            current = 0.0
        try:
            previous = float(previous)
        except Exception:
            previous = 0.0

        if previous == 0:
            return "▶", "trend-neutral"
        if current > previous:
            return "▲", "trend-up"
        elif current < previous:
            return "▼", "trend-down"
        else:
            return "▶", "trend-neutral"

    # For simplicity, previous values set to 0 (could be extended later)
    trend_admissions, trend_admissions_class = compute_trend(kpis.get("total_admissions", 0), 0)
    trend_idr, trend_idr_class = compute_trend(kpis.get("idr", 0), 0)
    trend_csr, trend_csr_class = compute_trend(kpis.get("csr", 0), 0)
    trend_mc, trend_mc_class = compute_trend(kpis.get("maternal_complications_total", 0), 0)
    trend_md, trend_md_class = compute_trend(maternal_deaths, 0)

    # ---------------- KPI Cards ----------------
    with col1:
        st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
        kpi_html = f"""
        <div class='kpi-row'>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis.get("total_admissions", 0)} <span class='{trend_admissions_class}'>{trend_admissions}</span></div>
                <div class='kpi-name'>Maternal Admissions</div>
                <div style='margin-top:10px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap;'>
                    <span class='metric-label metric-active'>Active: {kpis.get("active_count", 0)}</span>
                    <span class='metric-label metric-completed'>Completed: {kpis.get("completed_count", 0)}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis.get("idr", 0):.1f}% <span class='{trend_idr_class}'>{trend_idr}</span></div>
                <div class='kpi-name'>Instrumental Delivery Rate</div>
                <div style='margin-top:10px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap;'>
                    <span class='metric-label metric-instrumental'>Instrumental: {kpis.get("instrumental_deliveries", 0)}</span>
                    <span class='metric-label metric-total'>Total Deliveries: {kpis.get("total_deliveries", 0)}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis.get("csr", 0):.1f}% <span class='{trend_csr_class}'>{trend_csr}</span></div>
                <div class='kpi-name'>Cesarean Section Rate</div>
                <div style='margin-top:10px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap;'>
                    <span class='metric-label metric-csection'>C-section: {kpis.get("csection_deliveries", 0)}</span>
                    <span class='metric-label metric-total'>Total Deliveries: {kpis.get("total_deliveries", 0)}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis.get("maternal_complications_total", 0)} <span class='{trend_mc_class}'>{trend_mc}</span></div>
                <div class='kpi-name'>Maternal Complications</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{maternal_deaths} <span class='{trend_md_class}'>{trend_md}</span></div>
                <div class='kpi-name'>Maternal Deaths</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # ---------------- KPI Graphs ----------------
    with col1:
        st.markdown(f'<div class="section-header">📈 {kpi_selection} Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        if kpi_selection == "Maternal Admissions":
            status_filter = st.selectbox(
                "Show Admissions by Status", 
                ["ACTIVE", "COMPLETED"], 
                key="maternal_status",
                help="Filter admissions by enrollment status"
            )
            filtered_df = enrollments_df[enrollments_df["status"] == status_filter] if "status" in enrollments_df.columns else pd.DataFrame()
            trend_group = (
                filtered_df.groupby("period")["tei_id"].nunique().reset_index(name="value")
                if not filtered_df.empty else pd.DataFrame()
            )
            render_trend_chart(trend_group, "period", "value", f"Maternal Admissions Trend ({status_filter})", bg_color, text_color, chart_type="line")

        elif kpi_selection == "Instrumental Delivery Rate":
            if not delivery_df.empty:
                idr_group = delivery_df.groupby("period").apply(
                    lambda x: pd.Series({
                        "value": (
                            x[(x.get("dataElementName") == "Instrumental Delivery") &
                              (x["value"].astype(str).str.lower() == "true")]["tei_id"].nunique()
                            / max(1, x[x.get("dataElementName") == "Total Deliveries"]["tei_id"].nunique()) * 100
                        )
                    })
                ).reset_index()
            else:
                idr_group = pd.DataFrame()
            render_trend_chart(idr_group, "period", "value", f"Instrumental Delivery Rate (%) by {period_label}", bg_color, text_color, chart_type="bar")

        elif kpi_selection == "Cesarean Section Rate":
            if not delivery_df.empty:
                csr_group = delivery_df.groupby("period").apply(
                    lambda x: pd.Series({
                        "value": (
                            x[(x.get("dataElementName") == "Delivery Type") & (x["value"] == "C-Section")]["tei_id"].nunique()
                            / max(1, x[x.get("dataElementName") == "Delivery Type"]["tei_id"].nunique()) * 100
                        )
                    })
                ).reset_index()
            else:
                csr_group = pd.DataFrame()
            render_trend_chart(csr_group, "period", "value", f"Cesarean Section Rate (%) by {period_label}", bg_color, text_color, chart_type="bar")

        elif kpi_selection == "Maternal Complications":
            render_maternal_complications_chart(delivery_df, "period", bg_color, text_color)

        elif kpi_selection == "Maternal Deaths":
            if not delivery_df.empty:
                md_group = delivery_df[
                    (delivery_df.get("dataElement_uid")=="TjQOcW6tm8k") & (delivery_df.get("value")=="4")
                ].groupby("period")["tei_id"].nunique().reset_index(name="value")
            else:
                md_group = pd.DataFrame()

            # If no maternal deaths rows exist for the filtered period, create a one-row DF so table shows 0
            if md_group.empty:
                md_group = pd.DataFrame({
                    "period": [f"{start_date} to {end_date}"],
                    "value": [0]
                })
            render_trend_chart(md_group, "period", "value", "Maternal Deaths Trend", bg_color, text_color, chart_type="line")
            
        st.markdown('</div>', unsafe_allow_html=True)
