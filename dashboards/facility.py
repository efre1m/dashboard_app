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
    st.set_page_config(page_title="Facility Dashboard", page_icon="🏥", layout="wide")

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
    with st.spinner("Fetching complete hierarchical DHIS2 data..."):
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
    st.sidebar.subheader("Export Data")
    col_exp1, col_exp2 = st.sidebar.columns(2)

    with col_exp1:
        if st.button("Raw JSON"):
            json_str = json.dumps(raw_json, indent=2)
            st.download_button(
                label="Download Raw JSON",
                data=json_str,
                file_name=f"{facility_name}_raw.json",
                mime="application/json"
            )

    with col_exp2:
        if st.button("Export CSV"):
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

    # ---------------- Layout ----------------
    col1, col2 = st.columns([3, 1])

    # ---------------- Filters ----------------
    with col2:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-selector">Select KPI</div>', unsafe_allow_html=True)

        kpi_selection = st.selectbox(
            "KPI",
            ["Maternal Admissions", "Instrumental Delivery Rate", "Cesarean Section Rate", "Maternal Complications", "Maternal Deaths"]
        )

        _df_for_dates = enrollments_df if ("enrollmentDate" in enrollments_df.columns) else pd.DataFrame()

        quick_range = st.selectbox(
            "Quick Period",
            ["Custom Range","Today","This Week","Last Week","This Month","Last Month","This Year","Last Year"]
        )
        start_date, end_date = get_date_range(_df_for_dates, quick_range)

        period_label = st.selectbox("Time Aggregation", ["Daily", "Monthly", "Quarterly", "Annual"])
        bg_color = st.color_picker("Graph Background Color", "#C2C2CB")
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

    # ---------------- Compute Maternal Deaths KPI ----------------
    if not delivery_df.empty:
        maternal_deaths = delivery_df[
            (delivery_df["dataElement_uid"]=="TjQOcW6tm8k") & (delivery_df["value"]=="4")
        ]["tei_id"].nunique()
    else:
        maternal_deaths = 0

    # ---------------- Compute Trend Symbol ----------------
    def compute_trend(current, previous):
        if previous is None or previous == 0:
            return "▶"
        if current > previous:
            return "▲"
        elif current < previous:
            return "▼"
        else:
            return "▶"

    # For simplicity, previous values can be extended to last period
    trend_admissions = compute_trend(kpis["total_admissions"], 0)
    trend_idr = compute_trend(kpis["idr"], 0)
    trend_csr = compute_trend(kpis["csr"], 0)
    trend_mc = compute_trend(kpis["maternal_complications_total"], 0)
    trend_md = compute_trend(maternal_deaths, 0)

    # ---------------- KPI Cards ----------------
    with col1:
        st.subheader(f"📊 Facility KPIs - {facility_name}")
        kpi_html = f"""
        <div class='kpi-row'>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis["total_admissions"]} {trend_admissions}</div>
                <div class='kpi-name'>Maternal Admissions</div>
                <div style='margin-top:5px; display:flex; justify-content:center; gap:10px;'>
                    <span style='color:green'>● Active: {kpis["active_count"]}</span>
                    <span style='color:blue'>● Completed: {kpis["completed_count"]}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis["idr"]:.1f}% {trend_idr}</div>
                <div class='kpi-name'>Instrumental Delivery Rate</div>
                <div style='margin-top:5px; display:flex; justify-content:center; gap:10px;'>
                    <span style='color:purple'>● Instrumental: {kpis["instrumental_deliveries"]}</span>
                    <span style='color:orange'>● Total Deliveries: {kpis["total_deliveries"]}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis["csr"]:.1f}% {trend_csr}</div>
                <div class='kpi-name'>Cesarean Section Rate</div>
                <div style='margin-top:5px; display:flex; justify-content:center; gap:10px;'>
                    <span style='color:red'>● C-section: {kpis["csection_deliveries"]}</span>
                    <span style='color:orange'>● Total Deliveries: {kpis["total_deliveries"]}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis["maternal_complications_total"]} {trend_mc}</div>
                <div class='kpi-name'>Maternal Complications</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{maternal_deaths} {trend_md}</div>
                <div class='kpi-name'>Maternal Deaths</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # ---------------- KPI Graphs ----------------
    with col1:
        st.subheader(f"📈 {kpi_selection} Trend")

        if kpi_selection == "Maternal Admissions":
            status_filter = st.selectbox("Show Admissions by Status", ["ACTIVE","COMPLETED"], key="maternal_status")
            filtered_df = enrollments_df[enrollments_df["status"] == status_filter] if "status" in enrollments_df.columns else pd.DataFrame()
            trend_group = (
                filtered_df.groupby("period")["tei_id"].nunique().reset_index(name="value")
                if not filtered_df.empty else pd.DataFrame()
            )
            render_trend_chart(trend_group, "period", "value", f"Maternal Admissions Trend ({status_filter})", bg_color, text_color, chart_type="line")

        elif kpi_selection == "Instrumental Delivery Rate":
            idr_group = delivery_df.groupby("period").apply(
                lambda x: pd.Series({
                    "value": (
                        x[(x["dataElementName"] == "Instrumental Delivery") &
                          (x["value"].astype(str).str.lower() == "true")]["tei_id"].nunique()
                        / max(1, x[x["dataElementName"] == "Total Deliveries"]["tei_id"].nunique()) * 100
                    )
                })
            ).reset_index() if not delivery_df.empty else pd.DataFrame()
            render_trend_chart(idr_group, "period", "value", f"Instrumental Delivery Rate (%) by {period_label}", bg_color, text_color, chart_type="bar")

        elif kpi_selection == "Cesarean Section Rate":
            csr_group = delivery_df.groupby("period").apply(
                lambda x: pd.Series({
                    "value": (
                        x[(x["dataElementName"] == "Delivery Type") & (x["value"] == "C-Section")]["tei_id"].nunique()
                        / max(1, x[x["dataElementName"] == "Delivery Type"]["tei_id"].nunique()) * 100
                    )
                })
            ).reset_index() if not delivery_df.empty else pd.DataFrame()
            render_trend_chart(csr_group, "period", "value", f"Cesarean Section Rate (%) by {period_label}", bg_color, text_color, chart_type="bar")

        elif kpi_selection == "Maternal Complications":
            render_maternal_complications_chart(delivery_df, "period", bg_color, text_color)

        elif kpi_selection == "Maternal Deaths":
            md_group = delivery_df[
                (delivery_df["dataElement_uid"]=="TjQOcW6tm8k") & (delivery_df["value"]=="4")
            ].groupby("period")["tei_id"].nunique().reset_index(name="value") if not delivery_df.empty else pd.DataFrame()
            render_trend_chart(md_group, "period", "value", "Maternal Deaths Trend", bg_color, text_color, chart_type="line")
