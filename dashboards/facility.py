import streamlit as st
import pandas as pd
import json
import logging
from io import BytesIO
import zipfile
import concurrent.futures
import requests
from utils.data_service import fetch_program_data_for_user
from utils.time_filter import get_date_range, assign_period
from utils.kpi_utils import compute_kpis, render_trend_chart, auto_text_color

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes

# ---------------- Cache Data Fetch ----------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user)
        return future.result(timeout=180)

# ---------------- Render Dashboard ----------------
def render():
    st.set_page_config(
        page_title="Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---------------- Session State ----------------
    if "refresh_trigger" not in st.session_state:
        st.session_state["refresh_trigger"] = False

    st_autorefresh = getattr(st, "autorefresh", None)
    if st_autorefresh:
        st.autorefresh(interval=1800*1000, key="auto_refresh")

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

    # ---------------- Refresh Button ----------------
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]

    # ---------------- Fetch Data ----------------
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

    # ---------------- Safety Patches ----------------
    if not enrollments_df.empty and "enrollmentDate" in enrollments_df.columns:
        enrollments_df["enrollmentDate"] = pd.to_datetime(enrollments_df["enrollmentDate"], errors="coerce")

    # unified events dataframe
    copied_events_df = events_df.copy()
    if not copied_events_df.empty:
        if "event_date" not in copied_events_df.columns and "eventDate" in copied_events_df.columns:
            copied_events_df["event_date"] = pd.to_datetime(copied_events_df["eventDate"], errors="coerce")
        elif "event_date" not in copied_events_df.columns:
            copied_events_df["event_date"] = pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

    # ---------------- Export Buttons ----------------
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)
    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        if st.button("📥 Raw JSON"):
            st.download_button(
                "Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{facility_name}_raw.json",
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
                file_name=f"{facility_name}_dataframes.zip",
                mime="application/zip"
            )

    st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - {facility_name}</div>', unsafe_allow_html=True)

    # ---------------- Check if any data exists ----------------
    if copied_events_df.empty:
        st.markdown('<div class="no-data-warning">⚠️ No data available for the selected period. KPIs and charts are hidden.</div>', unsafe_allow_html=True)
        return

    # ---------------- Filters ----------------
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        kpi_selection = st.selectbox(
            "📊 Select KPI to Visualize",
            [
                "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)",
                "Stillbirth Rate (per 1000 births)"
            ]
        )
        _df_for_dates = copied_events_df if "event_date" in copied_events_df.columns else pd.DataFrame()
        quick_range = st.selectbox(
            "📅 Time Period",
            ["Custom Range", "Today", "This Week", "Last Week", "This Month",
             "Last Month", "This Year", "Last Year"]
        )
        start_date, end_date = get_date_range(_df_for_dates, quick_range)

        KPI_AGGREGATION = {
            "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": ["Monthly", "Quarterly"],
            "Stillbirth Rate (per 1000 births)": ["Monthly"]
        }
        allowed_periods = KPI_AGGREGATION.get(kpi_selection, ["Monthly"])
        period_label = st.selectbox("⏰ Aggregation Level", allowed_periods)
        bg_color = st.color_picker("🎨 Chart Background", "#FFFFFF")
        st.markdown('</div>', unsafe_allow_html=True)

    text_color = auto_text_color(bg_color)

    # ---------------- Apply Time Filters ----------------
    if not copied_events_df.empty:
        copied_events_df = copied_events_df[
            (copied_events_df["event_date"].dt.date >= start_date) &
            (copied_events_df["event_date"].dt.date <= end_date)
        ]
        copied_events_df = assign_period(copied_events_df, "event_date", period_label)

    # ---------------- Compute KPIs ----------------
    kpis = compute_kpis(copied_events_df)
    if not isinstance(kpis, dict):
        st.error("Error computing KPI. Please check data.")
        return

    # ---------------- KPI Cards ----------------
    with col1:
        st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
        
        # Create a grid layout for KPI cards
        kpi_html = f"""
        <div class='kpi-grid'>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis.get("ippcar",0):.1f}%</div>
                <div class='kpi-name'>IPPCAR (Immediate Postpartum Contraceptive Acceptance Rate)</div>
                <div class='kpi-metrics'>
                    <span class='metric-label metric-fp'>Accepted FP: {kpis.get("fp_acceptance",0)}</span>
                    <span class='metric-label metric-total'>Total Deliveries: {kpis.get("total_deliveries",0)}</span>
                </div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{kpis.get("stillbirth_rate",0):.1f}</div>
                <div class='kpi-name'>Stillbirth Rate (per 1000 births)</div>
                <div class='kpi-metrics'>
                    <span class='metric-label metric-stillbirth'>Stillbirths: {kpis.get("stillbirths",0)}</span>
                    <span class='metric-label metric-total'>Total Births: {kpis.get("total_births",0)}</span>
                </div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

    # ---------------- KPI Trend Chart ----------------
    with col1:
        st.markdown(f'<div class="section-header">📈 {kpi_selection} Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # ---------------- Prepare Trend Data ----------------
        if kpi_selection == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)":
            group = copied_events_df.groupby("period", as_index=False).apply(
                lambda x: pd.Series({
                    "value": (
                        x[(x["dataElement_uid"]=="Q1p7CxWGUoi") & 
                          (x["value"].isin(["sn2MGial4TT","aB5By4ATx8M","TAxj9iLvWQ0",
                                            "FyCtuLALNpY","ejFYFZlmlwT"]))]["tei_id"].nunique()
                        / max(1, x[(x["dataElement_uid"]=="lphtwP2ViZU") & (x["value"].notna())]["tei_id"].nunique())
                    ) * 100
                })
            ).reset_index(drop=True)
            render_trend_chart(group, "period", "value", "IPPCAR (%)", bg_color, text_color)

        elif kpi_selection == "Stillbirth Rate (per 1000 births)":
            group = copied_events_df.groupby("period", as_index=False).apply(
                lambda x: pd.Series({"value": compute_kpis(x)["stillbirth_rate"]})
            ).reset_index(drop=True)
            render_trend_chart(group, "period", "value", "Stillbirth Rate (per 1000 births)", bg_color, text_color)

        # ---------------- Trend Symbol ----------------
        if not group.empty and len(group) > 1:
            last_value = group["value"].iloc[-1]
            prev_value = group["value"].iloc[-2]
            if last_value > prev_value:
                trend_symbol = "▲"
                trend_class = "trend-up"
            elif last_value < prev_value:
                trend_symbol = "▼"
                trend_class = "trend-down"
            else:
                trend_symbol = "–"
                trend_class = "trend-neutral"
            st.markdown(
                f'<p style="font-size:1.2rem; font-weight:600;">Latest Value: {last_value:.1f} <span class="{trend_class}">{trend_symbol}</span></p>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)