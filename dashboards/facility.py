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


def calculate_trend(df: pd.DataFrame, kpi_type: str):
    """
    Compute ▲ ▼ – by comparing the last two period values.
    Assumes df already has a 'period' column.
    """
    if df.empty or "period" not in df.columns:
        return "–", "trend-neutral"

    # Build a compact period-level dataframe with a single 'value' column
    if kpi_type == "ippcar":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": (
                    x[(x["dataElement_uid"] == "Q1p7CxWGUoi") &
                      (x["value"].isin([
                          "sn2MGial4TT", "aB5By4ATx8M", "TAxj9iLvWQ0",
                          "FyCtuLALNpY", "ejFYFZlmlwT"
                      ]))]["tei_id"].nunique()
                    / max(1, x[(x["dataElement_uid"] == "lphtwP2ViZU") & (x["value"].notna())]["tei_id"].nunique())
                ) * 100
            })
        ).reset_index(drop=True)

    elif kpi_type == "stillbirth":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["stillbirth_rate"]
            })
        ).reset_index(drop=True)

    elif kpi_type == "pnc":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["pnc_coverage"]
            })
        ).reset_index(drop=True)

    elif kpi_type == "maternal_death":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["maternal_death_rate"]
            })
        ).reset_index(drop=True)

    elif kpi_type == "csection":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["csection_rate"]
            })
        ).reset_index(drop=True)

    else:
        return "–", "trend-neutral"

    if len(group) > 1:
        last_value = group["value"].iloc[-1]
        prev_value = group["value"].iloc[-2]
        if pd.notna(last_value) and pd.notna(prev_value):
            if last_value > prev_value:
                return "▲", "trend-up"
            elif last_value < prev_value:
                return "▼", "trend-down"

    return "–", "trend-neutral"


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
    facility_name = user.get("facility_name", "Unknown facility")

    st.sidebar.markdown(f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🗺️ Facility: {facility_name}</div>
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

    # MAIN HEADING - Moved to the top
    st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - {facility_name}</div>', unsafe_allow_html=True)

    # ---------------- KPI CARDS - Moved immediately after heading ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown('<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>', unsafe_allow_html=True)
        return

    # Compute KPIs on the full dataset for the cards
    kpis = compute_kpis(copied_events_df)
    if not isinstance(kpis, dict):
        st.error("Error computing KPI. Please check data.")
        return

    # Calculate trends for the cards
    ippcar_trend, ippcar_trend_class = calculate_trend(copied_events_df, "ippcar")
    stillbirth_trend, stillbirth_trend_class = calculate_trend(copied_events_df, "stillbirth")
    pnc_trend, pnc_trend_class = calculate_trend(copied_events_df, "pnc")
    maternal_death_trend, maternal_death_trend_class = calculate_trend(copied_events_df, "maternal_death")
    csection_trend, csection_trend_class = calculate_trend(copied_events_df, "csection")

    # KPI Cards
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    kpi_html = f"""
    <div class='kpi-grid'>
        <div class='kpi-card'>
            <div class='kpi-value'>{kpis.get("ippcar",0):.1f}% <span class='{ippcar_trend_class}'>{ippcar_trend}</span></div>
            <div class='kpi-name'>IPPCAR (Immediate Postpartum Contraceptive Acceptance Rate)</div>
            <div class='kpi-metrics'>
                <span class='metric-label metric-fp'>Accepted FP: {kpis.get("fp_acceptance",0)}</span>
                <span class='metric-label metric-total'>Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>{kpis.get("stillbirth_rate",0):.1f} <span class='{stillbirth_trend_class}'>{stillbirth_trend}</span></div>
            <div class='kpi-name'>Stillbirth Rate (per 1000 births)</div>
            <div class='kpi-metrics'>
                <span class='metric-label metric-stillbirth'>Stillbirths: {kpis.get("stillbirths",0)}</span>
                <span class='metric-label metric-total'>Total Births: {kpis.get("total_births",0)}</span>
            </div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>{kpis.get("pnc_coverage",0):.1f}% <span class='{pnc_trend_class}'>{pnc_trend}</span></div>
            <div class='kpi-name'>Early PNC Coverage (within 48 hrs)</div>
            <div class='kpi-metrics'>
               <span class='metric-label metric-fp'>PNC ≤48 hrs: {kpis.get("early_pnc",0)}</span>
               <span class='metric-label metric-total'>Total Deliveries: {kpis.get("total_deliveries_pnc",0)}</span>
            </div>
        </div>
        <div class = 'kpi-card'>
            <div class='kpi-value'>{kpis.get("maternal_death_rate",0):.1f} <span class='{maternal_death_trend_class}'>{maternal_death_trend}</span></div>
            <div class='kpi-name'>Maternal Death Rate (per 100,000 births)</div>
            <div class='kpi-metrics'>
               <span class='metric-label metric-maternal-death'>Maternal Deaths: {kpis.get("maternal_deaths",0)}</span>
               <span class='metric-label metric-total'>Live Births: {kpis.get("live_births",0)}</span>
            </div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>{kpis.get("csection_rate",0):.1f}% <span class='{csection_trend_class}'>{csection_trend}</span></div>
            <div class='kpi-name'>C-Section Rate</div>
            <div class='kpi-metrics'>
               <span class='metric-label metric-csection'>C-Sections: {kpis.get("csection_deliveries",0)}</span>
               <span class='metric-label metric-total'>Total Deliveries: {kpis.get("total_deliveries_cs",0)}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ---------------- Controls & Time Filter (AFTER KPI cards) ----------------
    # Filter controls sit in the small column
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

    # ---------------- APPLY FILTER (this is the critical fix) ----------------
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

    # Assign period AFTER filtering (so period aligns with the time window)
    filtered_events = assign_period(filtered_events, "event_date", period_label)

    # ---------------- KPI Trend Charts (use the filtered df) ----------------
    if filtered_events.empty:
        st.markdown('<div class="no-data-warning">⚠️ No data available for the selected period. Charts are hidden.</div>', unsafe_allow_html=True)
        return

    text_color = auto_text_color(bg_color)

    with col_chart:
        st.markdown(f'<div class="section-header">📈 {kpi_selection} Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Build a tiny df with 'period' and the chosen KPI value
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
                              facility_name, "FP Acceptances", "Total Deliveries")

        elif kpi_selection == "Stillbirth Rate (per 1000 births)":
            group = filtered_events.groupby("period", as_index=False).apply(
                lambda x: pd.Series({
                    "value": compute_kpis(x)["stillbirth_rate"],
                    "Stillbirths": compute_kpis(x)["stillbirths"],
                    "Total Births": compute_kpis(x)["total_births"]
                })
            ).reset_index(drop=True)
            render_trend_chart(group, "period", "value", "Stillbirth Rate (per 1000 births)", bg_color, text_color,
                              facility_name, "Stillbirths", "Total Births")

        elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
            group = filtered_events.groupby("period", as_index=False).apply(
                lambda x: pd.Series({
                    "value": compute_kpis(x)["pnc_coverage"],
                    "Early PNC (≤48 hrs)": compute_kpis(x)["early_pnc"],
                    "Total Deliveries": compute_kpis(x)["total_deliveries_pnc"]
                })
            ).reset_index(drop=True)
            render_trend_chart(group, "period", "value", "Early PNC Coverage (%)", bg_color, text_color,
                              facility_name, "Early PNC (≤48 hrs)", "Total Deliveries")

        elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
            group = filtered_events.groupby("period", as_index=False).apply(
                lambda x: pd.Series({
                    "value": compute_kpis(x)["maternal_death_rate"],
                    "Maternal Deaths": compute_kpis(x)["maternal_deaths"],
                    "Live Births": compute_kpis(x)["live_births"]
                })
            ).reset_index(drop=True)
            render_trend_chart(group, "period", "value", "Maternal Death Rate (per 100,000 births)", bg_color, text_color,
                              facility_name, "Maternal Deaths", "Live Births")

        elif kpi_selection == "C-Section Rate (%)":
            group = filtered_events.groupby("period", as_index=False).apply(
                lambda x: pd.Series({
                    "value": compute_kpis(x)["csection_rate"],
                    "C-Sections": compute_kpis(x)["csection_deliveries"],
                    "Total Deliveries": compute_kpis(x)["total_deliveries_cs"]
                })
            ).reset_index(drop=True)
            render_trend_chart(group, "period", "value", "C-Section Rate (%)", bg_color, text_color,
                              facility_name, "C-Sections", "Total Deliveries")

        st.markdown('</div>', unsafe_allow_html=True)