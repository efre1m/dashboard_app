import streamlit as st
import pandas as pd
import json
import logging
from io import BytesIO
import zipfile
import concurrent.futures
import requests

# Add this import near the other imports
from components.kpi_card import render_kpi_cards
from utils.data_service import fetch_program_data_for_user
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.kpi_utils import compute_kpis, render_trend_chart, auto_text_color
from utils.kpi_pph import (
    compute_pph_kpi,
    render_pph_trend_chart,
    render_obstetric_condition_pie_chart,
)

from utils.kpi_uterotonic import (
    compute_uterotonic_kpi,
    render_uterotonic_trend_chart,
    render_uterotonic_type_pie_chart,
)
from utils.kpi_arv import (
    compute_arv_kpi,
    render_arv_trend_chart,
)

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
    df["enrollmentDate"] = pd.to_datetime(
        df["enrollmentDate"], format="%m/%d/%Y", errors="coerce"
    )
    return df


# ---------------- Page Rendering ----------------
def render():
    st.set_page_config(
        page_title="Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
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
    facility_uid = user.get("facility_uid")  # Get facility UID from user session

    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🗺️ Facility: {facility_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

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

    # Filter data to only show this facility's data
    if facility_uid and not copied_events_df.empty:
        copied_events_df = copied_events_df[copied_events_df["orgUnit"] == facility_uid]

    # ---------------- Export Buttons ----------------
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="section-header">Export Data</div>', unsafe_allow_html=True
    )

    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        if st.button("📥 Raw JSON"):
            st.download_button(
                "Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{facility_name}_raw.json",
                mime="application/json",
            )
    with col_exp2:
        if st.button("📊 Export CSV"):
            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("tei.csv", tei_df.to_csv(index=False).encode("utf-8"))
                zf.writestr(
                    "enrollments.csv",
                    enrollments_df.to_csv(index=False).encode("utf-8"),
                )
                zf.writestr(
                    "events.csv", copied_events_df.to_csv(index=False).encode("utf-8")
                )
            buffer.seek(0)
            st.download_button(
                "Download All DataFrames (ZIP)",
                data=buffer,
                file_name=f"{facility_name}_dataframes.zip",
                mime="application/zip",
            )

    # MAIN HEADING
    st.markdown(
        f'<div class="main-header">🏥 Maternal Health Dashboard - {facility_name}</div>',
        unsafe_allow_html=True,
    )

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # Pass user_id into KPI card renderer so it can save/load previous values
    user_id = str(
        user.get("id", username)
    )  # Prefer numeric ID if available, fallback to username
    render_kpi_cards(
        copied_events_df,
        [facility_uid] if facility_uid else None,
        facility_name,
        user_id=user_id,
    )

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
                "C-Section Rate (%)",
                "Postpartum Hemorrhage (PPH) Rate (%)",
                "Delivered women who received uterotonic (%)",
                "ARV Prophylaxis Rate (%)",
            ],
        )

        # Build a minimal df for date range defaults
        _df_for_dates = (
            copied_events_df[["event_date"]]
            if "event_date" in copied_events_df.columns
            else pd.DataFrame()
        )

        quick_range = st.selectbox(
            "📅 Time Period",
            [
                "Custom Range",
                "Today",
                "This Week",
                "Last Week",
                "This Month",
                "Last Month",
                "This Year",
                "Last Year",
            ],
        )

        # Use your existing helper (returns Python date objects)
        start_date, end_date = get_date_range(_df_for_dates, quick_range)

        # Get available aggregation levels based on date range
        available_aggregations = get_available_aggregations(start_date, end_date)

        # If current period selection is not available, default to the widest available
        if (
            "period_label" not in st.session_state
            or st.session_state.period_label not in available_aggregations
        ):
            st.session_state.period_label = available_aggregations[
                -1
            ]  # Widest available

        period_label = st.selectbox(
            "⏰ Aggregation Level",
            available_aggregations,
            index=available_aggregations.index(st.session_state.period_label),
            key="period_selectbox",
        )

        # Update session state with current selection
        st.session_state.period_label = period_label

        bg_color = st.color_picker("🎨 Chart Background", "#FFFFFF")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- APPLY FILTER ----------------
    # Convert date objects to datetimes for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    # Filter events by selected range
    filtered_events = copied_events_df[
        (copied_events_df["event_date"] >= start_datetime)
        & (copied_events_df["event_date"] <= end_datetime)
    ].copy()

    # STORE FILTERED EVENTS FOR GAUGE CHART - ADD THIS LINE
    st.session_state["filtered_events"] = filtered_events.copy()

    # Assign period AFTER filtering (so period aligns with the time window)
    filtered_events = assign_period(filtered_events, "event_date", period_label)

    # ---------------- KPI Trend Charts ----------------
    if filtered_events.empty:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available for the selected period. Charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    text_color = auto_text_color(bg_color)

    with col_chart:
        st.markdown(
            f'<div class="section-header">📈 {kpi_selection} Trend</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Build aggregated trend data using standardized compute_kpis function
        if (
            kpi_selection
            == "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)"
        ):
            group = (
                filtered_events.groupby(["period", "period_display"], as_index=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "value": compute_kpis(x, facility_uid)["ippcar"],
                            "FP Acceptances": compute_kpis(x, facility_uid)[
                                "fp_acceptance"
                            ],
                            "Total Deliveries": compute_kpis(x, facility_uid)[
                                "total_deliveries"
                            ],
                        }
                    )
                )
                .reset_index(drop=True)
            )
            render_trend_chart(
                group,
                "period_display",
                "value",
                "IPPCAR (%)",
                bg_color,
                text_color,
                facility_name,
                "FP Acceptances",
                "Total Deliveries",
                facility_uid,
            )
        elif kpi_selection == "Stillbirth Rate (per 1000 births)":
            group = (
                filtered_events.groupby(["period", "period_display"], as_index=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "value": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["stillbirth_rate"],
                            "Stillbirths": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["stillbirths"],
                            "Total Births": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["total_births"],
                        }
                    )
                )
                .reset_index(drop=True)
            )
            render_trend_chart(
                group,
                "period_display",
                "value",
                "Stillbirth Rate (per 1000 births)",
                bg_color,
                text_color,
                [facility_name],
                "Stillbirths",
                "Total Births",
                [facility_uid] if facility_uid else None,
            )

        elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
            group = (
                filtered_events.groupby(["period", "period_display"], as_index=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "value": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["pnc_coverage"],
                            "Early PNC (≤48 hrs)": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["early_pnc"],
                            "Total Deliveries": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["total_deliveries_pnc"],
                        }
                    )
                )
                .reset_index(drop=True)
            )
            render_trend_chart(
                group,
                "period_display",
                "value",
                "Early PNC Coverage (%)",
                bg_color,
                text_color,
                [facility_name],
                "Early PNC (≤48 hrs)",
                "Total Deliveries",
                [facility_uid] if facility_uid else None,
            )

        elif kpi_selection == "Institutional Maternal Death Rate (per 100,000 births)":
            group = (
                filtered_events.groupby(["period", "period_display"], as_index=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "value": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["maternal_death_rate"],
                            "Maternal Deaths": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["maternal_deaths"],
                            "Live Births": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["live_births"],
                        }
                    )
                )
                .reset_index(drop=True)
            )
            render_trend_chart(
                group,
                "period_display",
                "value",
                "Maternal Death Rate (per 100,000 births)",
                bg_color,
                text_color,
                [facility_name],
                "Maternal Deaths",
                "Live Births",
                [facility_uid] if facility_uid else None,
            )

        elif kpi_selection == "C-Section Rate (%)":
            group = (
                filtered_events.groupby(["period", "period_display"], as_index=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "value": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["csection_rate"],
                            "C-Sections": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["csection_deliveries"],
                            "Total Deliverings": compute_kpis(
                                x, [facility_uid] if facility_uid else None
                            )["total_deliveries_cs"],
                        }
                    )
                )
                .reset_index(drop=True)
            )
            render_trend_chart(
                group,
                "period_display",
                "value",
                "C-Section Rate (%)",
                bg_color,
                text_color,
                [facility_name],
                "C-Sections",
                "Total Deliveries",
                [facility_uid] if facility_uid else None,
            )

        elif kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            # Use the same structure as other indicators
            group = (
                filtered_events.groupby(["period", "period_display"], as_index=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "value": compute_pph_kpi(x, facility_uid)[
                                "pph_rate"
                            ],  # Changed to compute_pph_kpi
                            "PPH Cases": compute_pph_kpi(x, facility_uid)[
                                "pph_count"
                            ],  # Changed to compute_pph_kpi
                            "Total Deliveries": compute_pph_kpi(x, facility_uid)[
                                "total_deliveries"
                            ],  # Changed to compute_pph_kpi
                        }
                    )
                )
                .reset_index(drop=True)
            )
            render_pph_trend_chart(  # Keep using PPH-specific render function
                group,
                "period_display",
                "value",
                "PPH Rate (%)",
                bg_color,
                text_color,
                facility_name,
                "PPH Cases",
                "Total Deliveries",
                facility_uid,
            )

        # Build aggregated trend data using standardized compute_kpis function
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            # First, let's compute the data for each period
            period_data = []
            for period in filtered_events["period"].unique():
                period_df = filtered_events[filtered_events["period"] == period]
                period_display = (
                    period_df["period_display"].iloc[0]
                    if not period_df.empty
                    else period
                )

                # Compute the KPI for this period
                kpi_data = compute_uterotonic_kpi(period_df, facility_uid)

                # Calculate percentage rates for each drug type
                total_deliveries = kpi_data["total_deliveries"]
                ergometrine_rate = (
                    (
                        kpi_data["uterotonic_types"]["Ergometrine"]
                        / total_deliveries
                        * 100
                    )
                    if total_deliveries > 0
                    else 0
                )
                oxytocin_rate = (
                    (kpi_data["uterotonic_types"]["Oxytocin"] / total_deliveries * 100)
                    if total_deliveries > 0
                    else 0
                )
                misoprostol_rate = (
                    (
                        kpi_data["uterotonic_types"]["Misoprostol"]
                        / total_deliveries
                        * 100
                    )
                    if total_deliveries > 0
                    else 0
                )

                period_data.append(
                    {
                        "period": period,
                        "period_display": period_display,
                        "value": kpi_data["uterotonic_rate"],
                        "Uterotonic Cases": kpi_data["uterotonic_count"],
                        "Total Deliveries": total_deliveries,
                        "ergometrine_rate": ergometrine_rate,
                        "oxytocin_rate": oxytocin_rate,
                        "misoprostol_rate": misoprostol_rate,
                        # Keep the counts as well for reference
                        "ergometrine_count": kpi_data["uterotonic_types"][
                            "Ergometrine"
                        ],
                        "oxytocin_count": kpi_data["uterotonic_types"]["Oxytocin"],
                        "misoprostol_count": kpi_data["uterotonic_types"][
                            "Misoprostol"
                        ],
                    }
                )

                # Convert to DataFrame
                group = pd.DataFrame(period_data)

            # Render the chart
            render_uterotonic_trend_chart(
                group,
                "period_display",
                "value",
                "Uterotonic Administration Rate (%)",
                bg_color,
                text_color,
                facility_name,
                "Uterotonic Cases",
                "Total Deliveries",
                facility_uid,
            )

        elif kpi_selection == "ARV Prophylaxis Rate (%)":
            # First, let's compute the data for each period
            period_data = []
            for period in filtered_events["period"].unique():
                period_df = filtered_events[filtered_events["period"] == period]
                period_display = (
                    period_df["period_display"].iloc[0]
                    if not period_df.empty
                    else period
                )

                # Compute the ARV KPI for this period
                arv_data = compute_arv_kpi(period_df, facility_uid)

                period_data.append(
                    {
                        "period": period,
                        "period_display": period_display,
                        "value": arv_data["arv_rate"],
                        "ARV Cases": arv_data["arv_count"],
                        "HIV-Exposed Infants": arv_data["hiv_exposed_infants"],
                    }
                )

            # Convert to DataFrame
            group = pd.DataFrame(period_data)

            # Render the ARV trend chart
            render_arv_trend_chart(
                group,
                "period_display",
                "value",
                "ARV Prophylaxis Rate (%)",
                bg_color,
                text_color,
                facility_name,
                "ARV Cases",
                "HIV-Exposed Infants",
                facility_uid,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Optional: Add additional PPH visualizations
        if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
            st.markdown("---")
            st.markdown(
                f'<div class="section-header">📊 Additional PPH Analytics</div>',
                unsafe_allow_html=True,
            )

            render_obstetric_condition_pie_chart(
                filtered_events, facility_uid, bg_color, text_color
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            st.markdown("---")
            st.markdown(
                f'<div class="section-header">📊 Additional Uterotonic Analytics</div>',
                unsafe_allow_html=True,
            )
            render_uterotonic_type_pie_chart(
                filtered_events, facility_uid, bg_color, text_color
            )
