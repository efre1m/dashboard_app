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
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
from utils.kpi_utils import (
    compute_kpis,
    render_trend_chart,
    auto_text_color,
    render_facility_comparison_chart,
    render_region_comparison_chart,
)

from utils.kpi_pph import (
    compute_pph_kpi,
    render_pph_trend_chart,
    render_pph_facility_comparison_chart,
    render_pph_region_comparison_chart,
    render_obstetric_condition_pie_chart,
)
from utils.kpi_uterotonic import (
    compute_uterotonic_kpi,
    render_uterotonic_trend_chart,
    render_uterotonic_facility_comparison_chart,
    render_uterotonic_region_comparison_chart,
    render_uterotonic_type_pie_chart,
)
from utils.queries import (
    get_facilities_grouped_by_region,
    get_facility_mapping_for_user,
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


# Initialize session state keys at the very top
if "all_facilities_checkbox" not in st.session_state:
    st.session_state.all_facilities_checkbox = False


# Helper functions for facility and region selection
def handle_all_facilities_change(all_facility_names):
    """Handle when 'All Facilities' checkbox is toggled"""
    if st.session_state.all_facilities_checkbox:
        # Select all facilities
        st.session_state.selected_facilities = ["All Facilities"] + all_facility_names
        # Deselect all regions
        st.session_state.selected_regions = []
    else:
        # Deselect all facilities
        st.session_state.selected_facilities = []


def handle_region_change(region_name, facility_names):
    """Handle when 'Select all in region' checkbox is toggled"""
    if st.session_state[f"select_all_{region_name}"]:
        # Add all facilities in this region
        for facility_name in facility_names:
            if facility_name not in st.session_state.selected_facilities:
                st.session_state.selected_facilities.append(facility_name)
        # Remove "All Facilities" if individual facilities are selected
        if "All Facilities" in st.session_state.selected_facilities:
            st.session_state.selected_facilities.remove("All Facilities")
    else:
        # Remove all facilities from this region
        for facility_name in facility_names:
            if facility_name in st.session_state.selected_facilities:
                st.session_state.selected_facilities.remove(facility_name)


def handle_facility_change(facility_name):
    """Handle when individual facility checkbox is toggled"""
    if st.session_state[f"facility_{facility_name}"]:
        # Add this facility
        if facility_name not in st.session_state.selected_facilities:
            st.session_state.selected_facilities.append(facility_name)
        # Remove "All Facilities" if individual facilities are selected
        if "All Facilities" in st.session_state.selected_facilities:
            st.session_state.selected_facilities.remove("All Facilities")
    else:
        # Remove this facility
        if facility_name in st.session_state.selected_facilities:
            st.session_state.selected_facilities.remove(facility_name)


def handle_region_selection_change(region_name):
    """Handle when region checkbox is toggled for region comparison"""
    if st.session_state[f"region_{region_name}"]:
        # Add this region to selected regions
        if region_name not in st.session_state.selected_regions:
            st.session_state.selected_regions.append(region_name)
        # Clear facility selection when selecting regions
        st.session_state.selected_facilities = []
        st.session_state.all_facilities_checkbox = False
    else:
        # Remove this region
        if region_name in st.session_state.selected_regions:
            st.session_state.selected_regions.remove(region_name)


def handle_all_regions_change(all_region_names):
    """Handle when 'All Regions' checkbox is toggled"""
    if st.session_state.all_regions_checkbox:
        # Select all regions
        st.session_state.selected_regions = all_region_names.copy()
        # Clear facility selection
        st.session_state.selected_facilities = []
        st.session_state.all_facilities_checkbox = False
    else:
        # Deselect all regions
        st.session_state.selected_regions = []


# ---------------- Page Rendering ----------------
def render():
    st.set_page_config(
        page_title="National Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "refresh_trigger" not in st.session_state:
        st.session_state["refresh_trigger"] = False

    # Load both CSS files - facility.css first, then national.css for overrides
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Facility CSS file not found: {e}")

    try:
        with open("utils/national.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"National CSS file not found: {e}")
        # Fallback to basic national styling
        st.markdown(
            """
            <style>
            /* National-specific fallback styles */
            .sidebar .sidebar-content {
                background: linear-gradient(135deg, #1a5fb4 0%, #1c71d8 100%);
                color: white;
            }
            .user-info {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .section-header {
                color: white !important;
                font-weight: 700;
                font-size: 18px;
                margin: 20px 0 15px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            }
            .stCheckbox [data-baseweb="checkbox"]:checked {
                background-color: #ffa348;
                border-color: #ffa348;
            }
            .stExpander {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .stExpander summary {
                color: white !important;
                font-weight: 600;
                padding: 12px 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

    # Sidebar user info
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    country_name = user.get("country_name", "Unknown country")

    st.sidebar.markdown(
        f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🌍 Country: {country_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]
        st.rerun()

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

    # ---------------- Facility and Region Filter ----------------
    # Get facilities grouped by region from database
    facilities_by_region = get_facilities_grouped_by_region(user)

    # Create facility mapping for UID lookup (from database)
    facility_mapping = get_facility_mapping_for_user(user)

    # Get all facility names for "All Facilities" selection
    all_facility_names = []
    for region_name, facilities in facilities_by_region.items():
        for facility_name, _ in facilities:
            all_facility_names.append(facility_name)

    # Get all region names
    all_region_names = list(facilities_by_region.keys())

    # Initialize session state for facility and region selection
    if "selected_facilities" not in st.session_state:
        st.session_state.selected_facilities = ["All Facilities"] + all_facility_names

    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = []

    if "expanded_regions" not in st.session_state:
        # Default all regions to collapsed (not expanded)
        st.session_state.expanded_regions = {
            region_name: False for region_name in facilities_by_region.keys()
        }

    # Facility and region selector in sidebar with expandable regions
    st.sidebar.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 8px;">🏥 Select Facilities or Regions</p>',
        unsafe_allow_html=True,
    )

    # Create a container for the selector
    selector_container = st.sidebar.container()

    with selector_container:
        # "All Facilities" checkbox
        all_facilities_selected = st.checkbox(
            "All Facilities",
            value="All Facilities" in st.session_state.selected_facilities,
            key="all_facilities_checkbox",
            on_change=lambda: handle_all_facilities_change(all_facility_names),
        )

        # "All Regions" checkbox
        all_regions_selected = st.checkbox(
            "All Regions",
            value=len(st.session_state.selected_regions) == len(all_region_names),
            key="all_regions_checkbox",
            on_change=lambda: handle_all_regions_change(all_region_names),
        )

        # Individual region checkboxes for region comparison
        st.markdown(
            '<p style="color: white; font-weight: 600; margin: 15px 0 8px 0;">Select Regions for Comparison:</p>',
            unsafe_allow_html=True,
        )

        for region_name in all_region_names:
            region_selected = st.checkbox(
                region_name,
                value=region_name in st.session_state.selected_regions,
                key=f"region_{region_name}",
                on_change=lambda rn=region_name: handle_region_selection_change(rn),
            )

        # Region expanders for facility selection
        st.markdown(
            '<p style="color: white; font-weight: 600; margin: 15px 0 8px 0;">Select Individual Facilities:</p>',
            unsafe_allow_html=True,
        )

        for region_name, facilities in facilities_by_region.items():
            # Create expander for each region
            with st.expander(
                f"{region_name} ({len(facilities)})",
                expanded=st.session_state.expanded_regions[region_name],
            ):
                # Update expanded state when user expands
                st.session_state.expanded_regions[region_name] = True

                # Select all facilities in this region
                all_in_region_selected = all(
                    [
                        facility_name in st.session_state.selected_facilities
                        for facility_name, _ in facilities
                    ]
                )
                all_in_region = st.checkbox(
                    f"Select all in {region_name}",
                    value=all_in_region_selected,
                    key=f"select_all_{region_name}",
                    on_change=lambda r=region_name, f=[
                        fac[0] for fac in facilities
                    ]: handle_region_change(r, f),
                )

                # Individual facility checkboxes
                for facility_name, _ in facilities:
                    facility_selected = st.checkbox(
                        facility_name,
                        value=facility_name in st.session_state.selected_facilities,
                        key=f"facility_{facility_name}",
                        on_change=lambda fn=facility_name: handle_facility_change(fn),
                    )

    # Get the selected facilities and regions
    selected_facilities = [
        f for f in st.session_state.selected_facilities if f != "All Facilities"
    ]
    selected_regions = st.session_state.selected_regions

    # Calculate total facilities count
    total_facilities = sum(
        len(facilities) for facilities in facilities_by_region.values()
    )

    # Display selection count
    if "All Facilities" in st.session_state.selected_facilities:
        display_text = f"Selected: All Facilities ({total_facilities})"
    elif selected_facilities:
        display_text = (
            f"Selected: {len(selected_facilities)} / {total_facilities} Facilities"
        )
    elif selected_regions:
        if len(selected_regions) == len(all_region_names):
            display_text = f"Selected: All Regions ({len(selected_regions)})"
        else:
            display_text = (
                f"Selected: {len(selected_regions)} / {len(all_region_names)} Regions"
            )
    else:
        display_text = "No selection"

    st.sidebar.markdown(
        f"<p style='color: white; font-size: 13px; margin-top: 10px;'>{display_text}</p>",
        unsafe_allow_html=True,
    )

    # Get the facility UIDs for selected facilities or regions
    facility_uids = None
    display_names = None
    comparison_mode = None

    if selected_facilities:
        # Use selected individual facilities
        facility_uids = [
            facility_mapping[facility]
            for facility in selected_facilities
            if facility in facility_mapping
        ]
        display_names = selected_facilities
        comparison_mode = "facility"
    elif selected_regions:
        # Use all facilities in selected regions
        facility_uids = []
        for region_name in selected_regions:
            if region_name in facilities_by_region:
                for facility_name, facility_uid in facilities_by_region[region_name]:
                    facility_uids.append(facility_uid)
        display_names = selected_regions
        comparison_mode = "region"
    else:
        # Default to all facilities
        facility_uids = list(facility_mapping.values())
        display_names = ["All Facilities"]
        comparison_mode = "facility"

    # ---------------- View Mode Selection ----------------
    view_mode = "Normal Trend"
    if (comparison_mode == "facility" and len(display_names) > 1) or (
        comparison_mode == "region" and len(display_names) > 1
    ):
        view_mode = st.sidebar.radio(
            "📊 View Mode",
            ["Normal Trend", "Comparison View"],
            index=0,
            help="Compare trends across multiple facilities or regions",
        )

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
                file_name=f"{country_name}_raw.json",
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
                file_name=f"{country_name}_dataframes.zip",
                mime="application/zip",
            )

    # MAIN HEADING
    if (
        comparison_mode == "facility"
        and "All Facilities" in st.session_state.selected_facilities
    ):
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {country_name}</div>',
            unsafe_allow_html=True,
        )
    elif comparison_mode == "facility" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {display_names[0]}</div>',
            unsafe_allow_html=True,
        )
    elif comparison_mode == "facility":
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Facilities ({len(display_names)})</div>',
            unsafe_allow_html=True,
        )
    elif comparison_mode == "region" and len(display_names) == 1:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - {display_names[0]} Region</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="main-header">🌍 National Maternal Health Dashboard - Multiple Regions ({len(display_names)})</div>',
            unsafe_allow_html=True,
        )

    # ---------------- KPI CARDS ----------------
    if copied_events_df.empty or "event_date" not in copied_events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # 🔒 Always national view
    display_name = country_name

    # Pass user_id into KPI card renderer so it can save/load previous values
    user_id = str(user.get("id", user.get("username", "default_user")))

    all_facility_uids = list(facility_mapping.values())  # all facilities in the DB
    render_kpi_cards(
        copied_events_df,  # full dataset
        all_facility_uids,  # force ALL facilities
        display_name,  # ✅ fixed to national label
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

        # Determine available aggregation levels based on selected date range
        available_aggregations = get_available_aggregations(start_date, end_date)

        # Apply safe default if previous selection is invalid
        if (
            "period_label" not in st.session_state
            or st.session_state.period_label not in available_aggregations
        ):
            st.session_state.period_label = available_aggregations[
                -1
            ]  # widest available

        # Show dropdown with safe default
        period_label = st.selectbox(
            "⏰ Aggregation Level",
            available_aggregations,
            index=available_aggregations.index(st.session_state.period_label),
        )

        bg_color = st.color_picker("🎨 Chart Background", "#FFFFFF")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- APPLY FILTER ----------------
    # Convert date objects to datetimes for comparison
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    # Filter events and enrollments by selected range
    filtered_events = copied_events_df[
        (copied_events_df["event_date"] >= start_datetime)
        & (copied_events_df["event_date"] <= end_datetime)
    ].copy()

    # Gauge
    st.session_state["filtered_events"] = filtered_events.copy()

    filtered_enrollments = enrollments_df.copy()
    if (
        not filtered_enrollments.empty
        and "enrollmentDate" in filtered_enrollments.columns
    ):
        filtered_enrollments = filtered_enrollments[
            (filtered_enrollments["enrollmentDate"] >= start_datetime)
            & (filtered_enrollments["enrollmentDate"] <= end_datetime)
        ]

    # Apply facility filter if selected (using dhis2_uid from database)
    if facility_uids:
        filtered_events = filtered_events[
            filtered_events["orgUnit"].isin(facility_uids)
        ]

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
        if view_mode == "Comparison View" and len(display_names) > 1:
            st.markdown(
                f'<div class="section-header">📈 {kpi_selection} - {comparison_mode.title()} Comparison</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

            # Map KPI selection to the appropriate parameters
            kpi_mapping = {
                "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
                    "title": "IPPCAR (%)",
                    "numerator_name": "FP Acceptances",
                    "denominator_name": "Total Deliveries",
                },
                "Stillbirth Rate (per 1000 births)": {
                    "title": "Stillbirth Rate (per 1000 births)",
                    "numerator_name": "Stillbirths",
                    "denominator_name": "Total Births",
                },
                "Early Postnatal Care (PNC) Coverage (%)": {
                    "title": "Early PNC Coverage (%)",
                    "numerator_name": "Early PNC (≤48 hrs)",
                    "denominator_name": "Total Deliveries",
                },
                "Institutional Maternal Death Rate (per 100,000 births)": {
                    "title": "Maternal Death Rate (per 100,000 births)",
                    "numerator_name": "Maternal Deaths",
                    "denominator_name": "Live Births",
                },
                "C-Section Rate (%)": {
                    "title": "C-Section Rate (%)",
                    "numerator_name": "C-Sections",
                    "denominator_name": "Total Deliveries",
                },
                "Postpartum Hemorrhage (PPH) Rate (%)": {
                    "title": "PPH Rate (%)",
                    "numerator_name": "PPH Cases",
                    "denominator_name": "Total Deliveries",
                },
                "Delivered women who received uterotonic (%)": {
                    "title": "Delivered women who received uterotonic (%)",
                    "numerator_name": "Women given uterotonic",
                    "denominator_name": "Deliveries",
                },
            }

            kpi_config = kpi_mapping.get(kpi_selection, {})

            if comparison_mode == "facility":
                if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
                    # Use PPH-specific facility comparison
                    render_pph_facility_comparison_chart(
                        df=filtered_events,
                        period_col="period_display",
                        value_col="value",
                        title="PPH Rate (%)",
                        bg_color=bg_color,
                        text_color=text_color,
                        facility_names=display_names,
                        facility_uids=facility_uids,
                        numerator_name="PPH Cases",
                        denominator_name="Total Deliveries",
                    )
                elif kpi_selection == "Delivered women who received uterotonic (%)":
                    # Use uterotonic-specific facility comparison
                    render_uterotonic_facility_comparison_chart(
                        df=filtered_events,
                        period_col="period_display",
                        value_col="value",
                        title="Uterotonic Administration Rate (%)",
                        bg_color=bg_color,
                        text_color=text_color,
                        facility_names=display_names,
                        facility_uids=facility_uids,
                        numerator_name="Uterotonic Cases",
                        denominator_name="Total Deliveries",
                    )
                else:
                    # Use generic facility comparison for other KPIs
                    render_facility_comparison_chart(
                        df=filtered_events,
                        period_col="period_display",
                        value_col="value",
                        title=kpi_config.get("title", kpi_selection),
                        bg_color=bg_color,
                        text_color=text_color,
                        facility_names=display_names,
                        facility_uids=facility_uids,
                        numerator_name=kpi_config.get("numerator_name", "Numerator"),
                        denominator_name=kpi_config.get(
                            "denominator_name", "Denominator"
                        ),
                    )
            else:  # comparison_mode == "region"
                if kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
                    # Use PPH-specific region comparison
                    render_pph_region_comparison_chart(
                        df=filtered_events,
                        period_col="period_display",
                        value_col="value",
                        title="PPH Rate (%)",
                        bg_color=bg_color,
                        text_color=text_color,
                        region_names=display_names,
                        region_mapping={},
                        facilities_by_region=facilities_by_region,
                        numerator_name="PPH Cases",
                        denominator_name="Total Deliveries",
                    )
                elif kpi_selection == "Delivered women who received uterotonic (%)":
                    # Use uterotonic-specific region comparison
                    render_uterotonic_region_comparison_chart(
                        df=filtered_events,
                        period_col="period_display",
                        value_col="value",
                        title="Uterotonic Administration Rate (%)",
                        bg_color=bg_color,
                        text_color=text_color,
                        region_names=display_names,
                        region_mapping={},
                        facilities_by_region=facilities_by_region,
                        numerator_name="Uterotonic Cases",
                        denominator_name="Total Deliveries",
                    )
                else:
                    # Use generic region comparison for other KPIs
                    render_region_comparison_chart(
                        df=filtered_events,
                        period_col="period_display",
                        value_col="value",
                        title=kpi_config.get("title", kpi_selection),
                        bg_color=bg_color,
                        text_color=text_color,
                        region_names=display_names,
                        region_mapping={},
                        facilities_by_region=facilities_by_region,
                        numerator_name=kpi_config.get("numerator_name", "Numerator"),
                        denominator_name=kpi_config.get(
                            "denominator_name", "Denominator"
                        ),
                    )

        else:
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
                    filtered_events.groupby(
                        ["period", "period_display"], as_index=False
                    )
                    .apply(
                        lambda x: pd.Series(
                            {
                                "value": compute_kpis(x, facility_uids)["ippcar"],
                                "FP Acceptances": compute_kpis(x, facility_uids)[
                                    "fp_acceptance"
                                ],
                                "Total Deliveries": compute_kpis(x, facility_uids)[
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
                    display_names,
                    "FP Acceptances",
                    "Total Deliveries",
                    facility_uids,
                )

            elif kpi_selection == "Stillbirth Rate (per 1000 births)":
                group = (
                    filtered_events.groupby(
                        ["period", "period_display"], as_index=False
                    )
                    .apply(
                        lambda x: pd.Series(
                            {
                                "value": compute_kpis(x, facility_uids)[
                                    "stillbirth_rate"
                                ],
                                "Stillbirths": compute_kpis(x, facility_uids)[
                                    "stillbirths"
                                ],
                                "Total Births": compute_kpis(x, facility_uids)[
                                    "total_births"
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
                    "Stillbirth Rate (per 1000 births)",
                    bg_color,
                    text_color,
                    display_names,
                    "Stillbirths",
                    "Total Births",
                    facility_uids,
                )

            elif kpi_selection == "Early Postnatal Care (PNC) Coverage (%)":
                group = (
                    filtered_events.groupby(
                        ["period", "period_display"], as_index=False
                    )
                    .apply(
                        lambda x: pd.Series(
                            {
                                "value": compute_kpis(x, facility_uids)["pnc_coverage"],
                                "Early PNC (≤48 hrs)": compute_kpis(x, facility_uids)[
                                    "early_pnc"
                                ],
                                "Total Deliveries": compute_kpis(x, facility_uids)[
                                    "total_deliveries_pnc"
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
                    "Early PNC Coverage (%)",
                    bg_color,
                    text_color,
                    display_names,
                    "Early PNC (≤48 hrs)",
                    "Total Deliveries",
                    facility_uids,
                )

            elif (
                kpi_selection
                == "Institutional Maternal Death Rate (per 100,000 births)"
            ):
                group = (
                    filtered_events.groupby(
                        ["period", "period_display"], as_index=False
                    )
                    .apply(
                        lambda x: pd.Series(
                            {
                                "value": compute_kpis(x, facility_uids)[
                                    "maternal_death_rate"
                                ],
                                "Maternal Deaths": compute_kpis(x, facility_uids)[
                                    "maternal_deaths"
                                ],
                                "Live Births": compute_kpis(x, facility_uids)[
                                    "live_births"
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
                    "Maternal Death Rate (per 100,000 births)",
                    bg_color,
                    text_color,
                    display_names,
                    "Maternal Deaths",
                    "Live Births",
                    facility_uids,
                )

            elif kpi_selection == "C-Section Rate (%)":
                group = (
                    filtered_events.groupby(
                        ["period", "period_display"], as_index=False
                    )
                    .apply(
                        lambda x: pd.Series(
                            {
                                "value": compute_kpis(x, facility_uids)[
                                    "csection_rate"
                                ],
                                "C-Sections": compute_kpis(x, facility_uids)[
                                    "csection_deliveries"
                                ],
                                "Total Deliveries": compute_kpis(x, facility_uids)[
                                    "total_deliveries_cs"
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
                    "C-Section Rate (%)",
                    bg_color,
                    text_color,
                    display_names,
                    "C-Sections",
                    "Total Deliveries",
                    facility_uids,
                )

            elif kpi_selection == "Postpartum Hemorrhage (PPH) Rate (%)":
                # Use the same structure as other indicators
                group = (
                    filtered_events.groupby(
                        ["period", "period_display"], as_index=False
                    )
                    .apply(
                        lambda x: pd.Series(
                            {
                                "value": compute_pph_kpi(x, facility_uids)[
                                    "pph_rate"
                                ],  # Changed to compute_pph_kpi
                                "PPH Cases": compute_pph_kpi(x, facility_uids)[
                                    "pph_count"
                                ],  # Changed to compute_pph_kpi
                                "Total Deliveries": compute_pph_kpi(x, facility_uids)[
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
                    display_names,
                    "PPH Cases",
                    "Total Deliveries",
                    facility_uids,
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
                    kpi_data = compute_uterotonic_kpi(period_df, facility_uids)

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
                        (
                            kpi_data["uterotonic_types"]["Oxytocin"]
                            / total_deliveries
                            * 100
                        )
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
                    display_names,
                    "Uterotonic Cases",
                    "Total Deliveries",
                    facility_uids,
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
                filtered_events, facility_uids, bg_color, text_color
            )
        elif kpi_selection == "Delivered women who received uterotonic (%)":
            st.markdown("---")
            st.markdown(
                f'<div class="section-header">📊 Additional Uterotonic Analytics</div>',
                unsafe_allow_html=True,
            )
            render_uterotonic_type_pie_chart(
                filtered_events, facility_uids, bg_color, text_color
            )
