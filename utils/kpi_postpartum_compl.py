import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt
import plotly.express as px
import hashlib

# Import only utility functions from kpi_utils
from utils.kpi_utils import (
    auto_text_color,
    format_period_month_year,
    compute_total_deliveries,
    get_relevant_date_column_for_kpi,
    render_trend_chart,
    render_facility_comparison_chart,
    render_region_comparison_chart,
)

# ---------------- Caching Setup ----------------
if "postpartum_compl_cache" not in st.session_state:
    st.session_state.postpartum_compl_cache = {}


def get_postpartum_compl_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Postpartum Complications computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_postpartum_compl_cache():
    """Clear the Postpartum Complications cache"""
    st.session_state.postpartum_compl_cache = {}


# Postpartum Complications KPI Configuration
POSTPARTUM_COMPL_COL = "obstetric_condition_at_delivery_delivery_summary"
# Postpartum Complications include all codes 1 to 10
POSTPARTUM_COMPL_CODES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# Using the complete OBSTETRIC_CONDITION_CODES mapping
OBSTETRIC_CONDITION_CODES = {
    "1": "Severe pre-eclampsia/eclampsia",
    "2": "Antepartum hemorrhage",
    "3": "Postpartum hemorrhage",
    "4": "Sepsis/severe systemic infection",
    "5": "Obstructed labor",
    "6": "Ruptured uterus",
    "7": "Severe anemia",
    "8": "Malaria with severe anemia",
    "9": "HIV-related conditions",
    "10": "Other complications",
}

# Standardized labels for distribution
POSTPARTUM_COMPL_LABELS = OBSTETRIC_CONDITION_CODES


def compute_postpartum_compl_count(df, facility_uids=None):
    """
    Count Postpartum Complications occurrences - any code from 1 to 10
    Handles multi-select string format like "1,2,3"
    """
    cache_key = get_postpartum_compl_cache_key(df, facility_uids, "postpartum_compl_count")

    if cache_key in st.session_state.postpartum_compl_cache:
        return st.session_state.postpartum_compl_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if POSTPARTUM_COMPL_COL not in filtered_df.columns:
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string and handle multiple codes row by row for robustness
            def has_postpartum_compl(val):
                if pd.isna(val):
                    return False
                # Normalize: string, remove .0, split, strip
                codes = str(val).replace(".0", "").split(",")
                codes = [c.strip() for c in codes]
                # Check if any valid postpartum code is present
                return any(c in POSTPARTUM_COMPL_CODES for c in codes)

            mask = filtered_df[POSTPARTUM_COMPL_COL].apply(has_postpartum_compl)
            result = int(mask.sum())

    st.session_state.postpartum_compl_cache[cache_key] = result
    return result


def compute_postpartum_distribution(df, facility_uids=None):
    """
    Compute distribution of ALL obstetric complications
    Returns counts for each complication type (codes 1-10)
    """
    if df is None or df.empty:
        return {name: 0 for name in POSTPARTUM_COMPL_LABELS.values()}

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if POSTPARTUM_COMPL_COL not in filtered_df.columns:
        return {name: 0 for name in POSTPARTUM_COMPL_LABELS.values()}

    # Initialize counts
    named_counts = {name: 0 for name in POSTPARTUM_COMPL_LABELS.values()}

    # ROBUST Instance-Based Vectorized computation:
    # 1. Convert to string and remove .0 decimals
    # 2. Split by comma and strip
    # 3. Explode to count EVERY instance of EVERY complication code
    # This provides clinical detail on all reported complications.
    
    s = filtered_df[POSTPARTUM_COMPL_COL].astype(str).str.replace(".0", "", regex=False).str.split(",")
    exploded = s.explode().str.strip()
    
    # Filter for valid postpartum codes 1-10
    exploded = exploded[exploded.isin(POSTPARTUM_COMPL_CODES)]
    dist = exploded.value_counts()
    
    total = 0
    for code, count in dist.items():
        if code in POSTPARTUM_COMPL_LABELS:
            name = POSTPARTUM_COMPL_LABELS[code]
            c_val = int(count)
            named_counts[name] = c_val
            total += c_val

    # Label as "Instances" to distinguish from the case-based numerator in trend charts
    named_counts["Total Complication Instances"] = total

    return named_counts


def compute_postpartum_compl_rate(df, facility_uids=None):
    """
    Compute Postpartum Complications Rate
    Returns: (rate, complication_cases, total_deliveries)
    """
    cache_key = get_postpartum_compl_cache_key(df, facility_uids, "postpartum_compl_rate")

    if cache_key in st.session_state.postpartum_compl_cache:
        return st.session_state.postpartum_compl_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for this KPI
        date_column = get_relevant_date_column_for_kpi(
            "Postpartum Complications Rate (%)"
        )

        # Count cases
        cases = compute_postpartum_compl_count(df, facility_uids)

        # Get total deliveries
        total_deliveries = compute_total_deliveries(df, facility_uids, date_column)

        # Calculate rate
        rate = (cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (rate, cases, total_deliveries)

    st.session_state.postpartum_compl_cache[cache_key] = result
    return result


def compute_postpartum_compl_kpi(df, facility_uids=None):
    """
    Compute KPI data for dashboard
    """
    rate, cases, total = compute_postpartum_compl_rate(df, facility_uids)

    # Get distribution
    distribution = compute_postpartum_distribution(df, facility_uids)

    return {
        "postpartum_compl_rate": float(rate),
        "postpartum_compl_cases": int(cases),
        "total_deliveries": int(total),
        "postpartum_compl_distribution": distribution,
    }


def get_numerator_denominator_for_postpartum_compl(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for the KPI with date range filtering
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column
    date_column = get_relevant_date_column_for_kpi("Postpartum Complications Rate (%)")

    # Filter by date range
    if date_column in filtered_df.columns:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                filtered_df = filtered_df[
                    (filtered_df[date_column] >= start_dt) & 
                    (filtered_df[date_column] < end_dt)
                ].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    rate, cases, total = compute_postpartum_compl_rate(filtered_df, facility_uids)
    return (cases, total, rate)


# ---------------- Chart Functions ----------------
def render_postpartum_compl_trend_chart(*args, **kwargs):
    return render_trend_chart(*args, **kwargs)


def render_postpartum_compl_facility_comparison_chart(*args, **kwargs):
    return render_facility_comparison_chart(*args, **kwargs)


def render_postpartum_compl_region_comparison_chart(*args, **kwargs):
    return render_region_comparison_chart(*args, **kwargs)


def render_postpartum_complication_type_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None, **kwargs
):
    """
    Render a pie chart showing distribution of all complication types
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    distribution = compute_postpartum_distribution(df, facility_uids)

    if distribution["Total Complication Instances"] == 0:
        st.info("⚠️ No complication data available for distribution.")
        return

    pie_data = []
    for name, count in distribution.items():
        if name != "Total Complication Instances" and count > 0:
            pie_data.append({"Complication": name, "Count": count})

    if not pie_data:
        st.info("⚠️ No complication data available.")
        return

    pie_df = pd.DataFrame(pie_data).sort_values("Count", ascending=False)
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

    st.markdown('<div style="text-align: center; font-weight: bold;">Distribution of Postpartum Complication Types</div>', unsafe_allow_html=True)
    
    fig = px.pie(
        pie_df,
        values="Count",
        names="Complication",
        hover_data=["Percentage"],
        height=400,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    # Use robust hash key
    uids_str = str(sorted(facility_uids)) if facility_uids else "all"
    key_hash = hashlib.md5(uids_str.encode()).hexdigest()[:8]
    unique_key = f"postpartum_compl_chart_{key_hash}_{kwargs.get('key_suffix', '')}"
    st.plotly_chart(fig, use_container_width=True, key=unique_key)


# ---------------- Additional Helper Functions ----------------
def prepare_postpartum_compl_data_for_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Postpartum Complications trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Postpartum Complications
    date_column = get_relevant_date_column_for_kpi(kpi_name)

    # Check if the SPECIFIC date column exists
    if date_column not in filtered_df.columns:
        # Try to use enrollment_date as fallback
        if "enrollment_date" in filtered_df.columns:
            date_column = "enrollment_date"
            st.warning(
                f"⚠️ KPI-specific date column not found for {kpi_name}, using 'enrollment_date' instead"
            )
        else:
            st.warning(
                f"⚠️ Required date column '{date_column}' not found for {kpi_name}"
            )
            return pd.DataFrame(), date_column

    # Create result dataframe
    result_df = filtered_df.copy()

    # Convert to datetime
    result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")

    # CRITICAL: Apply date range filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            # Convert to datetime for comparison
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

            # Filter by date range
            result_df = result_df[
                (result_df["event_date"] >= start_dt)
                & (result_df["event_date"] < end_dt)
            ].copy()

    # Filter out rows without valid dates
    result_df = result_df[result_df["event_date"].notna()].copy()

    if result_df.empty:
        st.info(f"⚠️ No data with valid dates in '{date_column}' for {kpi_name}")
        return pd.DataFrame(), date_column

    # Get period label
    period_label = st.session_state.get("period_label", "Monthly")
    if "filters" in st.session_state and "period_label" in st.session_state.filters:
        period_label = st.session_state.filters["period_label"]

    # Create period columns using time_filter utility
    from utils.time_filter import assign_period

    result_df = assign_period(result_df, "event_date", period_label)

    # Filter by facility if needed
    if facility_uids and "orgUnit" in result_df.columns:
        result_df = result_df[result_df["orgUnit"].isin(facility_uids)].copy()

    return result_df, date_column