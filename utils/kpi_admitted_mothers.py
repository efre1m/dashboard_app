import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import datetime as dt
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
if "admitted_mothers_cache" not in st.session_state:
    st.session_state.admitted_mothers_cache = {}


def get_admitted_mothers_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Admitted Mothers computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_admitted_mothers_cache():
    """Clear the Admitted Mothers cache"""
    st.session_state.admitted_mothers_cache = {}


# ---------------- KPI Computation Functions ----------------
def compute_admitted_mothers_count(df, facility_uids=None):
    """
    Count Admitted Mothers occurrences - SAME METHOD AS compute_csection_count in kpi_utils.py
    Counts unique TEI IDs with enrollment dates
    """
    cache_key = get_admitted_mothers_cache_key(
        df, facility_uids, "admitted_mothers_count"
    )

    if cache_key in st.session_state.admitted_mothers_cache:
        return st.session_state.admitted_mothers_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Use enrollment date directly for this KPI
        date_column = "enrollment_date"

        # Check if enrollment date column exists
        if date_column not in filtered_df.columns:
            result = 0
        else:
            # Filter to only include rows that have enrollment dates
            filtered_df[date_column] = pd.to_datetime(
                filtered_df[date_column], errors="coerce"
            )
            filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

            # Count unique TEI IDs with enrollment dates
            if "tei_id" in filtered_df.columns:
                result = filtered_df["tei_id"].dropna().nunique()
            else:
                result = len(filtered_df)

    st.session_state.admitted_mothers_cache[cache_key] = result
    return result


def compute_admitted_mothers_rate(df, facility_uids=None):
    """
    For Admitted Mothers, rate is just the count (since it's not a percentage)
    Returns: (count, count, 1) to match the pattern
    """
    cache_key = get_admitted_mothers_cache_key(
        df, facility_uids, "admitted_mothers_rate"
    )

    if cache_key in st.session_state.admitted_mothers_cache:
        return st.session_state.admitted_mothers_cache[cache_key]

    if df is None or df.empty:
        result = (0, 0, 0.0)  # (count, denominator, value)
    else:
        # Get date column for Admitted Mothers
        date_column = get_relevant_date_column_for_kpi("Admitted Mothers")

        # Count admitted mothers
        admitted_mothers = compute_admitted_mothers_count(df, facility_uids)

        # For Admitted Mothers, we just return the count as the value
        result = (admitted_mothers, 1, float(admitted_mothers))

    st.session_state.admitted_mothers_cache[cache_key] = result
    return result


def compute_admitted_mothers_kpi(df, facility_uids=None):
    """
    Compute Admitted Mothers KPI data
    This is the function your dashboard is calling
    """
    count, denominator, value = compute_admitted_mothers_rate(df, facility_uids)

    return {
        "admitted_mothers_count": int(count),
        "admitted_mothers_value": float(value),
        "admitted_mothers_denominator": int(denominator),
    }


def get_numerator_denominator_for_admitted_mothers(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Admitted Mothers KPI
    WITH DATE RANGE FILTERING
    Returns: (numerator, denominator, value)
    For Admitted Mothers: numerator = count, denominator = 1, value = count
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Admitted Mothers
    date_column = get_relevant_date_column_for_kpi("Admitted Mothers")

    # IMPORTANT: Filter to only include rows that have this specific date
    if date_column in filtered_df.columns:
        # Convert to datetime and filter out rows without this date
        filtered_df[date_column] = pd.to_datetime(
            filtered_df[date_column], errors="coerce"
        )
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        # Apply date range filtering if provided
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                filtered_df = filtered_df[
                    (filtered_df[date_column] >= start_dt)
                    & (filtered_df[date_column] < end_dt)
                ].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    # Compute Admitted Mothers count on date-filtered data
    count, denominator, value = compute_admitted_mothers_rate(
        filtered_df, facility_uids
    )

    return (count, denominator, value)


# ---------------- Chart Functions WITH TABLES ----------------
# In kpi_admitted_mothers.py - FIXED VERSION WITH UNIQUE KEYS


# ---------------- Chart Functions WITH TABLES ----------------
def render_admitted_mothers_trend_chart(*args, **kwargs):
    """Render trend chart for Admitted Mothers using standard utility"""
    return render_trend_chart(*args, **kwargs)


def render_admitted_mothers_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for Admitted Mothers using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)


def render_admitted_mothers_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for Admitted Mothers using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_admitted_mothers_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Admitted Mothers trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    SAME AS ASSISTED DELIVERY FUNCTION
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Admitted Mothers
    date_column = get_relevant_date_column_for_kpi(kpi_name)

    # Check if the SPECIFIC date column exists
    if date_column not in filtered_df.columns:
        # Try to use event_date as fallback
        if "event_date" in filtered_df.columns:
            date_column = "event_date"
            st.warning(
                f"⚠️ KPI-specific date column not found for {kpi_name}, using 'event_date' instead"
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


# ---------------- Main Entry Point ----------------
def get_admitted_mothers_kpi_data(df, facility_uids=None):
    """
    Main function to get Admitted Mothers KPI data for dashboard
    """
    # Get date range filters from session state if available
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Compute KPI
    count, denominator, value = get_numerator_denominator_for_admitted_mothers(
        df, facility_uids, date_range_filters
    )

    return {
        "admitted_mothers_count": int(count),
        "admitted_mothers_value": float(value),
    }
