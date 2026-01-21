"""
KPI: Missing Birth Outcome Documentation
Measures: Percentage of deliveries where birth outcome is not documented
Formula: (Deliveries with missing birth outcome documentation) ÷ (Total deliveries with enrollment dates) × 100
"""

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
if "missing_bo_cache" not in st.session_state:
    st.session_state.missing_bo_cache = {}


def get_missing_bo_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Missing Birth Outcome computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_missing_bo_cache():
    """Clear the Missing Birth Outcome cache"""
    st.session_state.missing_bo_cache = {}


# ---------------- KPI Constants ----------------
BIRTH_OUTCOME_COL = "birth_outcome_delivery_summary"
DELIVERY_DATE_COL = "enrollment_date"  # Date column for denominator

# Empty/Null indicators
EMPTY_INDICATORS = ["", "nan", "None", "null", "N/A", "n/a", "na", "NA", "undefined"]


def compute_missing_bo_count(df, facility_uids=None):
    """
    Count deliveries with missing birth outcome documentation
    SIMPLE: Check if birth_outcome_delivery_summary is empty/N/A
    """
    cache_key = get_missing_bo_cache_key(df, facility_uids, "missing_count")

    if cache_key in st.session_state.missing_bo_cache:
        return st.session_state.missing_bo_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Filter to only include deliveries with dates (denominator logic)
        # Use the same date column logic as other functions
        date_column = get_relevant_date_column_for_kpi(
            "Missing Birth Outcome Documentation"
        )
        if date_column in filtered_df.columns:
            filtered_df[date_column] = pd.to_datetime(
                filtered_df[date_column], errors="coerce"
            )
            filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        if filtered_df.empty:
            result = 0
        else:
            # SIMPLE CHECK: Birth outcome column must be empty/N/A
            # Convert to string for comparison
            birth_outcome_vals = (
                filtered_df[BIRTH_OUTCOME_COL].astype(str).str.strip()
                if BIRTH_OUTCOME_COL in filtered_df.columns
                else pd.Series([""] * len(filtered_df))
            )

            # Check if birth outcome is empty/N/A
            bo_empty = (
                birth_outcome_vals.isin(EMPTY_INDICATORS)
                | (birth_outcome_vals == "")
                | (birth_outcome_vals.str.upper() == "N/A")
            )

            # Missing if birth outcome is empty
            missing_mask = bo_empty
            result = int(missing_mask.sum())

    st.session_state.missing_bo_cache[cache_key] = result
    return result


def compute_missing_bo_rate(df, facility_uids=None):
    """
    Compute Missing Birth Outcome Rate
    Returns: (rate, missing_cases, total_deliveries)
    """
    cache_key = get_missing_bo_cache_key(df, facility_uids, "missing_rate")

    if cache_key in st.session_state.missing_bo_cache:
        return st.session_state.missing_bo_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Count missing cases
        missing_cases = compute_missing_bo_count(df, facility_uids)

        # Get total deliveries - ONLY COUNT ROWS THAT HAVE ENROLLMENT DATE
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Filter to only include deliveries with enrollment dates
        if DELIVERY_DATE_COL in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[DELIVERY_DATE_COL].notna()].copy()

        total_deliveries = len(filtered_df)

        # Calculate rate
        rate = (missing_cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (float(rate), int(missing_cases), int(total_deliveries))

    st.session_state.missing_bo_cache[cache_key] = result
    return result


def compute_missing_bo_kpi(df, facility_uids=None):
    """
    Compute Missing Birth Outcome KPI data
    This is the function your dashboard will call
    """
    rate, missing_cases, total_deliveries = compute_missing_bo_rate(df, facility_uids)

    return {
        "missing_bo_rate": float(rate),
        "missing_bo_cases": int(missing_cases),
        "total_deliveries": int(total_deliveries),
    }


def get_numerator_denominator_for_missing_bo(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Missing Birth Outcome KPI WITH DATE RANGE FILTERING
    FIXED VERSION: Properly handles "All Time" case
    Returns: (numerator, denominator, rate)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Missing Birth Outcome (delivery summary)
    date_column = get_relevant_date_column_for_kpi(
        "Missing Birth Outcome Documentation"
    )

    # IMPORTANT: Store whether we should filter by date range
    should_filter_by_date = (
        date_range_filters
        and date_range_filters.get("start_date")
        and date_range_filters.get("end_date")
    )

    # ALWAYS filter to only include rows that have the delivery date
    # This is the denominator requirement
    if date_column in filtered_df.columns:
        # Convert to datetime and filter out rows without this date
        filtered_df[date_column] = pd.to_datetime(
            filtered_df[date_column], errors="coerce"
        )
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()
    else:
        # If date column doesn't exist, we can't compute this KPI
        return (0, 0, 0.0)

    # Apply date range filtering ONLY if we have specific dates
    if should_filter_by_date:
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

    # CRITICAL: Use the SAME logic as compute_missing_bo_count
    # Check if birth outcome column is empty/N/A
    birth_outcome_vals = (
        filtered_df[BIRTH_OUTCOME_COL].astype(str).str.strip()
        if BIRTH_OUTCOME_COL in filtered_df.columns
        else pd.Series([""] * len(filtered_df))
    )

    # Check if birth outcome is empty/N/A
    bo_empty = (
        birth_outcome_vals.isin(EMPTY_INDICATORS)
        | (birth_outcome_vals == "")
        | (birth_outcome_vals.str.upper() == "N/A")
    )

    # Missing if birth outcome is empty
    missing_mask = bo_empty
    numerator = int(missing_mask.sum())
    denominator = len(filtered_df)

    # Calculate rate
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0

    return (numerator, denominator, rate)


# ---------------- Chart Functions WITH TABLES ----------------
def render_missing_bo_trend_chart(*args, **kwargs):
    """Render trend chart for Missing BO using standard utility"""
    return render_trend_chart(*args, **kwargs)


def render_missing_bo_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for Missing BO using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)


def render_missing_bo_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for Missing BO using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_missing_bo_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Missing Birth Outcome trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    SAME AS ASSISTED FUNCTION
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Missing Birth Outcome (enrollment_date)
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


# ---------------- Main Entry Point ----------------
def get_missing_bo_kpi_data(df, facility_uids=None):
    """
    Main function to get Missing Birth Outcome KPI data for dashboard
    """
    # Get date range filters from session state if available
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Compute KPI
    rate, missing_cases, total_deliveries = compute_missing_bo_rate(df, facility_uids)

    return {
        "missing_bo_rate": float(rate),
        "missing_bo_cases": int(missing_cases),
        "total_deliveries": int(total_deliveries),
    }
