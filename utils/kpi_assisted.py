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
    DELIVERY_MODE_COL,
)

# ---------------- Caching Setup ----------------
if "assisted_cache" not in st.session_state:
    st.session_state.assisted_cache = {}


def get_assisted_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Assisted Delivery computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_assisted_cache():
    """Clear the Assisted Delivery cache"""
    st.session_state.assisted_cache = {}


# Assisted Delivery KPI Configuration
INSTRUMENTAL_DELIVERY_COL = "instrumental_delivery_form"
ASSISTED_CODE = "1"  # Assisted Delivery code (value = 1)


def compute_assisted_count(df, facility_uids=None):
    """
    Count Assisted Delivery occurrences - SAME METHOD AS compute_csection_count in kpi_utils.py
    but checking for instrumental delivery code 1
    """
    cache_key = get_assisted_cache_key(df, facility_uids, "assisted_count")

    if cache_key in st.session_state.assisted_cache:
        return st.session_state.assisted_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if INSTRUMENTAL_DELIVERY_COL not in filtered_df.columns:
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string first, then extract numeric part
            df_copy["instrumental_clean"] = df_copy[INSTRUMENTAL_DELIVERY_COL].astype(
                str
            )
            df_copy["instrumental_numeric"] = pd.to_numeric(
                df_copy["instrumental_clean"].str.split(".").str[0], errors="coerce"
            )

            # --- Refined Logic: Exclude SVD (1) and C-section (2) to avoid double counting ---
            if DELIVERY_MODE_COL in df_copy.columns:
                df_copy["mode_clean"] = df_copy[DELIVERY_MODE_COL].astype(str)
                df_copy["mode_numeric"] = pd.to_numeric(
                    df_copy["mode_clean"].str.split(".").str[0], errors="coerce"
                )
                
                # Count Assisted Delivery (value = 1) ONLY if Mode of Delivery is EMPTY or N/A
                # (Ignore instrumental delivery if SVD or C-section is already recorded)
                assisted_mask = (df_copy["instrumental_numeric"] == 1) & (
                    df_copy["mode_numeric"].isna()
                )
            else:
                # Fallback if mode column missing
                assisted_mask = df_copy["instrumental_numeric"] == 1
                
            result = int(assisted_mask.sum())

    st.session_state.assisted_cache[cache_key] = result
    return result


def compute_assisted_rate(df, facility_uids=None):
    """
    Compute Assisted Delivery Rate - SAME PATTERN AS compute_csection_rate in kpi_utils.py
    Returns: (rate, assisted_deliveries, total_deliveries)
    """
    cache_key = get_assisted_cache_key(df, facility_uids, "assisted_rate")

    if cache_key in st.session_state.assisted_cache:
        return st.session_state.assisted_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for Assisted Delivery (enrollment_date)
        date_column = get_relevant_date_column_for_kpi("Assisted Delivery Rate (%)")

        # Count Assisted deliveries
        assisted_deliveries = compute_assisted_count(df, facility_uids)

        # Get total deliveries - USING SAME LOGIC AS C-SECTION
        total_deliveries = compute_total_deliveries(df, facility_uids, date_column)

        # Calculate rate
        rate = (
            (assisted_deliveries / total_deliveries * 100)
            if total_deliveries > 0
            else 0.0
        )
        result = (rate, assisted_deliveries, total_deliveries)

    st.session_state.assisted_cache[cache_key] = result
    return result


def compute_assisted_kpi(df, facility_uids=None):
    """
    Compute Assisted Delivery KPI data - SAME STRUCTURE AS compute_csection_rate
    This is the function your dashboard is calling
    """
    rate, assisted_deliveries, total_deliveries = compute_assisted_rate(
        df, facility_uids
    )

    return {
        "assisted_delivery_rate": float(rate),
        "assisted_deliveries": int(assisted_deliveries),
        "total_deliveries": int(total_deliveries),
    }


def get_numerator_denominator_for_assisted(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Assisted Delivery KPI - SAME PATTERN AS C-SECTION
    WITH DATE RANGE FILTERING
    Returns: (numerator, denominator, rate)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Assisted Delivery (enrollment_date)
    date_column = get_relevant_date_column_for_kpi("Assisted Delivery Rate (%)")

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

    # Compute Assisted Delivery rate on date-filtered data
    rate, assisted_deliveries, total_deliveries = compute_assisted_rate(
        filtered_df, facility_uids
    )

    return (assisted_deliveries, total_deliveries, rate)


# ---------------- Chart Functions WITH TABLES ----------------
def render_assisted_trend_chart(*args, **kwargs):
    """Render trend chart for Assisted Delivery using standard utility"""
    return render_trend_chart(*args, **kwargs)


def render_assisted_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for Assisted Delivery using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)


def render_assisted_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for Assisted Delivery using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_assisted_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Assisted Delivery trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    SAME AS SVD FUNCTION
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Assisted Delivery (enrollment_date)
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
