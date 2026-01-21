"""
KPI: Missing Condition of Discharge Documentation
Measures: Percentage of mothers with missing condition of discharge documentation
Formula: (Mothers with missing CoD AND (Status='Complete' OR Enrollment > 14 days old)) ÷ (Total Enrolled Mothers) × 100
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
if "missing_cod_cache" not in st.session_state:
    st.session_state.missing_cod_cache = {}


def get_missing_cod_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Missing Condition of Discharge computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_missing_cod_cache():
    """Clear the Missing Condition of Discharge cache"""
    st.session_state.missing_cod_cache = {}


# ---------------- KPI Constants ----------------
CONDITION_OF_DISCHARGE_COL = "condition_of_discharge_discharge_summary"
ENROLLMENT_STATUS_COL = "enrollment_status"
ENROLLMENT_DATE_COL = "enrollment_date"
DISCHARGE_DATE_COL = "enrollment_date"

# Empty/Null indicators
EMPTY_INDICATORS = ["", "nan", "None", "null", "N/A", "n/a", "na", "NA", "undefined"]

# Enrollment status values
COMPLETED_STATUSES = ["COMPLETED", "COMPLETE"]


def compute_missing_cod_count(df, facility_uids=None):
    """
    Count mothers meeting the Missing COD criteria:
    Condition 1: CoD is missing AND status is Complete
    OR
    Condition 2: CoD is missing AND enrollment is > 14 days old
    """
    cache_key = get_missing_cod_cache_key(df, facility_uids, "missing_count")

    if cache_key in st.session_state.missing_cod_cache:
        return st.session_state.missing_cod_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if filtered_df.empty:
            result = 0
        else:
            # Check for missing condition of discharge
            if (
                CONDITION_OF_DISCHARGE_COL in filtered_df.columns
                and ENROLLMENT_DATE_COL in filtered_df.columns
            ):
                cod_vals = (
                    filtered_df[CONDITION_OF_DISCHARGE_COL].astype(str).str.strip()
                )
                cod_missing = (
                    cod_vals.isin(EMPTY_INDICATORS)
                    | (cod_vals == "")
                    | (cod_vals.str.upper() == "N/A")
                )

                # Status masks
                status_complete = pd.Series(False, index=filtered_df.index)
                status_active = pd.Series(False, index=filtered_df.index)
                
                if ENROLLMENT_STATUS_COL in filtered_df.columns:
                    status_vals = (
                        filtered_df[ENROLLMENT_STATUS_COL].astype(str).str.strip().str.upper()
                    )
                    status_complete = status_vals.isin(COMPLETED_STATUSES)
                    status_active = status_vals == "ACTIVE"

                # Condition 2: Enrollment older than 14 days
                enrollment_dates = pd.to_datetime(
                    filtered_df[ENROLLMENT_DATE_COL], errors="coerce"
                )
                today = pd.Timestamp(dt.date.today())
                older_than_14_days = enrollment_dates <= (today - pd.Timedelta(days=14))

                # Logic: Missing CoD AND (Complete OR (Active AND >14 days))
                missing_mask = cod_missing & (status_complete | (status_active & older_than_14_days))

                # Count UNIQUE TEI IDs
                if "tei_id" in filtered_df.columns:
                    result = filtered_df[missing_mask]["tei_id"].dropna().nunique()
                else:
                    result = int(missing_mask.sum())
            else:
                result = 0

    st.session_state.missing_cod_cache[cache_key] = result
    return result


def compute_missing_cod_rate(df, facility_uids=None):
    """
    Compute Missing Condition of Discharge Rate
    Returns: (rate, missing_cases, total_enrolled)
    """
    cache_key = get_missing_cod_cache_key(df, facility_uids, "missing_rate")

    if cache_key in st.session_state.missing_cod_cache:
        return st.session_state.missing_cod_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Count missing cases based on new logic
        missing_cases = compute_missing_cod_count(df, facility_uids)

        # Get total ENROLLED mothers - COUNT UNIQUE TEI IDs
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Filter to only include mothers with enrollment dates
        if ENROLLMENT_DATE_COL in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[ENROLLMENT_DATE_COL].notna()].copy()

        # Count UNIQUE TEI IDs
        if "tei_id" in filtered_df.columns:
            total_enrolled = filtered_df["tei_id"].dropna().nunique()
        else:
            total_enrolled = len(filtered_df)

        # Calculate rate
        rate = (
            (missing_cases / total_enrolled * 100) if total_enrolled > 0 else 0.0
        )
        result = (float(rate), int(missing_cases), int(total_enrolled))

    st.session_state.missing_cod_cache[cache_key] = result
    return result


def compute_missing_cod_kpi(df, facility_uids=None):
    """
    Compute Missing Condition of Discharge KPI data
    """
    rate, missing_cases, total_enrolled = compute_missing_cod_rate(
        df, facility_uids
    )

    return {
        "missing_cod_rate": float(rate),
        "missing_cod_cases": int(missing_cases),
        "total_enrolled_mothers": int(total_enrolled),
    }


def get_numerator_denominator_for_missing_cod(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Missing Condition of Discharge KPI
    Uses ENROLLMENT DATE for period filtering and denominator
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Use enrollment date for period filtering
    date_column = ENROLLMENT_DATE_COL

    if date_column not in filtered_df.columns:
        return (0, 0, 0.0)

    # Filter by enrollment date existence
    filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
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

    # Denominator: Total UNIQUE enrolled mothers in this period
    if "tei_id" in filtered_df.columns:
        denominator = filtered_df["tei_id"].dropna().nunique()
    else:
        denominator = len(filtered_df)

    # Numerator: Count mothers meeting Condition 1 or Condition 2
    numerator = compute_missing_cod_count(filtered_df, facility_uids)

    # Calculate rate
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0

    return (numerator, denominator, rate)


# ---------------- Chart Functions WITH TABLES ----------------
def render_missing_cod_trend_chart(*args, **kwargs):
    """Render trend chart for Missing COD using standard utility"""
    return render_trend_chart(*args, **kwargs)

def render_missing_cod_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for Missing COD using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)

def render_missing_cod_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for Missing COD using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_missing_cod_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Missing Condition of Discharge trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    SAME AS ASSISTED FUNCTION
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Missing Condition of Discharge (enrollment_date)
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
def get_missing_cod_kpi_data(df, facility_uids=None):
    """
    Main function to get Missing Condition of Discharge KPI data for dashboard
    """
    # Get date range filters from session state if available
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Compute KPI
    rate, missing_cases, total_enrolled = compute_missing_cod_rate(
        df, facility_uids
    )

    return {
        "missing_cod_rate": float(rate),
        "missing_cod_cases": int(missing_cases),
        "total_mothers": int(total_enrolled),
    }
