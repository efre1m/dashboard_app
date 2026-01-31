"""
KPI: Missing Uterotonics Documentation
Measures: Percentage of mothers where the uterotonics given delivery summary variable is empty or N/A
Formula: (Mothers with missing uterotonics documentation) ÷ (Total deliveries) × 100
"""

import pandas as pd
import streamlit as st
import hashlib

# Import only utility functions from kpi_utils
from utils.kpi_utils import (
    get_relevant_date_column_for_kpi,
    render_trend_chart,
    render_facility_comparison_chart,
    render_region_comparison_chart,
)

# ---------------- Caching Setup ----------------
if "missing_uterotonic_cache" not in st.session_state:
    st.session_state.missing_uterotonic_cache = {}


def get_missing_uterotonic_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Missing Uterotonics computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_missing_uterotonic_cache():
    """Clear the Missing Uterotonics cache"""
    st.session_state.missing_uterotonic_cache = {}


# ---------------- KPI Constants ----------------
UTEROTONIC_COL = "uterotonics_given_delivery_summary"
DELIVERY_DATE_COL = "enrollment_date"

# Empty/Null indicators
EMPTY_INDICATORS = ["", "nan", "None", "null", "N/A", "n/a", "na", "NA", "undefined"]


def compute_missing_uterotonic_count(df, facility_uids=None):
    """
    Count missing uterotonics documentation - VECTORIZED for performance
    """
    cache_key = get_missing_uterotonic_cache_key(df, facility_uids, "missing_count")

    if cache_key in st.session_state.missing_uterotonic_cache:
        return st.session_state.missing_uterotonic_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Filter to only include deliveries with dates
        date_column = get_relevant_date_column_for_kpi("Missing Uterotonics Given at Delivery")
        if date_column in filtered_df.columns:
            filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
            filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        if filtered_df.empty:
            result = 0
        else:
            if UTEROTONIC_COL in filtered_df.columns:
                vals = filtered_df[UTEROTONIC_COL].astype(str).str.strip()
                # Check for empty indicators or just "N/A"
                missing_mask = vals.isin(EMPTY_INDICATORS) | (vals == "") | (vals.str.upper() == "N/A")
                result = int(missing_mask.sum())
            else:
                result = len(filtered_df) # All missing if column missing

    st.session_state.missing_uterotonic_cache[cache_key] = result
    return result


def compute_missing_uterotonic_rate(df, facility_uids=None):
    """
    Compute Missing Uterotonics Documentation Rate
    Returns: (rate, missing_cases, total_deliveries)
    """
    cache_key = get_missing_uterotonic_cache_key(df, facility_uids, "missing_rate")

    if cache_key in st.session_state.missing_uterotonic_cache:
        return st.session_state.missing_uterotonic_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Count missing cases
        missing_cases = compute_missing_uterotonic_count(df, facility_uids)

        # Get total deliveries
        from utils.kpi_utils import compute_total_deliveries
        total_deliveries = compute_total_deliveries(df, facility_uids, "enrollment_date")

        # Calculate rate
        rate = (missing_cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (float(rate), int(missing_cases), int(total_deliveries))

    st.session_state.missing_uterotonic_cache[cache_key] = result
    return result


def compute_missing_uterotonic_kpi(df, facility_uids=None):
    """
    Compute Missing Uterotonics KPI data
    """
    rate, missing_cases, total_deliveries = compute_missing_uterotonic_rate(df, facility_uids)

    return {
        "missing_uterotonic_rate": float(rate),
        "missing_uterotonic_cases": int(missing_cases),
        "total_deliveries": int(total_deliveries),
    }


def get_numerator_denominator_for_missing_uterotonic(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Missing Uterotonics KPI WITH DATE RANGE FILTERING
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column
    date_column = get_relevant_date_column_for_kpi("Missing Uterotonics Given at Delivery")

    # Filter to only include rows that have the delivery date
    if date_column in filtered_df.columns:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()
    else:
        return (0, 0, 0.0)

    # Apply date range filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            filtered_df = filtered_df[(filtered_df[date_column] >= start_dt) & (filtered_df[date_column] < end_dt)].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    numerator = compute_missing_uterotonic_count(filtered_df, facility_uids)
    
    from utils.kpi_utils import compute_total_deliveries
    denominator = compute_total_deliveries(filtered_df, facility_uids, date_column)
    
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0

    return (numerator, denominator, rate)


def render_missing_uterotonic_trend_chart(*args, **kwargs):
    return render_trend_chart(*args, **kwargs)


def render_missing_uterotonic_facility_comparison_chart(*args, **kwargs):
    return render_facility_comparison_chart(*args, **kwargs)


def render_missing_uterotonic_region_comparison_chart(*args, **kwargs):
    return render_region_comparison_chart(*args, **kwargs)


def get_missing_uterotonic_kpi_data(df, facility_uids=None):
    rate, missing_cases, total_deliveries = compute_missing_uterotonic_rate(df, facility_uids)
    return {
        "missing_uterotonic_rate": float(rate),
        "missing_uterotonic_cases": int(missing_cases),
        "total_deliveries": int(total_deliveries),
    }
