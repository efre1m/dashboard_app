import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import datetime as dt
import hashlib

# Import utility functions
from utils.kpi_utils import (
    auto_text_color,
    format_period_month_year,
    compute_total_deliveries,
    get_relevant_date_column_for_kpi,
    render_trend_chart,
    render_facility_comparison_chart,
    render_region_comparison_chart,
    DELIVERY_MODE_COL,
    DELIVERY_DATE_COL,
)
from utils.kpi_svd import compute_svd_count
from utils.kpi_assisted import compute_assisted_count

# ---------------- Caching Setup ----------------
if "episiotomy_cache" not in st.session_state:
    st.session_state.episiotomy_cache = {}


def get_episiotomy_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Episiotomy computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_episiotomy_cache():
    """Clear the Episiotomy cache"""
    st.session_state.episiotomy_cache = {}


# Episiotomy KPI Configuration
EPISIOTOMY_COL = "episiotomy_performed_delivery_summary"
EPISIOTOMY_YES_CODE = 1


def compute_episiotomy_count(df, facility_uids=None):
    """
    Count Episiotomy occurrences - only where episiotomy_performed_delivery_summary is 1
    """
    cache_key = get_episiotomy_cache_key(df, facility_uids, "episiotomy_count")

    if cache_key in st.session_state.episiotomy_cache:
        return st.session_state.episiotomy_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if EPISIOTOMY_COL not in filtered_df.columns:
            result = 0
        else:
            # Count where value is 1 (Yes)
            filtered_df["episiotomy_numeric"] = pd.to_numeric(
                filtered_df[EPISIOTOMY_COL], errors="coerce"
            )

            # --- Refined Logic: Ensure the case belongs to a Vaginal Delivery ---
            # (Matches the logic in compute_assisted_count and the denominator)
            if DELIVERY_MODE_COL in filtered_df.columns:
                filtered_df["mode_numeric"] = pd.to_numeric(
                    filtered_df[DELIVERY_MODE_COL].astype(str).str.split(".").str[0], errors="coerce"
                )
                
                # Check for instrumental column
                from utils.kpi_assisted import INSTRUMENTAL_DELIVERY_COL
                inst_col = INSTRUMENTAL_DELIVERY_COL if INSTRUMENTAL_DELIVERY_COL in filtered_df.columns else None
                
                if inst_col:
                    filtered_df["inst_numeric"] = pd.to_numeric(
                        filtered_df[inst_col].astype(str).str.split(".").str[0], errors="coerce"
                    )
                else:
                    filtered_df["inst_numeric"] = 0
                
                # A valid vaginal delivery is: SVD (1) OR (Instrumental (1) AND Mode is EMPTY)
                is_vaginal = (filtered_df["mode_numeric"] == 1) | (
                    (filtered_df["inst_numeric"] == 1) & (filtered_df["mode_numeric"].isna())
                )
                
                episiotomy_mask = (filtered_df["episiotomy_numeric"] == 1) & is_vaginal
            else:
                # Fallback if mode column missing
                episiotomy_mask = filtered_df["episiotomy_numeric"] == 1
                
            result = int(episiotomy_mask.sum())

    st.session_state.episiotomy_cache[cache_key] = result
    return result


def compute_episiotomy_rate(df, facility_uids=None):
    """
    Compute Episiotomy Rate
    Numerator: Episiotomy Count
    Denominator: Total Vaginal Deliveries (SVD + Instrumental)
    Returns: (rate, episiotomy_cases, total_vaginal_deliveries)
    """
    cache_key = get_episiotomy_cache_key(df, facility_uids, "episiotomy_rate")

    if cache_key in st.session_state.episiotomy_cache:
        return st.session_state.episiotomy_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Numerator
        episiotomy_cases = compute_episiotomy_count(df, facility_uids)

        # Denominator: SVD + Instrumental
        svd_count = compute_svd_count(df, facility_uids)
        assisted_count = compute_assisted_count(df, facility_uids)
        total_vaginal_deliveries = svd_count + assisted_count

        # Calculate rate
        rate = (
            (episiotomy_cases / total_vaginal_deliveries * 100) 
            if total_vaginal_deliveries > 0 else 0.0
        )
        result = (float(rate), int(episiotomy_cases), int(total_vaginal_deliveries))

    st.session_state.episiotomy_cache[cache_key] = result
    return result


def compute_episiotomy_kpi(df, facility_uids=None):
    """
    Compute Episiotomy KPI data for dashboard
    """
    rate, cases, total = compute_episiotomy_rate(df, facility_uids)

    return {
        "episiotomy_rate": float(rate),
        "episiotomy_cases": int(cases),
        "total_vaginal_deliveries": int(total),
    }


def get_numerator_denominator_for_episiotomy(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for Episiotomy KPI with date range filtering
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column (enrollment_date)
    date_column = get_relevant_date_column_for_kpi("Episiotomy Rate (%)")

    # Filter to only include rows that have this specific date
    if date_column in filtered_df.columns:
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

    # Compute rate on filtered data
    rate, cases, total = compute_episiotomy_rate(filtered_df, facility_uids)

    return (cases, total, rate)


# ---------------- Chart Functions ----------------
def render_episiotomy_trend_chart(*args, **kwargs):
    return render_trend_chart(*args, **kwargs)


def render_episiotomy_facility_comparison_chart(*args, **kwargs):
    return render_facility_comparison_chart(*args, **kwargs)


def render_episiotomy_region_comparison_chart(*args, **kwargs):
    return render_region_comparison_chart(*args, **kwargs)


def prepare_data_for_episiotomy_trend(df, kpi_name, facility_uids=None, date_range_filters=None):
    """
    Prepare patient-level data for Episiotomy trend chart
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    date_column = get_relevant_date_column_for_kpi(kpi_name)

    if date_column not in filtered_df.columns:
        if "enrollment_date" in filtered_df.columns:
            date_column = "enrollment_date"
        else:
            return pd.DataFrame(), date_column

    result_df = filtered_df.copy()
    result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")

    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            result_df = result_df[
                (result_df["event_date"] >= start_dt)
                & (result_df["event_date"] < end_dt)
            ].copy()

    result_df = result_df[result_df["event_date"].notna()].copy()

    if result_df.empty:
        return pd.DataFrame(), date_column

    period_label = st.session_state.get("period_label", "Monthly")
    if "filters" in st.session_state and "period_label" in st.session_state.filters:
        period_label = st.session_state.filters["period_label"]

    from utils.time_filter import assign_period
    result_df = assign_period(result_df, "event_date", period_label)

    return result_df, date_column
