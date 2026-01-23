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

# Birth outcome columns for multiple newborns
BIRTH_OUTCOME_NEWBORN_1_COL = "birth_outcome_newborn_delivery_summary"
BIRTH_OUTCOME_NEWBORN_2_COL = "birth_outcome_newborn_2_delivery_summary"
BIRTH_OUTCOME_NEWBORN_3_COL = "birth_outcome_newborn_3_delivery_summary"
BIRTH_OUTCOME_NEWBORN_4_COL = "birth_outcome_newborn_4_delivery_summary"

# Number of newborns columns
NUMBER_OF_NEWBORNS_COL = "number_of_newborns_delivery_summary"
OTHER_NUMBER_OF_NEWBORNS_COL = "other_number_of_newborns_delivery_summary"

# Empty/Null indicators
EMPTY_INDICATORS = ["", "nan", "None", "null", "N/A", "n/a", "na", "NA", "undefined"]


def compute_missing_bo_count(df, facility_uids=None):
    """
    Count missing birth outcome documentation across all newborns
    Checks appropriate number of birth outcome columns based on number_of_newborns
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
            df_copy = filtered_df.copy()
            
            # Initialize columns if they don't exist
            if NUMBER_OF_NEWBORNS_COL not in df_copy.columns:
                df_copy[NUMBER_OF_NEWBORNS_COL] = 0
            if OTHER_NUMBER_OF_NEWBORNS_COL not in df_copy.columns:
                df_copy[OTHER_NUMBER_OF_NEWBORNS_COL] = 0
            
            # Convert number of newborns to numeric
            df_copy[NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
                df_copy[NUMBER_OF_NEWBORNS_COL], errors="coerce"
            ).fillna(0)
            df_copy[OTHER_NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
                df_copy[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce"
            ).fillna(0)
            
            # Determine number of newborns (use OTHER if primary is empty)
            df_copy["num_newborns"] = df_copy[NUMBER_OF_NEWBORNS_COL]
            df_copy.loc[df_copy["num_newborns"] == 0, "num_newborns"] = df_copy[OTHER_NUMBER_OF_NEWBORNS_COL]
            
            # List of birth outcome columns to check
            birth_outcome_cols = [
                BIRTH_OUTCOME_NEWBORN_1_COL,
                BIRTH_OUTCOME_NEWBORN_2_COL,
                BIRTH_OUTCOME_NEWBORN_3_COL,
                BIRTH_OUTCOME_NEWBORN_4_COL,
            ]
            
            # SIMPLIFIED LOGIC:
            total_missing = 0
            
            for idx, row in df_copy.iterrows():
                # Get number of babies (Prioritized Logic)
                n1 = pd.to_numeric(row[NUMBER_OF_NEWBORNS_COL], errors="coerce")
                n2 = pd.to_numeric(row[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce")
                
                n1_val = int(n1) if pd.notna(n1) and n1 > 0 else 0
                n2_val = int(n2) if pd.notna(n2) and n2 > 0 else 0
                
                count_was_missing = (n1_val == 0 and n2_val == 0)
                
                if n1_val > 0:
                    num_newborns = n1_val
                elif n2_val > 0:
                    num_newborns = n2_val
                else:
                    num_newborns = 1
                
                # For each baby, check its outcome
                for i in range(min(num_newborns, 4)):
                    outcome_val = None
                    
                    if i == 0:
                        # Baby 1: PER USER REQUEST - Always use the main General column
                        gen_col = BIRTH_OUTCOME_COL
                        if gen_col in row and pd.notna(row[gen_col]):
                            outcome_val = str(row[gen_col])
                    else:
                        # Babies 2-4: PER USER REQUEST - Use specific newborn columns 2, 3, or 4
                        col = birth_outcome_cols[i]
                        if col in row and pd.notna(row[col]):
                            outcome_val = str(row[col])
                    
                    # Check if missing
                    cleaned_val = str(outcome_val).strip() if outcome_val else ""
                    if cleaned_val in EMPTY_INDICATORS or cleaned_val == "" or cleaned_val.upper() == "N/A":
                        total_missing += 1
            
            result = int(total_missing)

    st.session_state.missing_bo_cache[cache_key] = result
    return result


def compute_total_newborns_for_missing_bo(df, facility_uids=None):
    """Count total newborns using number_of_newborns columns"""
    if df is None or df.empty:
        return 0
    
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
    
    # Filter to only include deliveries with enrollment dates
    if DELIVERY_DATE_COL in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[DELIVERY_DATE_COL].notna()].copy()
    
    # Initialize columns if they don't exist
    if NUMBER_OF_NEWBORNS_COL not in filtered_df.columns:
        filtered_df[NUMBER_OF_NEWBORNS_COL] = 0
    if OTHER_NUMBER_OF_NEWBORNS_COL not in filtered_df.columns:
        filtered_df[OTHER_NUMBER_OF_NEWBORNS_COL] = 0
    
    # Convert to numeric
    filtered_df[NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
        filtered_df[NUMBER_OF_NEWBORNS_COL], errors="coerce"
    ).fillna(0)
    filtered_df[OTHER_NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
        filtered_df[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce"
    ).fillna(0)
    
    # Prioritized Logic:
    # 1. Main column (n1)
    # 2. Other column (n2) if n1 is missing/zero
    # 3. Default to 1 if both are missing/zero
    total = 0
    for idx, row in filtered_df.iterrows():
        n1 = pd.to_numeric(row[NUMBER_OF_NEWBORNS_COL], errors="coerce")
        n2 = pd.to_numeric(row[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce")
        
        n1_val = int(n1) if pd.notna(n1) and n1 > 0 else 0
        n2_val = int(n2) if pd.notna(n2) and n2 > 0 else 0
        
        if n1_val > 0:
            row_count = n1_val
        elif n2_val > 0:
            row_count = n2_val
        else:
            row_count = 1
            
        total += row_count
        
    return int(total)


def compute_missing_bo_rate(df, facility_uids=None):
    """
    Compute Missing Birth Outcome Rate
    Returns: (rate, missing_cases, total_newborns)
    """
    cache_key = get_missing_bo_cache_key(df, facility_uids, "missing_rate")

    if cache_key in st.session_state.missing_bo_cache:
        return st.session_state.missing_bo_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Count missing cases
        missing_cases = compute_missing_bo_count(df, facility_uids)

        # Get total newborns - ONLY COUNT ROWS THAT HAVE ENROLLMENT DATE
        total_newborns = compute_total_newborns_for_missing_bo(df, facility_uids)

        # Calculate rate
        rate = (missing_cases / total_newborns * 100) if total_newborns > 0 else 0.0
        result = (float(rate), int(missing_cases), int(total_newborns))

    st.session_state.missing_bo_cache[cache_key] = result
    return result


def compute_missing_bo_kpi(df, facility_uids=None):
    """
    Compute Missing Birth Outcome KPI data
    This is the function your dashboard will call
    """
    rate, missing_cases, total_newborns = compute_missing_bo_rate(df, facility_uids)

    return {
        "missing_bo_rate": float(rate),
        "missing_bo_cases": int(missing_cases),
        "total_newborns": int(total_newborns),
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
    # Check all birth outcome columns based on number of newborns
    df_copy = filtered_df.copy()
    
    # Initialize columns if they don't exist
    if NUMBER_OF_NEWBORNS_COL not in df_copy.columns:
        df_copy[NUMBER_OF_NEWBORNS_COL] = 0
    if OTHER_NUMBER_OF_NEWBORNS_COL not in df_copy.columns:
        df_copy[OTHER_NUMBER_OF_NEWBORNS_COL] = 0
    
    # Convert number of newborns to numeric
    df_copy[NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
        df_copy[NUMBER_OF_NEWBORNS_COL], errors="coerce"
    ).fillna(0)
    df_copy[OTHER_NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
        df_copy[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce"
    ).fillna(0)
    
    # Determine number of newborns (use OTHER if primary is empty)
    df_copy["num_newborns"] = df_copy[NUMBER_OF_NEWBORNS_COL]
    df_copy.loc[df_copy["num_newborns"] == 0, "num_newborns"] = df_copy[OTHER_NUMBER_OF_NEWBORNS_COL]
    
    # List of birth outcome columns to check
    birth_outcome_cols = [
        BIRTH_OUTCOME_NEWBORN_1_COL,
        BIRTH_OUTCOME_NEWBORN_2_COL,
        BIRTH_OUTCOME_NEWBORN_3_COL,
        BIRTH_OUTCOME_NEWBORN_4_COL,
    ]
    
    total_missing = 0
    total_newborns = 0
    
    # For each row, check the appropriate number of birth outcome columns
    for idx, row in df_copy.iterrows():
        # Get number of babies (Prioritized Logic)
        n1 = pd.to_numeric(row[NUMBER_OF_NEWBORNS_COL], errors="coerce")
        n2 = pd.to_numeric(row[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce")
        
        n1_val = int(n1) if pd.notna(n1) and n1 > 0 else 0
        n2_val = int(n2) if pd.notna(n2) and n2 > 0 else 0
        
        count_was_missing = (n1_val == 0 and n2_val == 0)
        
        if n1_val > 0:
            num_babies = n1_val
        elif n2_val > 0:
            num_babies = n2_val
        else:
            num_babies = 1
        
        total_newborns += num_babies
        
        # Check outcomes for each baby
        for i in range(min(num_babies, 4)):
            outcome_val = None
            
            if i == 0:
                # Baby 1: PER USER REQUEST - Always use the main General column
                if BIRTH_OUTCOME_COL in row and pd.notna(row[BIRTH_OUTCOME_COL]):
                    outcome_val = str(row[BIRTH_OUTCOME_COL])
            else:
                # Babies 2-4: PER USER REQUEST - Use specific newborn columns 2, 3, or 4
                col = birth_outcome_cols[i]
                if col in row and pd.notna(row[col]):
                    outcome_val = str(row[col])
            
            # Check if missing (ensure outcome_val is cleaned)
            cleaned_val = str(outcome_val).strip() if outcome_val else ""
            if cleaned_val in EMPTY_INDICATORS or cleaned_val == "" or cleaned_val.upper() == "N/A":
                total_missing += 1
    
    numerator = int(total_missing)
    denominator = int(total_newborns)

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
    rate, missing_cases, total_newborns = compute_missing_bo_rate(df, facility_uids)

    return {
        "missing_bo_rate": float(rate),
        "missing_bo_cases": int(missing_cases),
        "total_newborns": int(total_newborns),
    }
