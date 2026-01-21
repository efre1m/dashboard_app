import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import datetime as dt
import hashlib

# Import shared utilities
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
if "arv_cache" not in st.session_state:
    st.session_state.arv_cache = {}


def get_arv_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for ARV computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_arv_cache():
    """Clear the ARV cache"""
    st.session_state.arv_cache = {}


# ---------------- ARV KPI Configuration - UPDATED FOR SINGLE-PATIENT PER ROW ----------------
# Column names for single-patient-per-row dataset
HIV_RESULT_COL = "hiv_result_delivery_summary"
ARV_RX_COL = "arv_rx_for_newborn_by_type_pp_postpartum_care"
BIRTH_OUTCOME_COL = "birth_outcome_delivery_summary"
NUMBER_OF_NEWBORNS_COL = "number_of_newborns_delivery_summary"
OTHER_NUMBER_OF_NEWBORNS_COL = "other_number_of_newborns_delivery_summary"
ARV_DATE_COL = "enrollment_date"

# Code values
HIV_POSITIVE_CODES = {"1"}  # HIV positive result
BIRTH_OUTCOME_ALIVE = "1"  # Alive birth outcome


def compute_hiv_exposed_infants(df, facility_uids=None):
    """
    Compute denominator for ARV KPI: Count of live infants born to HIV+ women
    For single-patient-per-row dataset

    Returns: Count of HIV-exposed infants (live infants born to HIV+ mothers)
    """
    cache_key = get_arv_cache_key(df, facility_uids, "hiv_exposed_infants")

    if cache_key in st.session_state.arv_cache:
        return st.session_state.arv_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Step 1: Check if we have the HIV column
        if HIV_RESULT_COL not in filtered_df.columns:
            st.warning(f"❌ HIV result column '{HIV_RESULT_COL}' not found in dataset")
            result = 0
        else:
            # Step 2: Identify HIV+ mothers - FLEXIBLE MATCHING
            df_copy = filtered_df.copy()

            # Convert HIV result to string
            df_copy["hiv_str"] = (
                df_copy[HIV_RESULT_COL].astype(str).str.lower().str.strip()
            )

            # Define patterns for HIV positive
            hiv_positive_patterns = [
                r"^1$",  # Exact "1"
                r"^1\.",  # "1.0", "1.00"
                r"positive",  # Contains "positive"
                r"^pos$",  # Exact "pos"
                r"^yes$",  # Exact "yes"
                r"hiv\+",  # "hiv+"
                r"infected",  # Contains "infected"
                r"reactive",  # Contains "reactive"
            ]

            # Create mask for HIV+
            hiv_positive_mask = pd.Series(False, index=df_copy.index)
            for pattern in hiv_positive_patterns:
                hiv_positive_mask = hiv_positive_mask | df_copy["hiv_str"].str.contains(
                    pattern, na=False
                )

            # Also check for numeric 1
            try:
                df_copy["hiv_numeric"] = pd.to_numeric(
                    df_copy[HIV_RESULT_COL], errors="coerce"
                )
                numeric_hiv_mask = df_copy["hiv_numeric"] == 1
                hiv_positive_mask = hiv_positive_mask | numeric_hiv_mask.fillna(False)
            except:
                pass

            hiv_positive_df = df_copy[hiv_positive_mask].copy()

            # DEBUG: Log findings
            total_rows = len(df_copy)
            hiv_positive_count = len(hiv_positive_df)

            if hiv_positive_count == 0:
                # Try to see what HIV values we actually have
                unique_hiv_vals = df_copy["hiv_str"].dropna().unique()[:20]
                print(f"⚠️ No HIV+ mothers found. Total rows: {total_rows}")
                print(f"   Sample HIV values: {list(unique_hiv_vals)}")
                result = 0
            else:
                print(
                    f"✅ Found {hiv_positive_count} HIV+ mothers out of {total_rows} total"
                )

                # Step 3: Count infants - SIMPLIFIED LOGIC
                total_infants = 0

                for idx, row in hiv_positive_df.iterrows():
                    # Count newborns with error handling
                    try:
                        newborns = 0

                        # Primary newborn count
                        if NUMBER_OF_NEWBORNS_COL in row and pd.notna(
                            row[NUMBER_OF_NEWBORNS_COL]
                        ):
                            val = row[NUMBER_OF_NEWBORNS_COL]
                            if pd.notna(val):
                                newborns += int(float(val))

                        # Other newborn count
                        if OTHER_NUMBER_OF_NEWBORNS_COL in row and pd.notna(
                            row[OTHER_NUMBER_OF_NEWBORNS_COL]
                        ):
                            val = row[OTHER_NUMBER_OF_NEWBORNS_COL]
                            if pd.notna(val):
                                newborns += int(float(val))

                        # If still 0, assume at least 1 infant for HIV+ mother
                        if newborns == 0:
                            newborns = 1

                        total_infants += newborns

                    except Exception as e:
                        # If error, assume 1 infant
                        total_infants += 1
                        print(f"   Row {idx}: Error counting newborns, assuming 1: {e}")

                result = total_infants
                print(f"✅ Total HIV-exposed infants: {result}")

    st.session_state.arv_cache[cache_key] = result
    return result


def compute_arv_cases(df, facility_uids=None):
    """
    Compute numerator for ARV KPI: Count of HIV-exposed infants who received ARV prophylaxis

    Returns: Count where ARV Rx for newborn field has a valid value
    """
    cache_key = get_arv_cache_key(df, facility_uids, "arv_cases")

    if cache_key in st.session_state.arv_cache:
        return st.session_state.arv_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if ARV_RX_COL not in filtered_df.columns:
            st.warning(f"❌ ARV Rx column '{ARV_RX_COL}' not found")
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string and clean
            df_copy["arv_str"] = df_copy[ARV_RX_COL].astype(str).str.lower().str.strip()

            # Valid ARV codes (expanded)
            valid_arv_codes = {
                "1",
                "2",
                "3",
                "4",  # Numeric codes
                "nvp",
                "azt",  # Text codes
                "arv",
                "prophylaxis",
            }  # General terms

            # Count rows with valid ARV values
            arv_mask = (
                df_copy["arv_str"].notna()
                & (df_copy["arv_str"] != "")
                & (df_copy["arv_str"] != "nan")
                & (df_copy["arv_str"] != "null")
                & (df_copy["arv_str"] != "0")
                & (df_copy["arv_str"] != "no")
                & (df_copy["arv_str"] != "none")
                & ~df_copy["arv_str"].str.contains("not", na=False)
                & ~df_copy["arv_str"].str.contains("na", na=False)
            )

            # Also check if it contains valid codes
            for code in valid_arv_codes:
                arv_mask = arv_mask | df_copy["arv_str"].str.contains(code, na=False)

            result = int(arv_mask.sum())

            # DEBUG
            print(f"✅ ARV cases found: {result}")
            if result > 0:
                sample_arv_vals = df_copy[arv_mask]["arv_str"].unique()[:10]
                print(f"   Sample ARV values: {list(sample_arv_vals)}")

    st.session_state.arv_cache[cache_key] = result
    return result


def compute_arv_rate(df, facility_uids=None):
    """
    Compute ARV Prophylaxis Rate
    Returns: (rate, arv_cases, hiv_exposed_infants)
    """
    cache_key = get_arv_cache_key(df, facility_uids, "arv_rate")

    if cache_key in st.session_state.arv_cache:
        return st.session_state.arv_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for ARV (postpartum care)
        date_column = get_relevant_date_column_for_kpi("ARV Prophylaxis Rate (%)")

        # Use enrollment_date
        if date_column not in df.columns and ARV_DATE_COL in df.columns:
            date_column = ARV_DATE_COL

        # Count ARV cases
        arv_cases = compute_arv_cases(df, facility_uids)

        # Get HIV-exposed infants
        hiv_exposed_infants = compute_hiv_exposed_infants(df, facility_uids)

        # Calculate rate
        rate = (
            (arv_cases / hiv_exposed_infants * 100) if hiv_exposed_infants > 0 else 0.0
        )
        result = (rate, arv_cases, hiv_exposed_infants)

    st.session_state.arv_cache[cache_key] = result
    return result


def compute_arv_kpi(df, facility_uids=None):
    """
    Compute ARV KPI data
    Returns: Dictionary with ARV metrics
    """
    rate, arv_cases, hiv_exposed_infants = compute_arv_rate(df, facility_uids)

    return {
        "arv_rate": float(rate),
        "arv_cases": int(arv_cases),
        "hiv_exposed_infants": int(hiv_exposed_infants),
    }


def get_numerator_denominator_for_arv(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for ARV KPI
    WITH DATE RANGE FILTERING
    Returns: (numerator, denominator, rate)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the date column for ARV (enrollment_date)
    date_column = get_relevant_date_column_for_kpi("ARV Prophylaxis Rate (%)")

    # For single-patient per row, use enrollment_date
    if date_column not in filtered_df.columns and "enrollment_date" in filtered_df.columns:
        date_column = "enrollment_date"

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

    # Compute ARV rate on date-filtered data
    rate, arv_cases, hiv_exposed_infants = compute_arv_rate(filtered_df, facility_uids)

    return (arv_cases, hiv_exposed_infants, rate)


# ---------------- Chart Functions WITH TABLES ----------------
# ---------------- Chart Functions WITH TABLES ----------------
def render_arv_trend_chart(*args, **kwargs):
    """Render trend chart for ARV using standard utility"""
    return render_trend_chart(*args, **kwargs)


def render_arv_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for ARV using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)


def render_arv_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for ARV using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


def render_arv_type_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """
    Render a pie chart showing distribution of ARV types for HIV-exposed infants
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_arv_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for ARV trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the date column for ARV (enrollment_date)
    date_column = get_relevant_date_column_for_kpi(kpi_name)

    # For single-patient per row, use enrollment_date
    if date_column not in filtered_df.columns and "enrollment_date" in filtered_df.columns:
        date_column = "enrollment_date"

    # Check if the date column exists
    if date_column not in filtered_df.columns:
        # Try to use enrollment_date as fallback
        if "enrollment_date" in filtered_df.columns:
            date_column = "enrollment_date"
            st.warning(f"⚠️ ARV date column not found, using 'enrollment_date' instead")
        else:
            st.warning(f"⚠️ Required date column '{date_column}' not found for ARV data")
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
        st.info(f"⚠️ No data with valid dates in '{date_column}' for ARV")
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
