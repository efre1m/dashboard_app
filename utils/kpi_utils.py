import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import streamlit as st
import hashlib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ---------------- Caching Setup ----------------
if "kpi_cache" not in st.session_state:
    st.session_state.kpi_cache = {}


def get_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key based on data and filters"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_cache():
    """Clear the KPI cache - call this when you know data has changed"""
    st.session_state.kpi_cache = {}


# ---------------- Utility ----------------
def auto_text_color(bg):
    """Return black or white text depending on background brightness"""
    bg = bg.lstrip("#")
    try:
        r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return "#000000" if brightness > 150 else "#ffffff"
    except Exception:
        return "#000000"


def format_period_month_year(period_str):
    """Convert period string to proper month-year format (e.g., Sep-25)"""
    if not isinstance(period_str, str):
        return str(period_str)

    period_str = str(period_str).strip()

    # If already in month-year format like "Sep-25", return as is
    if len(period_str) == 6 and "-" in period_str:
        month_part, year_part = period_str.split("-")
        if len(month_part) == 3 and len(year_part) == 2:
            return period_str.capitalize()  # Capitalize month abbreviation

    # Try to parse various formats
    formats_to_try = [
        "%y-%b",  # "25-Aug" -> "Aug-25"
        "%Y-%b",  # "2025-Aug" -> "Aug-25"
        "%b-%y",  # "Aug-25" (already correct)
        "%B-%y",  # "August-25" -> "Aug-25"
        "%Y-%m",  # "2025-08" -> "Aug-25"
        "%m/%Y",  # "08/2025" -> "Aug-25"
        "%Y/%m",  # "2025/08" -> "Aug-25"
        "%Y-%m-%d",  # "2025-08-15" -> "Aug-25"
        "%d/%m/%Y",  # "15/08/2025" -> "Aug-25"
        "%m/%d/%Y",  # "08/15/2025" -> "Aug-25"
    ]

    for fmt in formats_to_try:
        try:
            dt_obj = dt.datetime.strptime(period_str, fmt)
            return dt_obj.strftime("%b-%y").capitalize()  # Convert to "Aug-25" format
        except (ValueError, TypeError):
            continue

    # If all parsing fails, return original
    return period_str


# ---------------- KPI Constants ----------------
# Patient-level data columns
FP_ACCEPTANCE_COL = "fp_counseling_and_method_provided_pp_postpartum_care"
FP_ACCEPTED_CODES = {"1", "2", "3", "4", "5"}

# Birth outcome columns
BIRTH_OUTCOME_COL = "birth_outcome_delivery_summary"
ALIVE_CODE = "1"
STILLBIRTH_CODE = "2"

# Delivery mode columns
DELIVERY_MODE_COL = "mode_of_delivery_maternal_delivery_summary"
CSECTION_CODE = "2"

# PNC timing columns
PNC_TIMING_COL = "date_stay_pp_postpartum_care"
PNC_EARLY_CODES = {"1", "2"}

# Condition of discharge columns
CONDITION_OF_DISCHARGE_COL = "condition_of_discharge_discharge_summary"
DEAD_CODE = "4"

# Number of newborns columns
NUMBER_OF_NEWBORNS_COL = "number_of_newborns_delivery_summary"
OTHER_NUMBER_OF_NEWBORNS_COL = "other_number_of_newborns_delivery_summary"

# Event date columns
DELIVERY_DATE_COL = "event_date_delivery_summary"
PNC_DATE_COL = "event_date_postpartum_care"
DISCHARGE_DATE_COL = "event_date_discharge_summary"

# Enrollment date column - KEPT FOR BACKWARD COMPATIBILITY ONLY
ENROLLMENT_DATE_COL = "enrollment_date"


def compute_birth_counts(df, facility_uids=None):
    """
    Compute birth counts accounting for multiple births (twins, triplets, etc.)
    Uses UID filtering
    Returns: total_births, live_births, stillbirths
    """
    cache_key = get_cache_key(df, facility_uids, "birth_counts")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0, 0, 0)
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        actual_events_df = filtered_df.copy()

        # Initialize columns with zeros if they don't exist
        if NUMBER_OF_NEWBORNS_COL not in actual_events_df.columns:
            actual_events_df[NUMBER_OF_NEWBORNS_COL] = 0

        if OTHER_NUMBER_OF_NEWBORNS_COL not in actual_events_df.columns:
            actual_events_df[OTHER_NUMBER_OF_NEWBORNS_COL] = 0

        if BIRTH_OUTCOME_COL not in actual_events_df.columns:
            actual_events_df[BIRTH_OUTCOME_COL] = np.nan

        # Convert to numeric and fill NaN
        actual_events_df[NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
            actual_events_df[NUMBER_OF_NEWBORNS_COL], errors="coerce"
        ).fillna(0)

        actual_events_df[OTHER_NUMBER_OF_NEWBORNS_COL] = pd.to_numeric(
            actual_events_df[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce"
        ).fillna(0)

        actual_events_df[BIRTH_OUTCOME_COL] = pd.to_numeric(
            actual_events_df[BIRTH_OUTCOME_COL], errors="coerce"
        )

        # Calculate total babies per row
        total_babies_per_row = (
            actual_events_df[NUMBER_OF_NEWBORNS_COL]
            + actual_events_df[OTHER_NUMBER_OF_NEWBORNS_COL]
        )

        # If total babies is 0 but birth outcome exists, count as 1 baby
        zero_babies_mask = (total_babies_per_row == 0) & actual_events_df[
            BIRTH_OUTCOME_COL
        ].notna()
        total_babies_per_row = total_babies_per_row.where(~zero_babies_mask, 1)

        # Vectorized outcome calculation
        outcome_mask = actual_events_df[BIRTH_OUTCOME_COL].notna()

        alive_mask = (
            actual_events_df[BIRTH_OUTCOME_COL] == float(ALIVE_CODE)
        ) & outcome_mask
        stillbirth_mask = (
            actual_events_df[BIRTH_OUTCOME_COL] == float(STILLBIRTH_CODE)
        ) & outcome_mask

        # For alive births: all babies are alive
        live_births = total_babies_per_row[alive_mask].sum()

        # For stillbirths: all babies are stillbirths
        stillbirths = total_babies_per_row[stillbirth_mask].sum()

        # Total births
        total_births = total_babies_per_row[outcome_mask].sum()

        result = (int(total_births), int(live_births), int(stillbirths))

    st.session_state.kpi_cache[cache_key] = result
    return result


# ---------------- SEPARATE NUMERATOR COMPUTATION FUNCTIONS ----------------
def compute_fp_acceptance_count(df, facility_uids=None):
    """Count FP acceptance occurrences - VECTORIZED - with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    actual_events_df = filtered_df.copy()

    if FP_ACCEPTANCE_COL not in actual_events_df.columns:
        return 0

    fp_series = actual_events_df[FP_ACCEPTANCE_COL].dropna()

    # Handle different data types
    if fp_series.dtype in [np.float64, np.int64]:
        fp_codes = fp_series.astype(int).astype(str)
    else:
        fp_codes = fp_series.astype(str).str.split(".").str[0]

    # Check if in accepted codes
    accepted_mask = fp_codes.isin(FP_ACCEPTED_CODES)

    return int(accepted_mask.sum())


def compute_early_pnc_count(df, facility_uids=None):
    """Count early PNC occurrences - VECTORIZED - with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    actual_events_df = filtered_df.copy()

    if PNC_TIMING_COL not in actual_events_df.columns:
        return 0

    pnc_series = actual_events_df[PNC_TIMING_COL].dropna()

    # Handle different data types
    if pnc_series.dtype in [np.float64, np.int64]:
        pnc_codes = pnc_series.astype(int).astype(str)
    else:
        pnc_codes = pnc_series.astype(str).str.split(".").str[0]

    # Check if in early codes
    early_mask = pnc_codes.isin(PNC_EARLY_CODES)

    return int(early_mask.sum())


def compute_csection_count(df, facility_uids=None):
    """Count C-section occurrences with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if DELIVERY_MODE_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Convert to string first, then extract numeric part
    df_copy["delivery_mode_clean"] = df_copy[DELIVERY_MODE_COL].astype(str)
    df_copy["delivery_mode_numeric"] = pd.to_numeric(
        df_copy["delivery_mode_clean"].str.split(".").str[0], errors="coerce"
    )

    # Count C-sections (value = 2)
    csection_mask = df_copy["delivery_mode_numeric"] == 2

    return int(csection_mask.sum())


def compute_maternal_death_count(df, facility_uids=None):
    """Count maternal death occurrences - VECTORIZED - with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    actual_events_df = filtered_df.copy()

    if CONDITION_OF_DISCHARGE_COL not in actual_events_df.columns:
        return 0

    condition_series = actual_events_df[CONDITION_OF_DISCHARGE_COL].dropna()

    # Convert to numeric and compare with DEAD_CODE
    condition_numeric = pd.to_numeric(condition_series, errors="coerce")
    death_mask = condition_numeric == float(DEAD_CODE)

    return int(death_mask.sum())


def compute_stillbirth_count(df, facility_uids=None):
    """Count stillbirth occurrences with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if BIRTH_OUTCOME_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Handle all possible data types: float, int, string, mixed
    df_copy["birth_outcome_clean"] = df_copy[BIRTH_OUTCOME_COL].astype(str)

    # Extract numeric part
    df_copy["birth_outcome_numeric"] = pd.to_numeric(
        df_copy["birth_outcome_clean"].str.split(".").str[0], errors="coerce"
    )

    # Count stillbirths (value = 2)
    stillbirth_mask = df_copy["birth_outcome_numeric"] == 2

    return int(stillbirth_mask.sum())


# ---------------- KPI Computation Functions ----------------
def compute_total_deliveries(df, facility_uids=None, date_column=None):
    """Count total deliveries - counts unique TEI IDs using UID filtering AND optional date filtering"""
    cache_key = get_cache_key(df, facility_uids, f"total_deliveries_{date_column}")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()

        # Filter by facility UIDs
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Filter by specific date column if provided
        if date_column and date_column in filtered_df.columns:
            # Convert to datetime and filter
            filtered_df[date_column] = pd.to_datetime(
                filtered_df[date_column], errors="coerce"
            )
            filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        if "tei_id" in filtered_df.columns:
            # Count unique TEI IDs that have this specific program stage date
            unique_tei_ids = filtered_df["tei_id"].dropna().nunique()
            result = unique_tei_ids
        else:
            result = len(filtered_df)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_fp_acceptance(df, facility_uids=None):
    cache_key = get_cache_key(df, facility_uids, "fp_acceptance")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    result = compute_fp_acceptance_count(df, facility_uids)
    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_stillbirth_rate(df, facility_uids=None):
    """Compute stillbirth rate (now as percentage, not per 1000)"""
    cache_key = get_cache_key(df, facility_uids, "stillbirth_rate")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)].copy()

        stillbirths = compute_stillbirth_count(df, facility_uids)
        total_deliveries = compute_total_deliveries(df, facility_uids)
        rate = (stillbirths / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (rate, stillbirths, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_early_pnc_coverage(df, facility_uids=None):
    cache_key = get_cache_key(df, facility_uids, "pnc_coverage")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)].copy()

        early_pnc = compute_early_pnc_count(df, facility_uids)
        total_deliveries = compute_total_deliveries(df, facility_uids)
        coverage = (early_pnc / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (coverage, early_pnc, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_maternal_death_rate(df, facility_uids=None):
    """Compute maternal death rate (now as percentage, not per 100,000)"""
    cache_key = get_cache_key(df, facility_uids, "maternal_death_rate")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)].copy()

        maternal_deaths = compute_maternal_death_count(df, facility_uids)
        total_deliveries = compute_total_deliveries(df, facility_uids)
        rate = (
            (maternal_deaths / total_deliveries * 100) if total_deliveries > 0 else 0.0
        )
        result = (rate, maternal_deaths, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_csection_rate(df, facility_uids=None):
    cache_key = get_cache_key(df, facility_uids, "csection_rate")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        csection_deliveries = compute_csection_count(df, facility_uids)
        total_deliveries = compute_total_deliveries(df, facility_uids)
        rate = (
            (csection_deliveries / total_deliveries * 100)
            if total_deliveries > 0
            else 0.0
        )
        result = (rate, csection_deliveries, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


# ---------------- Master KPI Function ----------------
def compute_kpis(df, facility_uids=None):
    """Compute all KPIs with optional date filtering"""
    cache_key = get_cache_key(df, facility_uids, "main_kpis")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # IMPORTANT: The compute_total_deliveries function now filters by date
    # Get the date column context if available (from get_numerator_denominator_for_kpi)
    date_column = None
    # We'll get this from the calling context

    total_deliveries = compute_total_deliveries(filtered_df, facility_uids, date_column)
    fp_acceptance = compute_fp_acceptance(filtered_df, facility_uids)
    ippcar = (fp_acceptance / total_deliveries * 100) if total_deliveries > 0 else 0.0

    stillbirth_rate, stillbirths, total_deliveries_sb = compute_stillbirth_rate(
        filtered_df, facility_uids
    )
    pnc_coverage, early_pnc, total_deliveries_pnc = compute_early_pnc_coverage(
        filtered_df, facility_uids
    )
    maternal_death_rate, maternal_deaths, total_deliveries_md = (
        compute_maternal_death_rate(filtered_df, facility_uids)
    )
    csection_rate, csection_deliveries, total_deliveries_cs = compute_csection_rate(
        filtered_df, facility_uids
    )
    total_births, live_births, stillbirths_count = compute_birth_counts(
        filtered_df, facility_uids
    )

    result = {
        "total_deliveries": int(total_deliveries),
        "fp_acceptance": int(fp_acceptance),
        "ippcar": float(ippcar),
        "stillbirth_rate": float(stillbirth_rate),
        "stillbirths": int(stillbirths),
        "total_deliveries_sb": int(total_deliveries),
        "pnc_coverage": float(pnc_coverage),
        "early_pnc": int(early_pnc),
        "total_deliveries_pnc": int(total_deliveries),
        "maternal_death_rate": float(maternal_death_rate),
        "maternal_deaths": int(maternal_deaths),
        "total_deliveries_md": int(total_deliveries),
        "live_births": int(live_births),
        "total_births": int(total_births),
        "stillbirths_count": int(stillbirths_count),
        "csection_rate": float(csection_rate),
        "csection_deliveries": int(csection_deliveries),
        "total_deliveries_cs": int(total_deliveries),
    }

    st.session_state.kpi_cache[cache_key] = result
    return result


# ---------------- Date Handling with Program Stage Specific Dates ----------------
def get_relevant_date_column_for_kpi(kpi_name):
    """
    Get the relevant event date column for a specific KPI based on program stage
    """
    # Mapping of KPIs to their program stages and date columns
    program_stage_date_mapping = {
        # Delivery Summary KPIs
        "Stillbirth Rate (%)": "event_date_delivery_summary",
        "C-Section Rate (%)": "event_date_delivery_summary",
        "Normal Vaginal Delivery (SVD) Rate (%)": "event_date_delivery_summary",
        "Postpartum Hemorrhage (PPH) Rate (%)": "event_date_delivery_summary",
        "Uterotonic Administration Rate (%)": "event_date_delivery_summary",
        "Missing Mode of Delivery Documentation Rate (%)": "event_date_delivery_summary",
        "ARV Prophylaxis Rate (%)": "event_date_postpartum_care",
        # Instrumental Delivery KPIs - UPDATED CORRECT COLUMN NAME
        "Assisted Delivery Rate (%)": "event_date_instrumental_delivery_form",
        # Postpartum Care KPIs
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": "event_date_postpartum_care",
        "FP Acceptance": "event_date_postpartum_care",
        "IPPCAR": "event_date_postpartum_care",
        "Early Postnatal Care (PNC) Coverage (%)": "event_date_postpartum_care",
        "PNC Coverage": "event_date_postpartum_care",
        "Postnatal Care": "event_date_postpartum_care",
        # Discharge Summary KPIs
        "Institutional Maternal Death Rate (%)": "event_date_discharge_summary",
        "Maternal Death Rate": "event_date_discharge_summary",
        "Low Birth Weight (LBW) Rate (%)": "event_date_delivery_summary",
    }

    # Try exact match first
    for key in program_stage_date_mapping:
        if key in kpi_name:
            return program_stage_date_mapping[key]

    # Fallback based on keywords
    if any(word in kpi_name for word in ["Assisted", "Instrumental"]):
        return "event_date_instrumental_delivery_form"
    elif any(
        word in kpi_name
        for word in ["Delivery", "Birth", "Section", "PPH", "Uterotonic", "SVD"]
    ):
        return "event_date_delivery_summary"
    elif any(
        word in kpi_name for word in ["PNC", "Postnatal", "Contraceptive", "Postpartum"]
    ):
        return "event_date_postpartum_care"
    elif any(word in kpi_name for word in ["Death", "Discharge", "Maternal"]):
        return "event_date_discharge_summary"

    # Default to delivery summary date
    return "event_date_delivery_summary"


def prepare_data_for_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for trend chart using ONLY program stage specific dates
    WITH DATE RANGE FILTERING
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
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


def extract_event_date_for_period(df, event_name):
    """
    Extract event date for period grouping with fallback to enrollment_date
    event_name should be one of: "Delivery summary", "Postpartum care", "Discharge Summary"
    """
    if df.empty:
        return pd.DataFrame()

    event_date_columns = {
        "Delivery summary": "event_date_delivery_summary",
        "Postpartum care": "event_date_postpartum_care",
        "Discharge Summary": "event_date_discharge_summary",
        "Instrumental delivery form": "event_date_instrumental_delivery_form",
    }

    result_df = df.copy()

    if event_date_columns.get(event_name) in df.columns:
        event_dates = pd.to_datetime(
            result_df[event_date_columns[event_name]], errors="coerce"
        )

        # Use only event date (no fallback)
        result_df["event_date"] = event_dates
        result_df["period"] = result_df["event_date"].dt.strftime("%Y-%m")
        result_df["period_display"] = result_df["event_date"].dt.strftime("%b-%y")
        result_df["period_sort"] = result_df["event_date"].dt.strftime("%Y%m")

    return result_df


def get_numerator_denominator_for_kpi(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    # SPECIAL HANDLING FOR SVD - MUST BE FIRST!
    if kpi_name == "Normal Vaginal Delivery (SVD) Rate (%)":
        from utils.kpi_svd import get_numerator_denominator_for_svd

        return get_numerator_denominator_for_svd(df, facility_uids, date_range_filters)

    # SPECIAL HANDLING FOR ASSISTED DELIVERY
    elif kpi_name == "Assisted Delivery Rate (%)":
        from utils.kpi_assisted import get_numerator_denominator_for_assisted

        return get_numerator_denominator_for_assisted(
            df, facility_uids, date_range_filters
        )

    elif kpi_name == "Postpartum Hemorrhage (PPH) Rate (%)":
        from utils.kpi_pph import get_numerator_denominator_for_pph

        return get_numerator_denominator_for_pph(df, facility_uids, date_range_filters)

    elif kpi_name == "Delivered women who received uterotonic (%)":
        from utils.kpi_uterotonic import get_numerator_denominator_for_uterotonic

        return get_numerator_denominator_for_uterotonic(
            df, facility_uids, date_range_filters
        )

    elif kpi_name == "Missing Mode of Delivery":
        from utils.kpi_missing_md import get_numerator_denominator_for_missing_md

        return get_numerator_denominator_for_missing_md(
            df, facility_uids, date_range_filters
        )
    elif kpi_name == "ARV Prophylaxis Rate (%)":
        from utils.kpi_arv import get_numerator_denominator_for_arv

        return get_numerator_denominator_for_arv(df, facility_uids, date_range_filters)

    elif kpi_name == "Low Birth Weight (LBW) Rate (%)":
        from utils.kpi_lbw import get_numerator_denominator_for_lbw

        return get_numerator_denominator_for_lbw(df, facility_uids, date_range_filters)

    """
    Get numerator and denominator for a specific KPI with UID filtering
    AND filtered by KPI-specific program stage dates AND date range
    Returns: (numerator, denominator, value)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_kpi(kpi_name)

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

    # Now compute KPI on date-filtered data
    kpi_data = compute_kpis(filtered_df, facility_uids)

    kpi_mapping = {
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
            "numerator": "fp_acceptance",
            "denominator": "total_deliveries",
            "value": "ippcar",
        },
        "Stillbirth Rate (%)": {
            "numerator": "stillbirths",
            "denominator": "total_deliveries_sb",
            "value": "stillbirth_rate",
        },
        "Early Postnatal Care (PNC) Coverage (%)": {
            "numerator": "early_pnc",
            "denominator": "total_deliveries",
            "value": "pnc_coverage",
        },
        "Institutional Maternal Death Rate (%)": {
            "numerator": "maternal_deaths",
            "denominator": "total_deliveries_md",
            "value": "maternal_death_rate",
        },
        "C-Section Rate (%)": {
            "numerator": "csection_deliveries",
            "denominator": "total_deliveries",
            "value": "csection_rate",
        },
    }

    if kpi_name in kpi_mapping:
        mapping = kpi_mapping[kpi_name]
        numerator = kpi_data.get(mapping["numerator"], 0)
        denominator = kpi_data.get(mapping["denominator"], 1)
        value = kpi_data.get(mapping["value"], 0.0)

        return (numerator, denominator, value)

    # Fallback mappings for partial matches
    if "IPPCAR" in kpi_name or "Contraceptive" in kpi_name:
        numerator = kpi_data.get("fp_acceptance", 0)
        denominator = kpi_data.get("total_deliveries", 1)
        value = kpi_data.get("ippcar", 0.0)
        return (numerator, denominator, value)
    elif "Stillbirth" in kpi_name:
        numerator = kpi_data.get("stillbirths", 0)
        denominator = kpi_data.get("total_deliveries_sb", 1)
        value = kpi_data.get("stillbirth_rate", 0.0)
        return (numerator, denominator, value)
    elif "PNC" in kpi_name or "Postnatal" in kpi_name:
        numerator = kpi_data.get("early_pnc", 0)
        denominator = kpi_data.get("total_deliveries", 1)
        value = kpi_data.get("pnc_coverage", 0.0)
        return (numerator, denominator, value)
    elif "Maternal Death" in kpi_name:
        numerator = kpi_data.get("maternal_deaths", 0)
        denominator = kpi_data.get("total_deliveries_md", 1)
        value = kpi_data.get("maternal_death_rate", 0.0)
        return (numerator, denominator, value)
    elif "C-Section" in kpi_name:
        numerator = kpi_data.get("csection_deliveries", 0)
        denominator = kpi_data.get("total_deliveries", 1)
        value = kpi_data.get("csection_rate", 0.0)
        return (numerator, denominator, value)

    return (0, 0, 0.0)


# ---------------- Period Aggregation Function ----------------
def aggregate_by_period_with_sorting(
    df, period_col, period_sort_col, facility_uids, kpi_function, kpi_name=None
):
    """
    Aggregate data by period while preserving chronological sorting
    Works with patient-level data
    """
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby([period_col, period_sort_col])

    result_data = []
    for (period_display, period_sort), group_df in grouped:
        if kpi_name:
            numerator, denominator, value = get_numerator_denominator_for_kpi(
                group_df, kpi_name, facility_uids
            )
        else:
            kpi_data = kpi_function(group_df, facility_uids)
            if isinstance(kpi_data, dict):
                value = kpi_data.get("value", 0)
                numerator = kpi_data.get("numerator", 0)
                denominator = kpi_data.get("denominator", 1)
            else:
                value = kpi_data
                numerator = 0
                denominator = 1

        result_data.append(
            {
                period_col: period_display,
                period_sort_col: period_sort,
                "value": value,
                "numerator": numerator,
                "denominator": denominator,
            }
        )

    result_df = pd.DataFrame(result_data)

    if not result_df.empty and period_sort_col in result_df.columns:
        result_df = result_df.sort_values(period_sort_col)

    return result_df


# ---------------- Chart Functions WITH TABLES ----------------
def render_trend_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names=None,
    numerator_name="Numerator",
    denominator_name="Denominator",
    facility_uids=None,
):
    """Render a trend chart for a single facility/region with numerator/denominator data AND TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    chart_options = ["Line", "Bar", "Area"]

    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_{title}_{str(facility_uids)}",
    ).lower()

    if "numerator" in df.columns and "denominator" in df.columns:
        df[numerator_name] = df["numerator"]
        df[denominator_name] = df["denominator"]
        hover_columns = [numerator_name, denominator_name]
        use_hover_data = True
    else:
        hover_columns = []
        use_hover_data = False

    try:
        if chart_type == "line":
            fig = px.line(
                df,
                x=x_axis_col,
                y=value_col,
                markers=True,
                line_shape="linear",
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=7),
            )
        elif chart_type == "bar":
            fig = px.bar(
                df,
                x=x_axis_col,
                y=value_col,
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
            )
        elif chart_type == "area":
            fig = px.area(
                df,
                x=x_axis_col,
                y=value_col,
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
            )
        else:
            fig = px.line(
                df,
                x=x_axis_col,
                y=value_col,
                markers=True,
                line_shape="linear",
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=7),
            )
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        fig = px.line(
            df,
            x=x_axis_col,
            y=value_col,
            markers=True,
            title=title,
            height=400,
        )

    is_categorical = (
        not all(isinstance(x, (dt.date, dt.datetime)) for x in df[period_col])
        if not df.empty
        else True
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title=period_col,
        yaxis_title=value_col,
        xaxis=dict(
            type="category" if is_categorical else None,
            tickangle=-45 if is_categorical else 0,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    if "Rate" in title or "%" in title:
        fig.update_layout(yaxis_tickformat=".2f")
    if any(k in title for k in ["Deliveries", "Acceptance"]):
        fig.update_layout(yaxis_tickformat=",")

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Data Table")

    # Create a clean display dataframe
    display_df = df.copy()

    # Select columns to show in table
    table_columns = [x_axis_col, value_col]

    # Add numerator and denominator if available
    if "numerator" in display_df.columns and "denominator" in display_df.columns:
        display_df[numerator_name] = display_df["numerator"]
        display_df[denominator_name] = display_df["denominator"]
        table_columns.extend([numerator_name, denominator_name])

    # Format the dataframe for display
    display_df = display_df[table_columns].copy()

    # Format numbers
    if "Rate" in title or "%" in title:
        display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:.2f}%")
    else:
        display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:,.0f}")

    if numerator_name in display_df.columns:
        display_df[numerator_name] = display_df[numerator_name].apply(
            lambda x: f"{x:,.0f}"
        )
    if denominator_name in display_df.columns:
        display_df[denominator_name] = display_df[denominator_name].apply(
            lambda x: f"{x:,.0f}"
        )

    # Add Overall/Total row
    if "numerator" in df.columns and "denominator" in df.columns:
        total_numerator = df["numerator"].sum()
        total_denominator = df["denominator"].sum()
        overall_value = (
            (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        )
    else:
        overall_value = df[value_col].mean() if not df.empty else 0
        total_numerator = df[value_col].sum() if not df.empty else 0
        total_denominator = len(df)

    # Create overall row with consistent date format
    overall_row = {
        x_axis_col: "Overall",
        value_col: f"{overall_value:.2f}%" if "%" in title else f"{overall_value:,.0f}",
    }

    if numerator_name in display_df.columns:
        overall_row[numerator_name] = f"{total_numerator:,.0f}"
    if denominator_name in display_df.columns:
        overall_row[denominator_name] = f"{total_denominator:,.0f}"

    # Convert to DataFrame and append
    overall_df = pd.DataFrame([overall_row])
    display_df = pd.concat([display_df, overall_df], ignore_index=True)

    # Display the table
    st.dataframe(display_df, use_container_width=True)

    # Add summary statistics
    if len(df) > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📈 Latest Value", f"{df[value_col].iloc[-1]:.2f}")

        with col2:
            st.metric("📊 Average", f"{df[value_col].mean():.2f}")

        with col3:
            # Calculate trend
            last_value = df[value_col].iloc[-1]
            prev_value = df[value_col].iloc[-2]
            trend_change = last_value - prev_value
            trend_symbol = (
                "▲" if trend_change > 0 else ("▼" if trend_change < 0 else "–")
            )
            st.metric("📈 Trend from Previous", f"{trend_change:.2f} {trend_symbol}")

    # Keep the download button - FIX DATE FORMAT ISSUE
    summary_df = df.copy().reset_index(drop=True)

    if "numerator" in summary_df.columns and "denominator" in summary_df.columns:
        summary_df = summary_df[
            [x_axis_col, "numerator", "denominator", value_col]
        ].copy()

        # FIX: Ensure period column is in proper format before exporting
        if x_axis_col in summary_df.columns:
            # Convert any date-like strings to proper month-year format
            summary_df[x_axis_col] = summary_df[x_axis_col].apply(
                format_period_month_year
            )

        summary_df = summary_df.rename(
            columns={
                "numerator": numerator_name,
                "denominator": denominator_name,
                value_col: title,
            }
        )

        total_numerator = summary_df[numerator_name].sum()
        total_denominator = summary_df[denominator_name].sum()

        overall_value = (
            (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        )

        overall_row = pd.DataFrame(
            {
                x_axis_col: ["Overall"],
                numerator_name: [total_numerator],
                denominator_name: [total_denominator],
                title: [overall_value],
            }
        )

        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    else:
        summary_df = summary_df[[x_axis_col, value_col]].copy()

        # FIX: Ensure period column is in proper format before exporting
        if x_axis_col in summary_df.columns:
            summary_df[x_axis_col] = summary_df[x_axis_col].apply(
                format_period_month_year
            )

        summary_df = summary_df.rename(columns={value_col: title})
        summary_table = summary_df.copy()

        overall_value = summary_table[title].mean() if not summary_table.empty else 0
        overall_row = pd.DataFrame({x_axis_col: ["Overall"], title: [overall_value]})
        summary_table = pd.concat([summary_table, overall_row], ignore_index=True)

    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_chart_data.csv",
        mime="text/csv",
        help="Download the exact x, y, and value components shown in the chart",
    )


def render_facility_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names,
    facility_uids,
    numerator_name,
    denominator_name,
):
    """Render a comparison chart showing each facility's performance over time WITH TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # STANDARDIZE COLUMN NAMES - UPDATED TO MATCH YOUR DATA STRUCTURE
    if "orgUnit" not in df.columns:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["orgunit", "facility_uid", "facility_id", "uid", "ou"]:
                df = df.rename(columns={col: "orgUnit"})

    # Check for facility name column - LOOK FOR orgUnit_name FIRST
    if "orgUnit_name" in df.columns:
        df = df.rename(columns={"orgUnit_name": "Facility"})
    elif "Facility" not in df.columns:
        # Try to find other facility name columns
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["facility_name", "facility", "name", "display_name"]:
                df = df.rename(columns={col: "Facility"})
                break

    if "orgUnit" not in df.columns or "Facility" not in df.columns:
        st.error(
            f"❌ Facility identifier columns not found in the data. Cannot perform facility comparison.\n"
            f"Available columns: {list(df.columns)}"
        )
        return

    if df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Create a mapping from orgUnit to facility name
    facility_mapping = {}
    for _, row in df.iterrows():
        if pd.notna(row["orgUnit"]) and pd.notna(row["Facility"]):
            facility_mapping[str(row["orgUnit"])] = str(row["Facility"])

    # If we have facility_names parameter, update the mapping
    if facility_names and len(facility_names) == len(facility_uids):
        for uid, name in zip(facility_uids, facility_names):
            facility_mapping[str(uid)] = name

    # Prepare comparison data
    comparison_data = []

    # Get unique periods in order
    if "period_sort" in df.columns:
        unique_periods = df[["period_display", "period_sort"]].drop_duplicates()
        unique_periods = unique_periods.sort_values("period_sort")
        period_order = unique_periods["period_display"].tolist()
    else:
        # Try to sort by month-year
        try:
            period_order = sorted(
                df["period_display"].unique().tolist(),
                key=lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if "-" in x
                    else x
                ),
            )
        except:
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order]

    # Prepare data for each facility and period
    for facility_uid, facility_name in facility_mapping.items():
        facility_df = df[df["orgUnit"] == facility_uid].copy()

        if facility_df.empty:
            continue

        # Group by period for this facility
        for period_display, period_group in facility_df.groupby("period_display"):
            if not period_group.empty:
                # Get the first row for this facility/period combination
                row = period_group.iloc[0]
                formatted_period = format_period_month_year(period_display)

                # Skip if both numerator and denominator are 0
                numerator_val = row.get("numerator", 0)
                denominator_val = row.get("denominator", 1)

                if numerator_val == 0 and denominator_val == 0:
                    continue  # Skip this period for this facility

                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Facility": facility_name,
                        "value": row.get(value_col, 0) if value_col in row else 0,
                        "numerator": numerator_val,
                        "denominator": denominator_val,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: dt.datetime.strptime(x, "%b-%y")
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: dt.datetime.strptime(x, "%b-%y"),
        )
    except:
        pass

    # Filter out facilities that have no data (all periods with 0 numerator and denominator)
    facilities_with_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        # Check if facility has any non-zero data
        if not (
            facility_data["numerator"].sum() == 0
            and facility_data["denominator"].sum() == 0
        ):
            facilities_with_data.append(facility_name)

    # Filter comparison_df to only include facilities with data
    comparison_df = comparison_df[
        comparison_df["Facility"].isin(facilities_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all facilities have zero data).")
        return

    # Create the chart
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=f"{title} - Facility Comparison",
        height=500,
        category_orders={"period_display": period_order},
        hover_data=["numerator", "denominator"],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Facility: %{{fullData.name}}<br>"
            f"{title}: %{{y:.2f}}<br>"
            f"{numerator_name}: %{{customdata[0]}}<br>"
            f"{denominator_name}: %{{customdata[1]}}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=title,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Facilities",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    if "Rate" in title or "%" in title:
        fig.update_layout(yaxis_tickformat=".2f")

    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Facility Comparison Data")

    # Create pivot table for better display with Overall row - REMOVED "Periods with Data" column
    pivot_data = []

    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            total_numerator = facility_data["numerator"].sum()
            total_denominator = facility_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )

            pivot_data.append(
                {
                    "Facility": facility_name,
                    numerator_name: f"{total_numerator:,.0f}",
                    denominator_name: f"{total_denominator:,.0f}",
                    "Overall Value": f"{overall_value:.2f}%",
                }
            )

    # Add Overall row for all facilities
    if pivot_data:
        all_numerators = comparison_df["numerator"].sum()
        all_denominators = comparison_df["denominator"].sum()
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )

        pivot_data.append(
            {
                "Facility": "Overall",
                numerator_name: f"{all_numerators:,.0f}",
                denominator_name: f"{all_denominators:,.0f}",
                "Overall Value": f"{grand_overall:.2f}%",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Keep download functionality
    csv_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            total_numerator = facility_data["numerator"].sum()
            total_denominator = facility_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )
            csv_data.append(
                {
                    "Facility": facility_name,
                    numerator_name: total_numerator,
                    denominator_name: total_denominator,
                    title: f"{overall_value:.2f}%",
                }
            )

    # Add overall row to CSV
    if csv_data:
        all_numerators = sum(item[numerator_name] for item in csv_data)
        all_denominators = sum(item[denominator_name] for item in csv_data)
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )
        csv_data.append(
            {
                "Facility": "Overall",
                numerator_name: all_numerators,
                denominator_name: all_denominators,
                title: f"{grand_overall:.2f}%",
            }
        )

        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Overall Comparison Data",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_facility_summary.csv",
            mime="text/csv",
            help="Download overall summary data for facility comparison",
        )


def render_region_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    region_names,
    region_mapping,
    facilities_by_region,
    numerator_name,
    denominator_name,
):
    """Render a comparison chart showing each region's performance over time WITH TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if "Region" not in df.columns:
        st.error(
            f"❌ Region column not found in the data. Cannot perform region comparison.\n"
            f"Available columns: {list(df.columns)}"
        )
        return

    if df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Prepare comparison data
    comparison_data = []

    # Get unique periods in order
    if "period_sort" in df.columns:
        unique_periods = df[["period_display", "period_sort"]].drop_duplicates()
        unique_periods = unique_periods.sort_values("period_sort")
        period_order = unique_periods["period_display"].tolist()
    else:
        try:
            period_order = sorted(
                df["period_display"].unique().tolist(),
                key=lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if "-" in x
                    else x
                ),
            )
        except:
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order]

    # Prepare data for each region and period
    for region_name in df["Region"].unique():
        region_df = df[df["Region"] == region_name].copy()

        if region_df.empty:
            continue

        # Group by period for this region
        for period_display, period_group in region_df.groupby("period_display"):
            if not period_group.empty:
                # Get aggregated values for this region/period
                avg_value = (
                    period_group[value_col].mean()
                    if value_col in period_group.columns
                    else 0
                )
                total_numerator = period_group["numerator"].sum()
                total_denominator = (
                    period_group["denominator"].sum()
                    if period_group["denominator"].sum() > 0
                    else 1
                )

                # Skip if both numerator and denominator are 0
                if total_numerator == 0 and total_denominator == 0:
                    continue

                formatted_period = format_period_month_year(period_display)
                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Region": region_name,
                        "value": avg_value,
                        "numerator": total_numerator,
                        "denominator": total_denominator,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available for regions.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: dt.datetime.strptime(x, "%b-%y")
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: dt.datetime.strptime(x, "%b-%y"),
        )
    except:
        pass

    # Filter out regions that have no data (all periods with 0 numerator and denominator)
    regions_with_data = []
    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        # Check if region has any non-zero data
        if not (
            region_data["numerator"].sum() == 0
            and region_data["denominator"].sum() == 0
        ):
            regions_with_data.append(region_name)

    # Filter comparison_df to only include regions with data
    comparison_df = comparison_df[
        comparison_df["Region"].isin(regions_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all regions have zero data).")
        return

    # Create the chart
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=f"{title} - Region Comparison",
        height=500,
        category_orders={"period_display": period_order},
        hover_data=["numerator", "denominator"],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Region: %{{fullData.name}}<br>"
            f"{title}: %{{y:.2f}}<br>"
            f"{numerator_name}: %{{customdata[0]}}<br>"
            f"{denominator_name}: %{{customdata[1]}}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=title,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Regions",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    if "Rate" in title or "%" in title:
        fig.update_layout(yaxis_tickformat=".2f")

    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Region Comparison Data")

    # Create pivot table for better display with Overall row - REMOVED "Periods with Data" column
    pivot_data = []

    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        if not region_data.empty:
            total_numerator = region_data["numerator"].sum()
            total_denominator = region_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )

            pivot_data.append(
                {
                    "Region": region_name,
                    numerator_name: f"{total_numerator:,.0f}",
                    denominator_name: f"{total_denominator:,.0f}",
                    "Overall Value": f"{overall_value:.2f}%",
                }
            )

    # Add Overall row for all regions
    if pivot_data:
        all_numerators = comparison_df["numerator"].sum()
        all_denominators = comparison_df["denominator"].sum()
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )

        pivot_data.append(
            {
                "Region": "Overall",
                numerator_name: f"{all_numerators:,.0f}",
                denominator_name: f"{all_denominators:,.0f}",
                "Overall Value": f"{grand_overall:.2f}%",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Keep download functionality
    csv_data = []
    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        if not region_data.empty:
            total_numerator = region_data["numerator"].sum()
            total_denominator = region_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )
            csv_data.append(
                {
                    "Region": region_name,
                    numerator_name: total_numerator,
                    denominator_name: total_denominator,
                    title: f"{overall_value:.2f}%",
                }
            )

    # Add overall row to CSV
    if csv_data:
        all_numerators = sum(item[numerator_name] for item in csv_data)
        all_denominators = sum(item[denominator_name] for item in csv_data)
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )
        csv_data.append(
            {
                "Region": "Overall",
                numerator_name: all_numerators,
                denominator_name: all_denominators,
                title: f"{grand_overall:.2f}%",
            }
        )

        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Overall Comparison Data",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_region_summary.csv",
            mime="text/csv",
            help="Download overall summary data for region comparison",
        )


# ---------------- Additional Helper Functions ----------------
def extract_period_columns(df, date_column):
    """
    SIMPLE VERSION: Assumes dates are already valid, just need proper grouping
    """
    if df.empty or date_column not in df.columns:
        return df

    result_df = df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
        result_df["event_date"] = pd.to_datetime(
            result_df[date_column], errors="coerce"
        )
    else:
        result_df["event_date"] = result_df[date_column]

    # Create period columns with proper month-year format
    result_df["period"] = result_df["event_date"].dt.strftime("%Y-%m")
    result_df["period_display"] = (
        result_df["event_date"].dt.strftime("%b-%y").str.capitalize()
    )
    result_df["period_sort"] = result_df["event_date"].dt.strftime("%Y%m")

    return result_df
