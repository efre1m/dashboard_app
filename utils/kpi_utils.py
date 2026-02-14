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


def get_current_period_label():
    """Get the current time aggregation label from session state."""
    period_label = st.session_state.get("period_label", "Monthly")
    if "filters" in st.session_state and "period_label" in st.session_state.filters:
        period_label = st.session_state.filters["period_label"]
    return str(period_label)


def format_period_for_download(period_value, period_label):
    """Format period value for CSV downloads, including monthly M/1/YYYY output."""
    if pd.isna(period_value):
        return ""

    raw = str(period_value).strip()
    if not raw:
        return ""

    parsed_date = None
    formats_to_try = [
        "%Y-%m",
        "%b-%y",
        "%B-%y",
        "%Y-%m-%d",
        "%m/%Y",
        "%Y/%m",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y%m",
    ]

    for fmt in formats_to_try:
        try:
            parsed_date = dt.datetime.strptime(raw, fmt)
            break
        except (ValueError, TypeError):
            continue

    if parsed_date is not None and str(period_label).lower() == "monthly":
        return f"{parsed_date.month}/1/{parsed_date.year}"

    return raw


def format_trend_period_for_download(period_value):
    """Format period for trend-style display (MMM YYYY) where parseable."""
    if pd.isna(period_value):
        return ""

    raw = str(period_value).strip()
    if not raw:
        return ""

    formats_to_try = [
        "%Y-%m",
        "%b-%y",
        "%B-%y",
        "%Y-%m-%d",
        "%m/%Y",
        "%Y/%m",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y%m",
    ]

    for fmt in formats_to_try:
        try:
            return dt.datetime.strptime(raw, fmt).strftime("%b %Y")
        except (ValueError, TypeError):
            continue

    return raw


def format_period_list_for_download(period_series, period_label):
    """Create a readable period string for download rows from available periods."""
    if period_series is None:
        return ""

    formatted_periods = []
    for period_value in pd.Series(period_series).dropna().tolist():
        formatted = format_period_for_download(period_value, period_label)
        if formatted:
            formatted_periods.append(formatted)

    unique_periods = list(dict.fromkeys(formatted_periods))
    return ", ".join(unique_periods)


def get_attractive_hover_template(
    kpi_name, numerator_name, denominator_name, is_count=False
):
    """
    Generate an attractive, well-styled hover template for Plotly charts.
    """
    if is_count:
        return (
            "<span style='font-size:13px; font-weight:bold; color:#1f2937;'>%{x}</span><br><br>"
            + f"<span style='color:#4b5563; font-weight:500;'>{kpi_name}:</span> <span style='font-weight:bold; color:#3b82f6; font-size:14px;'>%{{y:,.0f}}</span><extra></extra>"
        )

    return (
        "<span style='font-size:13px; font-weight:bold; color:#1f2937;'>%{x}</span><br><br>"
        + f"<span style='color:#4b5563; font-weight:500;'>{kpi_name}:</span> <span style='font-weight:bold; color:#2ecc71; font-size:14px;'>%{{y:.2f}}%</span><br>"
        + f"<span style='color:#e67e22; font-weight:500; font-size:12px;'>{numerator_name}:</span> <span style='font-weight:600; color:#374151; font-size:12px;'>%{{customdata[0]:,.0f}}</span><br>"
        + f"<span style='color:#9b59b6; font-weight:500; font-size:12px;'>{denominator_name}:</span> <span style='font-weight:600; color:#374151; font-size:12px;'>%{{customdata[1]:,.0f}}</span><extra></extra>"
    )


# ---------------- KPI Constants ----------------
# Patient-level data columns
FP_ACCEPTANCE_COL = "fp_counseling_and_method_provided_pp_postpartum_care"
FP_ACCEPTED_CODES = {"1", "2", "3", "4", "5"}

# Birth outcome columns
BIRTH_OUTCOME_COL = "birth_outcome_delivery_summary"
ALIVE_CODE = "1"
STILLBIRTH_CODE = "2"

# Birth outcome columns for multiple newborns
BIRTH_OUTCOME_NEWBORN_1_COL = "birth_outcome_newborn_delivery_summary"
BIRTH_OUTCOME_NEWBORN_2_COL = "birth_outcome_newborn_2_delivery_summary"
BIRTH_OUTCOME_NEWBORN_3_COL = "birth_outcome_newborn_3_delivery_summary"
BIRTH_OUTCOME_NEWBORN_4_COL = "birth_outcome_newborn_4_delivery_summary"

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

# Event date columns - UPDATED: ALL KPIs NOW USE ENROLLMENT DATE
DELIVERY_DATE_COL = "enrollment_date"
PNC_DATE_COL = "enrollment_date"
DISCHARGE_DATE_COL = "enrollment_date"

# Enrollment date column
ENROLLMENT_DATE_COL = "enrollment_date"


def compute_birth_counts(df, facility_uids=None):
    """
    Compute birth counts accounting for multiple births (twins, triplets, etc.)
    Uses UID filtering - VECTORIZED for performance
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

        # Initialize columns with zeros if they don't exist
        for col in [NUMBER_OF_NEWBORNS_COL, OTHER_NUMBER_OF_NEWBORNS_COL]:
            if col not in filtered_df.columns:
                filtered_df[col] = 0
            filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce").fillna(0)

        # PRIORITIZED LOGIC for baby count
        n1 = filtered_df[NUMBER_OF_NEWBORNS_COL]
        n2 = filtered_df[OTHER_NUMBER_OF_NEWBORNS_COL]
        
        # Determine total babies per row: n1 if > 0 else (n2 if > 0 else 1)
        total_babies_per_row = n1.where(n1 > 0, n2.where(n2 > 0, 1)).astype(int)
        total_births = total_babies_per_row.sum()

        live_births = 0
        stillbirths = 0
        
        # Baby 1: Always use BIRTH_OUTCOME_COL
        if BIRTH_OUTCOME_COL in filtered_df.columns:
            outcomes = pd.to_numeric(filtered_df[BIRTH_OUTCOME_COL], errors="coerce")
            live_births += (outcomes == 1).sum()
            stillbirths += (outcomes == 2).sum()

        # Babies 2-4: Use specific outcome columns
        birth_outcome_cols = [
            BIRTH_OUTCOME_NEWBORN_2_COL,
            BIRTH_OUTCOME_NEWBORN_3_COL,
            BIRTH_OUTCOME_NEWBORN_4_COL,
        ]
        
        for i, col in enumerate(birth_outcome_cols):
            if col in filtered_df.columns:
                # Only count if the row actually has at least (i+2) babies
                mask = total_babies_per_row >= (i + 2)
                outcomes = pd.to_numeric(filtered_df.loc[mask, col], errors="coerce")
                live_births += (outcomes == 1).sum()
                stillbirths += (outcomes == 2).sum()

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
    """Count stillbirth occurrences across all newborns - VECTORIZED for performance"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Initialize columns
    for col in [NUMBER_OF_NEWBORNS_COL, OTHER_NUMBER_OF_NEWBORNS_COL]:
        if col not in filtered_df.columns:
            filtered_df[col] = 0
    
    n1 = pd.to_numeric(filtered_df[NUMBER_OF_NEWBORNS_COL], errors="coerce").fillna(0)
    n2 = pd.to_numeric(filtered_df[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce").fillna(0)
    total_babies = n1.where(n1 > 0, n2.where(n2 > 0, 1)).astype(int)
    
    stillbirths = 0
    
    # Baby 1: Always use BIRTH_OUTCOME_COL
    if BIRTH_OUTCOME_COL in filtered_df.columns:
        outcomes = pd.to_numeric(filtered_df[BIRTH_OUTCOME_COL], errors="coerce")
        stillbirths += (outcomes == 2).sum()
        
    # Babies 2-4: Use specific outcome columns
    birth_outcome_cols = [BIRTH_OUTCOME_NEWBORN_2_COL, BIRTH_OUTCOME_NEWBORN_3_COL, BIRTH_OUTCOME_NEWBORN_4_COL]
    for i, col in enumerate(birth_outcome_cols):
        if col in filtered_df.columns:
            mask = total_babies >= (i + 2)
            outcomes = pd.to_numeric(filtered_df.loc[mask, col], errors="coerce")
            stillbirths += (outcomes == 2).sum()
            
    return int(stillbirths)



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


def compute_total_newborns(df, facility_uids=None):
    """Count total newborns - VECTORIZED for performance"""
    if df is None or df.empty:
        return 0
    
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
    
    for col in [NUMBER_OF_NEWBORNS_COL, OTHER_NUMBER_OF_NEWBORNS_COL]:
        if col not in filtered_df.columns:
            filtered_df[col] = 0
            
    n1 = pd.to_numeric(filtered_df[NUMBER_OF_NEWBORNS_COL], errors="coerce").fillna(0)
    n2 = pd.to_numeric(filtered_df[OTHER_NUMBER_OF_NEWBORNS_COL], errors="coerce").fillna(0)
    
    total_newborns = n1.where(n1 > 0, n2.where(n2 > 0, 1)).astype(int).sum()
    return int(total_newborns)


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
        total_newborns = compute_total_newborns(df, facility_uids)
        rate = (stillbirths / total_newborns * 100) if total_newborns > 0 else 0.0
        result = (rate, stillbirths, total_newborns)

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
    """Compute maternal death rate (per 100,000 live births)"""
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
            (maternal_deaths / total_deliveries * 100000) if total_deliveries > 0 else 0.0
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


def compute_fp_distribution(df, facility_uids=None):
    """Compute distribution of FP methods/counseling with Human Readable Labels"""
    if df is None or df.empty:
        return {}
    
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
        
    if FP_ACCEPTANCE_COL not in filtered_df.columns:
        return {}
        
    # Count values
    dist = filtered_df[FP_ACCEPTANCE_COL].dropna().astype(str).value_counts().to_dict()
    
    # Mapping for Human Readable Labels
    FP_LABELS = {
        "1": "Pills",
        "2": "Injectables",
        "3": "Implants",
        "4": "IUCD",
        "5": "Condom",
        "6": "Counseled only",
    }
    
    # Clean up keys and map to labels
    cleaned_dist = {}
    for k, v in dist.items():
        key = k.split('.')[0] if '.' in k else k
        
        # Use label if available, otherwise ignore or use key?
        # User implies only these should be shown ("human readable text not code")
        if key in FP_LABELS:
            label = FP_LABELS[key]
            cleaned_dist[label] = cleaned_dist.get(label, 0) + int(v)
        
    return cleaned_dist


# ---------------- Master KPI Function ----------------
def compute_kpis(df, facility_uids=None):
    """Compute all KPIs with optional date filtering"""
    cache_key = get_cache_key(df, facility_uids, "main_kpis_v4")

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
    
    # FP Distribution
    fp_distribution = compute_fp_distribution(filtered_df, facility_uids)

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
    
    # NEW: Episiotomy Rate
    from utils.kpi_episiotomy import compute_episiotomy_rate
    episiotomy_rate, episiotomy_cases, total_vaginal_deliveries = compute_episiotomy_rate(
        filtered_df, facility_uids
    )
    
    # NEW: SVD Rate
    from utils.kpi_svd import compute_svd_count
    svd_deliveries = compute_svd_count(filtered_df, facility_uids)
    svd_rate = (svd_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0
    
    # NEW: Assisted Delivery Rate
    from utils.kpi_assisted import compute_assisted_count
    assisted_deliveries = compute_assisted_count(filtered_df, facility_uids)
    assisted_rate = (assisted_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0

    # NEW: Antepartum Complications Rate
    from utils.kpi_antipartum_compl import compute_antipartum_compl_rate
    antipartum_rate, antipartum_cases, _ = compute_antipartum_compl_rate(
        filtered_df, facility_uids
    )

#    from utils.kpi_postpartum_compl import compute_postpartum_compl_rate, compute_postpartum_distribution
    from utils.kpi_postpartum_compl import compute_postpartum_compl_rate
    postpartum_rate, postpartum_cases, _ = compute_postpartum_compl_rate(
        filtered_df, facility_uids
    )
#    postpartum_distribution = compute_postpartum_distribution(filtered_df, facility_uids)

    result = {
        "total_deliveries": int(total_deliveries),
        "fp_acceptance": int(fp_acceptance),
        "ippcar": float(ippcar),
        "fp_distribution": fp_distribution,
        "stillbirth_rate": float(stillbirth_rate),
        "stillbirths": int(stillbirths),
        "total_deliveries_sb": int(total_deliveries_sb),
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
        "episiotomy_rate": float(episiotomy_rate),
        "episiotomy_cases": int(episiotomy_cases),
        "total_vaginal_deliveries": int(total_vaginal_deliveries),
        "svd_deliveries": int(svd_deliveries),
        "svd_rate": float(svd_rate),
        "assisted_deliveries": int(assisted_deliveries),
        "assisted_rate": float(assisted_rate),
        "antipartum_rate": float(antipartum_rate),
        "antipartum_cases": int(antipartum_cases),
        "postpartum_rate": float(postpartum_rate),
        "postpartum_cases": int(postpartum_cases),
#        "postpartum_distribution": postpartum_distribution,
    }

    st.session_state.kpi_cache[cache_key] = result
    return result


# ---------------- Date Handling with Program Stage Specific Dates ----------------
def get_relevant_date_column_for_kpi(kpi_name):
    """
    Get the relevant date column for a specific KPI.
    UPDATED: All indicators now use 'enrollment_date' to ensure consistent denominators.
    """
    return "enrollment_date"


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


def extract_event_date_for_period(df, event_name=None):
    """
    Extract event date for period grouping.
    UPDATED: All indicators now use 'enrollment_date' to ensure consistent denominators.
    """
    if df.empty:
        return pd.DataFrame()

    result_df = df.copy()

    # Use enrollment_date for all period grouping
    if "enrollment_date" in result_df.columns:
        result_df["event_date"] = pd.to_datetime(result_df["enrollment_date"], errors="coerce")
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
    elif kpi_name == "Missing Birth Outcome":
        from utils.kpi_missing_bo import get_numerator_denominator_for_missing_bo

        return get_numerator_denominator_for_missing_bo(
            df, facility_uids, date_range_filters
        )
    elif kpi_name == "Missing Obstetric Condition at Delivery":
        from utils.kpi_missing_postpartum import (
            get_numerator_denominator_for_missing_postpartum,
        )

        return get_numerator_denominator_for_missing_postpartum(
            df, facility_uids, date_range_filters
        )
    elif kpi_name == "Missing Obstetric Complications Diagnosis":
        from utils.kpi_missing_antepartum import (
            get_numerator_denominator_for_missing_antepartum,
        )

        return get_numerator_denominator_for_missing_antepartum(
            df, facility_uids, date_range_filters
        )
    elif kpi_name == "Missing Uterotonics Given at Delivery":
        from utils.kpi_missing_uterotonic import (
            get_numerator_denominator_for_missing_uterotonic,
        )

        return get_numerator_denominator_for_missing_uterotonic(
            df, facility_uids, date_range_filters
        )
    elif kpi_name == "Missing Condition of Discharge":
        from utils.kpi_missing_cod import get_numerator_denominator_for_missing_cod

        return get_numerator_denominator_for_missing_cod(
            df, facility_uids, date_range_filters
        )
    elif kpi_name == "Admitted Mothers":
        from utils.kpi_admitted_mothers import (
            get_numerator_denominator_for_admitted_mothers,
        )

        return get_numerator_denominator_for_admitted_mothers(
            df, facility_uids, date_range_filters
        )

    elif kpi_name == "Antepartum Complications Rate (%)":
        from utils.kpi_antipartum_compl import get_numerator_denominator_for_antipartum_compl

        return get_numerator_denominator_for_antipartum_compl(
            df, facility_uids, date_range_filters
        )

    elif kpi_name == "Postpartum Complications Rate (%)":
        from utils.kpi_postpartum_compl import get_numerator_denominator_for_postpartum_compl

        return get_numerator_denominator_for_postpartum_compl(
            df, facility_uids, date_range_filters
        )

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
        "Maternal Death Rate (per 100,000)": {
            "numerator": "maternal_deaths",
            "denominator": "total_deliveries_md",
            "value": "maternal_death_rate",
        },
        "C-Section Rate (%)": {
            "numerator": "csection_deliveries",
            "denominator": "total_deliveries",
            "value": "csection_rate",
        },
        "Episiotomy Rate (%)": {
            "numerator": "episiotomy_cases",
            "denominator": "total_vaginal_deliveries",
            "value": "episiotomy_rate",
        },
        "SVD Rate (%)": {
            "numerator": "svd_deliveries",
            "denominator": "total_deliveries",
            "value": "svd_rate",
        },
        "Assisted Delivery Rate (%)": {
            "numerator": "assisted_deliveries",
            "denominator": "total_deliveries",
            "value": "assisted_rate",
        },
        "Antepartum Complications Rate (%)": {
            "numerator": "antipartum_cases",
            "denominator": "total_deliveries",
            "value": "antipartum_rate",
        },
        "Postpartum Complications Rate (%)": {
            "numerator": "postpartum_cases",
            "denominator": "total_deliveries",
            "value": "postpartum_rate",
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
    elif "Episiotomy" in kpi_name:
        numerator = kpi_data.get("episiotomy_cases", 0)
        denominator = kpi_data.get("total_vaginal_deliveries", 1)
        value = kpi_data.get("episiotomy_rate", 0.0)
        return (numerator, denominator, value)
    elif "SVD" in kpi_name or "Normal Vaginal" in kpi_name:
        numerator = kpi_data.get("svd_deliveries", 0)
        denominator = kpi_data.get("total_deliveries", 1)
        value = kpi_data.get("svd_rate", 0.0)
        return (numerator, denominator, value)
    elif "Assisted" in kpi_name or "Instrumental" in kpi_name:
        numerator = kpi_data.get("assisted_deliveries", 0)
        denominator = kpi_data.get("total_deliveries", 1)
        value = kpi_data.get("assisted_rate", 0.0)
        return (numerator, denominator, value)
    elif "Antepartum" in kpi_name:
        numerator = kpi_data.get("antipartum_cases", 0)
        denominator = kpi_data.get("total_deliveries", 1)
        value = kpi_data.get("antipartum_rate", 0.0)
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
    key_suffix="",
):
    # Create unique key
    if facility_uids:
        facility_str = "_".join(str(uid) for uid in sorted(facility_uids))
        unique_key = f"download_{title.replace(' ', '_').lower()}_trend_{facility_str}_{key_suffix}"
    else:
        unique_key = f"download_{title.replace(' ', '_').lower()}_trend_overall_{key_suffix}"

    """Render a trend chart for a single facility/region with numerator/denominator data AND TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Chart type selection removed - enforcing Line chart only


    if "numerator" in df.columns and "denominator" in df.columns:
        df[numerator_name] = df["numerator"]
        df[denominator_name] = df["denominator"]
        hover_columns = [numerator_name, denominator_name]
        use_hover_data = True
    else:
        hover_columns = []
        use_hover_data = False

    try:
        fig = px.line(
            df,
            x=x_axis_col,
            y=value_col,
            markers=True,
            line_shape="linear",
            title=title,
            height=400,
            custom_data=[numerator_name, denominator_name] if use_hover_data else None,
        )
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
        )

        # Apply standardized hover template
        if use_hover_data:
            is_rate = "Rate" in title or "%" in title or "Missing" in title or "missing" in title.lower()
            fig.update_traces(
                hovertemplate=get_attractive_hover_template(
                    title, numerator_name, denominator_name, is_count=not is_rate
                )
            )
        else:
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>"
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

    # Display metrics at the TOP for immediate visibility
    if len(df) > 0:
        col1, col2, col3 = st.columns(3)
        val_format = ".2f" if ("Rate" in title or "%" in title) else ",.0f"
        suffix = "%" if ("Rate" in title or "%" in title) else ""
        
        with col1:
            st.metric("📈 Latest Value", f"{df[value_col].iloc[-1]:{val_format}}{suffix}")
        with col2:
            st.metric("📊 Average", f"{df[value_col].mean():{val_format}}{suffix}")
        with col3:
            if len(df) > 1:
                last_value = df[value_col].iloc[-1]
                prev_value = df[value_col].iloc[-2]
                trend_change = last_value - prev_value
                trend_symbol = "▲" if trend_change > 0 else ("▼" if trend_change < 0 else "–")
                st.metric("📈 Trend", f"{trend_change:.2f}{suffix} {trend_symbol}")
            else:
                st.metric("📈 Trend", "–")

    # Display the chart with reduced height
    fig.update_layout(height=260, margin=dict(t=20, b=20, l=10, r=10))
    # Generate unique key for plotly chart
    chart_key = f"trend_chart_{title.replace(' ', '_')}_{str(facility_uids) if facility_uids else 'overall'}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # =========== COMPACT TABLE ===========
    with st.expander("📊 View Detailed Data Table", expanded=True):
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
            display_df[numerator_name] = display_df[numerator_name].apply(lambda x: f"{x:,.0f}")
        if denominator_name in display_df.columns:
            display_df[denominator_name] = display_df[denominator_name].apply(lambda x: f"{x:,.0f}")

        # Add Overall/Total row
        if "numerator" in df.columns and "denominator" in df.columns:
            total_numerator = df["numerator"].sum()
            total_denominator = df["denominator"].sum()
            overall_value = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        else:
            overall_value = df[value_col].mean() if not df.empty else 0
            total_numerator = df[value_col].sum() if not df.empty else 0
            total_denominator = len(df)

        # Create overall row
        overall_row = {
            x_axis_col: "Overall",
            value_col: f"{overall_value:.2f}%" if ("Rate" in title or "%" in title) else f"{overall_value:,.0f}",
        }

        if numerator_name in display_df.columns:
            overall_row[numerator_name] = f"{total_numerator:,.0f}"
        if denominator_name in display_df.columns:
            overall_row[denominator_name] = f"{total_denominator:,.0f}"

        # Convert to DataFrame and append
        overall_df = pd.DataFrame([overall_row]).reset_index(drop=True)
        display_df = pd.concat([display_df, overall_df], ignore_index=True)

        # Display the table
        st.dataframe(display_df, use_container_width=True)

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
        ).reset_index(drop=True)

        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    else:
        summary_df = summary_df[[x_axis_col, value_col]].copy()

        # FIX: Ensure period column is in proper format before exporting
        if x_axis_col in summary_df.columns:
            summary_df[x_axis_col] = summary_df[x_axis_col].apply(
                format_period_month_year
            )
    if x_axis_col in summary_df.columns:
        summary_df[x_axis_col] = summary_df[x_axis_col].apply(format_period_month_year)

    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_data.csv",
        mime="text/csv",
        key=f"dl_btn_{title}_{str(facility_uids)}_{key_suffix}",
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
    suppress_plot=False,
    key_suffix="",
):
    # Ensure index is clean
    df = df.reset_index(drop=True)
    
    # Create unique key
    facility_str = "_".join(str(uid) for uid in sorted(facility_uids))
    unique_key = f"download_{title.replace(' ', '_').lower()}_facility_{facility_str}_{key_suffix}"
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

    # If we have facility_names parameter, update/restrict the mapping
    if facility_names and facility_uids and len(facility_names) == len(facility_uids):
        # STRICT MODE: Only include the requested UIDs
        scoped_mapping = {}
        for uid, name in zip(facility_uids, facility_names):
            scoped_mapping[str(uid)] = name
        facility_mapping = scoped_mapping
    elif facility_uids:
        # Filter existing mapping to only include requested UIDs
        facility_mapping = {uid: name for uid, name in facility_mapping.items() if uid in [str(u) for u in facility_uids]}

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
                # Ensure numeric types to prevent 'int + str' errors
                numerator_val = pd.to_numeric(row.get("numerator", 0), errors='coerce')
                denominator_val = pd.to_numeric(row.get("denominator", 1), errors='coerce')
                value_val = pd.to_numeric(row.get(value_col, 0) if value_col in row else 0, errors='coerce')
                
                # Replace NaN with 0
                numerator_val = 0 if pd.isna(numerator_val) else numerator_val
                denominator_val = 1 if pd.isna(denominator_val) or denominator_val == 0 else denominator_val
                value_val = 0 if pd.isna(value_val) else value_val

                if numerator_val == 0 and denominator_val == 0:
                    continue  # Skip this period for this facility

                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Facility": facility_name,
                        "value": float(value_val),
                        "numerator": int(numerator_val),
                        "denominator": int(denominator_val),
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

    # Display Grand Overall Metric at the TOP
    if not comparison_df.empty:
        all_numerators = comparison_df["numerator"].sum()
        all_denominators = comparison_df["denominator"].sum()
        grand_overall = (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        
        # Determine if this is a rate KPI based on title
        is_rate_kpi = "Rate" in title or "%" in title or "Missing" in title or "missing" in title.lower()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            val_format = ".2f" if is_rate_kpi else ",.0f"
            suffix = "%" if is_rate_kpi else ""
            st.metric("🌍 Grand Overall Value", f"{grand_overall:{val_format}}{suffix}")

    # Create the chart with reduced height
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=f"{title} - Facility Comparison",
        height=350,
        category_orders={"period_display": period_order},
        custom_data=["numerator", "denominator"],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=get_attractive_hover_template(
            title, numerator_name, denominator_name, is_count=not is_rate_kpi
        ).replace("%{y", f"%{{fullData.name}}<br>{title}: %{{y"),
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

    if is_rate_kpi:
        fig.update_layout(yaxis_tickformat=".2f")

    # Generate unique key for facility comparison chart
    chart_key = f"facility_comp_{title.replace(' ', '_')}_{len(facility_uids) if facility_uids else 0}_{key_suffix}"
    
    if not suppress_plot:
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    else:
        st.info(f"💡 Showing comparison table only for **{title}**.")

    # =========== COMPACT TABLE ===========
    with st.expander("📋 View Facility Comparison Data", expanded=True):
        # Create pivot table for better display
        pivot_data = []

        for facility_name in comparison_df["Facility"].unique():
            facility_data = comparison_df[comparison_df["Facility"] == facility_name]
            if not facility_data.empty:
                total_numerator = facility_data["numerator"].sum()
                total_denominator = facility_data["denominator"].sum()
                overall_value = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0

                pivot_data.append({
                    "Facility": facility_name,
                    numerator_name: f"{total_numerator:,.0f}",
                    denominator_name: f"{total_denominator:,.0f}",
                    "Overall Value": f"{overall_value:.2f}%" if is_rate_kpi else f"{overall_value:,.0f}",
                })

        # Add Overall row
        if pivot_data:
            pivot_data.append({
                "Facility": "Overall",
                numerator_name: f"{all_numerators:,.0f}",
                denominator_name: f"{all_denominators:,.0f}",
                "Overall Value": f"{grand_overall:.2f}%" if is_rate_kpi else f"{grand_overall:,.0f}",
            })

            pivot_df = pd.DataFrame(pivot_data)
            st.dataframe(pivot_df, use_container_width=True)

    # Keep download functionality (one row per facility per period)
    period_label = get_current_period_label()
    if not comparison_df.empty:
        csv_df = comparison_df.copy()
        csv_df["Time Period"] = csv_df["period_display"].apply(
            lambda p: format_period_for_download(p, period_label)
        )
        csv_df = csv_df.rename(
            columns={"numerator": numerator_name, "denominator": denominator_name}
        )
        csv_df[title] = csv_df["value"].apply(
            lambda v: f"{float(v):.2f}%" if is_rate_kpi else f"{float(v):,.0f}"
        )
        csv_df = csv_df[
            ["Facility", "Time Period", numerator_name, denominator_name, title]
        ].reset_index(drop=True)
        st.download_button(
            label="📥 Download Overall Comparison Data",
            data=csv_df.to_csv(index=False),
            file_name=f"{title.lower().replace(' ', '_')}_facility_summary.csv",
            mime="text/csv",
            help="Download overall summary data for facility comparison",
            key=unique_key,
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
    suppress_plot=False,
    key_suffix="",
):
    # Ensure index is clean
    df = df.reset_index(drop=True)

    # Create unique key
    if region_mapping:
        region_str = "_".join(str(key) for key in region_mapping.keys())
        unique_key = f"download_{title.replace(' ', '_').lower()}_region_{region_str}_{key_suffix}"
    else:
        unique_key = f"download_{title.replace(' ', '_').lower()}_region_all_{key_suffix}"

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
    
    # Filter target regions if provided
    target_regions = df["Region"].unique()
    if region_names:
        # Convert all to same case for comparison to be safe
        target_regions = [r for r in target_regions if r in region_names or r.lower() in [n.lower() for n in region_names]]

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
    for region_name in target_regions:
        region_df = df[df["Region"] == region_name].copy()

        if region_df.empty:
            continue

        # Group by period for this region
        for period_display, period_group in region_df.groupby("period_display"):
            if not period_group.empty:
                # Get aggregated values for this region/period
                # Ensure numeric types to prevent 'int + str' errors
                avg_value = (
                    pd.to_numeric(period_group[value_col], errors='coerce').mean()
                    if value_col in period_group.columns
                    else 0
                )
                total_numerator = pd.to_numeric(period_group["numerator"], errors='coerce').sum()
                total_denominator_sum = pd.to_numeric(period_group["denominator"], errors='coerce').sum()
                total_denominator = (
                    total_denominator_sum
                    if total_denominator_sum > 0
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
                        "value": float(avg_value),
                        "numerator": int(total_numerator),
                        "denominator": int(total_denominator),
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available for regions.")
        return

    comparison_df = pd.DataFrame(comparison_data).reset_index(drop=True)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: dt.datetime.strptime(x, "%b-%y")
        )
        comparison_df = comparison_df.sort_values("period_sort").reset_index(drop=True)
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
    ].copy().reset_index(drop=True)

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all regions have zero data).")
        return

    # Display Grand Overall Metric at the TOP
    if not comparison_df.empty:
        all_numerators = comparison_df["numerator"].sum()
        all_denominators = comparison_df["denominator"].sum()
        grand_overall = (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        
        # Determine if this is a rate KPI based on title
        is_rate_kpi = "Rate" in title or "%" in title or "Missing" in title or "missing" in title.lower()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            val_format = ".2f" if is_rate_kpi else ",.0f"
            suffix = "%" if is_rate_kpi else ""
            st.metric("🌍 Grand Overall Value", f"{grand_overall:{val_format}}{suffix}")

    # Create the chart with reduced height
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=f"{title} - Region Comparison",
        height=350,
        category_orders={"period_display": period_order},
        custom_data=["numerator", "denominator"],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=get_attractive_hover_template(
            title, numerator_name, denominator_name, is_count=not is_rate_kpi
        ).replace("%{y", f"%{{fullData.name}}<br>{title}: %{{y"),
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

    if is_rate_kpi:
        fig.update_layout(yaxis_tickformat=".2f")

    # Generate unique key for region comparison chart
    chart_key = f"region_comp_{title.replace(' ', '_')}_{len(region_names) if region_names else 0}_{key_suffix}"
    
    if not suppress_plot:
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    else:
        st.info(f"💡 Showing comparison table only for **{title}**.")

    # =========== COMPACT TABLE ===========
    with st.expander("📋 View Region Comparison Data", expanded=True):
        # Create pivot table for better display
        pivot_data = []

        for region_name in comparison_df["Region"].unique():
            region_data = comparison_df[comparison_df["Region"] == region_name]
            if not region_data.empty:
                total_numerator = region_data["numerator"].sum()
                total_denominator = region_data["denominator"].sum()
                overall_value = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0

                pivot_data.append({
                    "Region": region_name,
                    numerator_name: f"{total_numerator:,.0f}",
                    denominator_name: f"{total_denominator:,.0f}",
                    "Overall Value": f"{overall_value:.2f}%" if is_rate_kpi else f"{overall_value:,.0f}",
                })

        # Add Overall row
        if pivot_data:
            pivot_data.append({
                "Region": "Overall",
                numerator_name: f"{all_numerators:,.0f}",
                denominator_name: f"{all_denominators:,.0f}",
                "Overall Value": f"{grand_overall:.2f}%" if is_rate_kpi else f"{grand_overall:,.0f}",
            })

            pivot_df = pd.DataFrame(pivot_data).reset_index(drop=True)
            st.dataframe(pivot_df, use_container_width=True)

    # Keep download functionality (one row per region per period)
    period_label = get_current_period_label()
    if not comparison_df.empty:
        csv_df = comparison_df.copy()
        csv_df["Time Period"] = csv_df["period_display"].apply(
            lambda p: format_period_for_download(p, period_label)
        )
        csv_df = csv_df.rename(
            columns={"numerator": numerator_name, "denominator": denominator_name}
        )
        csv_df[title] = csv_df["value"].apply(
            lambda v: f"{float(v):.2f}%" if is_rate_kpi else f"{float(v):,.0f}"
        )
        csv_df = csv_df[
            ["Region", "Time Period", numerator_name, denominator_name, title]
        ].reset_index(drop=True)
        st.download_button(
            label="📥 Download Overall Comparison Data",
            data=csv_df.to_csv(index=False),
            file_name=f"{title.lower().replace(' ', '_')}_region_summary.csv",
            mime="text/csv",
            help="Download overall summary data for region comparison",
            key=unique_key,
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
