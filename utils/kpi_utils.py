import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import streamlit as st
import hashlib
import numpy as np

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


# ---------------- KPI Constants ----------------
# Patient-level data columns
FP_ACCEPTANCE_COL = (
    "Q1p7CxWGUoi_VpBHRE7FlJL"  # FP_Counseling_and_Method_Provided_pp_Postpartum_care
)
FP_ACCEPTED_CODES = {
    "1",
    "2",
    "3",
    "4",
    "5",
}  # Pills, Injectables, Implants, IUCD, Condom

BIRTH_OUTCOME_COL = "wZig9cek3Gv_mdw5BoS50mb"  # Birth_Outcome_Delivery_summary
ALIVE_CODE = "1"
STILLBIRTH_CODE = "2"

DELIVERY_MODE_COL = (
    "lphtwP2ViZU_mdw5BoS50mb"  # Mode_of_Delivery_maternal_Delivery_summary
)
CSECTION_CODE = "2"

PNC_TIMING_COL = "z7Eb2yFLOBI_VpBHRE7FlJL"  # z7Eb2yFLOBI_VpBHRE7FlJL
PNC_EARLY_CODES = {"1", "2"}  # 1 = 24 hrs stay, 2 = 25-48 hrs

CONDITION_OF_DISCHARGE_COL = (
    "TjQOcW6tm8k_DLVsIxjhwMj"  # Condition_of_Discharge_Discharge_Summary
)
DEAD_CODE = "4"

NUMBER_OF_NEWBORNS_COL = (
    "VzwnSBROvUm_mdw5BoS50mb"  # Number_of_Newborns_Delivery_summary
)
OTHER_NUMBER_OF_NEWBORNS_COL = (
    "tIa0WvbPGLk_mdw5BoS50mb"  # Other_Number_of_Newborns_Delivery_summary
)

# Event status columns
HAS_ACTUAL_DELIVERY_COL = "has_actual_event_mdw5BoS50mb"
HAS_ACTUAL_PNC_COL = "has_actual_event_VpBHRE7FlJL"
HAS_ACTUAL_DISCHARGE_COL = "has_actual_event_DLVsIxjhwMj"

# Event date columns
DELIVERY_DATE_COL = "event_date_Delivery summary"
PNC_DATE_COL = "event_date_Postpartum care"
DISCHARGE_DATE_COL = "event_date_Discharge Summary"


def compute_birth_counts(df, facility_uids=None):
    """
    Compute birth counts accounting for multiple births (twins, triplets, etc.)
    Returns: total_births, live_births, stillbirths
    """
    # Generate cache key for this specific computation
    cache_key = get_cache_key(df, facility_uids, "birth_counts")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    # Check if we already computed this
    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0, 0, 0)
    else:
        # Filter by facilities if specified
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)]

        # Filter out placeholder events for calculation
        # Use delivery event status column
        if HAS_ACTUAL_DELIVERY_COL in df.columns:
            actual_events_df = df[df[HAS_ACTUAL_DELIVERY_COL] == True]
        else:
            actual_events_df = df

        # Initialize counts
        total_births = 0
        live_births = 0
        stillbirths = 0

        # Process each patient (each row is a patient in patient-level format)
        for idx, row in actual_events_df.iterrows():
            # Get number of newborns - this should represent the total babies born
            num_newborns = 0
            if NUMBER_OF_NEWBORNS_COL in row and pd.notna(row[NUMBER_OF_NEWBORNS_COL]):
                try:
                    num_newborns = int(float(row[NUMBER_OF_NEWBORNS_COL]))
                except (ValueError, TypeError):
                    num_newborns = 0  # If invalid, count as 0

            num_other_newborns = 0
            if OTHER_NUMBER_OF_NEWBORNS_COL in row and pd.notna(
                row[OTHER_NUMBER_OF_NEWBORNS_COL]
            ):
                try:
                    num_other_newborns = int(float(row[OTHER_NUMBER_OF_NEWBORNS_COL]))
                except (ValueError, TypeError):
                    num_other_newborns = 0  # If invalid, count as 0

            total_babies = num_newborns + num_other_newborns

            # If no newborn count records exist, we cannot assume any births occurred
            # Only count births if we have actual data
            if total_babies == 0:
                # Check if we have birth outcome records as evidence of births
                if BIRTH_OUTCOME_COL in row and pd.notna(row[BIRTH_OUTCOME_COL]):
                    # For patient-level data, we can't easily count multiple outcomes per patient
                    # If there's a birth outcome, assume at least 1 birth
                    total_babies = 1
                else:
                    # No evidence of any births for this patient, skip
                    continue

            # Get birth outcome - in patient-level, we need to handle multiple outcomes
            # For simplicity, we'll assume single outcome per patient for now
            if BIRTH_OUTCOME_COL in row and pd.notna(row[BIRTH_OUTCOME_COL]):
                outcome = str(row[BIRTH_OUTCOME_COL])
                if outcome == ALIVE_CODE:
                    alive_count = total_babies  # Assume all babies are alive
                    stillbirth_count = 0
                elif outcome == STILLBIRTH_CODE:
                    alive_count = 0
                    stillbirth_count = total_babies  # Assume all are stillbirths
                else:
                    # Unknown outcome - can't determine
                    alive_count = 0
                    stillbirth_count = 0
            else:
                # No outcome recorded
                alive_count = 0
                stillbirth_count = 0

            # Update counts
            live_births += alive_count
            stillbirths += stillbirth_count
            total_births += total_babies

        result = (total_births, live_births, stillbirths)

    # Store result in cache
    st.session_state.kpi_cache[cache_key] = result
    return result


# ---------------- SEPARATE NUMERATOR COMPUTATION FUNCTIONS ----------------
def compute_fp_acceptance_count(df, facility_uids=None):
    """Count FP acceptance occurrences (not unique patients)"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for numerator calculation
    # Use PNC event status column
    if HAS_ACTUAL_PNC_COL in df.columns:
        actual_events_df = df[df[HAS_ACTUAL_PNC_COL] == True]
    else:
        actual_events_df = df

    # Count occurrences where FP acceptance is recorded with accepted codes
    fp_count = 0
    for _, row in actual_events_df.iterrows():
        if FP_ACCEPTANCE_COL in row and pd.notna(row[FP_ACCEPTANCE_COL]):
            if str(row[FP_ACCEPTANCE_COL]) in FP_ACCEPTED_CODES:
                fp_count += 1

    return fp_count


def compute_early_pnc_count(df, facility_uids=None):
    """Count early PNC occurrences (not unique patients)"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for numerator calculation
    if HAS_ACTUAL_PNC_COL in df.columns:
        actual_events_df = df[df[HAS_ACTUAL_PNC_COL] == True]
    else:
        actual_events_df = df

    # Count occurrences where early PNC timing is recorded
    pnc_count = 0
    for _, row in actual_events_df.iterrows():
        if PNC_TIMING_COL in row and pd.notna(row[PNC_TIMING_COL]):
            if str(row[PNC_TIMING_COL]) in PNC_EARLY_CODES:
                pnc_count += 1

    return pnc_count


def compute_csection_count(df, facility_uids=None):
    """Count C-section occurrences (not unique patients)"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for numerator calculation
    if HAS_ACTUAL_DELIVERY_COL in df.columns:
        actual_events_df = df[df[HAS_ACTUAL_DELIVERY_COL] == True]
    else:
        actual_events_df = df

    # Count occurrences where delivery mode is C-section
    csection_count = 0
    for _, row in actual_events_df.iterrows():
        if DELIVERY_MODE_COL in row and pd.notna(row[DELIVERY_MODE_COL]):
            if str(row[DELIVERY_MODE_COL]) == CSECTION_CODE:
                csection_count += 1

    return csection_count


def compute_maternal_death_count(df, facility_uids=None):
    """Count maternal death occurrences (not unique patients)"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for calculation
    if HAS_ACTUAL_DISCHARGE_COL in df.columns:
        actual_events_df = df[df[HAS_ACTUAL_DISCHARGE_COL] == True]
    else:
        actual_events_df = df

    # Maternal deaths - count occurrences where condition of discharge is DEAD
    deaths_count = 0
    for _, row in actual_events_df.iterrows():
        if CONDITION_OF_DISCHARGE_COL in row and pd.notna(
            row[CONDITION_OF_DISCHARGE_COL]
        ):
            if str(row[CONDITION_OF_DISCHARGE_COL]) == DEAD_CODE:
                deaths_count += 1

    return deaths_count


def compute_stillbirth_count(df, facility_uids=None):
    """Count stillbirth occurrences (not unique patients)"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for calculation
    if HAS_ACTUAL_DELIVERY_COL in df.columns:
        actual_events_df = df[df[HAS_ACTUAL_DELIVERY_COL] == True]
    else:
        actual_events_df = df

    # Count occurrences where birth outcome is stillbirth
    stillbirth_count = 0
    for _, row in actual_events_df.iterrows():
        if BIRTH_OUTCOME_COL in row and pd.notna(row[BIRTH_OUTCOME_COL]):
            if str(row[BIRTH_OUTCOME_COL]) == STILLBIRTH_CODE:
                stillbirth_count += 1

    return stillbirth_count


# ---------------- KPI Computation Functions ----------------
def compute_total_deliveries(df, facility_uids=None):
    """Count total deliveries - FIXED to count unique patients regardless of event status"""
    # Generate cache key for this specific computation
    cache_key = get_cache_key(df, facility_uids, "total_deliveries")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    # Check if we already computed this
    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    # Count unique patients (tei_id) regardless of has_actual_event status
    if df is None or df.empty:
        result = 0
    else:
        # Filter by facilities if specified
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)]

        # FIX: Count ALL unique patients in the dataset
        # (Assuming the dataset contains only patients with deliveries)
        if "tei_id" in df.columns:
            # Count unique patients (tei_id)
            result = df["tei_id"].nunique()
        else:
            # If no tei_id column, count all rows as fallback
            result = len(df)

    # Store result in cache
    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_fp_acceptance(df, facility_uids=None):
    cache_key = get_cache_key(df, facility_uids, "fp_acceptance")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    # Use the new separate numerator computation
    result = compute_fp_acceptance_count(df, facility_uids)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_stillbirth_rate(df, facility_uids=None):
    """Compute stillbirth rate per 1000 deliveries"""
    cache_key = get_cache_key(df, facility_uids, "stillbirth_rate")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Filter by facilities if specified
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)]

        # Count stillbirths
        stillbirths = compute_stillbirth_count(df, facility_uids)

        # Count total deliveries (consistent denominator for all KPIs)
        total_deliveries = compute_total_deliveries(df, facility_uids)

        # Calculate stillbirth rate per 1000 deliveries
        rate = (stillbirths / total_deliveries * 1000) if total_deliveries > 0 else 0.0
        result = (rate, stillbirths, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_early_pnc_coverage(df, facility_uids=None):
    cache_key = get_cache_key(df, facility_uids, "pnc_coverage")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)]

        # Use the separate numerator computation
        early_pnc = compute_early_pnc_count(df, facility_uids)

        # Use the SAME total deliveries logic for consistency
        total_deliveries = compute_total_deliveries(df, facility_uids)

        coverage = (early_pnc / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (coverage, early_pnc, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_maternal_death_rate(df, facility_uids=None):
    """Compute maternal death rate per 100,000 deliveries"""
    cache_key = get_cache_key(df, facility_uids, "maternal_death_rate")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Filter by facilities if specified
        if facility_uids:
            df = df[df["orgUnit"].isin(facility_uids)]

        # Use the separate numerator computation
        maternal_deaths = compute_maternal_death_count(df, facility_uids)

        # Use total deliveries as denominator (consistent with other KPIs)
        total_deliveries = compute_total_deliveries(df, facility_uids)

        # Compute rate per 100,000 deliveries
        rate = (
            (maternal_deaths / total_deliveries * 100000)
            if total_deliveries > 0
            else 0.0
        )
        result = (rate, maternal_deaths, total_deliveries)

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_csection_rate(df, facility_uids=None):
    cache_key = get_cache_key(df, facility_uids, "csection_rate")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Use the separate numerator computation
        csection_deliveries = compute_csection_count(df, facility_uids)

        # Use the SAME total deliveries logic for consistency
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
    cache_key = get_cache_key(df, facility_uids, "main_kpis")

    # ✅ SAFE ACCESS: Check if cache exists first
    if "kpi_cache" not in st.session_state:
        st.session_state.kpi_cache = {}

    # ✅ NOW safe to access
    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Use consistent total deliveries across all KPIs
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Use updated functions with separate numerator computation
    fp_acceptance = compute_fp_acceptance(df, facility_uids)
    ippcar = (fp_acceptance / total_deliveries * 100) if total_deliveries > 0 else 0.0

    # Use updated functions with consistent denominator
    stillbirth_rate, stillbirths, total_deliveries_sb = compute_stillbirth_rate(
        df, facility_uids
    )

    pnc_coverage, early_pnc, total_deliveries_pnc = compute_early_pnc_coverage(
        df, facility_uids
    )

    # Use updated maternal death rate function with total deliveries denominator
    maternal_death_rate, maternal_deaths, total_deliveries_md = (
        compute_maternal_death_rate(df, facility_uids)
    )

    csection_rate, csection_deliveries, total_deliveries_cs = compute_csection_rate(
        df, facility_uids
    )

    # Also compute birth counts for reference (but not used in main KPIs)
    total_births, live_births, stillbirths_count = compute_birth_counts(
        df, facility_uids
    )

    result = {
        "total_deliveries": int(total_deliveries),
        "fp_acceptance": int(fp_acceptance),
        "ippcar": float(ippcar),
        "stillbirth_rate": float(stillbirth_rate),
        "stillbirths": int(stillbirths),
        "total_deliveries_sb": int(total_deliveries),  # Same as total_deliveries
        "pnc_coverage": float(pnc_coverage),
        "early_pnc": int(early_pnc),
        "total_deliveries_pnc": int(total_deliveries),  # Same as total_deliveries
        "maternal_death_rate": float(maternal_death_rate),
        "maternal_deaths": int(maternal_deaths),
        "total_deliveries_md": int(total_deliveries),  # Same as total_deliveries
        "live_births": int(live_births),  # For reference only
        "total_births": int(total_births),  # For reference only
        "stillbirths_count": int(stillbirths_count),  # For reference only
        "csection_rate": float(csection_rate),
        "csection_deliveries": int(csection_deliveries),
        "total_deliveries_cs": int(total_deliveries),  # Same as total_deliveries
    }

    # Cache the results
    st.session_state.kpi_cache[cache_key] = result

    return result


# ---------------- Helper Functions for Patient-Level Data ----------------
def extract_event_date_for_period(df, event_name):
    """
    Extract event date for period grouping from patient-level data
    event_name should be one of: "Delivery summary", "Postpartum care", "Discharge Summary"
    """
    if df.empty:
        return pd.DataFrame()

    # Map event names to column names in patient-level data
    event_date_columns = {
        "Delivery summary": "event_date_Delivery summary",
        "Postpartum care": "event_date_Postpartum care",
        "Discharge Summary": "event_date_Discharge Summary",
    }

    result_df = df.copy()

    # Add period columns if they exist
    if event_date_columns.get(event_name) in df.columns:
        result_df["event_date"] = pd.to_datetime(
            result_df[event_date_columns[event_name]], errors="coerce"
        )
        result_df["period"] = result_df["event_date"].dt.strftime("%Y-%m")
        result_df["period_display"] = result_df["event_date"].dt.strftime("%b-%y")
        result_df["period_sort"] = result_df["event_date"].dt.strftime("%Y%m")

    return result_df


def get_numerator_denominator_for_kpi(df, kpi_name, facility_uids=None):
    """
    Get numerator and denominator for a specific KPI
    Returns: (numerator, denominator, value)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Compute KPIs
    kpi_data = compute_kpis(df, facility_uids)

    # Map KPI names to the right values
    kpi_mapping = {
        "Immediate Postpartum Contraceptive Acceptance Rate (IPPCAR %)": {
            "numerator": "fp_acceptance",
            "denominator": "total_deliveries",
            "value": "ippcar",
        },
        "Stillbirth Rate (per 1000 births)": {
            "numerator": "stillbirths",
            "denominator": "total_deliveries_sb",
            "value": "stillbirth_rate",
        },
        "Early Postnatal Care (PNC) Coverage (%)": {
            "numerator": "early_pnc",
            "denominator": "total_deliveries",
            "value": "pnc_coverage",
        },
        "Institutional Maternal Death Rate (per 100,000 births)": {
            "numerator": "maternal_deaths",
            "denominator": "total_deliveries_md",
            "value": "maternal_death_rate",
        },
        "C-Section Rate (%)": {
            "numerator": "csection_deliveries",
            "denominator": "total_deliveries",
            "value": "csection_rate",
        },
        "Postpartum Hemorrhage (PPH) Rate (%)": {
            "numerator": "pph_count",
            "denominator": "total_deliveries",
            "value": "pph_rate",
        },
        "Delivered women who received uterotonic (%)": {
            "numerator": "uterotonic_count",
            "denominator": "total_deliveries",
            "value": "uterotonic_rate",
        },
        "ARV Prophylaxis Rate (%)": {
            "numerator": "arv_count",
            "denominator": "hiv_exposed_infants",
            "value": "arv_rate",
        },
        "Low Birth Weight (LBW) Rate (%)": {
            "numerator": "lbw_count",
            "denominator": "total_weighed",
            "value": "lbw_rate",
        },
        "Assisted Delivery Rate (%)": {
            "numerator": "assisted_deliveries",
            "denominator": "total_deliveries",
            "value": "assisted_delivery_rate",
        },
        "Normal Vaginal Delivery (SVD) Rate (%)": {
            "numerator": "svd_deliveries",
            "denominator": "total_deliveries",
            "value": "svd_rate",
        },
        "Missing Mode of Delivery": {
            "numerator": "missing_md_count",
            "denominator": "total_deliveries",
            "value": "missing_md_rate",
        },
    }

    # Check if we have a mapping for this KPI
    if kpi_name in kpi_mapping:
        mapping = kpi_mapping[kpi_name]
        numerator = kpi_data.get(mapping["numerator"], 0)
        denominator = kpi_data.get(mapping["denominator"], 1)
        value = kpi_data.get(mapping["value"], 0.0)
        return (numerator, denominator, value)

    # For generic KPI matching
    if "IPPCAR" in kpi_name or "Contraceptive" in kpi_name:
        return (
            kpi_data.get("fp_acceptance", 0),
            kpi_data.get("total_deliveries", 1),
            kpi_data.get("ippcar", 0.0),
        )
    elif "Stillbirth" in kpi_name:
        return (
            kpi_data.get("stillbirths", 0),
            kpi_data.get("total_deliveries_sb", 1),
            kpi_data.get("stillbirth_rate", 0.0),
        )
    elif "PNC" in kpi_name or "Postnatal" in kpi_name:
        return (
            kpi_data.get("early_pnc", 0),
            kpi_data.get("total_deliveries", 1),
            kpi_data.get("pnc_coverage", 0.0),
        )
    elif "Maternal Death" in kpi_name:
        return (
            kpi_data.get("maternal_deaths", 0),
            kpi_data.get("total_deliveries_md", 1),
            kpi_data.get("maternal_death_rate", 0.0),
        )
    elif "C-Section" in kpi_name:
        return (
            kpi_data.get("csection_deliveries", 0),
            kpi_data.get("total_deliveries", 1),
            kpi_data.get("csection_rate", 0.0),
        )

    # Default fallback
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

    # Group by both display and sort columns to preserve order
    grouped = df.groupby([period_col, period_sort_col])

    result_data = []
    for (period_display, period_sort), group_df in grouped:
        # Compute KPI for this period group
        if kpi_name:
            # Use the specific KPI computation
            numerator, denominator, value = get_numerator_denominator_for_kpi(
                group_df, kpi_name, facility_uids
            )
        else:
            # Use generic KPI function
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

    # Sort by period_sort column
    if not result_df.empty and period_sort_col in result_df.columns:
        result_df = result_df.sort_values(period_sort_col)

    return result_df


# ---------------- Chart Functions ----------------
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
    show_table=False,
):
    """Render a trend chart for a single facility/region with numerator/denominator data"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Chart type selection
    chart_options = ["Line", "Bar", "Area"]

    if "Maternal Death Rate" in title:
        chart_options = ["Line", "Bar"]
    elif "Stillbirth Rate" in title:
        chart_options = ["Line", "Bar"]
    elif "C-Section Rate" in title:
        chart_options = ["Line", "Bar"]

    # Add radio button for chart type selection
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_{title}_{str(facility_uids)}",
    ).lower()

    # Prepare hover data
    if "numerator" in df.columns and "denominator" in df.columns:
        df[numerator_name] = df["numerator"]
        df[denominator_name] = df["denominator"]
        hover_columns = [numerator_name, denominator_name]
        use_hover_data = True
    else:
        hover_columns = []
        use_hover_data = False

    # Create chart based on selected type
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

    st.plotly_chart(fig, use_container_width=True)

    # Trend analysis
    if len(df) > 1:
        last_value = df[value_col].iloc[-1]
        prev_value = df[value_col].iloc[-2]
        trend_symbol = (
            "▲"
            if last_value > prev_value
            else ("▼" if last_value < prev_value else "–")
        )
        trend_class = (
            "trend-up"
            if last_value > prev_value
            else ("trend-down" if last_value < prev_value else "trend-neutral")
        )
        st.markdown(
            f'<p style="font-size:1.2rem;font-weight:600;">Latest Value: {last_value:.2f} <span class="{trend_class}">{trend_symbol}</span></p>',
            unsafe_allow_html=True,
        )

    # Prepare summary data for CSV download
    summary_df = df.copy().reset_index(drop=True)

    # Ensure we have the right columns
    if "numerator" in summary_df.columns and "denominator" in summary_df.columns:
        # Create a new DataFrame with properly named columns
        summary_df = summary_df[
            [x_axis_col, "numerator", "denominator", value_col]
        ].copy()

        # Use the passed numerator and denominator names
        summary_df = summary_df.rename(
            columns={
                "numerator": numerator_name,
                "denominator": denominator_name,
                value_col: title,
            }
        )

        # Calculate overall using the actual formula
        total_numerator = summary_df[numerator_name].sum()
        total_denominator = summary_df[denominator_name].sum()

        # Calculate overall based on KPI type
        if "IPPCAR" in title or "Coverage" in title or "C-Section Rate" in title:
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )
        elif "Stillbirth Rate" in title:
            overall_value = (
                (total_numerator / total_denominator * 1000)
                if total_denominator > 0
                else 0
            )
        elif "Maternal Death Rate" in title:
            overall_value = (
                (total_numerator / total_denominator * 100000)
                if total_denominator > 0
                else 0
            )
        else:
            overall_value = summary_df[title].mean() if not summary_df.empty else 0

        # Add overall row
        overall_row = pd.DataFrame(
            {
                x_axis_col: [f"Overall {title}"],
                numerator_name: [total_numerator],
                denominator_name: [total_denominator],
                title: [overall_value],
            }
        )

        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    else:
        # If no numerator/denominator columns, just show the value
        summary_df = summary_df[[x_axis_col, value_col]].copy()
        summary_df = summary_df.rename(columns={value_col: title})
        summary_table = summary_df.copy()

        # Calculate overall as average
        overall_value = summary_table[title].mean() if not summary_table.empty else 0
        overall_row = pd.DataFrame(
            {x_axis_col: [f"Overall {title}"], title: [overall_value]}
        )
        summary_table = pd.concat([summary_table, overall_row], ignore_index=True)

    # Add row numbering
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Only show table if show_table parameter is True
    if show_table:
        st.subheader(f"📋 {title} Summary Table")

        # Create a styled dataframe using st.dataframe
        styled_df = summary_table.copy()

        # Format numeric columns
        if (
            numerator_name in styled_df.columns
            and denominator_name in styled_df.columns
        ):
            # Apply formatting to the display dataframe
            display_df = styled_df.copy()

            # Apply formatting to numeric columns
            for col in [numerator_name, denominator_name]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{int(x):,}" if pd.notnull(x) else "0"
                    )

            # Format the title (KPI value) column
            if title in display_df.columns:
                display_df[title] = display_df[title].apply(
                    lambda x: f"{float(x):.2f}" if pd.notnull(x) else "0.00"
                )

            # Display the dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "No": st.column_config.NumberColumn("No", width="small"),
                    x_axis_col: st.column_config.TextColumn(x_axis_col, width="medium"),
                    numerator_name: st.column_config.TextColumn(
                        numerator_name, width="medium"
                    ),
                    denominator_name: st.column_config.TextColumn(
                        denominator_name, width="medium"
                    ),
                    title: st.column_config.TextColumn(title, width="medium"),
                },
            )
        else:
            # Simple display for value-only tables
            display_df = styled_df.copy()
            if title in display_df.columns:
                display_df[title] = display_df[title].apply(
                    lambda x: f"{float(x):.2f}" if pd.notnull(x) else "0.00"
                )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    # Always show download button for CSV
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
    show_table=False,
):
    """Render a comparison chart showing each facility's performance over time"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Check if orgUnit column exists - FIX for KeyError
    if "orgUnit" not in df.columns:
        st.error(
            "❌ Column 'orgUnit' not found in the data. Cannot perform facility comparison."
        )
        st.info("The data must contain an 'orgUnit' column with facility IDs.")
        return

    # Create a mapping from facility UID to name
    facility_uid_to_name = dict(zip(facility_uids, facility_names))

    # Filter to only include selected facilities
    filtered_df = df[df["orgUnit"].isin(facility_uids)].copy()

    if filtered_df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Group by period AND facility, then compute the KPI
    comparison_data = []

    # Extract period information from patient-level data
    # Use delivery date for most KPIs
    if DELIVERY_DATE_COL in filtered_df.columns:
        filtered_df["period_date"] = pd.to_datetime(
            filtered_df[DELIVERY_DATE_COL], errors="coerce"
        )
    elif PNC_DATE_COL in filtered_df.columns:
        filtered_df["period_date"] = pd.to_datetime(
            filtered_df[PNC_DATE_COL], errors="coerce"
        )
    else:
        # Fallback to any date column
        date_columns = [col for col in filtered_df.columns if "event_date" in col]
        if date_columns:
            filtered_df["period_date"] = pd.to_datetime(
                filtered_df[date_columns[0]], errors="coerce"
            )
        else:
            filtered_df["period_date"] = pd.NaT

    # Create period columns
    filtered_df["period"] = filtered_df["period_date"].dt.strftime("%Y-%m")
    filtered_df["period_display"] = filtered_df["period_date"].dt.strftime("%b-%y")
    filtered_df["period_sort"] = filtered_df["period_date"].dt.strftime("%Y%m")

    # Get all unique periods in chronological order
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    for period_display in period_order:
        period_data = all_periods[all_periods["period_display"] == period_display]
        if period_data.empty:
            continue

        period_sort = period_data["period_sort"].iloc[0]
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_df = period_df[period_df["orgUnit"] == facility_uid]
            if not facility_df.empty:
                kpi_value = compute_kpis(facility_df, [facility_uid])

                if "IPPCAR" in title:
                    value = kpi_value["ippcar"]
                    numerator = kpi_value["fp_acceptance"]
                    denominator = kpi_value["total_deliveries"]
                elif "Stillbirth Rate" in title:
                    value = kpi_value["stillbirth_rate"]
                    numerator = kpi_value["stillbirths"]
                    denominator = kpi_value["total_deliveries"]
                elif "PNC Coverage" in title:
                    value = kpi_value["pnc_coverage"]
                    numerator = kpi_value["early_pnc"]
                    denominator = kpi_value["total_deliveries"]
                elif "Maternal Death Rate" in title:
                    value = kpi_value["maternal_death_rate"]
                    numerator = kpi_value["maternal_deaths"]
                    denominator = kpi_value["total_deliveries"]
                elif "C-Section Rate" in title:
                    value = kpi_value["csection_rate"]
                    numerator = kpi_value["csection_deliveries"]
                    denominator = kpi_value["total_deliveries"]
                else:
                    value = 0
                    numerator = 0
                    denominator = 1

                comparison_data.append(
                    {
                        "period_display": period_display,
                        "period_sort": period_sort,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": value,
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Create the chart - FORCE the x-axis order
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=f"{title} - Facility Comparison",
        height=500,
        category_orders={"period_display": period_order},  # This forces the order
        hover_data=["numerator", "denominator"],  # Add hover data
    )

    # Update hover template to show proper labels
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
        xaxis_title="Period",
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

    # Show facility comparison table (summary) with styling only if requested
    if show_table:
        st.subheader("📋 Facility Comparison Summary")
        facility_table_data = []

        for facility_name, facility_uid in zip(facility_names, facility_uids):
            facility_df = df[df["orgUnit"] == facility_uid]
            if facility_df.empty:
                continue

            kpi_data = compute_kpis(facility_df, [facility_uid])

            if "IPPCAR" in title:
                numerator = kpi_data["fp_acceptance"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["ippcar"]
            elif "Stillbirth Rate" in title:
                numerator = kpi_data["stillbirths"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["stillbirth_rate"]
            elif "PNC Coverage" in title:
                numerator = kpi_data["early_pnc"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["pnc_coverage"]
            elif "Maternal Death Rate" in title:
                numerator = kpi_data["maternal_deaths"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["maternal_death_rate"]
            elif "C-Section Rate" in title:
                numerator = kpi_data["csection_deliveries"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["csection_rate"]
            else:
                # fallback
                numerator = 0
                denominator = 0
                kpi_value = 0

            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    numerator_name: numerator,
                    denominator_name: denominator,
                    title: kpi_value,
                }
            )

        if not facility_table_data:
            st.info("⚠️ No data available for facility comparison table.")
            return

        facility_table_df = pd.DataFrame(facility_table_data)

        # Compute overall aggregated KPI using table columns (robust to labels)
        overall_numerator = facility_table_df[numerator_name].sum()
        overall_denominator = facility_table_df[denominator_name].sum()

        if "IPPCAR" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 100)
                if overall_denominator > 0
                else 0
            )
        elif "Stillbirth Rate" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 1000)
                if overall_denominator > 0
                else 0
            )
        elif "PNC Coverage" in title or "C-Section Rate" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 100)
                if overall_denominator > 0
                else 0
            )
        elif "Maternal Death Rate" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 100000)
                if overall_denominator > 0
                else 0
            )
        else:
            overall_value = (
                facility_table_df[title].mean() if not facility_table_df.empty else 0
            )

        overall_row = {
            "Facility Name": f"Overall {title}",
            numerator_name: overall_numerator,
            denominator_name: overall_denominator,
            title: overall_value,
        }

        facility_table_df = pd.concat(
            [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
        )
        # Add row numbering
        facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

        # Format & display with CSS styling
        html = """
        <style>
        .facility-comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 10px 0;
        }
        .facility-comparison-table th {
            background-color: #2c3e50;
            color: white;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: bold;
        }
        .facility-comparison-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .facility-comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        .facility-comparison-table .overall-row {
            background-color: #e8f4fd;
            font-weight: bold;
        }
        .facility-comparison-table .numeric {
            text-align: right;
        }
        </style>
        <table class="facility-comparison-table">
        <thead>
            <tr>
                <th>No</th>
                <th>Facility Name</th>
                <th class="numeric">{num_name}</th>
                <th class="numeric">{den_name}</th>
                <th class="numeric">{title_name}</th>
            </tr>
        </thead>
        <tbody>
        """.format(
            num_name=numerator_name, den_name=denominator_name, title_name=title
        )

        for i, row in facility_table_df.iterrows():
            row_class = (
                "overall-row" if f"Overall {title}" in str(row["Facility Name"]) else ""
            )
            html += f"""
            <tr class="{row_class}">
                <td>{int(row['No'])}</td>
                <td>{row['Facility Name']}</td>
                <td class="numeric">{row[numerator_name]:,.0f}</td>
                <td class="numeric">{row[denominator_name]:,.0f}</td>
                <td class="numeric">{row[title]:.2f}</td>
            </tr>
            """

        html += "</tbody></table>"

        st.markdown(html, unsafe_allow_html=True)

    # Always show download button for CSV
    csv_data = []
    for period_display in period_order:
        for facility_uid, facility_name in zip(facility_uids, facility_names):
            # Find matching data
            matching_rows = comparison_df[
                (comparison_df["period_display"] == period_display)
                & (comparison_df["Facility"] == facility_name)
            ]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                csv_data.append(
                    {
                        "Period": period_display,
                        "Facility": facility_name,
                        numerator_name: row["numerator"],
                        denominator_name: row["denominator"],
                        title: row["value"],
                    }
                )

    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Data as CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_facility_comparison.csv",
            mime="text/csv",
            help="Download the exact facility comparison data shown in the chart",
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
    show_table=False,
):
    """Render a comparison chart showing each region's performance over time"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Check if orgUnit column exists
    if "orgUnit" not in df.columns:
        st.error(
            "❌ Column 'orgUnit' not found in the data. Cannot perform region comparison."
        )
        st.info("The data must contain an 'orgUnit' column with facility IDs.")
        return

    # Get all facility UIDs for selected regions
    all_facility_uids = []
    for region_name in region_names:
        facility_uids = [uid for _, uid in facilities_by_region.get(region_name, [])]
        all_facility_uids.extend(facility_uids)

    # Filter to only include facilities in selected regions
    filtered_df = df[df["orgUnit"].isin(all_facility_uids)].copy()

    if filtered_df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Group by period AND region, then compute the KPI
    comparison_data = []

    # Extract period information from patient-level data
    if DELIVERY_DATE_COL in filtered_df.columns:
        filtered_df["period_date"] = pd.to_datetime(
            filtered_df[DELIVERY_DATE_COL], errors="coerce"
        )
    elif PNC_DATE_COL in filtered_df.columns:
        filtered_df["period_date"] = pd.to_datetime(
            filtered_df[PNC_DATE_COL], errors="coerce"
        )
    else:
        # Fallback to any date column
        date_columns = [col for col in filtered_df.columns if "event_date" in col]
        if date_columns:
            filtered_df["period_date"] = pd.to_datetime(
                filtered_df[date_columns[0]], errors="coerce"
            )
        else:
            filtered_df["period_date"] = pd.NaT

    # Create period columns
    filtered_df["period"] = filtered_df["period_date"].dt.strftime("%Y-%m")
    filtered_df["period_display"] = filtered_df["period_date"].dt.strftime("%b-%y")
    filtered_df["period_sort"] = filtered_df["period_date"].dt.strftime("%Y%m")

    # Get all unique periods in chronological order
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    for period_display in period_order:
        period_data = all_periods[all_periods["period_display"] == period_display]
        if period_data.empty:
            continue

        period_sort = period_data["period_sort"].iloc[0]
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for region_name in region_names:
            # Get all facility UIDs for this region
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_df = period_df[period_df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                kpi_value = compute_kpis(region_df, region_facility_uids)

                if "IPPCAR" in title:
                    value = kpi_value["ippcar"]
                    numerator = kpi_value["fp_acceptance"]
                    denominator = kpi_value["total_deliveries"]
                elif "Stillbirth Rate" in title:
                    value = kpi_value["stillbirth_rate"]
                    numerator = kpi_value["stillbirths"]
                    denominator = kpi_value["total_deliveries"]
                elif "PNC Coverage" in title:
                    value = kpi_value["pnc_coverage"]
                    numerator = kpi_value["early_pnc"]
                    denominator = kpi_value["total_deliveries"]
                elif "Maternal Death Rate" in title:
                    value = kpi_value["maternal_death_rate"]
                    numerator = kpi_value["maternal_deaths"]
                    denominator = kpi_value["total_deliveries"]
                elif "C-Section Rate" in title:
                    value = kpi_value["csection_rate"]
                    numerator = kpi_value["csection_deliveries"]
                    denominator = kpi_value["total_deliveries"]
                else:
                    value = 0
                    numerator = 0
                    denominator = 1

                comparison_data.append(
                    {
                        "period_display": period_display,
                        "period_sort": period_sort,
                        "Region": region_name,
                        "value": value,
                        "numerator": numerator,
                        "denominator": denominator,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for region comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Create the chart - FORCE the x-axis order
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=f"{title} - Region Comparison",
        height=500,
        category_orders={"period_display": period_order},  # This forces the order
        hover_data=["numerator", "denominator"],  # Add hover data
    )

    # Update hover template to show proper labels
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
        xaxis_title="Period",
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

    # Region comparison summary table only if requested
    if show_table:
        st.subheader("📋 Region Comparison Summary")
        region_table_data = []
        for region_name in region_names:
            facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_df = df[df["orgUnit"].isin(facility_uids)]
            if region_df.empty:
                continue

            kpi_data = compute_kpis(region_df, facility_uids)
            if "IPPCAR" in title:
                numerator = kpi_data["fp_acceptance"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["ippcar"]
            elif "Stillbirth Rate" in title:
                numerator = kpi_data["stillbirths"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["stillbirth_rate"]
            elif "PNC Coverage" in title:
                numerator = kpi_data["early_pnc"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["pnc_coverage"]
            elif "Maternal Death Rate" in title:
                numerator = kpi_data["maternal_deaths"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["maternal_death_rate"]
            elif "C-Section Rate" in title:
                numerator = kpi_data["csection_deliveries"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["csection_rate"]
            else:
                numerator = 0
                denominator = 0
                kpi_value = 0

            region_table_data.append(
                {
                    "Region Name": region_name,
                    numerator_name: numerator,
                    denominator_name: denominator,
                    title: kpi_value,
                }
            )

        if not region_table_data:
            st.info("⚠️ No data available for region comparison table.")
            return

        region_table_df = pd.DataFrame(region_table_data)

        # Compute overall
        overall_numerator = region_table_df[numerator_name].sum()
        overall_denominator = region_table_df[denominator_name].sum()

        if "IPPCAR" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 100)
                if overall_denominator > 0
                else 0
            )
        elif "Stillbirth Rate" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 1000)
                if overall_denominator > 0
                else 0
            )
        elif "PNC Coverage" in title or "C-Section Rate" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 100)
                if overall_denominator > 0
                else 0
            )
        elif "Maternal Death Rate" in title:
            overall_value = (
                (overall_numerator / overall_denominator * 100000)
                if overall_denominator > 0
                else 0
            )
        else:
            overall_value = (
                region_table_df[title].mean() if not region_table_df.empty else 0
            )

        overall_row = {
            "Region Name": f"Overall {title}",
            numerator_name: overall_numerator,
            denominator_name: overall_denominator,
            title: overall_value,
        }

        region_table_df = pd.concat(
            [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
        )
        region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

        # Format with CSS styling
        html = """
        <style>
        .region-comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 10px 0;
        }
        .region-comparison-table th {
            background-color: #27ae60;
            color: white;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: bold;
        }
        .region-comparison-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .region-comparison-table tr:hover {
            background-color: #f5f5f5;
        }
        .region-comparison-table .overall-row {
            background-color: #e8f4fd;
            font-weight: bold;
        }
        .region-comparison-table .numeric {
            text-align: right;
        }
        </style>
        <table class="region-comparison-table">
        <thead>
            <tr>
                <th>No</th>
                <th>Region Name</th>
                <th class="numeric">{num_name}</th>
                <th class="numeric">{den_name}</th>
                <th class="numeric">{title_name}</th>
            </tr>
        </thead>
        <tbody>
        """.format(
            num_name=numerator_name, den_name=denominator_name, title_name=title
        )

        for i, row in region_table_df.iterrows():
            row_class = (
                "overall-row" if f"Overall {title}" in str(row["Region Name"]) else ""
            )
            html += f"""
            <tr class="{row_class}">
                <td>{int(row['No'])}</td>
                <td>{row['Region Name']}</td>
                <td class="numeric">{row[numerator_name]:,.0f}</td>
                <td class="numeric">{row[denominator_name]:,.0f}</td>
                <td class="numeric">{row[title]:.2f}</td>
            </tr>
            """

        html += "</tbody></table>"

        st.markdown(html, unsafe_allow_html=True)

    # Always show download button for CSV
    csv_data = []
    for period_display in period_order:
        for region_name in region_names:
            # Find matching data
            matching_rows = comparison_df[
                (comparison_df["period_display"] == period_display)
                & (comparison_df["Region"] == region_name)
            ]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                csv_data.append(
                    {
                        "Period": period_display,
                        "Region": region_name,
                        numerator_name: row["numerator"],
                        denominator_name: row["denominator"],
                        title: row["value"],
                    }
                )

    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Data as CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_region_comparison.csv",
            mime="text/csv",
            help="Download the exact region comparison data shown in the chart",
        )


# ---------------- Additional Helper Functions ----------------
def extract_period_columns(df, date_column):
    """
    Extract period columns from a date column in patient-level data
    """
    if df.empty or date_column not in df.columns:
        return df

    result_df = df.copy()
    result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")
    result_df["period"] = result_df["event_date"].dt.strftime("%Y-%m")
    result_df["period_display"] = result_df["event_date"].dt.strftime("%b-%y")
    result_df["period_sort"] = result_df["event_date"].dt.strftime("%Y%m")

    return result_df


def get_relevant_date_column_for_kpi(kpi_name):
    """
    Get the relevant date column for a specific KPI
    """
    if kpi_name == "IPPCAR" or "FP Acceptance" in kpi_name:
        return PNC_DATE_COL  # FP acceptance happens during PNC
    elif kpi_name == "Stillbirth Rate":
        return DELIVERY_DATE_COL  # Birth outcomes from delivery
    elif kpi_name == "PNC Coverage":
        return PNC_DATE_COL  # PNC timing
    elif kpi_name == "Maternal Death Rate":
        return DISCHARGE_DATE_COL  # Death at discharge
    elif kpi_name == "C-Section Rate":
        return DELIVERY_DATE_COL  # Delivery mode
    else:
        return DELIVERY_DATE_COL  # Default to delivery date


def prepare_data_for_trend_chart(df, kpi_name, facility_uids=None):
    """
    Prepare patient-level data for trend chart visualization
    """
    if df.empty:
        return pd.DataFrame()

    # Get relevant date column for the KPI
    date_column = get_relevant_date_column_for_kpi(kpi_name)

    # Extract period columns
    result_df = extract_period_columns(df, date_column)

    # Filter by facilities if specified
    if facility_uids:
        result_df = result_df[result_df["orgUnit"].isin(facility_uids)]

    # Remove rows without valid dates
    result_df = result_df[result_df["event_date"].notna()]

    return result_df
