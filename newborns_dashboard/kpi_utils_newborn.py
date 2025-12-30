import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import streamlit as st
import hashlib
import numpy as np
import logging
import warnings
from utils.kpi_utils import auto_text_color, format_period_month_year

warnings.filterwarnings("ignore")

# ---------------- Caching Setup ----------------
if "kpi_cache_newborn" not in st.session_state:
    st.session_state.kpi_cache_newborn = {}


def get_cache_key_newborn(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key based on data and filters"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_cache_newborn():
    """Clear the newborn KPI cache - call this when you know data has changed"""
    st.session_state.kpi_cache_newborn = {}


# ---------------- Newborn KPI Constants ----------------
# Admission Information columns
BIRTH_LOCATION_COL = "birth_location_admission_information"
INBORN_CODE = "1"
OUTBORN_CODE = "2"

# Observations and Nursing Care 1 columns
TEMPERATURE_ON_ADMISSION_COL = (
    "temperature_on_admission_degc_observations_and_nursing_care_1"
)
HYPOTHERMIA_THRESHOLD = 36.5  # °C

# Observations and Nursing Care 2 columns
LOWEST_TEMPERATURE_COL = (
    "lowest_recorded_temperature_celsius_observations_and_nursing_care_2"
)

# Discharge columns
NEWBORN_STATUS_COL = "newborn_status_at_discharge_discharge_and_final_diagnosis"
NEWBORN_DEAD_CODE = "0"  # dead code value

# Event date columns
ADMISSION_DATE_COL = "event_date_admission_information"
OBS1_DATE_COL = "event_date_observations_and_nursing_care_1"
OBS2_DATE_COL = "event_date_observations_and_nursing_care_2"
DISCHARGE_DATE_COL = "event_date_discharge_and_final_diagnosis"
ENROLLMENT_DATE_COL = "enrollment_date"  # For Admitted Newborns


# ---------------- Base Computation Functions ----------------
def compute_total_admitted_newborns(df, facility_uids=None, date_column=None):
    """
    Count total admitted newborns - counts unique TEI IDs using UID filtering AND optional date filtering
    """
    cache_key = get_cache_key_newborn(
        df, facility_uids, f"total_admitted_{date_column}"
    )

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

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

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_admitted_newborns_count(df, facility_uids=None):
    """
    Count Admitted Newborns occurrences - SAME METHOD AS compute_admitted_mothers_count
    Counts unique TEI IDs with enrollment dates
    """
    cache_key = get_cache_key_newborn(df, facility_uids, "admitted_newborns_count")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Use enrollment date directly for this KPI
        date_column = ENROLLMENT_DATE_COL

        # Check if enrollment date column exists
        if date_column not in filtered_df.columns:
            result = 0
        else:
            # Filter to only include rows that have enrollment dates
            filtered_df[date_column] = pd.to_datetime(
                filtered_df[date_column], errors="coerce"
            )
            filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

            # Count unique TEI IDs with enrollment dates
            if "tei_id" in filtered_df.columns:
                result = filtered_df["tei_id"].dropna().nunique()
            else:
                result = len(filtered_df)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_admitted_newborns_rate(df, facility_uids=None):
    """
    For Admitted Newborns, rate is just the count (since it's not a percentage)
    Returns: (count, count, 1) to match the pattern
    """
    cache_key = get_cache_key_newborn(df, facility_uids, "admitted_newborns_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0, 0, 0.0)  # (count, denominator, value)
    else:
        # Count admitted newborns
        admitted_newborns = compute_admitted_newborns_count(df, facility_uids)

        # For Admitted Newborns, we just return the count as the value
        result = (admitted_newborns, 1, float(admitted_newborns))

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


# ---------------- SEPARATE NUMERATOR COMPUTATION FUNCTIONS ----------------
def compute_inborn_count(df, facility_uids=None):
    """Count inborn occurrences - VECTORIZED - with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if BIRTH_LOCATION_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Convert to string first, then extract numeric part
    df_copy["birth_location_clean"] = df_copy[BIRTH_LOCATION_COL].astype(str)
    df_copy["birth_location_numeric"] = pd.to_numeric(
        df_copy["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )

    # Count inborn (value = 1)
    inborn_mask = df_copy["birth_location_numeric"] == 1

    return int(inborn_mask.sum())


def compute_outborn_count(df, facility_uids=None):
    """Count outborn occurrences - VECTORIZED - with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if BIRTH_LOCATION_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Convert to string first, then extract numeric part
    df_copy["birth_location_clean"] = df_copy[BIRTH_LOCATION_COL].astype(str)
    df_copy["birth_location_numeric"] = pd.to_numeric(
        df_copy["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )

    # Count outborn (value = 2)
    outborn_mask = df_copy["birth_location_numeric"] == 2

    return int(outborn_mask.sum())


def compute_hypothermia_on_admission_count(df, facility_uids=None):
    """Count hypothermia on admission occurrences (< 36.5°C)"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if TEMPERATURE_ON_ADMISSION_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Convert temperature to numeric
    df_copy["temp_numeric"] = pd.to_numeric(
        df_copy[TEMPERATURE_ON_ADMISSION_COL], errors="coerce"
    )

    # Count hypothermia (temperature < 36.5)
    hypothermia_mask = df_copy["temp_numeric"] < HYPOTHERMIA_THRESHOLD

    return int(hypothermia_mask.sum())


def compute_hypothermia_after_admission_count(df, facility_uids=None):
    """Count hypothermia after admission occurrences (lowest recorded temp < 36.5°C)"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if LOWEST_TEMPERATURE_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Convert temperature to numeric
    df_copy["lowest_temp_numeric"] = pd.to_numeric(
        df_copy[LOWEST_TEMPERATURE_COL], errors="coerce"
    )

    # Count hypothermia (lowest temperature < 36.5)
    hypothermia_mask = df_copy["lowest_temp_numeric"] < HYPOTHERMIA_THRESHOLD

    return int(hypothermia_mask.sum())


def compute_neonatal_death_count(df, facility_uids=None):
    """Count neonatal death occurrences - VECTORIZED - with UID filtering"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if NEWBORN_STATUS_COL not in filtered_df.columns:
        return 0

    df_copy = filtered_df.copy()

    # Convert to string first, then extract numeric part
    df_copy["newborn_status_clean"] = df_copy[NEWBORN_STATUS_COL].astype(str)
    df_copy["newborn_status_numeric"] = pd.to_numeric(
        df_copy["newborn_status_clean"].str.split(".").str[0], errors="coerce"
    )

    # Count deaths (value = 0)
    death_mask = df_copy["newborn_status_numeric"] == 0

    return int(death_mask.sum())


def compute_hypothermia_on_admission_for_inborn(df, facility_uids=None):
    """Count hypothermia on admission for INBORN babies only - FIXED"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if (
        BIRTH_LOCATION_COL not in filtered_df.columns
        or TEMPERATURE_ON_ADMISSION_COL not in filtered_df.columns
    ):
        return 0

    # Create a working copy
    df_work = filtered_df.copy()

    # Clean birth location - extract numeric part
    df_work["birth_location_clean"] = df_work[BIRTH_LOCATION_COL].astype(str)
    df_work["birth_location_numeric"] = pd.to_numeric(
        df_work["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )

    # Filter for inborn babies (value = 1)
    df_inborn = df_work[df_work["birth_location_numeric"] == 1].copy()

    if df_inborn.empty:
        return 0

    # Clean temperature - FIXED CLEANING LOGIC
    df_inborn["temp_clean"] = df_inborn[TEMPERATURE_ON_ADMISSION_COL].astype(str)

    # Remove ALL non-numeric characters except decimal point and negative sign
    df_inborn["temp_clean"] = df_inborn["temp_clean"].str.replace(
        r"[^\d\.\-]", "", regex=True
    )

    # Handle empty strings after cleaning
    df_inborn["temp_clean"] = df_inborn["temp_clean"].replace("", pd.NA)

    # Convert to numeric
    df_inborn["temp_numeric"] = pd.to_numeric(df_inborn["temp_clean"], errors="coerce")

    # Drop rows with NaN temperature (invalid or missing)
    df_inborn_valid = df_inborn[df_inborn["temp_numeric"].notna()].copy()

    if df_inborn_valid.empty:
        return 0

    # Count hypothermia (temperature < 36.5)
    hypothermia_mask = df_inborn_valid["temp_numeric"] < HYPOTHERMIA_THRESHOLD

    hypothermia_count = int(hypothermia_mask.sum())

    return hypothermia_count


def compute_hypothermia_on_admission_for_outborn(df, facility_uids=None):
    """Count hypothermia on admission for OUTBORN babies only - FIXED"""
    if df is None or df.empty:
        return 0

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if (
        BIRTH_LOCATION_COL not in filtered_df.columns
        or TEMPERATURE_ON_ADMISSION_COL not in filtered_df.columns
    ):
        return 0

    # Create a working copy
    df_work = filtered_df.copy()

    # Clean birth location - extract numeric part
    df_work["birth_location_clean"] = df_work[BIRTH_LOCATION_COL].astype(str)
    df_work["birth_location_numeric"] = pd.to_numeric(
        df_work["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )

    # Filter for outborn babies (value = 2)
    df_outborn = df_work[df_work["birth_location_numeric"] == 2].copy()

    if df_outborn.empty:
        return 0

    # Clean temperature - FIXED CLEANING LOGIC
    df_outborn["temp_clean"] = df_outborn[TEMPERATURE_ON_ADMISSION_COL].astype(str)
    df_outborn["temp_clean"] = df_outborn["temp_clean"].str.replace(
        r"[^\d\.\-]", "", regex=True
    )
    df_outborn["temp_clean"] = df_outborn["temp_clean"].replace("", pd.NA)
    df_outborn["temp_numeric"] = pd.to_numeric(
        df_outborn["temp_clean"], errors="coerce"
    )

    # Drop rows with NaN temperature
    df_outborn_valid = df_outborn[df_outborn["temp_numeric"].notna()].copy()

    if df_outborn_valid.empty:
        return 0

    # Count hypothermia (temperature < 36.5)
    hypothermia_mask = df_outborn_valid["temp_numeric"] < HYPOTHERMIA_THRESHOLD

    hypothermia_count = int(hypothermia_mask.sum())

    return hypothermia_count


# ---------------- KPI Computation Functions ----------------
def compute_inborn_rate(df, facility_uids=None):
    """Compute inborn rate"""
    cache_key = get_cache_key_newborn(df, facility_uids, "inborn_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        inborn_count = compute_inborn_count(df, facility_uids)
        total_admitted = compute_total_admitted_newborns(df, facility_uids)
        rate = (inborn_count / total_admitted * 100) if total_admitted > 0 else 0.0
        result = (rate, inborn_count, total_admitted)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_outborn_rate(df, facility_uids=None):
    """Compute outborn rate"""
    cache_key = get_cache_key_newborn(df, facility_uids, "outborn_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        outborn_count = compute_outborn_count(df, facility_uids)
        total_admitted = compute_total_admitted_newborns(df, facility_uids)
        rate = (outborn_count / total_admitted * 100) if total_admitted > 0 else 0.0
        result = (rate, outborn_count, total_admitted)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_hypothermia_on_admission_rate(df, facility_uids=None):
    """Compute hypothermia on admission rate"""
    cache_key = get_cache_key_newborn(
        df, facility_uids, "hypothermia_on_admission_rate"
    )

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        hypothermia_count = compute_hypothermia_on_admission_count(df, facility_uids)
        total_admitted = compute_total_admitted_newborns(df, facility_uids)
        rate = (hypothermia_count / total_admitted * 100) if total_admitted > 0 else 0.0
        result = (rate, hypothermia_count, total_admitted)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_hypothermia_after_admission_rate(df, facility_uids=None):
    """Compute hypothermia after admission rate"""
    cache_key = get_cache_key_newborn(
        df, facility_uids, "hypothermia_after_admission_rate"
    )

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        hypothermia_count = compute_hypothermia_after_admission_count(df, facility_uids)
        total_admitted = compute_total_admitted_newborns(df, facility_uids)
        rate = (hypothermia_count / total_admitted * 100) if total_admitted > 0 else 0.0
        result = (rate, hypothermia_count, total_admitted)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_neonatal_mortality_rate(df, facility_uids=None):
    """Compute neonatal mortality rate"""
    cache_key = get_cache_key_newborn(df, facility_uids, "neonatal_mortality_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        death_count = compute_neonatal_death_count(df, facility_uids)
        total_admitted = compute_total_admitted_newborns(df, facility_uids)
        rate = (death_count / total_admitted * 100) if total_admitted > 0 else 0.0
        result = (rate, death_count, total_admitted)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_inborn_hypothermia_rate(df, facility_uids=None):
    """Compute hypothermia rate specifically for INBORN babies"""
    cache_key = get_cache_key_newborn(df, facility_uids, "inborn_hypothermia_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        inborn_hypothermia_count = compute_hypothermia_on_admission_for_inborn(
            df, facility_uids
        )
        inborn_count = compute_inborn_count(df, facility_uids)
        rate = (
            (inborn_hypothermia_count / inborn_count * 100) if inborn_count > 0 else 0.0
        )
        result = (rate, inborn_hypothermia_count, inborn_count)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_outborn_hypothermia_rate(df, facility_uids=None):
    """Compute hypothermia rate specifically for OUTBORN babies"""
    cache_key = get_cache_key_newborn(df, facility_uids, "outborn_hypothermia_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        outborn_hypothermia_count = compute_hypothermia_on_admission_for_outborn(
            df, facility_uids
        )
        outborn_count = compute_outborn_count(df, facility_uids)
        rate = (
            (outborn_hypothermia_count / outborn_count * 100)
            if outborn_count > 0
            else 0.0
        )
        result = (rate, outborn_hypothermia_count, outborn_count)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_admitted_newborns_kpi(df, facility_uids=None):
    """
    Compute Admitted Newborns KPI data
    This is the function your dashboard is calling
    """
    count, denominator, value = compute_admitted_newborns_rate(df, facility_uids)

    return {
        "admitted_newborns_count": int(count),
        "admitted_newborns_value": float(value),
        "admitted_newborns_denominator": int(denominator),
    }


# ---------------- Master KPI Function ----------------
def compute_newborn_kpis(df, facility_uids=None, date_column=None):
    """Compute all newborn KPIs with optional date filtering"""
    cache_key = get_cache_key_newborn(df, facility_uids, f"main_kpis_{date_column}")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Compute all KPIs
    total_admitted = compute_total_admitted_newborns(
        filtered_df, facility_uids, date_column
    )

    inborn_rate, inborn_count, total_inborn = compute_inborn_rate(
        filtered_df, facility_uids
    )
    outborn_rate, outborn_count, total_outborn = compute_outborn_rate(
        filtered_df, facility_uids
    )

    (
        hypothermia_on_admission_rate,
        hypothermia_on_admission_count,
        total_hypo_admission,
    ) = compute_hypothermia_on_admission_rate(filtered_df, facility_uids)
    (
        hypothermia_after_admission_rate,
        hypothermia_after_admission_count,
        total_hypo_after,
    ) = compute_hypothermia_after_admission_rate(filtered_df, facility_uids)

    neonatal_mortality_rate, death_count, total_deaths = (
        compute_neonatal_mortality_rate(filtered_df, facility_uids)
    )

    inborn_hypothermia_rate, inborn_hypothermia_count, total_inborn_hypo = (
        compute_inborn_hypothermia_rate(filtered_df, facility_uids)
    )
    outborn_hypothermia_rate, outborn_hypothermia_count, total_outborn_hypo = (
        compute_outborn_hypothermia_rate(filtered_df, facility_uids)
    )

    # Compute Admitted Newborns
    admitted_newborns_count = compute_admitted_newborns_count(
        filtered_df, facility_uids
    )

    result = {
        "total_admitted": int(total_admitted),
        "inborn_rate": float(inborn_rate),
        "inborn_count": int(inborn_count),
        "total_inborn": int(total_admitted),
        "outborn_rate": float(outborn_rate),
        "outborn_count": int(outborn_count),
        "total_outborn": int(total_admitted),
        "hypothermia_on_admission_rate": float(hypothermia_on_admission_rate),
        "hypothermia_on_admission_count": int(hypothermia_on_admission_count),
        "total_hypo_admission": int(total_admitted),
        "hypothermia_after_admission_rate": float(hypothermia_after_admission_rate),
        "hypothermia_after_admission_count": int(hypothermia_after_admission_count),
        "total_hypo_after": int(total_admitted),
        "neonatal_mortality_rate": float(neonatal_mortality_rate),
        "death_count": int(death_count),
        "total_deaths": int(total_admitted),
        "inborn_hypothermia_rate": float(inborn_hypothermia_rate),
        "inborn_hypothermia_count": int(inborn_hypothermia_count),
        "total_inborn_hypo": int(inborn_count if inborn_count > 0 else 0),
        "outborn_hypothermia_rate": float(outborn_hypothermia_rate),
        "outborn_hypothermia_count": int(outborn_hypothermia_count),
        "total_outborn_hypo": int(outborn_count if outborn_count > 0 else 0),
        "admitted_newborns_count": int(admitted_newborns_count),
    }

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


# ---------------- Date Handling with Program Stage Specific Dates ----------------
def get_relevant_date_column_for_newborn_kpi(kpi_name):
    """
    Get the relevant event date column for a specific newborn KPI based on program stage
    """
    # Mapping of newborn KPIs to their program stages and date columns
    program_stage_date_mapping = {
        # Admission Information KPIs
        "Inborn Rate (%)": "event_date_admission_information",
        "Outborn Rate (%)": "event_date_admission_information",
        "Inborn Babies (%)": "event_date_admission_information",
        "Outborn Babies (%)": "event_date_admission_information",
        # Observations and Nursing Care 1 KPIs
        "Hypothermia on Admission Rate (%)": "event_date_observations_and_nursing_care_1",
        "Hypothermia on Admission (%)": "event_date_observations_and_nursing_care_1",
        # Observations and Nursing Care 2 KPIs
        "Hypothermia After Admission Rate (%)": "event_date_observations_and_nursing_care_2",
        "Hypothermia After Admission (%)": "event_date_observations_and_nursing_care_2",
        # Discharge KPIs
        "Neonatal Mortality Rate (%)": "event_date_discharge_and_final_diagnosis",
        "NMR (%)": "event_date_discharge_and_final_diagnosis",
        # Combined KPIs
        "Inborn Hypothermia Rate (%)": "event_date_admission_information",
        "Outborn Hypothermia Rate (%)": "event_date_admission_information",
        # Admitted Newborns KPI - uses enrollment date
        "Admitted Newborns": "enrollment_date",
        # Default fallback
        "Total Admitted Newborns": "event_date_admission_information",
    }

    # Try exact match first
    for key in program_stage_date_mapping:
        if key in kpi_name:
            return program_stage_date_mapping[key]

    # Fallback based on keywords
    if any(word in kpi_name for word in ["Inborn", "Outborn", "Birth Location"]):
        return "event_date_admission_information"
    elif (
        any(word in kpi_name for word in ["Hypothermia", "Temperature", "Admission"])
        and "After" not in kpi_name
    ):
        return "event_date_observations_and_nursing_care_1"
    elif any(word in kpi_name for word in ["Lowest Temperature", "After Admission"]):
        return "event_date_observations_and_nursing_care_2"
    elif any(word in kpi_name for word in ["Mortality", "Death", "Discharge"]):
        return "event_date_discharge_and_final_diagnosis"
    elif any(word in kpi_name for word in ["Admitted Newborns", "Admitted"]):
        return "enrollment_date"

    # Default to admission date
    return "event_date_admission_information"


def prepare_data_for_newborn_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for trend chart - FIXED DATE HANDLING
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_name)

    # Check if the SPECIFIC date column exists
    if date_column not in filtered_df.columns:
        # Try to use event_date as fallback (already normalized)
        if "event_date" in filtered_df.columns:
            date_column = "event_date"
            logging.info(
                f"⚠️ KPI-specific date column not found for {kpi_name}, using 'event_date' instead"
            )
        elif "enrollment_date" in filtered_df.columns:
            date_column = "enrollment_date"
            logging.info(
                f"⚠️ KPI-specific date column not found for {kpi_name}, using 'enrollment_date' instead"
            )
        else:
            logging.warning(
                f"⚠️ Required date column '{date_column}' not found for {kpi_name}"
            )
            return pd.DataFrame(), None

    # Create result dataframe
    result_df = filtered_df.copy()

    # Convert to datetime
    result_df["event_date_for_chart"] = pd.to_datetime(
        result_df[date_column], errors="coerce"
    )

    # CRITICAL: Apply date range filtering if provided
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            # Convert to datetime for comparison
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

            # Filter by date range
            date_mask = (result_df["event_date_for_chart"] >= start_dt) & (
                result_df["event_date_for_chart"] < end_dt
            )

            result_df = result_df[date_mask].copy()

            logging.info(f"📅 Trend chart date filter: {start_date} to {end_date}")
            logging.info(f"📅 Filtered to {len(result_df)} patients")

    # Filter out rows without valid dates
    result_df = result_df[result_df["event_date_for_chart"].notna()].copy()

    if result_df.empty:
        logging.info(f"⚠️ No data with valid dates in '{date_column}' for {kpi_name}")
        return pd.DataFrame(), date_column

    # Get period label
    period_label = st.session_state.get("period_label", "Monthly")
    if "filters" in st.session_state and "period_label" in st.session_state.filters:
        period_label = st.session_state.filters["period_label"]

    # Create period columns using time_filter utility
    from utils.time_filter import assign_period

    result_df = assign_period(result_df, "event_date_for_chart", period_label)

    # Filter by facility if needed
    if facility_uids and "orgUnit" in result_df.columns:
        result_df = result_df[result_df["orgUnit"].isin(facility_uids)].copy()

    logging.info(f"📊 Trend chart prepared: {len(result_df)} patients for {kpi_name}")

    # Rename back to event_date for consistency
    result_df = result_df.rename(columns={"event_date_for_chart": "event_date"})

    return result_df, date_column


def get_numerator_denominator_for_newborn_kpi(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for a specific newborn KPI with UID filtering
    AND filtered by KPI-specific program stage dates AND date range
    Returns: (numerator, denominator, value)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_name)

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
    kpi_data = compute_newborn_kpis(filtered_df, facility_uids, date_column)

    kpi_mapping = {
        # Admission KPIs
        "Inborn Rate (%)": {
            "numerator": "inborn_count",
            "denominator": "total_admitted",
            "value": "inborn_rate",
        },
        "Outborn Rate (%)": {
            "numerator": "outborn_count",
            "denominator": "total_admitted",
            "value": "outborn_rate",
        },
        "Inborn Babies (%)": {
            "numerator": "inborn_count",
            "denominator": "total_admitted",
            "value": "inborn_rate",
        },
        "Outborn Babies (%)": {
            "numerator": "outborn_count",
            "denominator": "total_admitted",
            "value": "outborn_rate",
        },
        # Hypothermia KPIs
        "Hypothermia on Admission Rate (%)": {
            "numerator": "hypothermia_on_admission_count",
            "denominator": "total_admitted",
            "value": "hypothermia_on_admission_rate",
        },
        "Hypothermia After Admission Rate (%)": {
            "numerator": "hypothermia_after_admission_count",
            "denominator": "total_admitted",
            "value": "hypothermia_after_admission_rate",
        },
        # Mortality KPI
        "Neonatal Mortality Rate (%)": {
            "numerator": "death_count",
            "denominator": "total_admitted",
            "value": "neonatal_mortality_rate",
        },
        "NMR (%)": {
            "numerator": "death_count",
            "denominator": "total_admitted",
            "value": "neonatal_mortality_rate",
        },
        # Combined KPIs
        "Inborn Hypothermia Rate (%)": {
            "numerator": "inborn_hypothermia_count",
            "denominator": "total_inborn_hypo",
            "value": "inborn_hypothermia_rate",
        },
        "Outborn Hypothermia Rate (%)": {
            "numerator": "outborn_hypothermia_count",
            "denominator": "total_outborn_hypo",
            "value": "outborn_hypothermia_rate",
        },
        # Admitted Newborns KPI
        "Admitted Newborns": {
            "numerator": "admitted_newborns_count",
            "denominator": 1,
            "value": "admitted_newborns_count",
        },
    }

    if kpi_name in kpi_mapping:
        mapping = kpi_mapping[kpi_name]
        numerator = kpi_data.get(mapping["numerator"], 0)
        denominator = kpi_data.get(mapping["denominator"], 1)
        value = kpi_data.get(mapping["value"], 0.0)

        return (numerator, denominator, value)

    # Fallback mappings for partial matches
    if "Admitted Newborns" in kpi_name or "Admitted" in kpi_name:
        numerator = kpi_data.get("admitted_newborns_count", 0)
        denominator = 1
        value = float(numerator)
        return (numerator, denominator, value)
    elif "Inborn" in kpi_name and "Hypothermia" in kpi_name:
        numerator = kpi_data.get("inborn_hypothermia_count", 0)
        denominator = kpi_data.get("total_inborn_hypo", 1)
        value = kpi_data.get("inborn_hypothermia_rate", 0.0)
        return (numerator, denominator, value)
    elif "Outborn" in kpi_name and "Hypothermia" in kpi_name:
        numerator = kpi_data.get("outborn_hypothermia_count", 0)
        denominator = kpi_data.get("total_outborn_hypo", 1)
        value = kpi_data.get("outborn_hypothermia_rate", 0.0)
        return (numerator, denominator, value)
    elif (
        "Hypothermia" in kpi_name
        and "Admission" in kpi_name
        and "After" not in kpi_name
    ):
        numerator = kpi_data.get("hypothermia_on_admission_count", 0)
        denominator = kpi_data.get("total_admitted", 1)
        value = kpi_data.get("hypothermia_on_admission_rate", 0.0)
        return (numerator, denominator, value)
    elif "Hypothermia" in kpi_name and "After" in kpi_name:
        numerator = kpi_data.get("hypothermia_after_admission_count", 0)
        denominator = kpi_data.get("total_admitted", 1)
        value = kpi_data.get("hypothermia_after_admission_rate", 0.0)
        return (numerator, denominator, value)
    elif "Inborn" in kpi_name:
        numerator = kpi_data.get("inborn_count", 0)
        denominator = kpi_data.get("total_admitted", 1)
        value = kpi_data.get("inborn_rate", 0.0)
        return (numerator, denominator, value)
    elif "Outborn" in kpi_name:
        numerator = kpi_data.get("outborn_count", 0)
        denominator = kpi_data.get("total_admitted", 1)
        value = kpi_data.get("outborn_rate", 0.0)
        return (numerator, denominator, value)
    elif "Mortality" in kpi_name or "NMR" in kpi_name:
        numerator = kpi_data.get("death_count", 0)
        denominator = kpi_data.get("total_admitted", 1)
        value = kpi_data.get("neonatal_mortality_rate", 0.0)
        return (numerator, denominator, value)

    return (0, 0, 0.0)


# ---------------- Chart Functions WITH TABLES ----------------
def render_newborn_trend_chart(
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
    # Use the same function from kpi_utils
    from utils.kpi_utils import render_trend_chart

    return render_trend_chart(
        df,
        period_col,
        value_col,
        title,
        bg_color,
        text_color,
        facility_names,
        numerator_name,
        denominator_name,
        facility_uids,
    )


def render_newborn_facility_comparison_chart(
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
    # Use the same function from kpi_utils
    from utils.kpi_utils import render_facility_comparison_chart

    return render_facility_comparison_chart(
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
    )


def render_newborn_region_comparison_chart(
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
    # Use the same function from kpi_utils
    from utils.kpi_utils import render_region_comparison_chart

    return render_region_comparison_chart(
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
    )


# ---------------- Admitted Newborns Chart Functions ----------------
def render_admitted_newborns_trend_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Admitted Newborns Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    value_name="Admitted Newborns",
    facility_uids=None,
):
    """Render trend chart for Admitted Newborns - FOLLOWING SAME PATTERN AS ADMITTED MOTHERS"""
    from utils.kpi_admitted_mothers import render_admitted_mothers_trend_chart

    # Reuse the admitted mothers chart function since they work the same way
    return render_admitted_mothers_trend_chart(
        df,
        period_col,
        value_col,
        title,
        bg_color,
        text_color,
        facility_names,
        value_name,
        facility_uids,
    )


def render_admitted_newborns_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Admitted Newborns - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    value_name="Admitted Newborns",
):
    """Render facility comparison chart for Admitted Newborns"""
    from utils.kpi_admitted_mothers import (
        render_admitted_mothers_facility_comparison_chart,
    )

    # Reuse the admitted mothers chart function since they work the same way
    return render_admitted_mothers_facility_comparison_chart(
        df,
        period_col,
        value_col,
        title,
        bg_color,
        text_color,
        facility_names,
        facility_uids,
        value_name,
    )


def render_admitted_newborns_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Admitted Newborns - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    value_name="Admitted Newborns",
):
    """Render region comparison chart for Admitted Newborns"""
    from utils.kpi_admitted_mothers import (
        render_admitted_mothers_region_comparison_chart,
    )

    # Reuse the admitted mothers chart function since they work the same way
    return render_admitted_mothers_region_comparison_chart(
        df,
        period_col,
        value_col,
        title,
        bg_color,
        text_color,
        region_names,
        region_mapping,
        facilities_by_region,
        value_name,
    )


# ---------------- Period Aggregation Function ----------------
def aggregate_by_period_with_sorting_newborn(
    df, period_col, period_sort_col, facility_uids, kpi_function, kpi_name=None
):
    """
    Aggregate data by period while preserving chronological sorting
    Works with patient-level data for newborn KPIs
    """
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby([period_col, period_sort_col])

    result_data = []
    for (period_display, period_sort), group_df in grouped:
        if kpi_name:
            numerator, denominator, value = get_numerator_denominator_for_newborn_kpi(
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


# ---------------- Additional Helper Functions ----------------
def extract_period_columns_newborn(df, date_column):
    """
    SIMPLE VERSION: Assumes dates are already valid, just need proper grouping for newborn data
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


# ---------------- Export all functions ----------------
__all__ = [
    # Cache functions
    "get_cache_key_newborn",
    "clear_cache_newborn",
    # Base computation
    "compute_total_admitted_newborns",
    "compute_admitted_newborns_count",
    "compute_admitted_newborns_rate",
    "compute_admitted_newborns_kpi",
    # Numerator computation functions
    "compute_inborn_count",
    "compute_outborn_count",
    "compute_hypothermia_on_admission_count",
    "compute_hypothermia_after_admission_count",
    "compute_neonatal_death_count",
    "compute_hypothermia_on_admission_for_inborn",
    "compute_hypothermia_on_admission_for_outborn",
    # KPI computation functions
    "compute_inborn_rate",
    "compute_outborn_rate",
    "compute_hypothermia_on_admission_rate",
    "compute_hypothermia_after_admission_rate",
    "compute_neonatal_mortality_rate",
    "compute_inborn_hypothermia_rate",
    "compute_outborn_hypothermia_rate",
    # Master KPI function
    "compute_newborn_kpis",
    # Date handling functions
    "get_relevant_date_column_for_newborn_kpi",
    "prepare_data_for_newborn_trend_chart",
    "get_numerator_denominator_for_newborn_kpi",
    # Chart rendering functions
    "render_newborn_trend_chart",
    "render_newborn_facility_comparison_chart",
    "render_newborn_region_comparison_chart",
    "render_admitted_newborns_trend_chart",
    "render_admitted_newborns_facility_comparison_chart",
    "render_admitted_newborns_region_comparison_chart",
    # Period aggregation functions
    "aggregate_by_period_with_sorting_newborn",
    "extract_period_columns_newborn",
]
