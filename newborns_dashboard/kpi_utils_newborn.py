import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import streamlit as st
import hashlib
import numpy as np
import logging
import warnings
from utils.kpi_utils import (
    auto_text_color,
    format_period_month_year,
    get_attractive_hover_template,
    get_current_period_label,
    format_period_for_download,
)

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
# UPDATED: Based on your dataset column names
# Admission Information columns
BIRTH_LOCATION_COL = "place_of_delivery_nicu_admission_careform"
INBORN_CODE = "1"
OUTBORN_CODE = "2"

# Observations and Nursing Care 1 columns
TEMPERATURE_ON_ADMISSION_COL = "temp_at_admission_nicu_admission_careform"
HYPOTHERMIA_THRESHOLD = 36.5  # °C

# Observations and Nursing Care 2 columns - NOT AVAILABLE
# LOWEST_TEMPERATURE_COL = "lowest_recorded_temperature_celsius_observations_and_nursing_care_2"

# Discharge columns
NEWBORN_STATUS_COL = "newborn_status_at_discharge_n_discharge_care_form"
NEWBORN_DEAD_CODE = "0"  # dead code value

# Event date columns - UPDATED
ADMISSION_DATE_COL = "event_date_nicu_admission_careform"
DISCHARGE_DATE_COL = "event_date_discharge_care_form"
ENROLLMENT_DATE_COL = "enrollment_date"  # For Admitted Newborns

# ============== ANTIBIOTICS COLUMNS - COMMENTED OUT ==============
# # Maternal medication during pregnancy and labor - value "1" means Antibiotics
# MATERNAL_MEDICATION_COL = (
#     "maternal_medication_during_pregnancy_and_labor_nicu_admission_careform"
# )
# ANTIBIOTICS_CODE = "1"  # Antibiotics code in maternal medication
#
# # Probable sepsis (same logic, different column name)
# SUBCATEGORIES_INFECTION_COL = "sub_categories_of_infection_n_discharge_care_form"
# PROBABLE_SEPSIS_CODE = "1"


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


# ============== ANTIBIOTICS FUNCTIONS - COMMENTED OUT ==============
# def compute_probable_sepsis_count(df, facility_uids=None):
#     """Count newborns with probable sepsis"""
#     cache_key = get_cache_key_newborn(df, facility_uids, "probable_sepsis_count")
#
#     if cache_key in st.session_state.kpi_cache_newborn:
#         return st.session_state.kpi_cache_newborn[cache_key]
#
#     if df is None or df.empty:
#         result = 0
#     else:
#         filtered_df = df.copy()
#         if facility_uids and "orgUnit" in filtered_df.columns:
#             filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
#
#         if SUBCATEGORIES_INFECTION_COL not in filtered_df.columns:
#             result = 0
#         else:
#             df_work = filtered_df.copy()
#             df_work["infection_clean"] = df_work[SUBCATEGORIES_INFECTION_COL].astype(
#                 str
#             )
#             df_work["infection_numeric"] = pd.to_numeric(
#                 df_work["infection_clean"].str.split(".").str[0], errors="coerce"
#             )
#
#             sepsis_mask = df_work["infection_numeric"] == float(PROBABLE_SEPSIS_CODE)
#
#             if "tei_id" in df_work.columns:
#                 sepsis_teis = df_work.loc[sepsis_mask, "tei_id"].dropna().unique()
#                 result = len(sepsis_teis)
#             else:
#                 result = int(sepsis_mask.sum())
#
#     st.session_state.kpi_cache_newborn[cache_key] = result
#     return result
#
#
# def compute_antibiotics_count(df, facility_uids=None):
#     """Count newborns with maternal antibiotics administered"""
#     cache_key = get_cache_key_newborn(df, facility_uids, "antibiotics_count")
#
#     if cache_key in st.session_state.kpi_cache_newborn:
#         return st.session_state.kpi_cache_newborn[cache_key]
#
#     if df is None or df.empty:
#         result = 0
#     else:
#         filtered_df = df.copy()
#         if facility_uids and "orgUnit" in filtered_df.columns:
#             filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
#
#         if MATERNAL_MEDICATION_COL not in filtered_df.columns:
#             result = 0
#         else:
#             df_work = filtered_df.copy()
#
#             # Clean maternal medication column
#             df_work["medication_clean"] = df_work[MATERNAL_MEDICATION_COL].astype(str)
#             df_work["medication_numeric"] = pd.to_numeric(
#                 df_work["medication_clean"].str.split(".").str[0], errors="coerce"
#             )
#
#             antibiotics_mask = df_work["medication_numeric"] == float(ANTIBIOTICS_CODE)
#
#             if "tei_id" in df_work.columns:
#                 eligible_teis = (
#                     df_work.loc[antibiotics_mask, "tei_id"].dropna().unique()
#                 )
#                 result = len(eligible_teis)
#             else:
#                 result = int(antibiotics_mask.sum())
#
#     st.session_state.kpi_cache_newborn[cache_key] = result
#     return result
#
#
# def compute_antibiotics_rate(df, facility_uids=None):
#     """Compute antibiotics rate for newborns with clinical sepsis"""
#     cache_key = get_cache_key_newborn(df, facility_uids, "antibiotics_rate")
#
#     if cache_key in st.session_state.kpi_cache_newborn:
#         return st.session_state.kpi_cache_newborn[cache_key]
#
#     if df is None or df.empty:
#         result = (0.0, 0, 0)
#     else:
#         antibiotics_count = compute_antibiotics_count(df, facility_uids)
#         total_admitted = compute_total_admitted_newborns(df, facility_uids)
#
#         rate = (antibiotics_count / total_admitted * 100) if total_admitted > 0 else 0.0
#         result = (float(rate), int(antibiotics_count), int(total_admitted))
#
#     st.session_state.kpi_cache_newborn[cache_key] = result
#     return result
#
#
# def compute_antibiotics_kpi(df, facility_uids=None):
#     """Compute antibiotics KPI data"""
#     rate, antibiotics_count, total_admitted = compute_antibiotics_rate(
#         df, facility_uids
#     )
#
#     return {
#         "antibiotics_rate": float(rate),
#         "antibiotics_count": int(antibiotics_count),
#         "total_admitted": int(total_admitted),
#     }
#
#
# def get_numerator_denominator_for_antibiotics(
#     df, facility_uids=None, date_range_filters=None
# ):
#     """Get numerator and denominator for Antibiotics KPI"""
#     if df is None or df.empty:
#         return (0, 0, 0.0)
#
#     filtered_df = df.copy()
#     if facility_uids and "orgUnit" in filtered_df.columns:
#         filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
#
#     # Use admission date for antibiotics (maternal medication during admission)
#     date_column = ADMISSION_DATE_COL
#
#     if date_column in filtered_df.columns:
#         filtered_df[date_column] = pd.to_datetime(
#             filtered_df[date_column], errors="coerce"
#         )
#         filtered_df = filtered_df[filtered_df[date_column].notna()].copy()
#
#         if date_range_filters:
#             start_date = date_range_filters.get("start_date")
#             end_date = date_range_filters.get("end_date")
#
#             if start_date and end_date:
#                 start_dt = pd.Timestamp(start_date)
#                 end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
#
#                 filtered_df = filtered_df[
#                     (filtered_df[date_column] >= start_dt)
#                     & (filtered_df[date_column] < end_dt)
#                 ].copy()
#
#     if filtered_df.empty:
#         return (0, 0, 0.0)
#
#     antibiotics_data = compute_antibiotics_kpi(filtered_df, facility_uids)
#
#     numerator = antibiotics_data.get("antibiotics_count", 0)
#     denominator = antibiotics_data.get("total_admitted", 1)
#     rate = antibiotics_data.get("antibiotics_rate", 0.0)
#
#     return (numerator, denominator, rate)
#


# ---------------- Existing Newborn Functions ----------------
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
    df_copy["birth_location_clean"] = df_copy[BIRTH_LOCATION_COL].astype(str)
    df_copy["birth_location_numeric"] = pd.to_numeric(
        df_copy["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )
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
    df_copy["birth_location_clean"] = df_copy[BIRTH_LOCATION_COL].astype(str)
    df_copy["birth_location_numeric"] = pd.to_numeric(
        df_copy["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )
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
    df_copy["temp_numeric"] = pd.to_numeric(
        df_copy[TEMPERATURE_ON_ADMISSION_COL], errors="coerce"
    )
    hypothermia_mask = df_copy["temp_numeric"] < HYPOTHERMIA_THRESHOLD
    return int(hypothermia_mask.sum())


# NEW FUNCTION: Inborn Hypothermia Count
def compute_inborn_hypothermia_count(df, facility_uids=None):
    """Count inborn newborns with hypothermia (<36.5°C)"""
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

    df_copy = filtered_df.copy()

    # Clean birth location
    df_copy["birth_location_clean"] = df_copy[BIRTH_LOCATION_COL].astype(str)
    df_copy["birth_location_numeric"] = pd.to_numeric(
        df_copy["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )

    # Clean temperature
    df_copy["temp_numeric"] = pd.to_numeric(
        df_copy[TEMPERATURE_ON_ADMISSION_COL], errors="coerce"
    )

    # Count inborn AND hypothermic
    inborn_hypo_mask = (df_copy["birth_location_numeric"] == 1) & (
        df_copy["temp_numeric"] < HYPOTHERMIA_THRESHOLD
    )

    return int(inborn_hypo_mask.sum())


# NEW FUNCTION: Outborn Hypothermia Count
def compute_outborn_hypothermia_count(df, facility_uids=None):
    """Count outborn newborns with hypothermia (<36.5°C)"""
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

    df_copy = filtered_df.copy()

    # Clean birth location
    df_copy["birth_location_clean"] = df_copy[BIRTH_LOCATION_COL].astype(str)
    df_copy["birth_location_numeric"] = pd.to_numeric(
        df_copy["birth_location_clean"].str.split(".").str[0], errors="coerce"
    )

    # Clean temperature
    df_copy["temp_numeric"] = pd.to_numeric(
        df_copy[TEMPERATURE_ON_ADMISSION_COL], errors="coerce"
    )

    # Count outborn AND hypothermic
    outborn_hypo_mask = (df_copy["birth_location_numeric"] == 2) & (
        df_copy["temp_numeric"] < HYPOTHERMIA_THRESHOLD
    )

    return int(outborn_hypo_mask.sum())


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
    df_copy["newborn_status_clean"] = df_copy[NEWBORN_STATUS_COL].astype(str)
    df_copy["newborn_status_numeric"] = pd.to_numeric(
        df_copy["newborn_status_clean"].str.split(".").str[0], errors="coerce"
    )
    death_mask = df_copy["newborn_status_numeric"] == 0
    return int(death_mask.sum())


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


# NEW FUNCTION: Inborn Hypothermia Rate
def compute_inborn_hypothermia_rate(df, facility_uids=None):
    """Compute inborn hypothermia rate - CORRECT: hypothermic inborn / total inborn"""
    cache_key = get_cache_key_newborn(df, facility_uids, "inborn_hypothermia_rate")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        inborn_hypo_count = compute_inborn_hypothermia_count(df, facility_uids)
        inborn_count = compute_inborn_count(df, facility_uids)  # CORRECT DENOMINATOR

        rate = (inborn_hypo_count / inborn_count * 100) if inborn_count > 0 else 0.0
        result = (rate, inborn_hypo_count, inborn_count)

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


# NEW FUNCTION: Outborn Hypothermia Rate
def compute_outborn_hypothermia_rate(df, facility_uids=None):
    """Compute outborn hypothermia rate - CORRECT: hypothermic outborn / total outborn"""
    cache_key = get_cache_key_newborn(df, facility_uids, "outborn_hypothermia_rate")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        outborn_hypo_count = compute_outborn_hypothermia_count(df, facility_uids)
        outborn_count = compute_outborn_count(df, facility_uids)  # CORRECT DENOMINATOR

        rate = (outborn_hypo_count / outborn_count * 100) if outborn_count > 0 else 0.0
        result = (rate, outborn_hypo_count, outborn_count)

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


# ============== COMMENTED OUT - This function doesn't exist or is not needed ==============
# def compute_antibiotics_rate(df, facility_uids=None):
#     """Compute antibiotics rate for newborns with clinical sepsis"""
#     cache_key = get_cache_key_newborn(df, facility_uids, "antibiotics_rate")
#
#     if cache_key in st.session_state.kpi_cache_newborn:
#         return st.session_state.kpi_cache_newborn[cache_key]
#
#     if df is None or df.empty:
#         result = (0.0, 0, 0)
#     else:
#         antibiotics_count = compute_antibiotics_count(df, facility_uids)
#         total_admitted = compute_total_admitted_newborns(df, facility_uids)
#
#         rate = (antibiotics_count / total_admitted * 100) if total_admitted > 0 else 0.0
#         result = (float(rate), int(antibiotics_count), int(total_admitted))
#
#     st.session_state.kpi_cache_newborn[cache_key] = result
#     return result


# ---------------- Data Quality (Missing Data) Functions ----------------
def compute_missing_temperature(df, facility_uids=None):
    """Compute rate of missing temperature on admission"""
    cache_key = get_cache_key_newborn(df, facility_uids, "missing_temperature")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
        
        total_admitted = len(filtered_df)
        if total_admitted == 0:
            result = (0.0, 0, 0)
        elif TEMPERATURE_ON_ADMISSION_COL not in filtered_df.columns:
            # If column is missing from this filtered slice, we can't compute it
            result = (0.0, 0, total_admitted)
        else:
            missing_count = filtered_df[TEMPERATURE_ON_ADMISSION_COL].isna().sum()
            rate = (missing_count / total_admitted * 100)
            result = (rate, int(missing_count), int(total_admitted))

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_missing_birth_weight(df, facility_uids=None):
    """Compute rate of missing birth weight"""
    cache_key = get_cache_key_newborn(df, facility_uids, "missing_birth_weight")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
        
        total_admitted = len(filtered_df)
        if total_admitted == 0:
            result = (0.0, 0, 0)
        else:
            # Import birth weight column name
            from newborns_dashboard.kpi_utils_newborn_simplified import BIRTH_WEIGHT_COL
            
            if BIRTH_WEIGHT_COL not in filtered_df.columns:
                # If column is missing from this filtered slice, we can't compute it
                result = (0.0, 0, total_admitted)
            else:
                missing_count = filtered_df[BIRTH_WEIGHT_COL].isna().sum()
                rate = (missing_count / total_admitted * 100)
                result = (rate, int(missing_count), int(total_admitted))

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_missing_discharge_status(df, facility_uids=None):
    """Compute rate of missing newborn status at discharge"""
    cache_key = get_cache_key_newborn(df, facility_uids, "missing_discharge_status")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
        
        total_admitted = len(filtered_df)
        if total_admitted == 0:
            result = (0.0, 0, 0)
        elif NEWBORN_STATUS_COL not in filtered_df.columns:
            # If column is missing from this filtered slice, we can't compute it
            result = (0.0, 0, total_admitted)
        else:
            missing_count = filtered_df[NEWBORN_STATUS_COL].isna().sum()
            rate = (missing_count / total_admitted * 100)
            result = (rate, int(missing_count), int(total_admitted))

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_missing_birth_location(df, facility_uids=None):
    """Compute rate of missing birth location"""
    cache_key = get_cache_key_newborn(df, facility_uids, "missing_birth_location")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()
        
        total_admitted = len(filtered_df)
        if total_admitted == 0:
            result = (0.0, 0, 0)
        elif BIRTH_LOCATION_COL not in filtered_df.columns:
            # If column is missing from this filtered slice, we can't compute it
            result = (0.0, 0, total_admitted)
        else:
            missing_count = filtered_df[BIRTH_LOCATION_COL].isna().sum()
            rate = (missing_count / total_admitted * 100)
            result = (rate, int(missing_count), int(total_admitted))

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


# ---------------- Master KPI Function ----------------
def compute_newborn_kpis(df, facility_uids=None, date_column=None):
    """Compute all newborn KPIs with optional date filtering"""
    cache_key = get_cache_key_newborn(df, facility_uids, f"main_kpis_{date_column}")
    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    total_admitted = compute_total_admitted_newborns(
        filtered_df, facility_uids, date_column
    )
    inborn_rate, inborn_count, total_inborn_denom = compute_inborn_rate(
        filtered_df, facility_uids
    )
    outborn_rate, outborn_count, total_outborn_denom = compute_outborn_rate(
        filtered_df, facility_uids
    )
    hypothermia_on_admission_rate, hypothermia_on_admission_count, _ = (
        compute_hypothermia_on_admission_rate(filtered_df, facility_uids)
    )

    # ADD THE NEW HYPOTHERMIA BY BIRTH LOCATION KPIs
    inborn_hypo_rate, inborn_hypo_count, total_inborn_hypo = (
        compute_inborn_hypothermia_rate(filtered_df, facility_uids)
    )
    outborn_hypo_rate, outborn_hypo_count, total_outborn_hypo = (
        compute_outborn_hypothermia_rate(filtered_df, facility_uids)
    )

    neonatal_mortality_rate, death_count, _ = compute_neonatal_mortality_rate(
        filtered_df, facility_uids
    )

    # ============== ANTIBIOTICS RATE - COMMENTED OUT ==============
    # antibiotics_rate, antibiotics_count, total_admitted_abx = compute_antibiotics_rate(
    #     filtered_df, facility_uids
    # )

    admitted_newborns_count = compute_admitted_newborns_count(
        filtered_df, facility_uids
    )

    # DATA QUALITY KPIs
    missing_temp_rate, missing_temp_count, _ = compute_missing_temperature(filtered_df, facility_uids)
    missing_bw_rate, missing_bw_count, _ = compute_missing_birth_weight(filtered_df, facility_uids)
    missing_status_rate, missing_status_count, _ = compute_missing_discharge_status(filtered_df, facility_uids)
    missing_loc_rate, missing_loc_count, _ = compute_missing_birth_location(filtered_df, facility_uids)

    result = {
        "total_admitted": int(total_admitted),
        "inborn_rate": float(inborn_rate),
        "inborn_count": int(inborn_count),
        "total_inborn": int(
            total_inborn_denom
        ),  # This is total_admitted for distribution
        "outborn_rate": float(outborn_rate),
        "outborn_count": int(outborn_count),
        "total_outborn": int(
            total_outborn_denom
        ),  # This is total_admitted for distribution
        "hypothermia_on_admission_rate": float(hypothermia_on_admission_rate),
        "hypothermia_on_admission_count": int(hypothermia_on_admission_count),
        "total_hypo_admission": int(total_admitted),
        # NEW: Hypothermia by birth location
        "inborn_hypothermia_rate": float(inborn_hypo_rate),
        "inborn_hypothermia_count": int(inborn_hypo_count),
        "total_inborn_hypo": int(total_inborn_hypo),  # This is inborn_count
        "outborn_hypothermia_rate": float(outborn_hypo_rate),
        "outborn_hypothermia_count": int(outborn_hypo_count),
        "total_outborn_hypo": int(total_outborn_hypo),  # This is outborn_count
        "neonatal_mortality_rate": float(neonatal_mortality_rate),
        "death_count": int(death_count),
        "total_deaths": int(total_admitted),
        # ============== ANTIBIOTICS DATA - COMMENTED OUT ==============
        # "antibiotics_rate": float(antibiotics_rate),
        # "antibiotics_count": int(antibiotics_count),
        # "total_admitted_abx": int(total_admitted_abx),
        "admitted_newborns_count": int(admitted_newborns_count),
        # DATA QUALITY
        "missing_temperature_rate": float(missing_temp_rate),
        "missing_temperature_count": int(missing_temp_count),
        "missing_birth_weight_rate": float(missing_bw_rate),
        "missing_birth_weight_count": int(missing_bw_count),
        "missing_birth_weight_count": int(missing_bw_count),
        "missing_discharge_status_rate": float(missing_status_rate),
        "missing_discharge_status_count": int(missing_status_count),
        "missing_birth_location_rate": float(missing_loc_rate),
        "missing_birth_location_count": int(missing_loc_count),
    }

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


# ---------------- Date Handling ----------------
def get_relevant_date_column_for_newborn_kpi(kpi_name):
    """Get the relevant event date column for a specific newborn KPI"""
    return "enrollment_date"


def prepare_data_for_newborn_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for trend chart using KPI-specific dates
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_name)

    # Check if the SPECIFIC date column exists
    if date_column not in filtered_df.columns:
        if "event_date" in filtered_df.columns:
            date_column = "event_date"
            st.warning(
                f"⚠️ KPI-specific date column not found for {kpi_name}, using 'event_date' instead"
            )
        elif "enrollment_date" in filtered_df.columns and "Admitted" in kpi_name:
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
    result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")

    # CRITICAL: Apply date range filtering
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


def get_numerator_denominator_for_newborn_kpi(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for a specific newborn KPI with date range filtering
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_name)

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

    # Now compute KPI on date-filtered data
    kpi_data = compute_newborn_kpis(filtered_df, facility_uids, date_column)

    kpi_mapping = {
        # Birth location distribution
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
        # General hypothermia (all newborns)
        "Hypothermia on Admission Rate (%)": {
            "numerator": "hypothermia_on_admission_count",
            "denominator": "total_admitted",
            "value": "hypothermia_on_admission_rate",
        },
        # NEW: Hypothermia by birth location
        "Inborn Hypothermia Rate (%)": {
            "numerator": "inborn_hypothermia_count",
            "denominator": "total_inborn_hypo",  # This is inborn_count
            "value": "inborn_hypothermia_rate",
        },
        "Outborn Hypothermia Rate (%)": {
            "numerator": "outborn_hypothermia_count",
            "denominator": "total_outborn_hypo",  # This is outborn_count
            "value": "outborn_hypothermia_rate",
        },
        # Other KPIs
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
        # DATA QUALITY
        "Missing Temperature (%)": {
            "numerator": "missing_temperature_count",
            "denominator": "total_admitted",
            "value": "missing_temperature_rate",
        },
        "Missing Birth Weight (%)": {
            "numerator": "missing_birth_weight_count",
            "denominator": "total_admitted",
            "value": "missing_birth_weight_rate",
        },
        "Missing Status of Discharge (%)": {
            "numerator": "missing_discharge_status_count",
            "denominator": "total_admitted",
            "value": "missing_discharge_status_rate",
        },
        "Missing Birth Location (%)": {
            "numerator": "missing_birth_location_count",
            "denominator": "total_admitted",
            "value": "missing_birth_location_rate",
        },
        # ============== ANTIBIOTICS MAPPING - COMMENTED OUT ==============
        # "Antibiotics for Clinical Sepsis (%)": {
        #     "numerator": "antibiotics_count",
        #     "denominator": "total_admitted_abx",
        #     "value": "antibiotics_rate",
        # },
        # "Antibiotics Rate (%)": {
        #     "numerator": "antibiotics_count",
        #     "denominator": "total_admitted_abx",
        #     "value": "antibiotics_rate",
        # },
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

    # Fallback mappings
    if "Inborn Hypothermia" in kpi_name:
        numerator = kpi_data.get("inborn_hypothermia_count", 0)
        denominator = kpi_data.get("total_inborn_hypo", 1)  # This is inborn_count
        value = kpi_data.get("inborn_hypothermia_rate", 0.0)
        return (numerator, denominator, value)
    elif "Outborn Hypothermia" in kpi_name:
        numerator = kpi_data.get("outborn_hypothermia_count", 0)
        denominator = kpi_data.get("total_outborn_hypo", 1)  # This is outborn_count
        value = kpi_data.get("outborn_hypothermia_rate", 0.0)
        return (numerator, denominator, value)
    elif "Admitted Newborns" in kpi_name or "Admitted" in kpi_name:
        numerator = kpi_data.get("admitted_newborns_count", 0)
        denominator = 1
        value = float(numerator)
        return (numerator, denominator, value)
    # ============== ANTIBIOTICS FALLBACK - COMMENTED OUT ==============
    # elif "Antibiotics" in kpi_name or "Sepsis" in kpi_name:
    #     numerator = kpi_data.get("antibiotics_count", 0)
    #     denominator = kpi_data.get("total_admitted_abx", 1)
    #     value = kpi_data.get("antibiotics_rate", 0.0)
    #     return (numerator, denominator, value)
    elif (
        "Hypothermia" in kpi_name
        and "Admission" in kpi_name
        and "After" not in kpi_name
    ):
        numerator = kpi_data.get("hypothermia_on_admission_count", 0)
        denominator = kpi_data.get("total_admitted", 1)
        value = kpi_data.get("hypothermia_on_admission_rate", 0.0)
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


# ---------------- Chart Functions ----------------
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
    **kwargs
):
    """Render a trend chart for newborn KPI"""
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
    **kwargs
):
    """Render facility comparison chart for newborn KPI"""
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
    period_col="period_display",
    value_col="value",
    title="Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="Numerator",
    denominator_name="Denominator",
    **kwargs
):
    """Render region comparison using multi-line chart (one line per region)."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    st.subheader(title)

    if df is None or df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    required_cols = ["Region", "numerator", "denominator", period_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.write("Available columns:", list(df.columns))
        return

    # Keep existing summary computation logic (overall weighted by summed num/den).
    region_totals = {}
    for region_name in (region_names or df["Region"].dropna().unique().tolist()):
        region_data = df[df["Region"] == region_name]
        if region_data.empty:
            continue

        total_numerator = pd.to_numeric(
            region_data["numerator"], errors="coerce"
        ).fillna(0).sum()
        total_denominator = pd.to_numeric(
            region_data["denominator"], errors="coerce"
        ).fillna(0).sum()
        weighted_rate = (
            (total_numerator / total_denominator) * 100 if total_denominator > 0 else 0
        )

        region_totals[region_name] = {
            "total_numerator": total_numerator,
            "total_denominator": total_denominator,
            "weighted_rate": weighted_rate,
        }

    if not region_totals:
        st.info("⚠️ No comparison data available for regions.")
        return

    overall_numerator = sum(v["total_numerator"] for v in region_totals.values())
    overall_denominator = sum(v["total_denominator"] for v in region_totals.values())
    overall_rate = (
        (overall_numerator / overall_denominator) * 100 if overall_denominator > 0 else 0
    )

    comparison_data = []
    for region_name, totals in region_totals.items():
        comparison_data.append(
            {
                "Region": region_name,
                numerator_name: totals["total_numerator"],
                denominator_name: totals["total_denominator"],
                "Rate (%)": f"{totals['weighted_rate']:.2f}%",
            }
        )
    comparison_data.append(
        {
            "Region": "Overall",
            numerator_name: overall_numerator,
            denominator_name: overall_denominator,
            "Rate (%)": f"{overall_rate:.2f}%",
        }
    )
    comparison_df = pd.DataFrame(comparison_data)

    # Period-by-region chart data (same chart style as facility comparison: multi-line).
    chart_df = df.copy()
    if region_names:
        chart_df = chart_df[chart_df["Region"].isin(region_names)].copy()

    chart_df["numerator"] = pd.to_numeric(chart_df["numerator"], errors="coerce").fillna(0)
    chart_df["denominator"] = pd.to_numeric(chart_df["denominator"], errors="coerce").fillna(0)

    grouped = (
        chart_df.groupby(["Region", period_col], as_index=False)
        .agg({"numerator": "sum", "denominator": "sum"})
        .copy()
    )
    grouped["value"] = grouped.apply(
        lambda r: (r["numerator"] / r["denominator"] * 100) if r["denominator"] > 0 else 0,
        axis=1,
    )

    # Normalize and strictly sort periods so lines do not zigzag.
    grouped["_period_label"] = grouped[period_col].apply(format_period_month_year)

    if "period_sort" in chart_df.columns:
        period_map = (
            chart_df[[period_col, "period_sort"]]
            .dropna(subset=[period_col])
            .drop_duplicates(subset=[period_col])
            .copy()
        )
        period_map["_period_label"] = period_map[period_col].apply(format_period_month_year)
        period_map["period_sort"] = pd.to_numeric(period_map["period_sort"], errors="coerce")
        period_map = period_map.sort_values("period_sort")
        period_order = period_map["_period_label"].dropna().tolist()
        grouped = grouped.merge(
            period_map[[period_col, "period_sort"]],
            on=period_col,
            how="left",
        )
    else:
        grouped["period_sort"] = grouped["_period_label"].apply(
            lambda x: dt.datetime.strptime(x, "%b-%y")
            if isinstance(x, str) and "-" in x
            else pd.NaT
        )
        period_order = (
            grouped[["_period_label", "period_sort"]]
            .drop_duplicates()
            .sort_values("period_sort")
            ["_period_label"]
            .dropna()
            .tolist()
        )

    grouped = grouped.sort_values(["Region", "period_sort", "_period_label"]).copy()

    fig = px.line(
        grouped,
        x="_period_label",
        y="value",
        color="Region",
        markers=True,
        title=f"{title} - Region Comparison",
        category_orders={"_period_label": period_order},
        custom_data=["numerator", "denominator"],
        height=350,
    )
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=get_attractive_hover_template(
            title, numerator_name, denominator_name
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
            categoryorder="array",
            categoryarray=period_order,
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(rangemode="tozero", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        legend=dict(title="Regions", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Regional Data Summary")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # Download details with one row per region per period
    period_label = get_current_period_label()
    csv_df = grouped.copy()
    csv_df["Time Period"] = csv_df["_period_label"].apply(
        lambda p: format_period_for_download(p, period_label)
    )
    csv_df["Rate (%)"] = csv_df["value"].apply(lambda v: f"{float(v):.2f}%")
    csv_df = csv_df[
        ["Region", "Time Period", "numerator", "denominator", "Rate (%)"]
    ].rename(
        columns={"numerator": numerator_name, "denominator": denominator_name}
    ).reset_index(drop=True)

    st.download_button(
        label="📥 Download Overall Comparison Data",
        data=csv_df.to_csv(index=False),
        file_name=f"{title.lower().replace(' ', '_')}_region_summary.csv",
        mime="text/csv",
        help="Download overall summary data for region comparison",
        key=f"newborn_region_summary_{title.lower().replace(' ', '_')}_{len(csv_df)}",
    )
    return comparison_df


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
    **kwargs
):
    """Render trend chart for Admitted Newborns - FIXED WITH UNIQUE KEYS"""

    import time

    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Sort periods chronologically
    if "period_sort" in df.columns:
        df = df.sort_values("period_sort")
    else:
        try:
            df["sort_key"] = df[period_col].apply(
                lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if isinstance(x, str) and "-" in x
                    else x
                )
            )
            df = df.sort_values("sort_key")
            df = df.drop(columns=["sort_key"])
        except Exception as e:
            df = df.sort_values(period_col)

    # Create chart
    fig = px.bar(
        df,
        x=x_axis_col,
        y=value_col,
        title=title,
        height=400,
        text=value_col,
        category_orders={x_axis_col: df[x_axis_col].tolist()},
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        hovertemplate=get_attractive_hover_template(value_name, "", "", is_count=True)
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=value_name,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            categoryorder="array",
            categoryarray=df[x_axis_col].tolist(),
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    # Format y-axis as integers with commas
    fig.update_layout(yaxis_tickformat=",")

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Display table below graph
    st.markdown("---")
    st.subheader("📋 Data Table")

    # Create a clean display dataframe
    display_df = df.copy()
    table_columns = [x_axis_col, value_col]
    display_df = display_df[table_columns].copy()

    # Format numbers with commas
    display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:,.0f}")

    # Add Overall/Total row
    total_value = df[value_col].sum() if not df.empty else 0
    overall_row = {
        x_axis_col: "Overall",
        value_col: f"{total_value:,.0f}",
    }

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
            st.metric("📈 Latest Count", f"{df[value_col].iloc[-1]:,.0f}")

        with col2:
            avg_value = df[value_col].mean()
            st.metric("📊 Average per Period", f"{avg_value:,.1f}")

        with col3:
            last_value = df[value_col].iloc[-1]
            prev_value = df[value_col].iloc[-2] if len(df) > 1 else 0
            trend_change = last_value - prev_value
            trend_symbol = (
                "▲" if trend_change > 0 else ("▼" if trend_change < 0 else "–")
            )
            st.metric("📈 Trend from Previous", f"{trend_change:,.0f} {trend_symbol}")

    # Download button with UNIQUE KEY
    summary_df = df.copy().reset_index(drop=True)
    summary_df = summary_df[[x_axis_col, value_col]].copy()
    period_label = get_current_period_label()

    # Format period column
    if x_axis_col in summary_df.columns:
        summary_df[x_axis_col] = summary_df[x_axis_col].apply(
            lambda p: format_period_for_download(p, period_label)
        )

    summary_df = summary_df.rename(columns={value_col: f"{value_name} Count"})
    summary_table = summary_df.copy()

    overall_row = pd.DataFrame(
        {x_axis_col: ["Overall"], f"{value_name} Count": [total_value]}
    )
    summary_table = pd.concat([summary_table, overall_row], ignore_index=True)
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    csv = summary_table.to_csv(index=False)

    # Generate UNIQUE key for newborn download button
    unique_key = f"newborn_admitted_trend_{int(time.time())}_{hash(str(df))}"

    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name="admitted_newborns_trend_data.csv",
        mime="text/csv",
        help="Download the exact data shown in the chart",
        key=unique_key,  # UNIQUE KEY for newborn
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
    **kwargs
):
    """Render facility comparison chart for Admitted Newborns - FIXED WITH UNIQUE KEYS"""

    import time

    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Ensure we have required columns
    if "orgUnit" not in df.columns:
        st.error(f"❌ 'orgUnit' column not found in comparison data")
        st.write("Available columns:", list(df.columns))
        return

    # Create facility mapping
    facility_mapping = {}
    if facility_names and facility_uids and len(facility_names) == len(facility_uids):
        for uid, name in zip(facility_uids, facility_names):
            facility_mapping[str(uid)] = name
    elif "Facility" in df.columns:
        # Extract mapping from data
        for _, row in df.iterrows():
            if "orgUnit" in row and pd.notna(row["orgUnit"]):
                facility_mapping[str(row["orgUnit"])] = row.get(
                    "Facility", str(row["orgUnit"])
                )
    else:
        # Create simple mapping from UIDs
        unique_orgunits = df["orgUnit"].dropna().unique()
        for uid in unique_orgunits:
            facility_mapping[str(uid)] = f"Facility {str(uid)[:8]}"

    if not facility_mapping:
        st.info("⚠️ No facility mapping available for comparison.")
        return

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
                    if isinstance(x, str) and "-" in x
                    else x
                ),
            )
        except:
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order if p is not None]

    # Prepare data for each facility and period
    for facility_uid, facility_name in facility_mapping.items():
        facility_df = df[df["orgUnit"] == facility_uid].copy()

        if facility_df.empty:
            # Skip facilities with no data
            continue

        # Group by period for this facility
        for period_display, period_group in facility_df.groupby("period_display"):
            if not period_group.empty:
                # Sum values for this facility/period
                total_value = (
                    period_group[value_col].sum()
                    if value_col in period_group.columns
                    else 0
                )

                # Skip if there's no data at all for this period
                if len(period_group) == 0:
                    continue

                formatted_period = format_period_month_year(period_display)

                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Facility": facility_name,
                        "value": total_value,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available (all facilities have zero data).")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: (
                dt.datetime.strptime(x, "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            )
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: (
                dt.datetime.strptime(x, "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            ),
        )
    except:
        # Sort alphabetically as fallback
        comparison_df = comparison_df.sort_values(["Facility", "period_display"])
        period_order = sorted(comparison_df["period_display"].unique().tolist())

    # Filter out facilities that have no data
    facilities_with_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            facilities_with_data.append(facility_name)

    # Filter comparison_df to only include facilities with data
    comparison_df = comparison_df[
        comparison_df["Facility"].isin(facilities_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all facilities have zero data).")
        return

    # Create the chart
    fig = px.bar(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        title=f"{title} - Facility Comparison",
        height=500,
        category_orders={"period_display": period_order},
        barmode="group",
        text="value",
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=get_attractive_hover_template(value_name, "", "", is_count=True).replace("%{x}", "%{x}<br>Facility: %{fullData.name}")
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=value_name,
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

    # Format y-axis as integers with commas
    fig.update_layout(yaxis_tickformat=",")

    st.plotly_chart(fig, use_container_width=True)

    # Display table below graph
    st.markdown("---")
    st.subheader("📋 Facility Comparison Data")

    # Create pivot table for better display with Overall row
    pivot_data = []

    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            total_count = facility_data["value"].sum()

            pivot_data.append(
                {
                    "Facility": facility_name,
                    value_name: f"{total_count:,.0f}",
                }
            )

    # Add Overall row for all facilities
    if pivot_data:
        all_counts = comparison_df["value"].sum()

        pivot_data.append(
            {
                "Facility": "Overall",
                value_name: f"{all_counts:,.0f}",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Download button with UNIQUE KEY
    period_label = get_current_period_label()
    csv_df = comparison_df.copy()
    csv_df["Time Period"] = csv_df["period_display"].apply(
        lambda p: format_period_for_download(p, period_label)
    )
    csv_df = csv_df[["Facility", "Time Period", "value"]].rename(
        columns={"value": f"{value_name} Count"}
    )

    # Generate UNIQUE key for newborn download button
    unique_key = f"newborn_admitted_facility_{int(time.time())}_{hash(str(df))}"

    st.download_button(
        label="📥 Download Facility Comparison Data",
        data=csv_df.to_csv(index=False),
        file_name="admitted_newborns_facility_comparison.csv",
        mime="text/csv",
        help="Download the facility comparison data",
        key=unique_key,  # UNIQUE KEY for newborn
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
    **kwargs
):
    """Use the exact admitted-mothers region comparison implementation."""
    from utils.kpi_admitted_mothers import render_admitted_mothers_region_comparison_chart

    return render_admitted_mothers_region_comparison_chart(
        df=df,
        period_col=period_col,
        value_col=value_col,
        title=title,
        bg_color=bg_color,
        text_color=text_color,
        region_names=region_names,
        region_mapping=region_mapping,
        facilities_by_region=facilities_by_region,
        value_name=value_name,
        suppress_plot=kwargs.get("suppress_plot", False),
        **kwargs,
    )


# ---------------- Additional Helper Functions ----------------
def extract_period_columns_newborn(df, date_column):
    """
    SIMPLE VERSION: Assumes dates are already valid, just need proper grouping for newborn data
    """
    if df.empty or date_column not in df.columns:
        return df

    result_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
        result_df["event_date"] = pd.to_datetime(
            result_df[date_column], errors="coerce"
        )
    else:
        result_df["event_date"] = result_df[date_column]

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
    # ============== ANTIBIOTICS FUNCTIONS - REMOVED FROM EXPORTS ==============
    # # Antibiotics functions
    # "compute_probable_sepsis_count",
    # "compute_antibiotics_count",
    # "compute_antibiotics_rate",
    # "compute_antibiotics_kpi",
    # "get_numerator_denominator_for_antibiotics",
    # Numerator computation functions
    "compute_inborn_count",
    "compute_outborn_count",
    "compute_hypothermia_on_admission_count",
    "compute_inborn_hypothermia_count",  # NEW
    "compute_outborn_hypothermia_count",  # NEW
    "compute_neonatal_death_count",
    # KPI computation functions
    "compute_inborn_rate",
    "compute_outborn_rate",
    "compute_hypothermia_on_admission_rate",
    "compute_inborn_hypothermia_rate",  # NEW
    "compute_outborn_hypothermia_rate",  # NEW
    "compute_neonatal_mortality_rate",
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
    "extract_period_columns_newborn",
]
