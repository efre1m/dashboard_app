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

# ANTIBIOTICS COLUMNS - ADDED
SUBCATEGORIES_INFECTION_COL = (
    "sub_categories_of_infection_discharge_and_final_diagnosis"
)
ANTIBIOTICS_ADMINISTERED_COL = "were_antibiotics_administered?_interventions"
PROBABLE_SEPSIS_CODE = "1"
YES_CODE = "1"


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


# ---------------- ANTIBIOTICS FUNCTIONS - ADDED ----------------
def compute_probable_sepsis_count(df, facility_uids=None):
    """Count newborns with probable sepsis"""
    cache_key = get_cache_key_newborn(df, facility_uids, "probable_sepsis_count")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if SUBCATEGORIES_INFECTION_COL not in filtered_df.columns:
            result = 0
        else:
            df_work = filtered_df.copy()
            df_work["infection_clean"] = df_work[SUBCATEGORIES_INFECTION_COL].astype(
                str
            )
            df_work["infection_numeric"] = pd.to_numeric(
                df_work["infection_clean"].str.split(".").str[0], errors="coerce"
            )

            sepsis_mask = df_work["infection_numeric"] == float(PROBABLE_SEPSIS_CODE)

            if "tei_id" in df_work.columns:
                sepsis_teis = df_work.loc[sepsis_mask, "tei_id"].dropna().unique()
                result = len(sepsis_teis)
            else:
                result = int(sepsis_mask.sum())

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_antibiotics_count(df, facility_uids=None):
    """Count newborns with probable sepsis AND antibiotics administered"""
    cache_key = get_cache_key_newborn(df, facility_uids, "antibiotics_count")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if (
            SUBCATEGORIES_INFECTION_COL not in filtered_df.columns
            or ANTIBIOTICS_ADMINISTERED_COL not in filtered_df.columns
        ):
            result = 0
        else:
            df_work = filtered_df.copy()

            # Clean infection column
            df_work["infection_clean"] = df_work[SUBCATEGORIES_INFECTION_COL].astype(
                str
            )
            df_work["infection_numeric"] = pd.to_numeric(
                df_work["infection_clean"].str.split(".").str[0], errors="coerce"
            )

            # Clean antibiotics column
            df_work["antibiotics_clean"] = df_work[ANTIBIOTICS_ADMINISTERED_COL].astype(
                str
            )
            df_work["antibiotics_numeric"] = pd.to_numeric(
                df_work["antibiotics_clean"].str.split(".").str[0], errors="coerce"
            )

            sepsis_mask = df_work["infection_numeric"] == float(PROBABLE_SEPSIS_CODE)
            antibiotics_mask = df_work["antibiotics_numeric"] == float(YES_CODE)
            combined_mask = sepsis_mask & antibiotics_mask

            if "tei_id" in df_work.columns:
                eligible_teis = df_work.loc[combined_mask, "tei_id"].dropna().unique()
                result = len(eligible_teis)
            else:
                result = int(combined_mask.sum())

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_antibiotics_rate(df, facility_uids=None):
    """Compute antibiotics rate for newborns with clinical sepsis"""
    cache_key = get_cache_key_newborn(df, facility_uids, "antibiotics_rate")

    if cache_key in st.session_state.kpi_cache_newborn:
        return st.session_state.kpi_cache_newborn[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        antibiotics_count = compute_antibiotics_count(df, facility_uids)
        probable_sepsis_count = compute_probable_sepsis_count(df, facility_uids)

        rate = (
            (antibiotics_count / probable_sepsis_count * 100)
            if probable_sepsis_count > 0
            else 0.0
        )
        result = (float(rate), int(antibiotics_count), int(probable_sepsis_count))

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


def compute_antibiotics_kpi(df, facility_uids=None):
    """Compute antibiotics KPI data"""
    rate, antibiotics_count, probable_sepsis_count = compute_antibiotics_rate(
        df, facility_uids
    )

    return {
        "antibiotics_rate": float(rate),
        "antibiotics_count": int(antibiotics_count),
        "probable_sepsis_count": int(probable_sepsis_count),
    }


def get_numerator_denominator_for_antibiotics(
    df, facility_uids=None, date_range_filters=None
):
    """Get numerator and denominator for Antibiotics KPI"""
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Use discharge date for antibiotics
    date_column = DISCHARGE_DATE_COL

    if date_column in filtered_df.columns:
        filtered_df[date_column] = pd.to_datetime(
            filtered_df[date_column], errors="coerce"
        )
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

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

    antibiotics_data = compute_antibiotics_kpi(filtered_df, facility_uids)

    numerator = antibiotics_data.get("antibiotics_count", 0)
    denominator = antibiotics_data.get("probable_sepsis_count", 1)
    rate = antibiotics_data.get("antibiotics_rate", 0.0)

    return (numerator, denominator, rate)


# ---------------- Existing Newborn Functions (Keep as is) ----------------
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
    df_copy["lowest_temp_numeric"] = pd.to_numeric(
        df_copy[LOWEST_TEMPERATURE_COL], errors="coerce"
    )
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
    inborn_rate, inborn_count, _ = compute_inborn_rate(filtered_df, facility_uids)
    outborn_rate, outborn_count, _ = compute_outborn_rate(filtered_df, facility_uids)
    hypothermia_on_admission_rate, hypothermia_on_admission_count, _ = (
        compute_hypothermia_on_admission_rate(filtered_df, facility_uids)
    )
    hypothermia_after_admission_rate, hypothermia_after_admission_count, _ = (
        compute_hypothermia_after_admission_rate(filtered_df, facility_uids)
    )
    neonatal_mortality_rate, death_count, _ = compute_neonatal_mortality_rate(
        filtered_df, facility_uids
    )
    antibiotics_rate, antibiotics_count, probable_sepsis_count = (
        compute_antibiotics_rate(filtered_df, facility_uids)
    )
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
        "antibiotics_rate": float(antibiotics_rate),
        "antibiotics_count": int(antibiotics_count),
        "probable_sepsis_count": int(probable_sepsis_count),
        "admitted_newborns_count": int(admitted_newborns_count),
    }

    st.session_state.kpi_cache_newborn[cache_key] = result
    return result


# ---------------- Date Handling ----------------
def get_relevant_date_column_for_newborn_kpi(kpi_name):
    """Get the relevant event date column for a specific newborn KPI"""
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
        "Antibiotics for Clinical Sepsis (%)": "event_date_discharge_and_final_diagnosis",
        "Antibiotics Rate (%)": "event_date_discharge_and_final_diagnosis",
        # Admitted Newborns KPI
        "Admitted Newborns": "enrollment_date",
        "Total Admitted Newborns": "event_date_admission_information",
    }

    for key in program_stage_date_mapping:
        if key in kpi_name:
            return program_stage_date_mapping[key]

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
    elif any(word in kpi_name for word in ["Antibiotics", "Sepsis"]):
        return "event_date_discharge_and_final_diagnosis"
    elif any(word in kpi_name for word in ["Admitted Newborns", "Admitted"]):
        return "enrollment_date"

    return "event_date_admission_information"


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
        "Antibiotics for Clinical Sepsis (%)": {
            "numerator": "antibiotics_count",
            "denominator": "probable_sepsis_count",
            "value": "antibiotics_rate",
        },
        "Antibiotics Rate (%)": {
            "numerator": "antibiotics_count",
            "denominator": "probable_sepsis_count",
            "value": "antibiotics_rate",
        },
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
    if "Admitted Newborns" in kpi_name or "Admitted" in kpi_name:
        numerator = kpi_data.get("admitted_newborns_count", 0)
        denominator = 1
        value = float(numerator)
        return (numerator, denominator, value)
    elif "Antibiotics" in kpi_name or "Sepsis" in kpi_name:
        numerator = kpi_data.get("antibiotics_count", 0)
        denominator = kpi_data.get("probable_sepsis_count", 1)
        value = kpi_data.get("antibiotics_rate", 0.0)
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
):
    """Render region comparison chart - CHART ABOVE TABLE VERSION"""

    if text_color is None:
        text_color = auto_text_color(bg_color)

    st.subheader(title)

    if df is None or df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Check for required columns
    required_cols = ["Region", "numerator", "denominator"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.write("Available columns:", list(df.columns))
        return

    # Calculate region totals
    region_totals = {}
    for region_name in region_names:
        region_data = df[df["Region"] == region_name]

        if not region_data.empty:
            # SUM the numerator and denominator columns directly
            total_numerator = region_data["numerator"].sum()
            total_denominator = region_data["denominator"].sum()

            # Calculate weighted average
            weighted_rate = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )

            region_totals[region_name] = {
                "total_numerator": total_numerator,
                "total_denominator": total_denominator,
                "weighted_rate": weighted_rate,
            }

    # Calculate overall totals
    overall_numerator = sum(data["total_numerator"] for data in region_totals.values())
    overall_denominator = sum(
        data["total_denominator"] for data in region_totals.values()
    )
    overall_rate = (
        (overall_numerator / overall_denominator * 100)
        if overall_denominator > 0
        else 0
    )

    # Create comparison table data
    comparison_data = []
    for region_name, totals in region_totals.items():
        comparison_data.append(
            {
                "Region": region_name,
                f"{numerator_name}": totals["total_numerator"],
                f"{denominator_name}": totals["total_denominator"],
                "Rate (%)": f"{totals['weighted_rate']:.2f}%",
            }
        )

    # Add overall row
    comparison_data.append(
        {
            "Region": "Overall",
            f"{numerator_name}": overall_numerator,
            f"{denominator_name}": overall_denominator,
            "Rate (%)": f"{overall_rate:.2f}%",
        }
    )

    comparison_df = pd.DataFrame(comparison_data)

    # *** 1. DISPLAY CHART FIRST ***
    chart_data = comparison_df[comparison_df["Region"] != "Overall"].copy()

    if not chart_data.empty:
        chart_data["Rate_float"] = (
            chart_data["Rate (%)"].str.replace("%", "").astype(float)
        )

        fig = px.bar(
            chart_data,
            x="Region",
            y="Rate_float",
            title=f"{title} - Regional Rates",
            color="Region",
            text_auto=".1f",
        )

        fig.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font_color=text_color,
            showlegend=False,
            yaxis_title="Rate (%)",
            height=400,  # Fixed height for consistency
        )

        # Display chart at the top
        st.plotly_chart(fig, use_container_width=True)

    # Add some spacing between chart and table
    st.markdown("---")

    # *** 2. DISPLAY TABLE SECOND ***
    st.markdown("### Regional Data Summary")

    # Display the table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        height=300,  # Limit table height
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
        hover_data=[value_col],
        text=value_col,
        category_orders={x_axis_col: df[x_axis_col].tolist()},
    )

    fig.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        hovertemplate=(
            f"<b>%{{x}}</b><br>" f"{value_name}: %{{y:,.0f}}<br>" f"<extra></extra>"
        ),
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

    # Format period column
    if x_axis_col in summary_df.columns:
        summary_df[x_axis_col] = summary_df[x_axis_col].apply(format_period_month_year)

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
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Facility: %{{fullData.name}}<br>"
            f"{value_name}: %{{y:,.0f}}<extra></extra>"
        ),
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
    csv = comparison_df.to_csv(index=False)

    # Generate UNIQUE key for newborn download button
    unique_key = f"newborn_admitted_facility_{int(time.time())}_{hash(str(df))}"

    st.download_button(
        label="📥 Download Facility Comparison Data",
        data=csv,
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
):
    """Render region comparison chart for Admitted Newborns - FIXED DOWNLOAD BUTTONS"""

    import io  # Add this import at the top of the function

    if text_color is None:
        text_color = auto_text_color(bg_color)

    st.subheader(title)

    if df is None or df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Check for required columns
    if "Region" not in df.columns or value_col not in df.columns:
        st.error("❌ Missing required columns for region comparison.")
        return

    # Calculate region totals
    region_totals = {}
    for region_name in region_names:
        region_data = df[df["Region"] == region_name]

        if not region_data.empty:
            total_value = region_data[value_col].sum()
            region_totals[region_name] = total_value

    # Create comparison table data
    comparison_data = []
    for region_name, total_value in region_totals.items():
        comparison_data.append(
            {
                "Region": region_name,
                f"{value_name}": total_value,
            }
        )

    # Add overall total
    overall_total = sum(region_totals.values())
    comparison_data.append(
        {
            "Region": "Overall",
            f"{value_name}": overall_total,
        }
    )

    comparison_df = pd.DataFrame(comparison_data)

    # *** 1. DISPLAY CHART FIRST ***
    chart_data = comparison_df[comparison_df["Region"] != "Overall"].copy()

    if not chart_data.empty:
        fig = px.bar(
            chart_data,
            x="Region",
            y=value_name,
            title=f"{title} - Regional Comparison",
            color="Region",
            text_auto=True,
        )

        fig.update_layout(
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font_color=text_color,
            showlegend=False,
            yaxis_title=value_name,
            height=400,
        )

        # Display chart first
        st.plotly_chart(fig, use_container_width=True)

    # *** 2. DISPLAY TABLE SECOND ***
    st.markdown("### Regional Summary")
    st.dataframe(
        comparison_df,
        use_container_width=True,
        height=300,
    )

    # *** 3. FIXED DOWNLOAD BUTTONS WITH UNIQUE KEYS ***
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Create CSV data - FIXED WITH UNIQUE KEY
        csv_data = comparison_df.to_csv(index=False)

        # Generate a UNIQUE key using title and timestamp
        import time

        unique_key = (
            f"admitted_newborns_csv_{int(time.time())}_{hash(str(comparison_df))}"
        )

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"admitted_newborns_region_comparison_{title.replace(' ', '_')}.csv",
            mime="text/csv",
            key=unique_key,  # UNIQUE KEY for Admitted Newborns
            use_container_width=True,
        )

    return comparison_df


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
    "compute_admitted_newborns_kpi",
    # Antibiotics functions
    "compute_probable_sepsis_count",
    "compute_antibiotics_count",
    "compute_antibiotics_rate",
    "compute_antibiotics_kpi",
    "get_numerator_denominator_for_antibiotics",
    # Numerator computation functions
    "compute_inborn_count",
    "compute_outborn_count",
    "compute_hypothermia_on_admission_count",
    "compute_hypothermia_after_admission_count",
    "compute_neonatal_death_count",
    # KPI computation functions
    "compute_inborn_rate",
    "compute_outborn_rate",
    "compute_hypothermia_on_admission_rate",
    "compute_hypothermia_after_admission_rate",
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
    "aggregate_by_period_with_sorting_newborn",
    "extract_period_columns_newborn",
]
