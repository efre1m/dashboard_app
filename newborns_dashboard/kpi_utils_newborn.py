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
    """Render region comparison chart for newborn KPI"""
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
    """Render trend chart for Admitted Newborns"""
    from utils.kpi_admitted_mothers import render_admitted_mothers_trend_chart

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
