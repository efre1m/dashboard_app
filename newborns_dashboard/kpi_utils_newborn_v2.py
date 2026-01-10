# # kpi_utils_newborn_v2.py - FIXED COUNTING FOR REGIONAL COMPARISON

# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import datetime as dt
# import streamlit as st
# import hashlib
# import numpy as np
# import logging
# import warnings
# from utils.kpi_utils import auto_text_color, format_period_month_year

# # Import existing newborn functions to avoid duplication
# from newborns_dashboard.kpi_utils_newborn import (
#     # Antibiotics functions
#     compute_probable_sepsis_count,
#     compute_newborn_kpis,
#     # Date handling functions
#     get_relevant_date_column_for_newborn_kpi,
#     get_numerator_denominator_for_newborn_kpi,
#     # Chart rendering functions
#     render_newborn_trend_chart,
#     render_newborn_facility_comparison_chart,
#     render_newborn_region_comparison_chart,
# )

# warnings.filterwarnings("ignore")

# # ---------------- Caching Setup for V2 ----------------
# if "kpi_cache_newborn_v2" not in st.session_state:
#     st.session_state.kpi_cache_newborn_v2 = {}


# def get_cache_key_newborn_v2(df, facility_uids=None, computation_type=""):
#     """Generate a unique cache key based on data and filters"""
#     key_data = {
#         "computation_type": computation_type,
#         "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
#         "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
#         "data_shape": df.shape,
#     }
#     return str(key_data)


# def clear_cache_newborn_v2():
#     """Clear the newborn KPI v2 cache - call this when you know data has changed"""
#     st.session_state.kpi_cache_newborn_v2 = {}


# # ---------------- Newborn KPI Constants for V2 ----------------
# # CULTURE COLUMNS
# BLOOD_CULTURE_COL = "blood_culture_for_suspected_sepsis_microbiology_and_labs"
# CULTURE_DONE_VALUES = ["1", "2", "3"]  # 1=Negative, 2=Positive, 3=Unknown
# CULTURE_RESULT_RECORDED_VALUES = ["1", "2"]  # Negative or Positive only

# # ANTIBIOTICS COLUMNS (reused from original)
# ANTIBIOTICS_ADMINISTERED_COL = "were_antibiotics_administered?_interventions"
# ANTIBIOTICS_YES_VALUE = "1"
# SUBCATEGORIES_INFECTION_COL = (
#     "sub_categories_of_infection_discharge_and_final_diagnosis"
# )
# PROBABLE_SEPSIS_CODE = "1"
# YES_CODE = "1"

# # Event date columns for culture
# MICROBIOLOGY_DATE_COL = "event_date_microbiology_and_labs"


# # ---------------- CULTURE DONE KPI Functions (V2) - FIXED COUNTING ----------------
# def compute_culture_done_numerator_v2(df, facility_uids=None):
#     """
#     Compute numerator for Culture Done KPI:
#     Count of UNIQUE babies on antibiotics who had blood culture done
#     """
#     cache_key = get_cache_key_newborn_v2(df, facility_uids, "culture_done_numerator_v2")

#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     if df is None or df.empty:
#         result = 0
#     else:
#         filtered_df = df.copy()

#         # CRITICAL FIX: Apply facility filter if specified
#         if facility_uids and "orgUnit" in filtered_df.columns:
#             if not isinstance(facility_uids, list):
#                 facility_uids = [facility_uids]
#             if facility_uids and facility_uids != ["All Facilities"]:
#                 filtered_df = filtered_df[
#                     filtered_df["orgUnit"].isin(facility_uids)
#                 ].copy()

#         # Count UNIQUE babies with blood culture done (any result)
#         if BLOOD_CULTURE_COL not in filtered_df.columns:
#             result = 0
#         else:
#             # Clean the culture column
#             filtered_df["culture_clean"] = filtered_df[BLOOD_CULTURE_COL].astype(str)
#             filtered_df["culture_numeric"] = pd.to_numeric(
#                 filtered_df["culture_clean"].str.split(".").str[0], errors="coerce"
#             )

#             # Filter rows where culture was done
#             culture_rows = filtered_df[
#                 filtered_df["culture_numeric"].isin(
#                     [float(x) for x in CULTURE_DONE_VALUES]
#                 )
#             ]

#             # Count UNIQUE patients (tei_id) who had culture done
#             if "tei_id" in culture_rows.columns:
#                 # Get unique patients with culture done
#                 unique_culture_patients = culture_rows["tei_id"].dropna().unique()
#                 result = len(unique_culture_patients)
#             else:
#                 # If no tei_id column, count rows (fallback)
#                 result = len(culture_rows)

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# def compute_antibiotics_denominator_v2(df, facility_uids=None):
#     """
#     Compute denominator for Culture Done KPI:
#     Total count of UNIQUE babies who received antibiotics
#     """
#     cache_key = get_cache_key_newborn_v2(
#         df, facility_uids, "antibiotics_denominator_v2"
#     )

#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     if df is None or df.empty:
#         result = 0
#     else:
#         filtered_df = df.copy()

#         # CRITICAL FIX: Apply facility filter if specified
#         if facility_uids and "orgUnit" in filtered_df.columns:
#             if not isinstance(facility_uids, list):
#                 facility_uids = [facility_uids]
#             if facility_uids and facility_uids != ["All Facilities"]:
#                 filtered_df = filtered_df[
#                     filtered_df["orgUnit"].isin(facility_uids)
#                 ].copy()

#         if ANTIBIOTICS_ADMINISTERED_COL not in filtered_df.columns:
#             result = 0
#         else:
#             # Clean the antibiotics column
#             filtered_df["antibiotics_clean"] = filtered_df[
#                 ANTIBIOTICS_ADMINISTERED_COL
#             ].astype(str)
#             filtered_df["antibiotics_numeric"] = pd.to_numeric(
#                 filtered_df["antibiotics_clean"].str.split(".").str[0], errors="coerce"
#             )

#             # Filter rows where antibiotics were administered
#             antibiotics_rows = filtered_df[
#                 filtered_df["antibiotics_numeric"] == float(YES_CODE)
#             ]

#             # Count UNIQUE patients (tei_id) who received antibiotics
#             if "tei_id" in antibiotics_rows.columns:
#                 unique_antibiotics_patients = (
#                     antibiotics_rows["tei_id"].dropna().unique()
#                 )
#                 result = len(unique_antibiotics_patients)
#             else:
#                 # If no tei_id column, count rows (fallback)
#                 result = len(antibiotics_rows)

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# def compute_culture_done_kpi_v2(df, facility_uids=None):
#     """
#     Compute Culture Done KPI for the given dataframe
#     Formula: % Culture Done for Babies on Antibiotics =
#              (UNIQUE babies on antibiotics with blood culture done) ÷ (UNIQUE babies on antibiotics) × 100
#     """
#     cache_key = get_cache_key_newborn_v2(df, facility_uids, "culture_done_kpi_v2")

#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     if df is None or df.empty:
#         result = {
#             "culture_rate": 0.0,
#             "culture_count": 0,
#             "antibiotics_count": 0,
#         }
#     else:
#         # Count UNIQUE babies with culture done (numerator)
#         culture_count = compute_culture_done_numerator_v2(df, facility_uids)

#         # Count UNIQUE babies on antibiotics (denominator)
#         antibiotics_count = compute_antibiotics_denominator_v2(df, facility_uids)

#         # Calculate culture rate
#         culture_rate = (
#             (culture_count / antibiotics_count * 100) if antibiotics_count > 0 else 0.0
#         )

#         result = {
#             "culture_rate": float(culture_rate),
#             "culture_count": int(culture_count),
#             "antibiotics_count": int(antibiotics_count),
#         }

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# # ---------------- CULTURE RESULT RECORDED KPI Functions (V2) - FIXED COUNTING ----------------
# def compute_culture_result_recorded_numerator_v2(df, facility_uids=None):
#     """
#     Compute numerator for Culture Result Recorded KPI:
#     Count of UNIQUE babies with blood culture done AND result recorded (Negative or Positive only)
#     """
#     cache_key = get_cache_key_newborn_v2(
#         df, facility_uids, "culture_result_recorded_numerator_v2"
#     )

#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     if df is None or df.empty:
#         result = 0
#     else:
#         filtered_df = df.copy()

#         # CRITICAL FIX: Apply facility filter if specified
#         if facility_uids and "orgUnit" in filtered_df.columns:
#             if not isinstance(facility_uids, list):
#                 facility_uids = [facility_uids]
#             if facility_uids and facility_uids != ["All Facilities"]:
#                 filtered_df = filtered_df[
#                     filtered_df["orgUnit"].isin(facility_uids)
#                 ].copy()

#         # Count UNIQUE babies with blood culture result recorded
#         if BLOOD_CULTURE_COL not in filtered_df.columns:
#             result = 0
#         else:
#             # Clean the culture column
#             filtered_df["culture_clean"] = filtered_df[BLOOD_CULTURE_COL].astype(str)
#             filtered_df["culture_numeric"] = pd.to_numeric(
#                 filtered_df["culture_clean"].str.split(".").str[0], errors="coerce"
#             )

#             # Filter rows where culture result was recorded (Negative or Positive only)
#             result_rows = filtered_df[
#                 filtered_df["culture_numeric"].isin(
#                     [float(x) for x in CULTURE_RESULT_RECORDED_VALUES]
#                 )
#             ]

#             # Count UNIQUE patients (tei_id) with recorded results
#             if "tei_id" in result_rows.columns:
#                 unique_result_patients = result_rows["tei_id"].dropna().unique()
#                 result = len(unique_result_patients)
#             else:
#                 # If no tei_id column, count rows (fallback)
#                 result = len(result_rows)

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# def compute_culture_result_recorded_kpi_v2(df, facility_uids=None):
#     """
#     Compute Culture Result Recorded KPI for the given dataframe
#     Formula: % Blood culture result recorded =
#              (UNIQUE babies with recorded result: Negative or Positive) ÷
#              (UNIQUE babies with culture done: Negative, Positive, or Unknown) × 100
#     """
#     cache_key = get_cache_key_newborn_v2(
#         df, facility_uids, "culture_result_recorded_kpi_v2"
#     )

#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     if df is None or df.empty:
#         result = {
#             "culture_result_rate": 0.0,
#             "culture_result_count": 0,
#             "culture_done_count": 0,
#         }
#     else:
#         # Count UNIQUE babies with culture result recorded (numerator)
#         culture_result_count = compute_culture_result_recorded_numerator_v2(
#             df, facility_uids
#         )

#         # Count UNIQUE babies with culture done (denominator)
#         culture_done_count = compute_culture_done_numerator_v2(df, facility_uids)

#         # Calculate culture result recorded rate
#         culture_result_rate = (
#             (culture_result_count / culture_done_count * 100)
#             if culture_done_count > 0
#             else 0.0
#         )

#         result = {
#             "culture_result_rate": float(culture_result_rate),
#             "culture_result_count": int(culture_result_count),
#             "culture_done_count": int(culture_done_count),
#         }

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# # ---------------- CULTURE DONE FOR SEPSIS KPI Functions (V2) - FIXED COUNTING ----------------
# def compute_culture_done_sepsis_kpi_v2(df, facility_uids=None):
#     """
#     Compute Culture Done for Sepsis KPI for the given dataframe
#     Formula: % Culture done for babies with clinical sepsis =
#              (UNIQUE babies with sepsis who had blood culture done) ÷
#              (UNIQUE babies with Probable Sepsis) × 100
#     """
#     cache_key = get_cache_key_newborn_v2(
#         df, facility_uids, "culture_done_sepsis_kpi_v2"
#     )

#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     if df is None or df.empty:
#         result = {
#             "culture_sepsis_rate": 0.0,
#             "culture_count": 0,
#             "sepsis_count": 0,
#         }
#     else:
#         # Count UNIQUE babies with culture done (numerator)
#         culture_count = compute_culture_done_numerator_v2(df, facility_uids)

#         # Count UNIQUE babies with Probable Sepsis (denominator)
#         sepsis_count = compute_probable_sepsis_count(df, facility_uids)

#         # Calculate culture rate for sepsis cases
#         culture_sepsis_rate = (
#             (culture_count / sepsis_count * 100) if sepsis_count > 0 else 0.0
#         )

#         result = {
#             "culture_sepsis_rate": float(culture_sepsis_rate),
#             "culture_count": int(culture_count),
#             "sepsis_count": int(sepsis_count),
#         }

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# # ---------------- Master KPI Function V2 - FIXED COUNTING ----------------
# def compute_newborn_kpis_v2(df, facility_uids=None, date_column=None):
#     """
#     Compute all newborn KPIs including culture KPIs with optional date filtering
#     """
#     cache_key = get_cache_key_newborn_v2(
#         df, facility_uids, f"main_kpis_v2_{date_column}"
#     )
#     if cache_key in st.session_state.kpi_cache_newborn_v2:
#         return st.session_state.kpi_cache_newborn_v2[cache_key]

#     # Get existing KPIs with proper counting
#     existing_kpis = compute_newborn_kpis(df, facility_uids, date_column)

#     # Compute new culture KPIs with proper counting
#     culture_done_data = compute_culture_done_kpi_v2(df, facility_uids)
#     culture_result_data = compute_culture_result_recorded_kpi_v2(df, facility_uids)
#     culture_sepsis_data = compute_culture_done_sepsis_kpi_v2(df, facility_uids)

#     # Merge all results
#     result = {
#         **existing_kpis,
#         "culture_done_rate": float(culture_done_data.get("culture_rate", 0.0)),
#         "culture_done_count": int(culture_done_data.get("culture_count", 0)),
#         "antibiotics_count_for_culture": int(
#             culture_done_data.get("antibiotics_count", 0)
#         ),
#         "culture_result_recorded_rate": float(
#             culture_result_data.get("culture_result_rate", 0.0)
#         ),
#         "culture_result_recorded_count": int(
#             culture_result_data.get("culture_result_count", 0)
#         ),
#         "culture_done_total_count": int(
#             culture_result_data.get("culture_done_count", 0)
#         ),
#         "culture_sepsis_rate": float(
#             culture_sepsis_data.get("culture_sepsis_rate", 0.0)
#         ),
#         "culture_sepsis_count": int(culture_sepsis_data.get("culture_count", 0)),
#         "sepsis_total_count": int(culture_sepsis_data.get("sepsis_count", 0)),
#     }

#     st.session_state.kpi_cache_newborn_v2[cache_key] = result
#     return result


# # ---------------- Date Handling for Culture KPIs ----------------
# def get_relevant_date_column_for_newborn_kpi_v2(kpi_name):
#     """
#     Get the relevant event date column for a specific newborn KPI (V2 with culture support)
#     """
#     # First check with original function
#     original_date = get_relevant_date_column_for_newborn_kpi(kpi_name)
#     if original_date:
#         return original_date

#     # Add mappings for culture KPIs
#     program_stage_date_mapping = {
#         # Culture KPIs use microbiology date
#         "Culture Done for Babies on Antibiotics (%)": "event_date_microbiology_and_labs",
#         "Blood Culture Result Recorded (%)": "event_date_microbiology_and_labs",
#         "Culture Done for Babies with Clinical Sepsis (%)": "event_date_microbiology_and_labs",
#         "Culture Done Rate (%)": "event_date_microbiology_and_labs",
#         "Culture Result Recorded Rate (%)": "event_date_microbiology_and_labs",
#         "Culture for Sepsis Rate (%)": "event_date_microbiology_and_labs",
#     }

#     for key in program_stage_date_mapping:
#         if key in kpi_name:
#             return program_stage_date_mapping[key]

#     if any(word in kpi_name for word in ["Culture", "Blood Culture", "Microbiology"]):
#         return "event_date_microbiology_and_labs"

#     return "event_date_admission_information"  # Default fallback


# def prepare_data_for_newborn_trend_chart_v2(
#     df, kpi_name, facility_uids=None, date_range_filters=None
# ):
#     """
#     Prepare patient-level data for trend chart using KPI-specific dates (V2 with culture support)
#     """
#     if df.empty:
#         return pd.DataFrame(), None

#     filtered_df = df.copy()

#     # Apply facility filter if specified
#     if facility_uids and "orgUnit" in filtered_df.columns:
#         if not isinstance(facility_uids, list):
#             facility_uids = [facility_uids]
#         if facility_uids and facility_uids != ["All Facilities"]:
#             filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

#     # Get the SPECIFIC date column for this KPI
#     date_column = get_relevant_date_column_for_newborn_kpi_v2(kpi_name)

#     # Check if the SPECIFIC date column exists
#     if date_column not in filtered_df.columns:
#         if "event_date" in filtered_df.columns:
#             date_column = "event_date"
#             st.warning(
#                 f"⚠️ KPI-specific date column not found for {kpi_name}, using 'event_date' instead"
#             )
#         elif "enrollment_date" in filtered_df.columns and "Admitted" in kpi_name:
#             date_column = "enrollment_date"
#             st.warning(
#                 f"⚠️ KPI-specific date column not found for {kpi_name}, using 'enrollment_date' instead"
#             )
#         else:
#             st.warning(
#                 f"⚠️ Required date column '{date_column}' not found for {kpi_name}"
#             )
#             return pd.DataFrame(), date_column

#     # Create result dataframe
#     result_df = filtered_df.copy()
#     result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")

#     # Apply date range filtering
#     if date_range_filters:
#         start_date = date_range_filters.get("start_date")
#         end_date = date_range_filters.get("end_date")

#         if start_date and end_date:
#             start_dt = pd.Timestamp(start_date)
#             end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

#             result_df = result_df[
#                 (result_df["event_date"] >= start_dt)
#                 & (result_df["event_date"] < end_dt)
#             ].copy()

#     # Filter out rows without valid dates
#     result_df = result_df[result_df["event_date"].notna()].copy()

#     if result_df.empty:
#         st.info(f"⚠️ No data with valid dates in '{date_column}' for {kpi_name}")
#         return pd.DataFrame(), date_column

#     # Get period label
#     period_label = st.session_state.get("period_label", "Monthly")
#     if "filters" in st.session_state and "period_label" in st.session_state.filters:
#         period_label = st.session_state.filters["period_label"]

#     # Create period columns using time_filter utility
#     from utils.time_filter import assign_period

#     result_df = assign_period(result_df, "event_date", period_label)

#     return result_df, date_column


# def get_numerator_denominator_for_newborn_kpi_v2(
#     df, kpi_name, facility_uids=None, date_range_filters=None
# ):
#     """
#     Get numerator and denominator for a specific newborn KPI with date range filtering (V2 with culture support)
#     """
#     if df is None or df.empty:
#         return (0, 0, 0.0)

#     filtered_df = df.copy()

#     # Apply facility filter if specified
#     if facility_uids and "orgUnit" in filtered_df.columns:
#         if not isinstance(facility_uids, list):
#             facility_uids = [facility_uids]
#         if facility_uids and facility_uids != ["All Facilities"]:
#             filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

#     # Get the SPECIFIC date column for this KPI
#     date_column = get_relevant_date_column_for_newborn_kpi_v2(kpi_name)

#     # Filter to only include rows that have this specific date
#     if date_column in filtered_df.columns:
#         filtered_df[date_column] = pd.to_datetime(
#             filtered_df[date_column], errors="coerce"
#         )
#         filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

#         # Apply date range filtering if provided
#         if date_range_filters:
#             start_date = date_range_filters.get("start_date")
#             end_date = date_range_filters.get("end_date")

#             if start_date and end_date:
#                 start_dt = pd.Timestamp(start_date)
#                 end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

#                 filtered_df = filtered_df[
#                     (filtered_df[date_column] >= start_dt)
#                     & (filtered_df[date_column] < end_dt)
#                 ].copy()

#     if filtered_df.empty:
#         return (0, 0, 0.0)

#     # Now compute KPI on date-filtered AND facility-filtered data
#     kpi_data = compute_newborn_kpis_v2(filtered_df, facility_uids, date_column)

#     kpi_mapping = {
#         # Existing KPIs
#         "Inborn Rate (%)": {
#             "numerator": "inborn_count",
#             "denominator": "total_admitted",
#             "value": "inborn_rate",
#         },
#         "Outborn Rate (%)": {
#             "numerator": "outborn_count",
#             "denominator": "total_admitted",
#             "value": "outborn_rate",
#         },
#         "Hypothermia on Admission Rate (%)": {
#             "numerator": "hypothermia_on_admission_count",
#             "denominator": "total_admitted",
#             "value": "hypothermia_on_admission_rate",
#         },
#         "Hypothermia After Admission Rate (%)": {
#             "numerator": "hypothermia_after_admission_count",
#             "denominator": "total_admitted",
#             "value": "hypothermia_after_admission_rate",
#         },
#         "Neonatal Mortality Rate (%)": {
#             "numerator": "death_count",
#             "denominator": "total_admitted",
#             "value": "neonatal_mortality_rate",
#         },
#         "NMR (%)": {
#             "numerator": "death_count",
#             "denominator": "total_admitted",
#             "value": "neonatal_mortality_rate",
#         },
#         "Antibiotics for Clinical Sepsis (%)": {
#             "numerator": "antibiotics_count",
#             "denominator": "probable_sepsis_count",
#             "value": "antibiotics_rate",
#         },
#         "Antibiotics Rate (%)": {
#             "numerator": "antibiotics_count",
#             "denominator": "probable_sepsis_count",
#             "value": "antibiotics_rate",
#         },
#         "Admitted Newborns": {
#             "numerator": "admitted_newborns_count",
#             "denominator": 1,
#             "value": "admitted_newborns_count",
#         },
#         # New Culture KPIs
#         "Culture Done for Babies on Antibiotics (%)": {
#             "numerator": "culture_done_count",
#             "denominator": "antibiotics_count_for_culture",
#             "value": "culture_done_rate",
#         },
#         "Blood Culture Result Recorded (%)": {
#             "numerator": "culture_result_recorded_count",
#             "denominator": "culture_done_total_count",
#             "value": "culture_result_recorded_rate",
#         },
#         "Culture Done for Babies with Clinical Sepsis (%)": {
#             "numerator": "culture_sepsis_count",
#             "denominator": "sepsis_total_count",
#             "value": "culture_sepsis_rate",
#         },
#         "Culture Done Rate (%)": {
#             "numerator": "culture_done_count",
#             "denominator": "antibiotics_count_for_culture",
#             "value": "culture_done_rate",
#         },
#         "Culture Result Recorded Rate (%)": {
#             "numerator": "culture_result_recorded_count",
#             "denominator": "culture_done_total_count",
#             "value": "culture_result_recorded_rate",
#         },
#         "Culture for Sepsis Rate (%)": {
#             "numerator": "culture_sepsis_count",
#             "denominator": "sepsis_total_count",
#             "value": "culture_sepsis_rate",
#         },
#     }

#     if kpi_name in kpi_mapping:
#         mapping = kpi_mapping[kpi_name]
#         numerator = kpi_data.get(mapping["numerator"], 0)
#         denominator = kpi_data.get(mapping["denominator"], 1)
#         value = kpi_data.get(mapping["value"], 0.0)
#         return (numerator, denominator, value)

#     # Fallback mappings for culture KPIs
#     if any(word in kpi_name for word in ["Culture Done", "Culture Rate"]):
#         if "Sepsis" in kpi_name:
#             numerator = kpi_data.get("culture_sepsis_count", 0)
#             denominator = kpi_data.get("sepsis_total_count", 1)
#             value = kpi_data.get("culture_sepsis_rate", 0.0)
#         elif "Result" in kpi_name or "Recorded" in kpi_name:
#             numerator = kpi_data.get("culture_result_recorded_count", 0)
#             denominator = kpi_data.get("culture_done_total_count", 1)
#             value = kpi_data.get("culture_result_recorded_rate", 0.0)
#         else:
#             numerator = kpi_data.get("culture_done_count", 0)
#             denominator = kpi_data.get("antibiotics_count_for_culture", 1)
#             value = kpi_data.get("culture_done_rate", 0.0)
#         return (numerator, denominator, value)

#     # Fallback to original function for other KPIs
#     return get_numerator_denominator_for_newborn_kpi(
#         df, kpi_name, facility_uids, date_range_filters
#     )


# # ---------------- Chart Functions for Culture KPIs ----------------
# def render_culture_done_trend_chart_v2(
#     df,
#     period_col="period_display",
#     value_col="value",
#     title="Culture Done for Babies on Antibiotics",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     numerator_name="Culture Done",
#     denominator_name="Total babies on Antibiotics",
#     facility_uids=None,
# ):
#     """Render trend chart for Culture Done KPI"""
#     # Use existing chart rendering function with custom parameters
#     return render_newborn_trend_chart(
#         df,
#         period_col,
#         value_col,
#         title,
#         bg_color,
#         text_color,
#         facility_names,
#         numerator_name,
#         denominator_name,
#         facility_uids,
#     )


# def render_culture_done_facility_comparison_chart_v2(
#     df,
#     period_col="period_display",
#     value_col="value",
#     title="Culture Done for Babies on Antibiotics - Facility Comparison",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     facility_uids=None,
#     numerator_name="Culture Done",
#     denominator_name="Total babies on Antibiotics",
# ):
#     """Render facility comparison chart for Culture Done KPI"""
#     return render_newborn_facility_comparison_chart(
#         df,
#         period_col,
#         value_col,
#         title,
#         bg_color,
#         text_color,
#         facility_names,
#         facility_uids,
#         numerator_name,
#         denominator_name,
#     )


# def render_culture_done_region_comparison_chart_v2(
#     df,
#     period_col="period_display",
#     value_col="value",
#     title="Culture Done for Babies on Antibiotics - Region Comparison",
#     bg_color="#FFFFFF",
#     text_color=None,
#     region_names=None,
#     region_mapping=None,
#     facilities_by_region=None,
#     numerator_name="Culture Done",
#     denominator_name="Total babies on Antibiotics",
# ):
#     """Render region comparison chart for Culture Done KPI"""
#     return render_newborn_region_comparison_chart(
#         df,
#         period_col,
#         value_col,
#         title,
#         bg_color,
#         text_color,
#         region_names,
#         region_mapping,
#         facilities_by_region,
#         numerator_name,
#         denominator_name,
#     )


# def render_culture_result_recorded_trend_chart_v2(
#     df,
#     period_col="period_display",
#     value_col="value",
#     title="Blood Culture Result Recorded",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     numerator_name="Result Recorded (Negative/Positive)",
#     denominator_name="Total Culture Done (All Results)",
#     facility_uids=None,
# ):
#     """Render trend chart for Culture Result Recorded KPI"""
#     return render_newborn_trend_chart(
#         df,
#         period_col,
#         value_col,
#         title,
#         bg_color,
#         text_color,
#         facility_names,
#         numerator_name,
#         denominator_name,
#         facility_uids,
#     )


# def render_culture_done_sepsis_trend_chart_v2(
#     df,
#     period_col="period_display",
#     value_col="value",
#     title="Culture Done for Babies with Clinical Sepsis",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     numerator_name="Culture Done",
#     denominator_name="Probable Sepsis Cases",
#     facility_uids=None,
# ):
#     """Render trend chart for Culture Done for Sepsis KPI"""
#     return render_newborn_trend_chart(
#         df,
#         period_col,
#         value_col,
#         title,
#         bg_color,
#         text_color,
#         facility_names,
#         numerator_name,
#         denominator_name,
#         facility_uids,
#     )


# # ---------------- Debugging Functions ----------------
# def debug_kpi_computation(df, kpi_name, facility_uids=None):
#     """
#     Debug function to show how KPI is being computed
#     """
#     st.write(f"🔍 **DEBUG KPI Computation: {kpi_name}**")
#     st.write(f"Data shape: {df.shape}")
#     st.write(f"Facility UIDs: {facility_uids}")

#     # Show unique patients count
#     if "tei_id" in df.columns:
#         unique_patients = df["tei_id"].nunique()
#         st.write(f"Unique patients in data: {unique_patients}")

#     # Show column names
#     st.write(f"Columns in data: {list(df.columns)}")

#     # For culture KPIs, show specific column values
#     if "Culture" in kpi_name:
#         if "blood_culture_for_suspected_sepsis_microbiology_and_labs" in df.columns:
#             culture_values = (
#                 df["blood_culture_for_suspected_sepsis_microbiology_and_labs"]
#                 .dropna()
#                 .unique()[:10]
#             )
#             st.write(f"Culture values (first 10): {culture_values}")

#         if "were_antibiotics_administered?_interventions" in df.columns:
#             abx_values = (
#                 df["were_antibiotics_administered?_interventions"]
#                 .dropna()
#                 .unique()[:10]
#             )
#             st.write(f"Antibiotics values (first 10): {abx_values}")


# # ---------------- Comprehensive Summary Functions ----------------
# def render_culture_done_comprehensive_summary_v2(
#     df,
#     title="Culture Done for Babies on Antibiotics - Summary",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     facility_uids=None,
# ):
#     """Render a comprehensive summary for Culture Done KPI"""
#     if text_color is None:
#         text_color = auto_text_color(bg_color)

#     st.subheader(title)

#     if df is None or df.empty:
#         st.info("⚠️ No data available for summary.")
#         return

#     # Compute KPI WITH FACILITY FILTER
#     culture_data = compute_culture_done_kpi_v2(df, facility_uids)

#     # Create summary metrics
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             label="Culture Done Rate",
#             value=f"{culture_data['culture_rate']:.1f}%",
#             help=f"Percentage of babies on antibiotics who had blood culture done",
#         )

#     with col2:
#         st.metric(
#             label="Culture Done Cases",
#             value=f"{culture_data['culture_count']:,}",
#             help=f"Number of UNIQUE babies on antibiotics with blood culture done",
#         )

#     with col3:
#         st.metric(
#             label="Total babies on Antibiotics",
#             value=f"{culture_data['antibiotics_count']:,}",
#             help=f"Total number of UNIQUE babies who received antibiotics",
#         )

#     # Create detailed summary table
#     st.subheader("📊 Detailed Summary")

#     summary_data = {
#         "Metric": [
#             "Culture Done Rate",
#             "Culture Done Cases",
#             "Total babies on Antibiotics",
#         ],
#         "Value": [
#             f"{culture_data['culture_rate']:.1f}%",
#             f"{culture_data['culture_count']:,}",
#             f"{culture_data['antibiotics_count']:,}",
#         ],
#         "Description": [
#             "Percentage of UNIQUE babies on antibiotics who had blood culture done",
#             "Number of UNIQUE babies on antibiotics with blood culture done",
#             "Total number of UNIQUE babies who received antibiotics",
#         ],
#     }

#     summary_df = pd.DataFrame(summary_data)

#     styled_summary = summary_df.style.set_table_attributes(
#         'class="summary-table"'
#     ).hide(axis="index")

#     st.markdown(styled_summary.to_html(), unsafe_allow_html=True)


# def render_culture_result_recorded_comprehensive_summary_v2(
#     df,
#     title="Blood Culture Result Recorded - Summary",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     facility_uids=None,
# ):
#     """Render a comprehensive summary for Culture Result Recorded KPI"""
#     if text_color is None:
#         text_color = auto_text_color(bg_color)

#     st.subheader(title)

#     if df is None or df.empty:
#         st.info("⚠️ No data available for summary.")
#         return

#     # Compute KPI WITH FACILITY FILTER
#     culture_data = compute_culture_result_recorded_kpi_v2(df, facility_uids)

#     # Create summary metrics
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             label="Result Recorded Rate",
#             value=f"{culture_data['culture_result_rate']:.1f}%",
#             help=f"Percentage of blood cultures with recorded results (Negative or Positive)",
#         )

#     with col2:
#         st.metric(
#             label="Result Recorded Cases",
#             value=f"{culture_data['culture_result_count']:,}",
#             help=f"Number of UNIQUE babies with recorded results (Negative or Positive)",
#         )

#     with col3:
#         st.metric(
#             label="Total Culture Done Cases",
#             value=f"{culture_data['culture_done_count']:,}",
#             help=f"Total number of UNIQUE babies with culture done (all results)",
#         )

#     # Create detailed summary table
#     st.subheader("📊 Detailed Summary")

#     summary_data = {
#         "Metric": [
#             "Result Recorded Rate",
#             "Result Recorded Cases",
#             "Total Culture Done Cases",
#         ],
#         "Value": [
#             f"{culture_data['culture_result_rate']:.1f}%",
#             f"{culture_data['culture_result_count']:,}",
#             f"{culture_data['culture_done_count']:,}",
#         ],
#         "Description": [
#             "Percentage of UNIQUE babies with recorded results (Negative or Positive)",
#             "Number of UNIQUE babies with recorded results (Negative or Positive)",
#             "Total number of UNIQUE babies with culture done (Negative, Positive, or Unknown)",
#         ],
#     }

#     summary_df = pd.DataFrame(summary_data)

#     styled_summary = summary_df.style.set_table_attributes(
#         'class="summary-table"'
#     ).hide(axis="index")

#     st.markdown(styled_summary.to_html(), unsafe_allow_html=True)


# def render_culture_done_sepsis_comprehensive_summary_v2(
#     df,
#     title="Culture Done for Babies with Clinical Sepsis - Summary",
#     bg_color="#FFFFFF",
#     text_color=None,
#     facility_names=None,
#     facility_uids=None,
# ):
#     """Render a comprehensive summary for Culture Done for Sepsis KPI"""
#     if text_color is None:
#         text_color = auto_text_color(bg_color)

#     st.subheader(title)

#     if df is None or df.empty:
#         st.info("⚠️ No data available for summary.")
#         return

#     # Compute KPI WITH FACILITY FILTER
#     culture_data = compute_culture_done_sepsis_kpi_v2(df, facility_uids)

#     # Create summary metrics
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             label="Culture Done for Sepsis Rate",
#             value=f"{culture_data['culture_sepsis_rate']:.1f}%",
#             help=f"Percentage of babies with clinical sepsis who had blood culture done",
#         )

#     with col2:
#         st.metric(
#             label="Culture Done Cases",
#             value=f"{culture_data['culture_count']:,}",
#             help=f"Number of UNIQUE babies with clinical sepsis who had blood culture done",
#         )

#     with col3:
#         st.metric(
#             label="Probable Sepsis Cases",
#             value=f"{culture_data['sepsis_count']:,}",
#             help=f"Total number of UNIQUE babies with clinical sepsis",
#         )

#     # Create detailed summary table
#     st.subheader("📊 Detailed Summary")

#     summary_data = {
#         "Metric": [
#             "Culture Done for Sepsis Rate",
#             "Culture Done Cases",
#             "Probable Sepsis Cases",
#         ],
#         "Value": [
#             f"{culture_data['culture_sepsis_rate']:.1f}%",
#             f"{culture_data['culture_count']:,}",
#             f"{culture_data['sepsis_count']:,}",
#         ],
#         "Description": [
#             "Percentage of UNIQUE babies with clinical sepsis who had blood culture done",
#             "Number of UNIQUE babies with clinical sepsis who had blood culture done",
#             "Total number of UNIQUE babies with clinical sepsis",
#         ],
#     }

#     summary_df = pd.DataFrame(summary_data)

#     styled_summary = summary_df.style.set_table_attributes(
#         'class="summary-table"'
#     ).hide(axis="index")

#     st.markdown(styled_summary.to_html(), unsafe_allow_html=True)


# # ---------------- Export all functions ----------------
# __all__ = [
#     # V2 Cache functions
#     "get_cache_key_newborn_v2",
#     "clear_cache_newborn_v2",
#     # Culture Done KPI functions - FIXED COUNTING
#     "compute_culture_done_numerator_v2",
#     "compute_antibiotics_denominator_v2",
#     "compute_culture_done_kpi_v2",
#     # Culture Result Recorded KPI functions - FIXED COUNTING
#     "compute_culture_result_recorded_numerator_v2",
#     "compute_culture_result_recorded_kpi_v2",
#     # Culture Done for Sepsis KPI functions - FIXED COUNTING
#     "compute_culture_done_sepsis_kpi_v2",
#     # Master KPI function V2 - FIXED COUNTING
#     "compute_newborn_kpis_v2",
#     # Date handling functions V2
#     "get_relevant_date_column_for_newborn_kpi_v2",
#     "prepare_data_for_newborn_trend_chart_v2",
#     "get_numerator_denominator_for_newborn_kpi_v2",
#     # Chart rendering functions for Culture KPIs
#     "render_culture_done_trend_chart_v2",
#     "render_culture_done_facility_comparison_chart_v2",
#     "render_culture_done_region_comparison_chart_v2",
#     "render_culture_result_recorded_trend_chart_v2",
#     "render_culture_done_sepsis_trend_chart_v2",
#     # Comprehensive summary functions
#     "render_culture_done_comprehensive_summary_v2",
#     "render_culture_result_recorded_comprehensive_summary_v2",
#     "render_culture_done_sepsis_comprehensive_summary_v2",
#     # Debugging functions
#     "debug_kpi_computation",
#     # Constants
#     "BLOOD_CULTURE_COL",
#     "CULTURE_DONE_VALUES",
#     "CULTURE_RESULT_RECORDED_VALUES",
#     "ANTIBIOTICS_ADMINISTERED_COL",
#     "ANTIBIOTICS_YES_VALUE",
#     "SUBCATEGORIES_INFECTION_COL",
#     "PROBABLE_SEPSIS_CODE",
#     "MICROBIOLOGY_DATE_COL",
# ]
