# kpi_utils_newborn_simplified.py - COMPLETE UPDATED WITH DATASET VARIABLE NAMES

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import hashlib
import logging
from datetime import datetime

# Import shared utilities
from utils.kpi_utils import auto_text_color

# Set up logging
logger = logging.getLogger(__name__)

# ---------------- Caching Setup ----------------
if "kpi_cache_newborn_simplified" not in st.session_state:
    st.session_state.kpi_cache_newborn_simplified = {}


def get_cache_key_simplified(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key based on data and filters"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_cache_simplified():
    """Clear the newborn KPI cache"""
    st.session_state.kpi_cache_newborn_simplified = {}


# ---------------- Helper Function for CSV Download ----------------
def download_csv_button(
    df, filename, button_label="📥 Download CSV", help_text="Download data as CSV"
):
    """Helper function to create download button with proper encoding"""
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label=button_label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        help=help_text,
    )


# ---------------- Newborn KPI Constants - EXACT DATASET COLUMN NAMES ----------------
# Birth weight column (from NICU Admission Careform) - CORRECT
BIRTH_WEIGHT_COL = "birth_weight_n_nicu_admission_careform"

# KMC columns (from Nurse followup Sheet) - UPDATED: Multiple columns exist
KMC_COLUMNS = [
    "kmc_done_nurse_followup_sheet",
    "kmc_done_nurse_followup_sheet_v2",
    "kmc_done_nurse_followup_sheet_v3",
    "kmc_done_nurse_followup_sheet_v4",
    "kmc_done_nurse_followup_sheet_v5",
]
KMC_YES_CODE = "1"

# CPAP columns (from Neonatal referral form) - CORRECT
CPAP_ADMINISTERED_COL = "baby_placed_on_cpap_neonatal_referral_form"
CPAP_YES_CODE = "1"

# RDS Diagnosis columns (from Discharge care form) - UPDATED: lowercase 'n'
RDS_DIAGNOSIS_COL = "sub_categories_of_prematurity_n_discharge_care_form"
RDS_YES_CODE = "1"  # "1" means "Respiratory Distress Syndrome (of Prematurity)"

# ---------------- Birth Weight Categories with Gradient Colors ----------------
BIRTH_WEIGHT_CATEGORIES = {
    "lt_1000": {
        "name": "<1000 g",
        "min": 0,
        "max": 999,
        "sort_order": 1,
        "color": "#e74c3c",  # Red for lowest weight
        "short_name": "<1000g",
    },
    "1000_1499": {
        "name": "1000-1499 g",
        "min": 1000,
        "max": 1499,
        "sort_order": 2,
        "color": "#e67e22",  # Orange
        "short_name": "1000-1499g",
    },
    "1500_1999": {
        "name": "1500-1999 g",
        "min": 1500,
        "max": 1999,
        "sort_order": 3,
        "color": "#f1c40f",  # Yellow
        "short_name": "1500-1999g",
    },
    "2000_2499": {
        "name": "2000-2499 g",
        "min": 2000,
        "max": 2499,
        "sort_order": 4,
        "color": "#2ecc71",  # Light Green
        "short_name": "2000-2499g",
    },
    "2500_4000": {
        "name": "2500-4000 g",
        "min": 2500,
        "max": 4000,
        "sort_order": 5,
        "color": "#27ae60",  # Green
        "short_name": "2500-4000g",
    },
    "gt_4000": {
        "name": "4001+ g",
        "min": 4001,
        "max": 8000,
        "sort_order": 6,
        "color": "#1e8449",  # Dark Green
        "short_name": "4001+g",
    },
}

# Get category names in correct order for sorting
BIRTH_WEIGHT_CATEGORY_NAMES = [
    BIRTH_WEIGHT_CATEGORIES[key]["name"]
    for key in sorted(
        BIRTH_WEIGHT_CATEGORIES.keys(),
        key=lambda x: BIRTH_WEIGHT_CATEGORIES[x]["sort_order"],
    )
]


# ---------------- HELPER FUNCTIONS ----------------
def safe_convert_numeric(value, default=0):
    """Safely convert value to numeric"""
    try:
        if pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.strip()
            if "." in value:
                return float(value.split(".")[0])
            else:
                return float(value)
        return float(value)
    except:
        return default


def filter_by_facility(df, facility_uids):
    """Filter dataframe by facility UIDs"""
    if df is None or df.empty:
        return df

    if facility_uids and "orgUnit" in df.columns:
        return df[df["orgUnit"].isin(facility_uids)].copy()
    return df.copy()


def clean_category_name(category_name):
    """Clean category name for CSV download (replace special characters)"""
    return category_name.replace("–", "-").replace("+", "plus")


def deduplicate_by_tei(df):
    """Deduplicate dataframe by TEI ID"""
    if df is None or df.empty:
        return df

    if "tei_id" in df.columns:
        # Keep first record per TEI ID
        return df.drop_duplicates(subset=["tei_id"], keep="first").copy()
    return df.copy()


# ---------------- ENHANCED KMC FUNCTION TO HANDLE MULTIPLE COLUMNS ----------------
def get_kmc_status_for_tei(row):
    """Check all KMC columns to determine if KMC was done for a TEI"""
    for kmc_col in KMC_COLUMNS:
        if kmc_col in row and pd.notna(row[kmc_col]):
            try:
                kmc_value = str(row[kmc_col]).strip()
                if kmc_value == KMC_YES_CODE:
                    return True
                elif "." in kmc_value:
                    # Handle values like "1.0"
                    if kmc_value.split(".")[0] == KMC_YES_CODE:
                        return True
            except:
                continue
    return False


# ---------------- DATE SORTING HELPER FUNCTIONS ----------------
def parse_period_to_datetime(period_str):
    """Parse period string like 'Apr-18', 'Jan-26' to datetime"""
    try:
        # Handle "Overall" separately
        if period_str == "Overall":
            return datetime(2099, 12, 31)  # Far future date to put at end

        # Parse month-year format (e.g., "Apr-18", "Jan-26")
        month_str, year_str = period_str.split("-")

        # Convert month abbreviation to number
        month_dict = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }

        month = month_dict.get(month_str, 1)

        # Handle 2-digit years (assume 2000s for years < 50, 1900s for years >= 50)
        year = int(year_str)
        if year < 50:
            year += 2000
        elif year < 100:
            year += 1900

        return datetime(year, month, 1)
    except Exception as e:
        logger.warning(f"Could not parse period '{period_str}': {e}")
        return datetime(1900, 1, 1)  # Default for invalid dates


def sort_periods_chronologically(periods):
    """Sort period strings chronologically"""
    # Create list of tuples (datetime, period_string)
    period_tuples = [(parse_period_to_datetime(p), p) for p in periods]

    # Sort by datetime
    sorted_tuples = sorted(period_tuples, key=lambda x: x[0])

    # Extract sorted period strings
    sorted_periods = [p[1] for p in sorted_tuples]

    return sorted_periods


# ---------------- BIRTH WEIGHT KPI Functions ----------------
def compute_birth_weight_by_category(df, facility_uids=None):
    """Compute distribution of birth weights by all categories - FIXED with deduplication"""
    cache_key = get_cache_key_simplified(df, facility_uids, "birth_weight_by_category")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    filtered_df = filter_by_facility(df, facility_uids)

    # CRITICAL: Deduplicate by TEI ID to prevent overcounting
    filtered_df = deduplicate_by_tei(filtered_df)

    if BIRTH_WEIGHT_COL not in filtered_df.columns:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    try:
        # Convert birth weight to numeric
        filtered_df = filtered_df.copy()
        filtered_df["birth_weight_numeric"] = pd.to_numeric(
            filtered_df[BIRTH_WEIGHT_COL], errors="coerce"
        )

        # Filter valid weights
        valid_weights = filtered_df[
            (filtered_df["birth_weight_numeric"].notna())
            & (filtered_df["birth_weight_numeric"] > 0)
            & (filtered_df["birth_weight_numeric"] <= 8000)
        ].copy()

        if valid_weights.empty:
            result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        else:
            result = {}
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                mask = (
                    valid_weights["birth_weight_numeric"] >= category_info["min"]
                ) & (valid_weights["birth_weight_numeric"] <= category_info["max"])
                # Count unique TEI IDs in this category
                if "tei_id" in valid_weights.columns:
                    category_newborns = (
                        valid_weights.loc[mask, "tei_id"].dropna().unique()
                    )
                    result[category_key] = len(category_newborns)
                else:
                    result[category_key] = int(mask.sum())

    except Exception as e:
        logger.error(f"Error computing birth weight by category: {e}")
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


def compute_total_with_birth_weight(df, facility_uids=None):
    """Compute total newborns with valid birth weight - FIXED with deduplication"""
    cache_key = get_cache_key_simplified(df, facility_uids, "total_with_birth_weight")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = filter_by_facility(df, facility_uids)

        # CRITICAL: Deduplicate by TEI ID
        filtered_df = deduplicate_by_tei(filtered_df)

        if BIRTH_WEIGHT_COL not in filtered_df.columns:
            result = 0
        else:
            filtered_df = filtered_df.copy()
            filtered_df["birth_weight_numeric"] = pd.to_numeric(
                filtered_df[BIRTH_WEIGHT_COL], errors="coerce"
            )

            valid_weights = filtered_df[
                (filtered_df["birth_weight_numeric"].notna())
                & (filtered_df["birth_weight_numeric"] > 0)
                & (filtered_df["birth_weight_numeric"] <= 8000)
            ]

            if "tei_id" in valid_weights.columns:
                result = valid_weights["tei_id"].dropna().nunique()
            else:
                result = len(valid_weights)

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


def compute_birth_weight_kpi(df, facility_uids=None):
    """Compute birth weight KPI for the given dataframe - FIXED"""
    cache_key = get_cache_key_simplified(df, facility_uids, "birth_weight_kpi")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {
            "bw_category_counts": {
                category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
            "total_with_birth_weight": 0,
        }
    else:
        category_counts = compute_birth_weight_by_category(df, facility_uids)
        total_with_weight = compute_total_with_birth_weight(df, facility_uids)

        # Calculate sum for verification
        category_sum = sum(category_counts.values())

        # Log for debugging
        logger.info(
            f"Birth weight - Total with weight: {total_with_weight}, Category sum: {category_sum}"
        )

        # Ensure consistency
        if category_sum != total_with_weight:
            logger.warning(
                f"Birth weight data inconsistent! Category sum: {category_sum}, Total: {total_with_weight}"
            )
            # Use the larger value
            total_with_weight = max(total_with_weight, category_sum)

        result = {
            "bw_category_counts": category_counts,
            "total_with_birth_weight": int(total_with_weight),
            "category_sum": int(category_sum),
        }

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


# ---------------- KMC COVERAGE KPI Functions ----------------
def compute_kmc_by_weight_category(df, facility_uids=None):
    """Compute KMC administered by birth weight category - ENHANCED with multiple KMC columns"""
    cache_key = get_cache_key_simplified(df, facility_uids, "kmc_by_weight_category")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    filtered_df = filter_by_facility(df, facility_uids)

    # CRITICAL: Deduplicate by TEI ID
    filtered_df = deduplicate_by_tei(filtered_df)

    if BIRTH_WEIGHT_COL not in filtered_df.columns:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    try:
        filtered_df = filtered_df.copy()

        # Convert birth weight
        filtered_df["birth_weight_numeric"] = pd.to_numeric(
            filtered_df[BIRTH_WEIGHT_COL], errors="coerce"
        )

        # Check for KMC in any column
        filtered_df["kmc_done"] = filtered_df.apply(get_kmc_status_for_tei, axis=1)

        # Filter KMC cases
        kmc_df = filtered_df[filtered_df["kmc_done"]].copy()

        if kmc_df.empty:
            result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        else:
            result = {}
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                mask = (
                    (kmc_df["birth_weight_numeric"].notna())
                    & (kmc_df["birth_weight_numeric"] >= category_info["min"])
                    & (kmc_df["birth_weight_numeric"] <= category_info["max"])
                )

                if "tei_id" in kmc_df.columns:
                    category_kmc = kmc_df.loc[mask, "tei_id"].dropna().unique()
                    result[category_key] = len(category_kmc)
                else:
                    result[category_key] = int(mask.sum())

    except Exception as e:
        logger.error(f"Error computing KMC by weight category: {e}")
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


def compute_kmc_coverage_kpi(df, facility_uids=None):
    """Compute KMC coverage rate by birth weight category - ENHANCED with multiple KMC columns"""
    cache_key = get_cache_key_simplified(df, facility_uids, "kmc_coverage_kpi")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {
            "kmc_counts_by_category": {
                category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
            "kmc_total_by_category": {
                category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
            "kmc_rates_by_category": {
                category: 0.0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
        }
    else:
        kmc_counts = compute_kmc_by_weight_category(df, facility_uids)
        total_by_category = compute_birth_weight_by_category(df, facility_uids)

        rates_by_category = {}
        for category_key in BIRTH_WEIGHT_CATEGORIES.keys():
            kmc_count = kmc_counts.get(category_key, 0)
            total_count = total_by_category.get(category_key, 0)

            # Calculate rate with safe division
            if total_count > 0:
                rate = (kmc_count / total_count) * 100
            else:
                rate = 0.0

            rates_by_category[category_key] = float(rate)

        result = {
            "kmc_counts_by_category": kmc_counts,
            "kmc_total_by_category": total_by_category,
            "kmc_rates_by_category": rates_by_category,
        }

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


# ---------------- CPAP COVERAGE KPI Functions - ENHANCED RDS LOGIC ----------------
def get_rds_newborns(df, facility_uids=None):
    """Identify newborns with RDS diagnosis - ENHANCED with exact column name"""
    if df is None or df.empty:
        return set()

    filtered_df = filter_by_facility(df, facility_uids)

    # Deduplicate before identifying RDS
    filtered_df = deduplicate_by_tei(filtered_df)

    # Check if required column exists
    if RDS_DIAGNOSIS_COL not in filtered_df.columns:
        logger.warning(f"Missing RDS column: {RDS_DIAGNOSIS_COL}")
        return set()

    try:
        # Convert RDS column to string and check for RDS code (value "1" means RDS)
        # Check if sub_categories_of_prematurity_n_discharge_care_form contains "1"
        rds_mask = (
            filtered_df[RDS_DIAGNOSIS_COL]
            .astype(str)
            .str.contains(RDS_YES_CODE, na=False)
        )

        if "tei_id" in filtered_df.columns:
            rds_newborns = filtered_df.loc[rds_mask, "tei_id"].dropna().unique()
            return set(rds_newborns)
        else:
            # Count rows if no TEI ID
            rds_indices = filtered_df.index[rds_mask]
            return set(rds_indices)

    except Exception as e:
        logger.error(f"Error identifying RDS newborns: {e}")
        return set()


def compute_cpap_general_kpi(df, facility_uids=None):
    """Compute general CPAP coverage rate - FIXED with deduplication"""
    cache_key = get_cache_key_simplified(df, facility_uids, "cpap_general_kpi")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {
            "cpap_general_rate": 0.0,
            "cpap_general_count": 0,
            "total_admitted": 0,
        }
    else:
        filtered_df = filter_by_facility(df, facility_uids)

        # CRITICAL: Deduplicate by TEI ID
        filtered_df = deduplicate_by_tei(filtered_df)

        if CPAP_ADMINISTERED_COL not in filtered_df.columns:
            result = {
                "cpap_general_rate": 0.0,
                "cpap_general_count": 0,
                "total_admitted": 0,
            }
        else:
            # Count unique TEIs
            if "tei_id" in filtered_df.columns:
                total_admitted = filtered_df["tei_id"].dropna().nunique()

                # Filter for CPAP cases
                filtered_df["cpap_numeric"] = pd.to_numeric(
                    filtered_df[CPAP_ADMINISTERED_COL]
                    .astype(str)
                    .str.split(".")
                    .str[0],
                    errors="coerce",
                )
                cpap_mask = filtered_df["cpap_numeric"] == float(CPAP_YES_CODE)

                cpap_teis = filtered_df.loc[cpap_mask, "tei_id"].dropna().unique()
                cpap_general_count = len(cpap_teis)
            else:
                total_admitted = len(filtered_df)
                cpap_general_count = (
                    filtered_df[CPAP_ADMINISTERED_COL]
                    .astype(str)
                    .str.contains(CPAP_YES_CODE)
                    .sum()
                )

            cpap_general_rate = (
                (cpap_general_count / total_admitted * 100)
                if total_admitted > 0
                else 0.0
            )

            # Log for debugging
            logger.info(
                f"General CPAP - Total admitted: {total_admitted}, CPAP count: {cpap_general_count}, Rate: {cpap_general_rate:.1f}%"
            )

            result = {
                "cpap_general_rate": float(cpap_general_rate),
                "cpap_general_count": int(cpap_general_count),
                "total_admitted": int(total_admitted),
            }

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


def compute_cpap_for_rds_kpi(df, facility_uids=None):
    """Compute CPAP coverage for RDS newborns - FIXED with deduplication and ENHANCED RDS LOGIC"""
    cache_key = get_cache_key_simplified(df, facility_uids, "cpap_for_rds_kpi")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {"cpap_rate": 0.0, "cpap_count": 0, "total_rds": 0}
    else:
        filtered_df = filter_by_facility(df, facility_uids)

        # Deduplicate before identifying RDS
        filtered_df = deduplicate_by_tei(filtered_df)

        rds_newborns = get_rds_newborns(filtered_df, facility_uids)

        if not rds_newborns:
            result = {"cpap_rate": 0.0, "cpap_count": 0, "total_rds": 0}
        else:
            # Filter for RDS newborns
            if "tei_id" in filtered_df.columns:
                rds_df = filtered_df[filtered_df["tei_id"].isin(rds_newborns)].copy()
                total_rds = len(rds_newborns)
            else:
                rds_df = filtered_df[filtered_df.index.isin(rds_newborns)].copy()
                total_rds = len(rds_newborns)

            cpap_count = 0

            if CPAP_ADMINISTERED_COL in rds_df.columns and not rds_df.empty:
                # Convert CPAP column
                rds_df["cpap_numeric"] = pd.to_numeric(
                    rds_df[CPAP_ADMINISTERED_COL].astype(str).str.split(".").str[0],
                    errors="coerce",
                )
                cpap_mask = rds_df["cpap_numeric"] == float(CPAP_YES_CODE)

                if "tei_id" in rds_df.columns:
                    cpap_newborns = rds_df.loc[cpap_mask, "tei_id"].dropna().unique()
                    cpap_count = len(cpap_newborns)
                else:
                    cpap_count = int(cpap_mask.sum())

            cpap_rate = (cpap_count / total_rds * 100) if total_rds > 0 else 0.0

            # Log for debugging
            logger.info(
                f"CPAP for RDS - Total RDS: {total_rds}, CPAP count: {cpap_count}, Rate: {cpap_rate:.1f}%"
            )

            result = {
                "cpap_rate": float(cpap_rate),
                "cpap_count": int(cpap_count),
                "total_rds": int(total_rds),
            }

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


# ---------------- CPAP BY BIRTH WEIGHT CATEGORY FUNCTION ----------------
def compute_cpap_by_weight_category(df, facility_uids=None):
    """Compute CPAP administered by birth weight category - FIXED with deduplication"""
    cache_key = get_cache_key_simplified(df, facility_uids, "cpap_by_weight_category")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    filtered_df = filter_by_facility(df, facility_uids)

    # CRITICAL: Deduplicate by TEI ID
    filtered_df = deduplicate_by_tei(filtered_df)

    if (
        BIRTH_WEIGHT_COL not in filtered_df.columns
        or CPAP_ADMINISTERED_COL not in filtered_df.columns
    ):
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    try:
        filtered_df = filtered_df.copy()

        # Convert birth weight
        filtered_df["birth_weight_numeric"] = pd.to_numeric(
            filtered_df[BIRTH_WEIGHT_COL], errors="coerce"
        )

        # Convert CPAP column
        filtered_df["cpap_numeric"] = pd.to_numeric(
            filtered_df[CPAP_ADMINISTERED_COL].astype(str).str.split(".").str[0],
            errors="coerce",
        )

        # Filter CPAP cases
        cpap_mask = filtered_df["cpap_numeric"] == float(CPAP_YES_CODE)
        cpap_df = filtered_df[cpap_mask].copy()

        if cpap_df.empty:
            result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        else:
            result = {}
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                mask = (
                    (cpap_df["birth_weight_numeric"].notna())
                    & (cpap_df["birth_weight_numeric"] >= category_info["min"])
                    & (cpap_df["birth_weight_numeric"] <= category_info["max"])
                )

                if "tei_id" in cpap_df.columns:
                    category_cpap = cpap_df.loc[mask, "tei_id"].dropna().unique()
                    result[category_key] = len(category_cpap)
                else:
                    result[category_key] = int(mask.sum())

    except Exception as e:
        logger.error(f"Error computing CPAP by weight category: {e}")
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


def compute_cpap_coverage_by_weight_kpi(df, facility_uids=None):
    """Compute CPAP coverage rate by birth weight category - FIXED"""
    cache_key = get_cache_key_simplified(
        df, facility_uids, "cpap_coverage_by_weight_kpi"
    )

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {
            "cpap_counts_by_category": {
                category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
            "cpap_total_by_category": {
                category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
            "cpap_rates_by_category": {
                category: 0.0 for category in BIRTH_WEIGHT_CATEGORIES.keys()
            },
        }
    else:
        cpap_counts = compute_cpap_by_weight_category(df, facility_uids)
        total_by_category = compute_birth_weight_by_category(df, facility_uids)

        rates_by_category = {}
        for category_key in BIRTH_WEIGHT_CATEGORIES.keys():
            cpap_count = cpap_counts.get(category_key, 0)
            total_count = total_by_category.get(category_key, 0)

            # Calculate rate with safe division
            if total_count > 0:
                rate = (cpap_count / total_count) * 100
            else:
                rate = 0.0

            rates_by_category[category_key] = float(rate)

        result = {
            "cpap_counts_by_category": cpap_counts,
            "cpap_total_by_category": total_by_category,
            "cpap_rates_by_category": rates_by_category,
        }

    st.session_state.kpi_cache_newborn_simplified[cache_key] = result
    return result


# ---------------- CHART FUNCTIONS (UPDATED WITH SINGLE TABLE) ----------------
def render_birth_weight_trend_chart(
    df,
    period_col="period_display",
    title="Birth Weight Distribution Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render trend chart for Birth Weight KPI - WITH SINGLE TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Get unique periods
    periods = df[period_col].unique()

    # FIX: Sort periods chronologically using helper function
    periods = sort_periods_chronologically(periods)

    # Compute BW distribution for each period
    trend_data = []

    for period in periods:
        period_df = df[df[period_col] == period]
        bw_data = compute_birth_weight_kpi(period_df, facility_uids)

        period_row = {
            period_col: period,
            "total_with_birth_weight": bw_data["total_with_birth_weight"],
        }

        # Add each category count
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
            period_row[f"{category_key}_count"] = bw_data["bw_category_counts"].get(
                category_key, 0
            )
            period_row[f"{category_key}_name"] = category_info["name"]

        trend_data.append(period_row)

    if not trend_data:
        st.warning("⚠️ No birth weight data available for the selected period.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create stacked bar chart for birth weight distribution
    fig = go.Figure()

    # Add bars for each BW category IN REVERSE ORDER (for proper stacking)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(),
        key=lambda x: x[1]["sort_order"],
        reverse=True,  # Reverse for stacking order
    ):
        count_col = f"{category_key}_count"

        if count_col in trend_df.columns:
            fig.add_trace(
                go.Bar(
                    x=trend_df[period_col],
                    y=trend_df[count_col],
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.0f}} newborns<extra></extra>",
                )
            )

    # Calculate Y-axis range PROPERLY
    # We need to calculate the TOTAL height of each stacked bar (sum of all categories for each period)
    period_totals = []
    for period in periods:
        period_data = trend_df[trend_df[period_col] == period].iloc[0]
        total = 0
        for category_key in BIRTH_WEIGHT_CATEGORIES.keys():
            count_col = f"{category_key}_count"
            if count_col in period_data:
                total += period_data[count_col]
        period_totals.append(total)

    if period_totals:
        max_total = max(period_totals)
        # Add GENEROUS padding to ensure no cutting off
        y_max = max_total * 1.3  # 30% padding instead of 10%

        # Always ensure minimum height for visibility
        y_max = max(y_max, 20)  # Minimum of 20 to see small values
    else:
        y_max = 20  # Default minimum height

    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Period",
        yaxis_title="Number of Newborns",
        barmode="stack",
        showlegend=True,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,  # Ensure chronological order
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],  # FIXED: Now using calculated y_max
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            traceorder="reversed",  # Reverse legend to match stacking order
            title="Birth Weight Categories",
        ),
        # Add margin to ensure no cutting
        margin=dict(l=50, r=50, t=50, b=100),
    )

    st.plotly_chart(fig, use_container_width=True)

    # SINGLE COMPARISON TABLE
    st.subheader("📊 Birth Weight Distribution Table")

    # Create a table with all categories for each period
    table_data = []

    for period in periods:
        period_data = trend_df[trend_df[period_col] == period].iloc[0]
        row = {"Period": period}

        # Add counts for each category IN ORDER
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            count_col = f"{category_key}_count"
            if count_col in period_data:
                row[category_info["short_name"]] = int(period_data[count_col])

        # Add total
        row["Total"] = int(period_data["total_with_birth_weight"])
        table_data.append(row)

    # Add overall row
    overall_row = {"Period": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        count_col = f"{category_key}_count"
        overall_row[category_info["short_name"]] = int(trend_df[count_col].sum())

    overall_row["Total"] = int(trend_df["total_with_birth_weight"].sum())
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table without expander
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    # Create a simplified version for download
    download_df = trend_df.copy()

    # Select only the period and count columns IN ORDER
    download_cols = [period_col, "total_with_birth_weight"]
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        download_cols.append(f"{category_key}_count")

    download_df = download_df[download_cols]

    # Calculate totals for "Overall" row
    overall_row = {period_col: "Overall"}
    overall_row["total_with_birth_weight"] = download_df[
        "total_with_birth_weight"
    ].sum()

    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        col_name = f"{category_key}_count"
        if col_name in download_df.columns:
            overall_row[col_name] = download_df[col_name].sum()

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Rename columns for better readability with clean names IN ORDER
    column_names = {
        period_col: "Period",
        "total_with_birth_weight": "Total Newborns with Birth Weight",
    }
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        clean_name = clean_category_name(category_info["name"])
        column_names[f"{category_key}_count"] = f"{clean_name} Newborns"

    download_df = download_df.rename(columns=column_names)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"birth_weight_distribution_{timestamp}.csv",
        "📥 Download CSV",
        "Download the birth weight distribution data as CSV",
    )


def render_birth_weight_facility_comparison(
    df,
    period_col="period_display",
    title="Birth Weight Distribution - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for birth weight - WITH SINGLE TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # Get periods (for information only, not for selection)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED data for each facility (sum across all periods)
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            # Aggregate data across ALL periods
            bw_data = compute_birth_weight_kpi(facility_df, [facility_uid])

            row_data = {
                "Facility": facility_name,
                "Total with Birth Weight": bw_data["total_with_birth_weight"],
            }

            # Add each category count IN ORDER
            for category_key, category_info in sorted(
                BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
            ):
                row_data[category_info["short_name"]] = bw_data[
                    "bw_category_counts"
                ].get(category_key, 0)

            facility_data.append(row_data)

    if not facility_data:
        st.warning("⚠️ No data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create stacked bar chart WITH SORTED CATEGORIES
    fig = go.Figure()

    # Add bars for each BW category IN REVERSE ORDER (for proper stacking)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(),
        key=lambda x: x[1]["sort_order"],
        reverse=True,  # Reverse for stacking order
    ):
        short_name = category_info["short_name"]
        if short_name in facility_df.columns:
            fig.add_trace(
                go.Bar(
                    x=facility_df["Facility"],
                    y=facility_df[short_name],
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.0f}<extra></extra>",
                )
            )

    # Calculate Y-axis range
    all_values = []
    for category_info in BIRTH_WEIGHT_CATEGORIES.values():
        short_name = category_info["short_name"]
        if short_name in facility_df.columns:
            all_values.extend(facility_df[short_name].tolist())

    if all_values:
        max_value = max(all_values)
        y_max = max_value * 1.1  # Add 10% padding
    else:
        y_max = None

    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Facility",
        yaxis_title="Number of Newborns",
        barmode="stack",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max] if y_max else None,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            traceorder="reversed",  # Reverse legend to match stacking order
            title="Birth Weight Categories",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # SINGLE COMPARISON TABLE
    st.subheader("📊 Facility Comparison Table")

    # Display the table with all categories
    display_df = facility_df.copy()

    # Add overall row
    overall_row = {"Facility": "Overall"}
    for col in display_df.columns:
        if col != "Facility":
            overall_row[col] = display_df[col].sum()

    overall_df = pd.DataFrame([overall_row])
    display_df = pd.concat([display_df, overall_df], ignore_index=True)

    # Format the table with all categories
    st.dataframe(display_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    download_df = facility_df.copy()

    # Clean column names for download IN ORDER
    column_names = {
        "Facility": "Facility",
        "Total with Birth Weight": "Total Newborns with Birth Weight",
    }
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        clean_name = clean_category_name(category_info["name"])
        column_names[category_info["short_name"]] = f"{clean_name} Newborns"

    download_df = download_df.rename(columns=column_names)

    # Calculate totals for "Overall" row
    overall_row = {"Facility": "Overall"}
    for col in download_df.columns:
        if col != "Facility":
            overall_row[col] = download_df[col].sum()

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"birth_weight_facility_comparison_{timestamp}.csv",
        "📥 Download CSV",
        "Download birth weight facility comparison data as CSV",
    )


def render_birth_weight_region_comparison(
    df,
    period_col="period_display",
    title="Birth Weight Distribution - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
):
    """Render region comparison for birth weight - WITH SINGLE TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED data for each region (sum across all periods)
    region_data = []
    for region_name in region_names:
        facility_uids = []
        for facility_name, facility_uid in facilities_by_region.get(region_name, []):
            facility_uids.append(facility_uid)

        if facility_uids:
            # Filter for this region's facilities
            region_df = df[df["orgUnit"].isin(facility_uids)]

            # CRITICAL: Deduplicate by TEI ID for the entire region
            region_df = deduplicate_by_tei(region_df)

            if not region_df.empty:
                # Aggregate data across ALL periods
                bw_data = compute_birth_weight_kpi(region_df, facility_uids)

                row_data = {
                    "Region": region_name,
                    "Total with Birth Weight": bw_data["total_with_birth_weight"],
                }

                # Add each category count IN ORDER
                for category_key, category_info in sorted(
                    BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
                ):
                    row_data[category_info["short_name"]] = bw_data[
                        "bw_category_counts"
                    ].get(category_key, 0)

                region_data.append(row_data)

    if not region_data:
        st.warning("⚠️ No data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create stacked bar chart
    fig = go.Figure()

    # Add bars for each BW category IN CORRECT ORDER (from smallest to largest)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        if short_name in region_df.columns:
            fig.add_trace(
                go.Bar(
                    x=region_df["Region"],
                    y=region_df[short_name],
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.0f}<extra></extra>",
                )
            )

    # Calculate Y-axis range
    all_values = []
    for category_info in BIRTH_WEIGHT_CATEGORIES.values():
        short_name = category_info["short_name"]
        if short_name in region_df.columns:
            all_values.extend(region_df[short_name].tolist())

    if all_values:
        max_value = max(all_values)
        y_max = max_value * 1.1  # Add 10% padding
    else:
        y_max = None

    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Region",
        yaxis_title="Number of Newborns",
        barmode="stack",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max] if y_max else None,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # SINGLE COMPARISON TABLE
    st.subheader("📊 Region Comparison Table")

    # Display the table with all categories
    display_df = region_df.copy()

    # Add overall row
    overall_row = {"Region": "Overall"}
    for col in display_df.columns:
        if col != "Region":
            overall_row[col] = display_df[col].sum()

    overall_df = pd.DataFrame([overall_row])
    display_df = pd.concat([display_df, overall_df], ignore_index=True)

    # Format the table with all categories
    st.dataframe(display_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    download_df = region_df.copy()

    # Clean column names for download IN ORDER
    column_names = {
        "Region": "Region",
        "Total with Birth Weight": "Total Newborns with Birth Weight",
    }
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        clean_name = clean_category_name(category_info["name"])
        column_names[category_info["short_name"]] = f"{clean_name} Newborns"

    download_df = download_df.rename(columns=column_names)

    # Calculate totals for "Overall" row
    overall_row = {"Region": "Overall"}
    for col in download_df.columns:
        if col != "Region":
            overall_row[col] = download_df[col].sum()

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"birth_weight_region_comparison_{timestamp}.csv",
        "📥 Download CSV",
        "Download birth weight region comparison data as CSV",
    )


# ---------------- SEPARATE CPAP CHART FUNCTIONS (UPDATED WITH SINGLE TABLE) ----------------
def render_cpap_general_trend_chart(
    df,
    period_col="period_display",
    title="General CPAP Coverage Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render general CPAP coverage trend chart - WITH SINGLE TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Get unique periods
    periods = df[period_col].unique()

    # FIX: Sort periods chronologically using helper function
    periods = sort_periods_chronologically(periods)

    # Compute CPAP data for each period
    trend_data = []

    for period in periods:
        period_df = df[df[period_col] == period]
        cpap_general_data = compute_cpap_general_kpi(period_df, facility_uids)

        period_row = {
            period_col: period,
            "cpap_general_rate": cpap_general_data.get("cpap_general_rate", 0.0),
            "cpap_general_count": cpap_general_data.get("cpap_general_count", 0),
            "cpap_general_total": cpap_general_data.get("total_admitted", 0),
        }

        trend_data.append(period_row)

    if not trend_data:
        st.warning("⚠️ No CPAP data available for the selected period.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create bar chart for general CPAP
    fig = go.Figure()

    # Prepare hover data as numpy array
    hover_data = np.column_stack(
        (trend_df["cpap_general_count"], trend_df["cpap_general_total"])
    )

    fig.add_trace(
        go.Bar(
            x=trend_df[period_col],
            y=trend_df["cpap_general_rate"],
            name="General CPAP",
            marker_color="#3498db",  # Blue for General CPAP
            hovertemplate=(
                "<b>%{x}</b><br>"
                + "General CPAP: %{y:.1f}%<br>"
                + "CPAP Cases: %{customdata[0]:.0f}<br>"  # FIXED: Use .0f formatting
                + "Total Admitted: %{customdata[1]:.0f}<extra></extra>"  # FIXED: Use .0f formatting
            ),
            customdata=hover_data,
        )
    )

    # Calculate Y-axis range for percentage charts
    all_rates = trend_df["cpap_general_rate"].tolist()
    if all_rates:
        max_rate = max(all_rates)
        y_max = min(100, max_rate * 1.1)  # Cap at 100% for percentages
    else:
        y_max = 100

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage (%)",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,  # Ensure chronological order
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE
    st.subheader("📊 General CPAP Coverage Table")

    # Create a detailed table
    table_data = []

    for _, row in trend_df.iterrows():
        table_row = {
            "Period": row[period_col],
            "Rate (%)": f"{row['cpap_general_rate']:.1f}%",
            "CPAP Cases": int(row["cpap_general_count"]),
            "Total Admitted": int(row["cpap_general_total"]),
        }
        table_data.append(table_row)

    # Add overall row
    total_cases = trend_df["cpap_general_count"].sum()
    total_admitted = trend_df["cpap_general_total"].sum()
    overall_rate = (total_cases / total_admitted * 100) if total_admitted > 0 else 0

    overall_row = {
        "Period": "Overall",
        "Rate (%)": f"{overall_rate:.1f}%",
        "CPAP Cases": int(total_cases),
        "Total Admitted": int(total_admitted),
    }
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # Add description
    st.info(
        "**General CPAP Coverage**: Shows the percentage of all admitted newborns who received CPAP (Continuous Positive Airway Pressure) therapy."
    )

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    # Create a simplified version for download
    download_df = trend_df.copy()

    # Select and rename columns
    download_df = download_df[
        [
            period_col,
            "cpap_general_rate",
            "cpap_general_count",
            "cpap_general_total",
        ]
    ]

    # Calculate totals for "Overall" row
    overall_row = {period_col: "Overall"}
    overall_row["cpap_general_count"] = download_df["cpap_general_count"].sum()
    overall_row["cpap_general_total"] = download_df["cpap_general_total"].sum()
    overall_row["cpap_general_rate"] = (
        overall_row["cpap_general_count"] / overall_row["cpap_general_total"] * 100
        if overall_row["cpap_general_total"] > 0
        else 0
    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    column_names = {
        period_col: "Period",
        "cpap_general_rate": "General CPAP Rate (%)",
        "cpap_general_count": "General CPAP Cases",
        "cpap_general_total": "Total Admitted Newborns",
    }

    download_df = download_df.rename(columns=column_names)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"general_cpap_coverage_{timestamp}.csv",
        "📥 Download CSV",
        "Download general CPAP coverage data as CSV",
    )


def render_cpap_rds_trend_chart(
    df,
    period_col="period_display",
    title="CPAP for RDS Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render CPAP for RDS trend chart - WITH SINGLE TABLE"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Get unique periods
    periods = df[period_col].unique()

    # FIX: Sort periods chronologically using helper function
    periods = sort_periods_chronologically(periods)

    # Compute CPAP data for each period
    trend_data = []

    for period in periods:
        period_df = df[df[period_col] == period]
        cpap_rds_data = compute_cpap_for_rds_kpi(period_df, facility_uids)

        period_row = {
            period_col: period,
            "cpap_rds_rate": cpap_rds_data.get("cpap_rate", 0.0),
            "cpap_rds_count": cpap_rds_data.get("cpap_count", 0),
            "cpap_rds_total": cpap_rds_data.get("total_rds", 0),
        }

        trend_data.append(period_row)

    if not trend_data:
        st.warning("⚠️ No CPAP for RDS data available for the selected period.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create bar chart for CPAP for RDS
    fig = go.Figure()

    # Prepare hover data as numpy array
    hover_data = np.column_stack(
        (trend_df["cpap_rds_count"], trend_df["cpap_rds_total"])
    )

    fig.add_trace(
        go.Bar(
            x=trend_df[period_col],
            y=trend_df["cpap_rds_rate"],
            name="CPAP for RDS",
            marker_color="#3498db",  # BLUE for CPAP for RDS (same as general CPAP)
            hovertemplate=(
                "<b>%{x}</b><br>"
                + "CPAP for RDS: %{y:.1f}%<br>"
                + "CPAP Cases: %{customdata[0]:.0f}<br>"  # FIXED: Use .0f formatting
                + "Total RDS: %{customdata[1]:.0f}<extra></extra>"  # FIXED: Use .0f formatting
            ),
            customdata=hover_data,
        )
    )

    # Calculate Y-axis range for percentage charts
    all_rates = trend_df["cpap_rds_rate"].tolist()
    if all_rates:
        max_rate = max(all_rates)
        y_max = min(100, max_rate * 1.1)  # Cap at 100% for percentages
    else:
        y_max = 100

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage (%)",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,  # Ensure chronological order
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE
    st.subheader("📊 CPAP for RDS Table")

    # Create a detailed table
    table_data = []

    for _, row in trend_df.iterrows():
        table_row = {
            "Period": row[period_col],
            "Rate (%)": f"{row['cpap_rds_rate']:.1f}%",
            "CPAP Cases": int(row["cpap_rds_count"]),
            "Total RDS": int(row["cpap_rds_total"]),
        }
        table_data.append(table_row)

    # Add overall row
    total_cases = trend_df["cpap_rds_count"].sum()
    total_rds = trend_df["cpap_rds_total"].sum()
    overall_rate = (total_cases / total_rds * 100) if total_rds > 0 else 0

    overall_row = {
        "Period": "Overall",
        "Rate (%)": f"{overall_rate:.1f}%",
        "CPAP Cases": int(total_cases),
        "Total RDS": int(total_rds),
    }
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # Add description
    st.info(
        "**CPAP for Respiratory Distress Syndrome (RDS)**: Shows the percentage of newborns diagnosed with RDS who received CPAP therapy."
    )

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    # Create a simplified version for download
    download_df = trend_df.copy()

    # Select and rename columns
    download_df = download_df[
        [
            period_col,
            "cpap_rds_rate",
            "cpap_rds_count",
            "cpap_rds_total",
        ]
    ]

    # Calculate totals for "Overall" row
    overall_row = {period_col: "Overall"}
    overall_row["cpap_rds_count"] = download_df["cpap_rds_count"].sum()
    overall_row["cpap_rds_total"] = download_df["cpap_rds_total"].sum()
    overall_row["cpap_rds_rate"] = (
        overall_row["cpap_rds_count"] / overall_row["cpap_rds_total"] * 100
        if overall_row["cpap_rds_total"] > 0
        else 0
    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    column_names = {
        period_col: "Period",
        "cpap_rds_rate": "CPAP for RDS Rate (%)",
        "cpap_rds_count": "CPAP for RDS Cases",
        "cpap_rds_total": "Total RDS Newborns",
    }

    download_df = download_df.rename(columns=column_names)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"cpap_for_rds_{timestamp}.csv",
        "📥 Download CSV",
        "Download CPAP for RDS data as CSV",
    )


# ---------------- NEW CPAP COMPARISON FUNCTIONS FOR FACILITY ----------------
def render_cpap_general_facility_comparison(
    df,
    period_col="period_display",
    title="General CPAP Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for general CPAP coverage - FIXED VERSION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED CPAP GENERAL data for each facility (sum across all periods)
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            # Aggregate CPAP GENERAL data across ALL periods
            cpap_data = compute_cpap_general_kpi(facility_df, [facility_uid])

            row_data = {
                "Facility": facility_name,
                "CPAP Rate (%)": cpap_data.get("cpap_general_rate", 0.0),
                "CPAP Cases": cpap_data.get("cpap_general_count", 0),
                "Total Admitted": cpap_data.get("total_admitted", 0),
            }

            facility_data.append(row_data)

    if not facility_data:
        st.warning("⚠️ No CPAP data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create color palette for facilities
    color_palette = px.colors.qualitative.Set3  # You can change this palette

    # Create bar chart for general CPAP rates
    fig = go.Figure()

    for i, (_, row) in enumerate(facility_df.iterrows()):
        # Assign a unique color to each facility
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]

        fig.add_trace(
            go.Bar(
                x=[row["Facility"]],
                y=[row["CPAP Rate (%)"]],
                name=row["Facility"],
                marker_color=color,
                hovertemplate=(
                    f"<b>{row['Facility']}</b><br>"
                    + "General CPAP: %{y:.1f}%<br>"
                    + f"CPAP Cases: {row['CPAP Cases']}<br>"  # FIXED: Direct value access
                    + f"Total Admitted: {row['Total Admitted']}<extra></extra>"  # FIXED: Direct value access
                ),
                showlegend=False,  # Don't show legend for single bars
            )
        )

    # Calculate Y-axis range for percentage charts
    all_values = facility_df["CPAP Rate (%)"].tolist()
    if all_values:
        max_value = max(all_values)
        y_max = min(100, max_value * 1.1)  # Cap at 100% for percentages
    else:
        y_max = 100

    fig.update_layout(
        title=f"{title}",
        height=400,
        xaxis_title="Facility",
        yaxis_title="General CPAP Coverage (%)",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE
    st.subheader("📊 General CPAP Facility Comparison Table")

    # Create a detailed table
    table_data = []

    for _, row in facility_df.iterrows():
        table_row = {
            "Facility": row["Facility"],
            "Rate (%)": f"{row['CPAP Rate (%)']:.1f}%",
            "CPAP Cases": int(row["CPAP Cases"]),
            "Total Admitted": int(row["Total Admitted"]),
        }
        table_data.append(table_row)

    # Add overall row
    total_cases = facility_df["CPAP Cases"].sum()
    total_admitted = facility_df["Total Admitted"].sum()
    overall_rate = (total_cases / total_admitted * 100) if total_admitted > 0 else 0

    overall_row = {
        "Facility": "Overall",
        "Rate (%)": f"{overall_rate:.1f}%",
        "CPAP Cases": int(total_cases),
        "Total Admitted": int(total_admitted),
    }
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    download_df = facility_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Facility": "Overall"}
    overall_row["CPAP Cases"] = download_df["CPAP Cases"].sum()
    overall_row["Total Admitted"] = download_df["Total Admitted"].sum()
    overall_row["CPAP Rate (%)"] = (
        overall_row["CPAP Cases"] / overall_row["Total Admitted"] * 100
        if overall_row["Total Admitted"] > 0
        else 0
    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"general_cpap_facility_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download general CPAP facility comparison data as CSV",
    )


def render_cpap_rds_facility_comparison(
    df,
    period_col="period_display",
    title="CPAP for RDS - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for CPAP for RDS - FIXED VERSION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED CPAP for RDS data for each facility (sum across all periods)
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            # Aggregate CPAP for RDS data across ALL periods
            cpap_data = compute_cpap_for_rds_kpi(facility_df, [facility_uid])

            row_data = {
                "Facility": facility_name,
                "CPAP Rate (%)": cpap_data.get("cpap_rate", 0.0),
                "CPAP Cases": cpap_data.get("cpap_count", 0),
                "Total RDS": cpap_data.get("total_rds", 0),
            }

            facility_data.append(row_data)

    if not facility_data:
        st.warning("⚠️ No CPAP for RDS data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create color palette for facilities (SAME as General CPAP for consistency)
    color_palette = px.colors.qualitative.Set3  # Same palette as General CPAP

    # Create bar chart for CPAP for RDS rates
    fig = go.Figure()

    for i, (_, row) in enumerate(facility_df.iterrows()):
        # Assign the SAME color to each facility as in General CPAP chart
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]

        fig.add_trace(
            go.Bar(
                x=[row["Facility"]],
                y=[row["CPAP Rate (%)"]],
                name=row["Facility"],
                marker_color=color,
                hovertemplate=(
                    f"<b>{row['Facility']}</b><br>"
                    + "CPAP for RDS: %{y:.1f}%<br>"
                    + f"CPAP Cases: {row['CPAP Cases']}<br>"  # FIXED: Direct value access
                    + f"Total RDS: {row['Total RDS']}<extra></extra>"  # FIXED: Direct value access
                ),
                showlegend=False,  # Don't show legend for single bars
            )
        )

    # Calculate Y-axis range for percentage charts
    all_values = facility_df["CPAP Rate (%)"].tolist()
    if all_values:
        max_value = max(all_values)
        y_max = min(100, max_value * 1.1)  # Cap at 100% for percentages
    else:
        y_max = 100

    fig.update_layout(
        title=f"{title}",
        height=400,
        xaxis_title="Facility",
        yaxis_title="CPAP for RDS Coverage (%)",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE
    st.subheader("📊 CPAP for RDS Facility Comparison Table")

    # Create a detailed table
    table_data = []

    for _, row in facility_df.iterrows():
        table_row = {
            "Facility": row["Facility"],
            "Rate (%)": f"{row['CPAP Rate (%)']:.1f}%",
            "CPAP Cases": int(row["CPAP Cases"]),
            "Total RDS": int(row["Total RDS"]),
        }
        table_data.append(table_row)

    # Add overall row
    total_cases = facility_df["CPAP Cases"].sum()
    total_rds = facility_df["Total RDS"].sum()
    overall_rate = (total_cases / total_rds * 100) if total_rds > 0 else 0

    overall_row = {
        "Facility": "Overall",
        "Rate (%)": f"{overall_rate:.1f}%",
        "CPAP Cases": int(total_cases),
        "Total RDS": int(total_rds),
    }
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    download_df = facility_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Facility": "Overall"}
    overall_row["CPAP Cases"] = download_df["CPAP Cases"].sum()
    overall_row["Total RDS"] = download_df["Total RDS"].sum()
    overall_row["CPAP Rate (%)"] = (
        overall_row["CPAP Cases"] / overall_row["Total RDS"] * 100
        if overall_row["Total RDS"] > 0
        else 0
    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"cpap_for_rds_facility_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download CPAP for RDS facility comparison data as CSV",
    )


# ---------------- NEW CPAP COMPARISON FUNCTIONS FOR REGION ----------------
def render_cpap_general_region_comparison(
    df,
    period_col="period_display",
    title="General CPAP Coverage - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
):
    """Render region comparison for general CPAP coverage - FIXED VERSION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED CPAP GENERAL data for each region (sum across all periods)
    region_data = []
    for region_name in region_names:
        facility_uids = []
        for facility_name, facility_uid in facilities_by_region.get(region_name, []):
            facility_uids.append(facility_uid)

        if facility_uids:
            # Filter for this region's facilities
            region_df = df[df["orgUnit"].isin(facility_uids)]

            # CRITICAL: Deduplicate by TEI ID for the entire region
            region_df = deduplicate_by_tei(region_df)

            if not region_df.empty:
                # Aggregate CPAP GENERAL data across ALL periods
                cpap_data = compute_cpap_general_kpi(region_df, facility_uids)

                row_data = {
                    "Region": region_name,
                    "CPAP Rate (%)": cpap_data.get("cpap_general_rate", 0.0),
                    "CPAP Cases": cpap_data.get("cpap_general_count", 0),
                    "Total Admitted": cpap_data.get("total_admitted", 0),
                }

                region_data.append(row_data)

    if not region_data:
        st.warning("⚠️ No CPAP data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create color palette for regions
    color_palette = px.colors.qualitative.Pastel  # You can change this palette

    # Create bar chart for general CPAP rates
    fig = go.Figure()

    for i, (_, row) in enumerate(region_df.iterrows()):
        # Assign a unique color to each region
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]

        fig.add_trace(
            go.Bar(
                x=[row["Region"]],
                y=[row["CPAP Rate (%)"]],
                name=row["Region"],
                marker_color=color,
                hovertemplate=(
                    f"<b>{row['Region']}</b><br>"
                    + "General CPAP: %{y:.1f}%<br>"
                    + f"CPAP Cases: {row['CPAP Cases']}<br>"  # FIXED: Direct value access
                    + f"Total Admitted: {row['Total Admitted']}<extra></extra>"  # FIXED: Direct value access
                ),
                showlegend=False,  # Don't show legend for single bars
            )
        )

    # Calculate Y-axis range for percentage charts
    all_values = region_df["CPAP Rate (%)"].tolist()
    if all_values:
        max_value = max(all_values)
        y_max = min(100, max_value * 1.1)  # Cap at 100% for percentages
    else:
        y_max = 100

    fig.update_layout(
        title=f"{title}",
        height=400,
        xaxis_title="Region",
        yaxis_title="General CPAP Coverage (%)",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE
    st.subheader("📊 General CPAP Region Comparison Table")

    # Create a detailed table
    table_data = []

    for _, row in region_df.iterrows():
        table_row = {
            "Region": row["Region"],
            "Rate (%)": f"{row['CPAP Rate (%)']:.1f}%",
            "CPAP Cases": int(row["CPAP Cases"]),
            "Total Admitted": int(row["Total Admitted"]),
        }
        table_data.append(table_row)

    # Add overall row
    total_cases = region_df["CPAP Cases"].sum()
    total_admitted = region_df["Total Admitted"].sum()
    overall_rate = (total_cases / total_admitted * 100) if total_admitted > 0 else 0

    overall_row = {
        "Region": "Overall",
        "Rate (%)": f"{overall_rate:.1f}%",
        "CPAP Cases": int(total_cases),
        "Total Admitted": int(total_admitted),
    }
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    download_df = region_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Region": "Overall"}
    overall_row["CPAP Cases"] = download_df["CPAP Cases"].sum()
    overall_row["Total Admitted"] = download_df["Total Admitted"].sum()
    overall_row["CPAP Rate (%)"] = (
        overall_row["CPAP Cases"] / overall_row["Total Admitted"] * 100
        if overall_row["Total Admitted"] > 0
        else 0
    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"general_cpap_region_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download general CPAP region comparison data as CSV",
    )


def render_cpap_rds_region_comparison(
    df,
    period_col="period_display",
    title="CPAP for RDS - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
):
    """Render region comparison for CPAP for RDS - FIXED VERSION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED CPAP for RDS data for each region (sum across all periods)
    region_data = []
    for region_name in region_names:
        facility_uids = []
        for facility_name, facility_uid in facilities_by_region.get(region_name, []):
            facility_uids.append(facility_uid)

        if facility_uids:
            # Filter for this region's facilities
            region_df = df[df["orgUnit"].isin(facility_uids)]

            # CRITICAL: Deduplicate by TEI ID for the entire region
            region_df = deduplicate_by_tei(region_df)

            if not region_df.empty:
                # Aggregate CPAP for RDS data across ALL periods
                cpap_data = compute_cpap_for_rds_kpi(region_df, facility_uids)

                row_data = {
                    "Region": region_name,
                    "CPAP Rate (%)": cpap_data.get("cpap_rate", 0.0),
                    "CPAP Cases": cpap_data.get("cpap_count", 0),
                    "Total RDS": cpap_data.get("total_rds", 0),
                }

                region_data.append(row_data)

    if not region_data:
        st.warning("⚠️ No CPAP for RDS data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create color palette for regions (SAME as General CPAP for consistency)
    color_palette = px.colors.qualitative.Pastel  # Same palette as General CPAP

    # Create bar chart for CPAP for RDS rates
    fig = go.Figure()

    for i, (_, row) in enumerate(region_df.iterrows()):
        # Assign the SAME color to each region as in General CPAP chart
        color_idx = i % len(color_palette)
        color = color_palette[color_idx]

        fig.add_trace(
            go.Bar(
                x=[row["Region"]],
                y=[row["CPAP Rate (%)"]],
                name=row["Region"],
                marker_color=color,
                hovertemplate=(
                    f"<b>{row['Region']}</b><br>"
                    + "CPAP for RDS: %{y:.1f}%<br>"
                    + f"CPAP Cases: {row['CPAP Cases']}<br>"  # FIXED: Direct value access
                    + f"Total RDS: {row['Total RDS']}<extra></extra>"  # FIXED: Direct value access
                ),
                showlegend=False,  # Don't show legend for single bars
            )
        )

    # Calculate Y-axis range for percentage charts
    all_values = region_df["CPAP Rate (%)"].tolist()
    if all_values:
        max_value = max(all_values)
        y_max = min(100, max_value * 1.1)  # Cap at 100% for percentages
    else:
        y_max = 100

    fig.update_layout(
        title=f"{title}",
        height=400,
        xaxis_title="Region",
        yaxis_title="CPAP for RDS Coverage (%)",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE
    st.subheader("📊 CPAP for RDS Region Comparison Table")

    # Create a detailed table
    table_data = []

    for _, row in region_df.iterrows():
        table_row = {
            "Region": row["Region"],
            "Rate (%)": f"{row['CPAP Rate (%)']:.1f}%",
            "CPAP Cases": int(row["CPAP Cases"]),
            "Total RDS": int(row["Total RDS"]),
        }
        table_data.append(table_row)

    # Add overall row
    total_cases = region_df["CPAP Cases"].sum()
    total_rds = region_df["Total RDS"].sum()
    overall_rate = (total_cases / total_rds * 100) if total_rds > 0 else 0

    overall_row = {
        "Region": "Overall",
        "Rate (%)": f"{overall_rate:.1f}%",
        "CPAP Cases": int(total_cases),
        "Total RDS": int(total_rds),
    }
    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION
    st.subheader("📥 Download Data")

    download_df = region_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Region": "Overall"}
    overall_row["CPAP Cases"] = download_df["CPAP Cases"].sum()
    overall_row["Total RDS"] = download_df["Total RDS"].sum()
    overall_row["CPAP Rate (%)"] = (
        overall_row["CPAP Cases"] / overall_row["Total RDS"] * 100
        if overall_row["Total RDS"] > 0
        else 0
    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"cpap_for_rds_region_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download CPAP for RDS region comparison data as CSV",
    )


# ---------------- CHART FUNCTIONS (UPDATED WITH GROUP BARS) ----------------
def render_kmc_coverage_trend_chart(
    df,
    period_col="period_display",
    title="KMC Coverage by Birth Weight Category",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render KMC coverage trend - WITH GROUP BAR CHART"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Get unique periods
    periods = df[period_col].unique()

    # FIX: Sort periods chronologically using helper function
    periods = sort_periods_chronologically(periods)

    # Compute KMC coverage for each period
    trend_data = []

    for period in periods:
        period_df = df[df[period_col] == period]
        kmc_data = compute_kmc_coverage_kpi(period_df, facility_uids)

        period_row = {period_col: period}

        # Add KMC rate for each category IN ORDER
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            period_row[f"{category_key}_rate"] = kmc_data["kmc_rates_by_category"].get(
                category_key, 0
            )
            period_row[f"{category_key}_count"] = kmc_data[
                "kmc_counts_by_category"
            ].get(category_key, 0)
            period_row[f"{category_key}_total"] = kmc_data["kmc_total_by_category"].get(
                category_key, 0
            )

        trend_data.append(period_row)

    if not trend_data:
        st.warning("⚠️ No KMC data available for the selected period.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create GROUP bar chart for KMC rates by weight category
    fig = go.Figure()

    # Add bars for each BW category as separate traces (GROUP BARS)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        rate_col = f"{category_key}_rate"
        count_col = f"{category_key}_count"
        total_col = f"{category_key}_total"

        if rate_col in trend_df.columns:
            fig.add_trace(
                go.Bar(
                    x=trend_df[period_col],
                    y=trend_df[rate_col],
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.1f}}%<br>KMC Cases: %{{customdata[0]}}<br>Total Newborns: %{{customdata[1]}}<extra></extra>",
                    customdata=np.column_stack(
                        (trend_df[count_col], trend_df[total_col])
                    ),
                )
            )

    # Calculate Y-axis range for percentage charts
    all_rates = []
    for category_key in BIRTH_WEIGHT_CATEGORIES.keys():
        rate_col = f"{category_key}_rate"
        if rate_col in trend_df.columns:
            all_rates.extend(trend_df[rate_col].tolist())

    if all_rates:
        max_rate = max(all_rates)
        y_max = min(100, max_rate * 1.2)  # Add more padding for group bars
    else:
        y_max = 100

    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Period",
        yaxis_title="KMC Coverage Rate (%)",
        barmode="group",  # CHANGED FROM "stack" TO "group"
        showlegend=True,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,  # Ensure chronological order
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
        bargap=0.15,  # Gap between bars of different categories
        bargroupgap=0.1,  # Gap between groups of bars (periods)
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE (unchanged)
    st.subheader("📊 KMC Coverage Table")

    # Create a detailed table with rates, counts, and totals for each category
    table_data = []

    for period in periods:
        period_data = trend_df[trend_df[period_col] == period].iloc[0]
        row = {"Period": period}

        # Add data for each category
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            rate_col = f"{category_key}_rate"
            count_col = f"{category_key}_count"
            total_col = f"{category_key}_total"

            if rate_col in period_data:
                rate = period_data[rate_col]
                count = period_data[count_col]
                total = period_data[total_col]

                # Format as "Rate% (Count/Total)"
                row[category_info["short_name"]] = (
                    f"{rate:.1f}% ({int(count)}/{int(total)})"
                )

        table_data.append(row)

    # Add overall row
    overall_row = {"Period": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        count_col = f"{category_key}_count"
        total_col = f"{category_key}_total"

        total_count = trend_df[count_col].sum()
        total_denom = trend_df[total_col].sum()
        overall_rate = (total_count / total_denom * 100) if total_denom > 0 else 0

        overall_row[category_info["short_name"]] = (
            f"{overall_rate:.1f}% ({int(total_count)}/{int(total_denom)})"
        )

    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.write("Format: Rate% (KMC Cases / Total Newborns)")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # Add description
    st.info(
        "**KMC Coverage by Birth Weight Category**: Shows the percentage of newborns receiving Kangaroo Mother Care (KMC), grouped by birth weight categories. Color gradient from red (lowest weight) to green (highest weight) indicates birth weight categories."
    )

    # SINGLE DOWNLOAD SECTION (unchanged)
    st.subheader("📥 Download Data")

    # Create a simplified version for download
    download_df = trend_df.copy()

    # Select only the period and category columns IN ORDER
    download_cols = [period_col]
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        download_cols.extend(
            [
                f"{category_key}_rate",
                f"{category_key}_count",
                f"{category_key}_total",
            ]
        )

    download_df = download_df[download_cols]

    # Calculate totals for "Overall" row
    overall_row = {period_col: "Overall"}
    for col in download_df.columns:
        if col != period_col:
            if "_rate" in col:
                # For rates, calculate weighted average
                rate_col = col
                count_col = col.replace("_rate", "_count")
                total_col = col.replace("_rate", "_total")

                if (
                    count_col in download_df.columns
                    and total_col in download_df.columns
                ):
                    total_cases = download_df[count_col].sum()
                    total_newborns = download_df[total_col].sum()
                    overall_row[rate_col] = (
                        (total_cases / total_newborns * 100)
                        if total_newborns > 0
                        else 0
                    )
            elif "_count" in col or "_total" in col:
                # For counts and totals, sum them
                overall_row[col] = download_df[col].sum()

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Rename columns for better readability with clean names IN ORDER
    column_names = {period_col: "Period"}

    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        clean_name = clean_category_name(category_info["name"])
        column_names[f"{category_key}_rate"] = f"{clean_name} KMC Rate (%)"
        column_names[f"{category_key}_count"] = f"{clean_name} KMC Cases"
        column_names[f"{category_key}_total"] = f"{clean_name} Total Newborns"

    download_df = download_df.rename(columns=column_names)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"kmc_coverage_{timestamp}.csv",
        "📥 Download CSV",
        f"Download KMC coverage data as CSV",
    )


def render_cpap_by_weight_trend_chart(
    df,
    period_col="period_display",
    title="CPAP Coverage by Birth Weight Category",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render CPAP coverage trend by birth weight category - WITH GROUP BAR CHART"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Get unique periods
    periods = df[period_col].unique()

    # FIX: Sort periods chronologically using helper function
    periods = sort_periods_chronologically(periods)

    # Compute CPAP coverage for each period
    trend_data = []

    for period in periods:
        period_df = df[df[period_col] == period]
        cpap_data = compute_cpap_coverage_by_weight_kpi(period_df, facility_uids)

        period_row = {period_col: period}

        # Add CPAP rate for each category IN ORDER
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            period_row[f"{category_key}_rate"] = cpap_data[
                "cpap_rates_by_category"
            ].get(category_key, 0)
            period_row[f"{category_key}_count"] = cpap_data[
                "cpap_counts_by_category"
            ].get(category_key, 0)
            period_row[f"{category_key}_total"] = cpap_data[
                "cpap_total_by_category"
            ].get(category_key, 0)

        trend_data.append(period_row)

    if not trend_data:
        st.warning("⚠️ No CPAP data available for the selected period.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create GROUP bar chart for CPAP rates by weight category
    fig = go.Figure()

    # Add bars for each BW category as separate traces (GROUP BARS)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        rate_col = f"{category_key}_rate"
        count_col = f"{category_key}_count"
        total_col = f"{category_key}_total"

        if rate_col in trend_df.columns:
            fig.add_trace(
                go.Bar(
                    x=trend_df[period_col],
                    y=trend_df[rate_col],
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.1f}}%<br>CPAP Cases: %{{customdata[0]}}<br>Total Newborns: %{{customdata[1]}}<extra></extra>",
                    customdata=np.column_stack(
                        (trend_df[count_col], trend_df[total_col])
                    ),
                )
            )

    # Calculate Y-axis range for percentage charts
    all_rates = []
    for category_key in BIRTH_WEIGHT_CATEGORIES.keys():
        rate_col = f"{category_key}_rate"
        if rate_col in trend_df.columns:
            all_rates.extend(trend_df[rate_col].tolist())

    if all_rates:
        max_rate = max(all_rates)
        y_max = min(100, max_rate * 1.2)  # Add more padding for group bars
    else:
        y_max = 100

    fig.update_layout(
        title=title,
        height=500,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage Rate (%)",
        barmode="group",  # CHANGED FROM "stack" TO "group"
        showlegend=True,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=periods,  # Ensure chronological order
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
        bargap=0.15,  # Gap between bars of different categories
        bargroupgap=0.1,  # Gap between groups of bars (periods)
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE (unchanged)
    st.subheader("📊 CPAP Coverage Table")

    # Create a detailed table with rates, counts, and totals for each category
    table_data = []

    for period in periods:
        period_data = trend_df[trend_df[period_col] == period].iloc[0]
        row = {"Period": period}

        # Add data for each category
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            rate_col = f"{category_key}_rate"
            count_col = f"{category_key}_count"
            total_col = f"{category_key}_total"

            if rate_col in period_data:
                rate = period_data[rate_col]
                count = period_data[count_col]
                total = period_data[total_col]

                # Format as "Rate% (Count/Total)"
                row[category_info["short_name"]] = (
                    f"{rate:.1f}% ({int(count)}/{int(total)})"
                )

        table_data.append(row)

    # Add overall row
    overall_row = {"Period": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        count_col = f"{category_key}_count"
        total_col = f"{category_key}_total"

        total_count = trend_df[count_col].sum()
        total_denom = trend_df[total_col].sum()
        overall_rate = (total_count / total_denom * 100) if total_denom > 0 else 0

        overall_row[category_info["short_name"]] = (
            f"{overall_rate:.1f}% ({int(total_count)}/{int(total_denom)})"
        )

    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.write("Format: Rate% (CPAP Cases / Total Newborns)")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # Add description
    st.info(
        "**CPAP Coverage by Birth Weight Category**: Shows the percentage of newborns receiving CPAP therapy, grouped by birth weight categories. Color gradient from red (lowest weight) to green (highest weight) indicates birth weight categories."
    )

    # SINGLE DOWNLOAD SECTION (unchanged)
    st.subheader("📥 Download Data")

    # Create a simplified version for download
    download_df = trend_df.copy()

    # Select only the period and category columns IN ORDER
    download_cols = [period_col]
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        download_cols.extend(
            [
                f"{category_key}_rate",
                f"{category_key}_count",
                f"{category_key}_total",
            ]
        )

    download_df = download_df[download_cols]

    # Calculate totals for "Overall" row
    overall_row = {period_col: "Overall"}
    for col in download_df.columns:
        if col != period_col:
            if "_rate" in col:
                # For rates, calculate weighted average
                rate_col = col
                count_col = col.replace("_rate", "_count")
                total_col = col.replace("_rate", "_total")

                if (
                    count_col in download_df.columns
                    and total_col in download_df.columns
                ):
                    total_cases = download_df[count_col].sum()
                    total_newborns = download_df[total_col].sum()
                    overall_row[rate_col] = (
                        (total_cases / total_newborns * 100)
                        if total_newborns > 0
                        else 0
                    )
            elif "_count" in col or "_total" in col:
                # For counts and totals, sum them
                overall_row[col] = download_df[col].sum()

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Rename columns for better readability with clean names IN ORDER
    column_names = {period_col: "Period"}

    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        clean_name = clean_category_name(category_info["name"])
        column_names[f"{category_key}_rate"] = f"{clean_name} CPAP Rate (%)"
        column_names[f"{category_key}_count"] = f"{clean_name} CPAP Cases"
        column_names[f"{category_key}_total"] = f"{clean_name} Total Newborns"

    download_df = download_df.rename(columns=column_names)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"cpap_coverage_{timestamp}.csv",
        "📥 Download CSV",
        f"Download CPAP coverage data as CSV",
    )


def render_kmc_facility_comparison(
    df,
    period_col="period_display",
    title="KMC Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for KMC coverage - WITH GROUP BAR CHART"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED KMC data for each facility (sum across all periods)
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            # Aggregate KMC data across ALL periods
            kmc_data = compute_kmc_coverage_kpi(facility_df, [facility_uid])

            row_data = {"Facility": facility_name}

            # Store both rates and counts for each category
            for category_key, category_info in sorted(
                BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
            ):
                rate = kmc_data["kmc_rates_by_category"].get(category_key, 0)
                count = kmc_data["kmc_counts_by_category"].get(category_key, 0)
                total = kmc_data["kmc_total_by_category"].get(category_key, 0)

                # Store rate for chart
                row_data[category_info["short_name"]] = rate
                # Store count and total for table
                row_data[f"{category_info['short_name']}_count"] = count
                row_data[f"{category_info['short_name']}_total"] = total

            facility_data.append(row_data)

    if not facility_data:
        st.warning("⚠️ No KMC data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create GROUP bar chart for KMC RATES
    fig = go.Figure()

    # Add bars for each BW category as separate traces (GROUP BARS)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        if short_name in facility_df.columns:
            fig.add_trace(
                go.Bar(
                    x=facility_df["Facility"],
                    y=facility_df[short_name],  # This is the RATE value
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>",
                )
            )

    # Calculate Y-axis range for percentage charts - ALWAYS 0-100%
    y_max = 100  # Always show up to 100% for rates

    fig.update_layout(
        title=f"{title}",
        height=500,
        xaxis_title="Facility",
        yaxis_title="KMC Coverage Rate (%)",
        barmode="group",  # CHANGED FROM "stack" TO "group"
        bargap=0.15,  # Gap between bars of different categories
        bargroupgap=0.1,  # Gap between groups of bars (facilities)
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],  # FIXED: Always 0-100%
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE (unchanged)
    st.subheader("📊 KMC Facility Comparison Table")

    # Create a detailed table with rates, counts, and totals for each category
    table_data = []

    for _, row in facility_df.iterrows():
        table_row = {"Facility": row["Facility"]}

        # Add data for each category
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            short_name = category_info["short_name"]
            count_col = f"{short_name}_count"
            total_col = f"{short_name}_total"

            if short_name in row:
                rate = row[short_name]
                count = row[count_col] if count_col in row else 0
                total = row[total_col] if total_col in row else 0

                # Format as "Rate% (Count/Total)"
                table_row[short_name] = f"{rate:.1f}% ({int(count)}/{int(total)})"

        table_data.append(table_row)

    # Add overall row
    overall_row = {"Facility": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        count_col = f"{short_name}_count"
        total_col = f"{short_name}_total"

        total_count = (
            facility_df[count_col].sum() if count_col in facility_df.columns else 0
        )
        total_denom = (
            facility_df[total_col].sum() if total_col in facility_df.columns else 0
        )
        overall_rate = (total_count / total_denom * 100) if total_denom > 0 else 0

        overall_row[short_name] = (
            f"{overall_rate:.1f}% ({int(total_count)}/{int(total_denom)})"
        )

    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.write("Format: Rate% (KMC Cases / Total Newborns)")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION (unchanged)
    st.subheader("📥 Download Data")

    download_df = facility_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Facility": "Overall"}
    for col in download_df.columns:
        if col != "Facility":
            if "_count" in col or "_total" in col:
                # For counts and totals, sum them
                overall_row[col] = (
                    download_df[col].sum() if col in download_df.columns else 0
                )
            else:
                # For rates, calculate weighted average (not simple sum)
                # Find corresponding count and total columns
                base_name = col
                count_col = f"{col}_count"
                total_col = f"{col}_total"

                if (
                    count_col in download_df.columns
                    and total_col in download_df.columns
                ):
                    total_count = download_df[count_col].sum()
                    total_denom = download_df[total_col].sum()
                    overall_row[col] = (
                        (total_count / total_denom * 100) if total_denom > 0 else 0
                    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"kmc_rates_facility_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download KMC rates facility comparison data as CSV",
    )


def render_cpap_facility_comparison(
    df,
    period_col="period_display",
    title="CPAP Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for CPAP coverage (by weight categories) - WITH GROUP BAR CHART"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED CPAP data for each facility (sum across all periods)
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            # Aggregate CPAP data across ALL periods
            cpap_data = compute_cpap_coverage_by_weight_kpi(facility_df, [facility_uid])

            row_data = {"Facility": facility_name}

            # Store both rates and counts for each category
            for category_key, category_info in sorted(
                BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
            ):
                rate = cpap_data["cpap_rates_by_category"].get(category_key, 0)
                count = cpap_data["cpap_counts_by_category"].get(category_key, 0)
                total = cpap_data["cpap_total_by_category"].get(category_key, 0)

                # Store rate for chart
                row_data[category_info["short_name"]] = rate
                # Store count and total for table
                row_data[f"{category_info['short_name']}_count"] = count
                row_data[f"{category_info['short_name']}_total"] = total

            facility_data.append(row_data)

    if not facility_data:
        st.warning("⚠️ No CPAP data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create GROUP bar chart for CPAP RATES
    fig = go.Figure()

    # Add bars for each BW category as separate traces (GROUP BARS)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        if short_name in facility_df.columns:
            fig.add_trace(
                go.Bar(
                    x=facility_df["Facility"],
                    y=facility_df[short_name],  # This is the RATE value
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>",
                )
            )

    # Calculate Y-axis range for percentage charts - ALWAYS 0-100%
    y_max = 100  # Always show up to 100% for rates

    fig.update_layout(
        title=f"{title}",
        height=500,
        xaxis_title="Facility",
        yaxis_title="CPAP Coverage Rate (%)",
        barmode="group",  # CHANGED FROM "stack" TO "group"
        bargap=0.15,  # Gap between bars of different categories
        bargroupgap=0.1,  # Gap between groups of bars (facilities)
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],  # FIXED: Always 0-100%
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE (unchanged)
    st.subheader("📊 CPAP Facility Comparison Table")

    # Create a detailed table with rates, counts, and totals for each category
    table_data = []

    for _, row in facility_df.iterrows():
        table_row = {"Facility": row["Facility"]}

        # Add data for each category
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            short_name = category_info["short_name"]
            count_col = f"{short_name}_count"
            total_col = f"{short_name}_total"

            if short_name in row:
                rate = row[short_name]  # This is the rate
                count = row[count_col] if count_col in row else 0
                total = row[total_col] if total_col in row else 0

                # Format as "Rate% (Count/Total)"
                table_row[short_name] = f"{rate:.1f}% ({int(count)}/{int(total)})"

        table_data.append(table_row)

    # Add overall row
    overall_row = {"Facility": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        count_col = f"{short_name}_count"
        total_col = f"{short_name}_total"

        total_count = (
            facility_df[count_col].sum() if count_col in facility_df.columns else 0
        )
        total_denom = (
            facility_df[total_col].sum() if total_col in facility_df.columns else 0
        )
        overall_rate = (total_count / total_denom * 100) if total_denom > 0 else 0

        overall_row[short_name] = (
            f"{overall_rate:.1f}% ({int(total_count)}/{int(total_denom)})"
        )

    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.write("Format: Rate% (CPAP Cases / Total Newborns)")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION (unchanged)
    st.subheader("📥 Download Data")

    download_df = facility_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Facility": "Overall"}
    for col in download_df.columns:
        if col != "Facility":
            if "_count" in col or "_total" in col:
                # For counts and totals, sum them
                overall_row[col] = (
                    download_df[col].sum() if col in download_df.columns else 0
                )
            else:
                # For rates, calculate weighted average (not simple sum)
                # Find corresponding count and total columns
                base_name = col
                count_col = f"{col}_count"
                total_col = f"{col}_total"

                if (
                    count_col in download_df.columns
                    and total_col in download_df.columns
                ):
                    total_count = download_df[count_col].sum()
                    total_denom = download_df[total_col].sum()
                    overall_row[col] = (
                        (total_count / total_denom * 100) if total_denom > 0 else 0
                    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"cpap_rates_facility_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download CPAP rates facility comparison data as CSV",
    )


def render_kmc_region_comparison(
    df,
    period_col="period_display",
    title="KMC Coverage - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
):
    """Render region comparison for KMC coverage - WITH GROUP BAR CHART"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED KMC data for each region (sum across all periods)
    region_data = []
    for region_name in region_names:
        facility_uids = []
        for facility_name, facility_uid in facilities_by_region.get(region_name, []):
            facility_uids.append(facility_uid)

        if facility_uids:
            # Filter for this region's facilities
            region_df = df[df["orgUnit"].isin(facility_uids)]

            # CRITICAL: Deduplicate by TEI ID for the entire region
            region_df = deduplicate_by_tei(region_df)

            if not region_df.empty:
                # Aggregate KMC data across ALL periods
                kmc_data = compute_kmc_coverage_kpi(region_df, facility_uids)

                row_data = {"Region": region_name}

                # Store both rates and counts for each category
                for category_key, category_info in sorted(
                    BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
                ):
                    rate = kmc_data["kmc_rates_by_category"].get(category_key, 0)
                    count = kmc_data["kmc_counts_by_category"].get(category_key, 0)
                    total = kmc_data["kmc_total_by_category"].get(category_key, 0)

                    # Store rate for chart
                    row_data[category_info["short_name"]] = rate
                    # Store count and total for table
                    row_data[f"{category_info['short_name']}_count"] = count
                    row_data[f"{category_info['short_name']}_total"] = total

                region_data.append(row_data)

    if not region_data:
        st.warning("⚠️ No KMC data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create GROUP bar chart for KMC RATES
    fig = go.Figure()

    # Add bars for each BW category as separate traces (GROUP BARS)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        if short_name in region_df.columns:
            fig.add_trace(
                go.Bar(
                    x=region_df["Region"],
                    y=region_df[short_name],  # This is the RATE value
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>",
                )
            )

    # Calculate Y-axis range for percentage charts - ALWAYS 0-100%
    y_max = 100  # Always show up to 100% for rates

    fig.update_layout(
        title=f"{title}",
        height=500,
        xaxis_title="Region",
        yaxis_title="KMC Coverage Rate (%)",
        barmode="group",  # CHANGED FROM "stack" TO "group"
        bargap=0.15,  # Gap between bars of different categories
        bargroupgap=0.1,  # Gap between groups of bars (regions)
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],  # FIXED: Always 0-100%
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE (unchanged)
    st.subheader("📊 KMC Region Comparison Table")

    # Create a detailed table with rates, counts, and totals for each category
    table_data = []

    for _, row in region_df.iterrows():
        table_row = {"Region": row["Region"]}

        # Add data for each category
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            short_name = category_info["short_name"]
            count_col = f"{short_name}_count"
            total_col = f"{short_name}_total"

            if short_name in row:
                rate = row[short_name]  # This is the rate
                count = row[count_col] if count_col in row else 0
                total = row[total_col] if total_col in row else 0

                # Format as "Rate% (Count/Total)"
                table_row[short_name] = f"{rate:.1f}% ({int(count)}/{int(total)})"

        table_data.append(table_row)

    # Add overall row
    overall_row = {"Region": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        count_col = f"{short_name}_count"
        total_col = f"{short_name}_total"

        total_count = (
            region_df[count_col].sum() if count_col in region_df.columns else 0
        )
        total_denom = (
            region_df[total_col].sum() if total_col in region_df.columns else 0
        )
        overall_rate = (total_count / total_denom * 100) if total_denom > 0 else 0

        overall_row[short_name] = (
            f"{overall_rate:.1f}% ({int(total_count)}/{int(total_denom)})"
        )

    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.write("Format: Rate% (KMC Cases / Total Newborns)")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION (unchanged)
    st.subheader("📥 Download Data")

    download_df = region_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Region": "Overall"}
    for col in download_df.columns:
        if col != "Region":
            if "_count" in col or "_total" in col:
                # For counts and totals, sum them
                overall_row[col] = (
                    download_df[col].sum() if col in download_df.columns else 0
                )
            else:
                # For rates, calculate weighted average (not simple sum)
                # Find corresponding count and total columns
                base_name = col
                count_col = f"{col}_count"
                total_col = f"{col}_total"

                if (
                    count_col in download_df.columns
                    and total_col in download_df.columns
                ):
                    total_count = download_df[count_col].sum()
                    total_denom = download_df[total_col].sum()
                    overall_row[col] = (
                        (total_count / total_denom * 100) if total_denom > 0 else 0
                    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"kmc_rates_region_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download KMC rates region comparison data as CSV",
    )


def render_cpap_region_comparison(
    df,
    period_col="period_display",
    title="CPAP Coverage - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
):
    """Render region comparison for CPAP coverage (by weight categories) - WITH GROUP BAR CHART"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Get periods (for information only)
    periods = df[period_col].unique() if not df.empty else []

    # FIX: Check if periods is empty properly
    if periods is None or len(periods) == 0:
        st.warning("⚠️ No data available for comparison.")
        return

    # Compute AGGREGATED CPAP data for each region (sum across all periods)
    region_data = []
    for region_name in region_names:
        facility_uids = []
        for facility_name, facility_uid in facilities_by_region.get(region_name, []):
            facility_uids.append(facility_uid)

        if facility_uids:
            # Filter for this region's facilities
            region_df = df[df["orgUnit"].isin(facility_uids)]

            # CRITICAL: Deduplicate by TEI ID for the entire region
            region_df = deduplicate_by_tei(region_df)

            if not region_df.empty:
                # Aggregate CPAP data across ALL periods
                cpap_data = compute_cpap_coverage_by_weight_kpi(
                    region_df, facility_uids
                )

                row_data = {"Region": region_name}

                # Store both rates and counts for each category
                for category_key, category_info in sorted(
                    BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
                ):
                    rate = cpap_data["cpap_rates_by_category"].get(category_key, 0)
                    count = cpap_data["cpap_counts_by_category"].get(category_key, 0)
                    total = cpap_data["cpap_total_by_category"].get(category_key, 0)

                    # Store rate for chart
                    row_data[category_info["short_name"]] = rate
                    # Store count and total for table
                    row_data[f"{category_info['short_name']}_count"] = count
                    row_data[f"{category_info['short_name']}_total"] = total

                region_data.append(row_data)

    if not region_data:
        st.warning("⚠️ No CPAP data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create GROUP bar chart for CPAP RATES
    fig = go.Figure()

    # Add bars for each BW category as separate traces (GROUP BARS)
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        if short_name in region_df.columns:
            fig.add_trace(
                go.Bar(
                    x=region_df["Region"],
                    y=region_df[short_name],  # This is the RATE value
                    name=category_info["name"],
                    marker_color=category_info["color"],
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<extra></extra>",
                )
            )

    # Calculate Y-axis range for percentage charts - ALWAYS 0-100%
    y_max = 100  # Always show up to 100% for rates

    fig.update_layout(
        title=f"{title}",
        height=500,
        xaxis_title="Region",
        yaxis_title="CPAP Coverage Rate (%)",
        barmode="group",  # CHANGED FROM "stack" TO "group"
        bargap=0.15,  # Gap between bars of different categories
        bargroupgap=0.1,  # Gap between groups of bars (regions)
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            rangemode="tozero",
            range=[0, y_max],  # FIXED: Always 0-100%
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        legend=dict(
            title="Birth Weight Categories",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # SINGLE TABLE (unchanged)
    st.subheader("📊 CPAP Region Comparison Table")

    # Create a detailed table with rates, counts, and totals for each category
    table_data = []

    for _, row in region_df.iterrows():
        table_row = {"Region": row["Region"]}

        # Add data for each category
        for category_key, category_info in sorted(
            BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
        ):
            short_name = category_info["short_name"]
            count_col = f"{short_name}_count"
            total_col = f"{short_name}_total"

            if short_name in row:
                rate = row[short_name]  # This is the rate
                count = row[count_col] if count_col in row else 0
                total = row[total_col] if total_col in row else 0

                # Format as "Rate% (Count/Total)"
                table_row[short_name] = f"{rate:.1f}% ({int(count)}/{int(total)})"

        table_data.append(table_row)

    # Add overall row
    overall_row = {"Region": "Overall"}
    for category_key, category_info in sorted(
        BIRTH_WEIGHT_CATEGORIES.items(), key=lambda x: x[1]["sort_order"]
    ):
        short_name = category_info["short_name"]
        count_col = f"{short_name}_count"
        total_col = f"{short_name}_total"

        total_count = (
            region_df[count_col].sum() if count_col in region_df.columns else 0
        )
        total_denom = (
            region_df[total_col].sum() if total_col in region_df.columns else 0
        )
        overall_rate = (total_count / total_denom * 100) if total_denom > 0 else 0

        overall_row[short_name] = (
            f"{overall_rate:.1f}% ({int(total_count)}/{int(total_denom)})"
        )

    table_data.append(overall_row)

    comparison_df = pd.DataFrame(table_data)

    # Display the table
    st.write("Format: Rate% (CPAP Cases / Total Newborns)")
    st.dataframe(comparison_df, use_container_width=True, height=300)

    # SINGLE DOWNLOAD SECTION (unchanged)
    st.subheader("📥 Download Data")

    download_df = region_df.copy()

    # Calculate totals for "Overall" row
    overall_row = {"Region": "Overall"}
    for col in download_df.columns:
        if col != "Region":
            if "_count" in col or "_total" in col:
                # For counts and totals, sum them
                overall_row[col] = (
                    download_df[col].sum() if col in download_df.columns else 0
                )
            else:
                # For rates, calculate weighted average (not simple sum)
                # Find corresponding count and total columns
                base_name = col
                count_col = f"{col}_count"
                total_col = f"{col}_total"

                if (
                    count_col in download_df.columns
                    and total_col in download_df.columns
                ):
                    total_count = download_df[count_col].sum()
                    total_denom = download_df[total_col].sum()
                    overall_row[col] = (
                        (total_count / total_denom * 100) if total_denom > 0 else 0
                    )

    # Add "Overall" row to the dataframe
    overall_df = pd.DataFrame([overall_row])
    download_df = pd.concat([download_df, overall_df], ignore_index=True)

    # Use helper function for download
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    download_csv_button(
        download_df,
        f"cpap_rates_region_comparison_{timestamp}.csv",
        "📥 Download CSV",
        f"Download CPAP rates region comparison data as CSV",
    )


# ---------------- NEW RATE COMPARISON FUNCTIONS (ALIASES) ----------------
def render_kmc_rate_facility_comparison(*args, **kwargs):
    """Render facility comparison for KMC coverage - STACKED BAR CHART FOR RATES"""
    # This is an alias for the main function
    return render_kmc_facility_comparison(*args, **kwargs)


def render_kmc_rate_region_comparison(*args, **kwargs):
    """Render region comparison for KMC coverage - STACKED BAR CHART FOR RATES"""
    # This is an alias for the main function
    return render_kmc_region_comparison(*args, **kwargs)


def render_cpap_rate_facility_comparison(*args, **kwargs):
    """Render facility comparison for CPAP coverage - STACKED BAR CHART FOR RATES"""
    # This is an alias for the main function
    return render_cpap_facility_comparison(*args, **kwargs)


def render_cpap_rate_region_comparison(*args, **kwargs):
    """Render region comparison for CPAP coverage - STACKED BAR CHART FOR RATES"""
    # This is an alias for the main function
    return render_cpap_region_comparison(*args, **kwargs)


# ---------------- Export all functions ----------------
__all__ = [
    # Cache functions
    "get_cache_key_simplified",
    "clear_cache_simplified",
    # Helper functions
    "download_csv_button",
    "clean_category_name",
    "deduplicate_by_tei",
    "get_kmc_status_for_tei",  # NEW: Enhanced KMC function
    # Date sorting helper functions
    "parse_period_to_datetime",
    "sort_periods_chronologically",
    # Computation functions
    "compute_birth_weight_by_category",
    "compute_total_with_birth_weight",
    "compute_birth_weight_kpi",
    "compute_kmc_by_weight_category",  # ENHANCED with multiple columns
    "compute_kmc_coverage_kpi",  # ENHANCED with multiple columns
    # CPAP computation functions
    "get_rds_newborns",  # ENHANCED with exact column name
    "compute_cpap_for_rds_kpi",  # ENHANCED RDS logic
    "compute_cpap_general_kpi",
    "compute_cpap_by_weight_category",
    "compute_cpap_coverage_by_weight_kpi",
    # Chart functions
    "render_birth_weight_trend_chart",
    "render_birth_weight_facility_comparison",
    "render_birth_weight_region_comparison",
    # CPAP chart functions
    "render_cpap_general_trend_chart",
    "render_cpap_rds_trend_chart",
    "render_cpap_by_weight_trend_chart",
    # Facility comparison functions
    "render_kmc_facility_comparison",
    "render_cpap_facility_comparison",
    "render_cpap_general_facility_comparison",
    "render_cpap_rds_facility_comparison",
    # Region comparison functions
    "render_kmc_region_comparison",
    "render_cpap_region_comparison",
    "render_cpap_general_region_comparison",
    "render_cpap_rds_region_comparison",
    # NEW RATE COMPARISON FUNCTIONS (ALIASES)
    "render_kmc_rate_facility_comparison",
    "render_kmc_rate_region_comparison",
    "render_cpap_rate_facility_comparison",
    "render_cpap_rate_region_comparison",
    # Constants - UPDATED with exact dataset names
    "BIRTH_WEIGHT_CATEGORIES",
    "BIRTH_WEIGHT_CATEGORY_NAMES",
    "BIRTH_WEIGHT_COL",
    "KMC_COLUMNS",  # List of all KMC columns
    "KMC_YES_CODE",
    "CPAP_ADMINISTERED_COL",
    "CPAP_YES_CODE",
    "RDS_DIAGNOSIS_COL",  # Exact column name from dataset
    "RDS_YES_CODE",
]
