# kpi_utils_newborn_simplified.py - FIXED WITH ALL ISSUES RESOLVED

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import hashlib
import logging
import io

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


# ---------------- Newborn KPI Constants ----------------
# Birth weight column (from maternal birth and infant details)
BIRTH_WEIGHT_COL = "birth_weight_grams_maternal_birth_and_infant_details"

# KMC columns
KMC_ADMINISTERED_COL = "kmc_administered_interventions"
KMC_YES_CODE = "1"

# CPAP columns
CPAP_ADMINISTERED_COL = "cpap_administered_interventions"
CPAP_YES_CODE = "1"

# RDS Diagnosis columns
FIRST_REASON_ADMISSION_COL = "first_reason_for_admission_admission_information"
SECOND_REASON_ADMISSION_COL = "second_reason_for_admission_admission_information"
THIRD_REASON_ADMISSION_COL = "third_reason_for_admission_admission_information"
RDS_CODE = "5"  # RDS diagnosis code

# ---------------- Birth Weight Categories ----------------
BIRTH_WEIGHT_CATEGORIES = {
    "lt_1000": {"name": "<1000 g", "min": 0, "max": 999},
    "1000_1499": {"name": "1000-1499 g", "min": 1000, "max": 1499},
    "1500_1999": {"name": "1500-1999 g", "min": 1500, "max": 1999},
    "2000_2499": {"name": "2000-2499 g", "min": 2000, "max": 2499},
    "2500_4000": {"name": "2500-4000 g", "min": 2500, "max": 4000},
    "gt_4000": {"name": "4001+ g", "min": 4001, "max": 8000},
}


# ---------------- HELPER FUNCTIONS ----------------
def safe_convert_numeric(value, default=0):
    """Safely convert value to numeric"""
    try:
        if pd.isna(value):
            return default
        # Handle string values like "1.0", "2", etc.
        if isinstance(value, str):
            # Extract numeric part (handle cases like "1.0", "2", "3.0", etc.)
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


# ---------------- BIRTH WEIGHT KPI Functions ----------------
def compute_birth_weight_by_category(df, facility_uids=None):
    """Compute distribution of birth weights by all categories - FIXED"""
    cache_key = get_cache_key_simplified(df, facility_uids, "birth_weight_by_category")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    filtered_df = filter_by_facility(df, facility_uids)

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
    """Compute total newborns with valid birth weight - FIXED"""
    cache_key = get_cache_key_simplified(df, facility_uids, "total_with_birth_weight")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = filter_by_facility(df, facility_uids)

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
    """Compute KMC administered by birth weight category - FIXED"""
    cache_key = get_cache_key_simplified(df, facility_uids, "kmc_by_weight_category")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    filtered_df = filter_by_facility(df, facility_uids)

    if (
        BIRTH_WEIGHT_COL not in filtered_df.columns
        or KMC_ADMINISTERED_COL not in filtered_df.columns
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

        # Convert KMC column
        filtered_df["kmc_numeric"] = pd.to_numeric(
            filtered_df[KMC_ADMINISTERED_COL].astype(str).str.split(".").str[0],
            errors="coerce",
        )

        # Filter KMC cases
        kmc_mask = filtered_df["kmc_numeric"] == float(KMC_YES_CODE)
        kmc_df = filtered_df[kmc_mask].copy()

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
    """Compute KMC coverage rate by birth weight category - FIXED"""
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


# ---------------- CPAP COVERAGE KPI Functions (UPDATED) ----------------
def get_rds_newborns(df, facility_uids=None):
    """Identify newborns with RDS diagnosis - FIXED"""
    if df is None or df.empty:
        return set()

    filtered_df = filter_by_facility(df, facility_uids)

    required_cols = [
        FIRST_REASON_ADMISSION_COL,
        SECOND_REASON_ADMISSION_COL,
        THIRD_REASON_ADMISSION_COL,
    ]

    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in filtered_df.columns]
    if missing_cols:
        logger.warning(f"Missing RDS columns: {missing_cols}")
        return set()

    try:
        # Convert all RDS columns to string and check for RDS code
        rds_mask = (
            (
                filtered_df[FIRST_REASON_ADMISSION_COL]
                .astype(str)
                .str.split(".")
                .str[0]
                == RDS_CODE
            )
            | (
                filtered_df[SECOND_REASON_ADMISSION_COL]
                .astype(str)
                .str.split(".")
                .str[0]
                == RDS_CODE
            )
            | (
                filtered_df[THIRD_REASON_ADMISSION_COL]
                .astype(str)
                .str.split(".")
                .str[0]
                == RDS_CODE
            )
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
    """Compute general CPAP coverage rate - FIXED"""
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
    """Compute CPAP coverage for RDS newborns - FIXED"""
    cache_key = get_cache_key_simplified(df, facility_uids, "cpap_for_rds_kpi")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {"cpap_rate": 0.0, "cpap_count": 0, "total_rds": 0}
    else:
        filtered_df = filter_by_facility(df, facility_uids)

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
    """Compute CPAP administered by birth weight category"""
    cache_key = get_cache_key_simplified(df, facility_uids, "cpap_by_weight_category")

    if cache_key in st.session_state.kpi_cache_newborn_simplified:
        return st.session_state.kpi_cache_newborn_simplified[cache_key]

    if df is None or df.empty:
        result = {category: 0 for category in BIRTH_WEIGHT_CATEGORIES.keys()}
        st.session_state.kpi_cache_newborn_simplified[cache_key] = result
        return result

    filtered_df = filter_by_facility(df, facility_uids)

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
    """Compute CPAP coverage rate by birth weight category"""
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


# ---------------- CHART FUNCTIONS (FIXED) ----------------
def render_birth_weight_trend_chart(
    df,
    period_col="period_display",
    title="Birth Weight Distribution Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render trend chart for Birth Weight KPI - FIXED"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Compute BW distribution for each period
    trend_data = []
    periods = sorted(df[period_col].unique())

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

    # Create line chart with multiple lines (all categories)
    fig = go.Figure()

    # Colors for each line
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    # Add lines for each BW category
    for i, (category_key, category_info) in enumerate(BIRTH_WEIGHT_CATEGORIES.items()):
        count_col = f"{category_key}_count"

        if count_col in trend_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trend_df[period_col],
                    y=trend_df[count_col],
                    mode="lines+markers",
                    name=f"{category_info['name']}",
                    line=dict(width=2, color=colors[i]),
                    marker=dict(size=5),
                    hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.0f}} newborns<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Period",
        yaxis_title="Number of Newborns",
        showlegend=True,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON FOR DATA
    with st.expander(f"📥 Download Birth Weight Data", expanded=False):
        # Create a simplified version for download
        download_df = trend_df.copy()

        # Select only the period and count columns
        download_cols = [period_col, "total_with_birth_weight"]
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
            download_cols.append(f"{category_key}_count")

        download_df = download_df[download_cols]

        # Rename columns for better readability with clean names
        column_names = {
            period_col: "Period",
            "total_with_birth_weight": "Total Newborns with Birth Weight",
        }
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
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

        # Show a preview of all data
        st.write("Data Preview (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


def render_kmc_coverage_trend_chart(
    df,
    period_col="period_display",
    title="KMC Coverage by Birth Weight Category",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render KMC coverage trend - SIMPLIFIED WITH CATEGORY SELECTION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Let user select which birth weight category to view
    category_options = [info["name"] for info in BIRTH_WEIGHT_CATEGORIES.values()]

    # Add "All Categories" option
    category_options_with_all = ["All Categories"] + category_options

    selected_category = st.selectbox(
        "Select Birth Weight Category to View:",
        options=category_options_with_all,
        index=0,  # Default to "All Categories"
        key=f"kmc_category_select_{hash(str(facility_uids))}",
    )

    # Compute KMC coverage for each period
    trend_data = []
    periods = sorted(df[period_col].unique())

    for period in periods:
        period_df = df[df[period_col] == period]
        kmc_data = compute_kmc_coverage_kpi(period_df, facility_uids)

        period_row = {period_col: period}

        # Add KMC rate for each category
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
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

    # Create line chart
    fig = go.Figure()

    if selected_category == "All Categories":
        # Show all categories
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

        # Add lines for each BW category
        for i, (category_key, category_info) in enumerate(
            BIRTH_WEIGHT_CATEGORIES.items()
        ):
            rate_col = f"{category_key}_rate"
            count_col = f"{category_key}_count"
            total_col = f"{category_key}_total"

            if rate_col in trend_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[period_col],
                        y=trend_df[rate_col],
                        mode="lines+markers",
                        name=f"{category_info['name']}",
                        line=dict(width=2, color=colors[i]),
                        marker=dict(size=5),
                        hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.1f}}%<br>KMC Cases: %{{customdata[0]}}<br>Total Newborns: %{{customdata[1]}}<extra></extra>",
                        customdata=np.column_stack(
                            (trend_df[count_col], trend_df[total_col])
                        ),
                    )
                )
    else:
        # Show only selected category
        # Find the category key for the selected category
        selected_category_key = None
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
            if category_info["name"] == selected_category:
                selected_category_key = category_key
                break

        if selected_category_key:
            rate_col = f"{selected_category_key}_rate"
            count_col = f"{selected_category_key}_count"
            total_col = f"{selected_category_key}_total"

            fig.add_trace(
                go.Scatter(
                    x=trend_df[period_col],
                    y=trend_df[rate_col],
                    mode="lines+markers",
                    name=selected_category,
                    line=dict(width=3, color="#3498db"),
                    marker=dict(size=7),
                    hovertemplate=f"<b>%{{x}}</b><br>{selected_category}: %{{y:.1f}}%<br>KMC Cases: %{{customdata[0]}}<br>Total Newborns: %{{customdata[1]}}<extra></extra>",
                    customdata=np.column_stack(
                        (trend_df[count_col], trend_df[total_col])
                    ),
                )
            )

    fig.update_layout(
        title=f"{title} - {selected_category}",
        height=400,
        xaxis_title="Period",
        yaxis_title="KMC Coverage (%)",
        showlegend=True,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON INSTEAD OF EXPANDER WITH LARGE TABLE
    with st.expander(f"📥 Download KMC Coverage Data", expanded=False):
        # Create a simplified version for download
        download_df = trend_df.copy()

        # Select only the period and selected category columns
        if selected_category == "All Categories":
            # Include all categories for download
            download_cols = [period_col]
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                download_cols.extend(
                    [
                        f"{category_key}_rate",
                        f"{category_key}_count",
                        f"{category_key}_total",
                    ]
                )
        else:
            # Only include selected category
            download_cols = [period_col]
            if selected_category_key:
                download_cols.extend(
                    [
                        f"{selected_category_key}_rate",
                        f"{selected_category_key}_count",
                        f"{selected_category_key}_total",
                    ]
                )

        download_df = download_df[download_cols]

        # Rename columns for better readability with clean names
        column_names = {period_col: "Period"}

        if selected_category == "All Categories":
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                clean_name = clean_category_name(category_info["name"])
                column_names[f"{category_key}_rate"] = f"{clean_name} KMC Rate (%)"
                column_names[f"{category_key}_count"] = f"{clean_name} KMC Cases"
                column_names[f"{category_key}_total"] = f"{clean_name} Total Newborns"
        elif selected_category_key:
            category_info = BIRTH_WEIGHT_CATEGORIES[selected_category_key]
            clean_name = clean_category_name(selected_category)
            column_names[f"{selected_category_key}_rate"] = f"{clean_name} KMC Rate (%)"
            column_names[f"{selected_category_key}_count"] = f"{clean_name} KMC Cases"
            column_names[f"{selected_category_key}_total"] = (
                f"{clean_name} Total Newborns"
            )

        download_df = download_df.rename(columns=column_names)

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = (
            selected_category.replace(" ", "_")
            .replace("–", "-")
            .replace("+", "plus")
            .lower()
        )
        download_csv_button(
            download_df,
            f"kmc_coverage_{clean_filename}_{timestamp}.csv",
            "📥 Download CSV",
            f"Download KMC coverage data for {selected_category} as CSV",
        )

        # Show a preview of the data
        st.write(f"Data Preview for {selected_category} (All Columns):")
        if selected_category == "All Categories":
            # Show subset for All Categories preview
            preview_cols = ["Period"]
            for i, (category_key, category_info) in enumerate(
                BIRTH_WEIGHT_CATEGORIES.items()
            ):
                if i < 3:  # Show only first 3 categories in preview
                    clean_name = clean_category_name(category_info["name"])
                    preview_cols.extend(
                        [
                            f"{clean_name} KMC Rate (%)",
                            f"{clean_name} KMC Cases",
                            f"{clean_name} Total Newborns",
                        ]
                    )
            preview_df = download_df[preview_cols].head(10)
            st.dataframe(preview_df, use_container_width=True)
            if len(BIRTH_WEIGHT_CATEGORIES) > 3:
                st.info(
                    f"Note: Showing first 3 of {len(BIRTH_WEIGHT_CATEGORIES)} categories in preview. All {len(BIRTH_WEIGHT_CATEGORIES)} categories are included in the download."
                )
        else:
            # Show all columns for single category
            st.dataframe(download_df.head(10), use_container_width=True)


# ---------------- SEPARATE CPAP CHART FUNCTIONS ----------------
def render_cpap_general_trend_chart(
    df,
    period_col="period_display",
    title="General CPAP Coverage Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render general CPAP coverage trend chart - SINGLE PLOT"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Compute CPAP data for each period
    trend_data = []
    periods = sorted(df[period_col].unique())

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

    # Create line chart for general CPAP
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df[period_col],
            y=trend_df["cpap_general_rate"],
            mode="lines+markers",
            name="General CPAP",
            line=dict(width=2, color="#3498db"),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>General CPAP: %{y:.1f}%<br>Cases: %{customdata[0]}<br>Total Admitted: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (trend_df["cpap_general_count"], trend_df["cpap_general_total"])
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage (%)",
        showlegend=True,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander(f"📥 Download General CPAP Data", expanded=False):
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

        # Show a preview of the data
        st.write("Data Preview (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


def render_cpap_rds_trend_chart(
    df,
    period_col="period_display",
    title="CPAP for RDS Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render CPAP for RDS trend chart - SINGLE PLOT"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Compute CPAP data for each period
    trend_data = []
    periods = sorted(df[period_col].unique())

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

    # Create line chart for CPAP for RDS
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend_df[period_col],
            y=trend_df["cpap_rds_rate"],
            mode="lines+markers",
            name="CPAP for RDS",
            line=dict(width=2, color="#e74c3c"),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>CPAP for RDS: %{y:.1f}%<br>Cases: %{customdata[0]}<br>Total RDS: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (trend_df["cpap_rds_count"], trend_df["cpap_rds_total"])
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage (%)",
        showlegend=True,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander(f"📥 Download CPAP for RDS Data", expanded=False):
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

        # Show a preview of the data
        st.write("Data Preview (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


def render_cpap_by_weight_trend_chart(
    df,
    period_col="period_display",
    title="CPAP Coverage by Birth Weight Category",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render CPAP coverage trend by birth weight category - SIMPLIFIED WITH CATEGORY SELECTION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.warning("⚠️ No data available for the selected period.")
        return

    # Let user select which birth weight category to view
    category_options = [info["name"] for info in BIRTH_WEIGHT_CATEGORIES.values()]

    # Add "All Categories" option
    category_options_with_all = ["All Categories"] + category_options

    selected_category = st.selectbox(
        "Select Birth Weight Category to View:",
        options=category_options_with_all,
        index=0,  # Default to "All Categories"
        key=f"cpap_category_select_{hash(str(facility_uids))}",
    )

    # Compute CPAP coverage for each period
    trend_data = []
    periods = sorted(df[period_col].unique())

    for period in periods:
        period_df = df[df[period_col] == period]
        cpap_data = compute_cpap_coverage_by_weight_kpi(period_df, facility_uids)

        period_row = {period_col: period}

        # Add CPAP rate for each category
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
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

    # Create line chart
    fig = go.Figure()

    if selected_category == "All Categories":
        # Show all categories
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

        # Add lines for each BW category
        for i, (category_key, category_info) in enumerate(
            BIRTH_WEIGHT_CATEGORIES.items()
        ):
            rate_col = f"{category_key}_rate"
            count_col = f"{category_key}_count"
            total_col = f"{category_key}_total"

            if rate_col in trend_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=trend_df[period_col],
                        y=trend_df[rate_col],
                        mode="lines+markers",
                        name=f"{category_info['name']}",
                        line=dict(width=2, color=colors[i]),
                        marker=dict(size=5),
                        hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.1f}}%<br>CPAP Cases: %{{customdata[0]}}<br>Total Newborns: %{{customdata[1]}}<extra></extra>",
                        customdata=np.column_stack(
                            (trend_df[count_col], trend_df[total_col])
                        ),
                    )
                )
    else:
        # Show only selected category
        # Find the category key for the selected category
        selected_category_key = None
        for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
            if category_info["name"] == selected_category:
                selected_category_key = category_key
                break

        if selected_category_key:
            rate_col = f"{selected_category_key}_rate"
            count_col = f"{selected_category_key}_count"
            total_col = f"{selected_category_key}_total"

            fig.add_trace(
                go.Scatter(
                    x=trend_df[period_col],
                    y=trend_df[rate_col],
                    mode="lines+markers",
                    name=selected_category,
                    line=dict(width=3, color="#e74c3c"),
                    marker=dict(size=7),
                    hovertemplate=f"<b>%{{x}}</b><br>{selected_category}: %{{y:.1f}}%<br>CPAP Cases: %{{customdata[0]}}<br>Total Newborns: %{{customdata[1]}}<extra></extra>",
                    customdata=np.column_stack(
                        (trend_df[count_col], trend_df[total_col])
                    ),
                )
            )

    fig.update_layout(
        title=f"{title} - {selected_category}",
        height=400,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage (%)",
        showlegend=True,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON INSTEAD OF EXPANDER WITH LARGE TABLE
    with st.expander(f"📥 Download CPAP Coverage Data", expanded=False):
        # Create a simplified version for download
        download_df = trend_df.copy()

        # Select only the period and selected category columns
        if selected_category == "All Categories":
            # Include all categories for download
            download_cols = [period_col]
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                download_cols.extend(
                    [
                        f"{category_key}_rate",
                        f"{category_key}_count",
                        f"{category_key}_total",
                    ]
                )
        else:
            # Only include selected category
            download_cols = [period_col]
            if selected_category_key:
                download_cols.extend(
                    [
                        f"{selected_category_key}_rate",
                        f"{selected_category_key}_count",
                        f"{selected_category_key}_total",
                    ]
                )

        download_df = download_df[download_cols]

        # Rename columns for better readability with clean names
        column_names = {period_col: "Period"}

        if selected_category == "All Categories":
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                clean_name = clean_category_name(category_info["name"])
                column_names[f"{category_key}_rate"] = f"{clean_name} CPAP Rate (%)"
                column_names[f"{category_key}_count"] = f"{clean_name} CPAP Cases"
                column_names[f"{category_key}_total"] = f"{clean_name} Total Newborns"
        elif selected_category_key:
            category_info = BIRTH_WEIGHT_CATEGORIES[selected_category_key]
            clean_name = clean_category_name(selected_category)
            column_names[f"{selected_category_key}_rate"] = (
                f"{clean_name} CPAP Rate (%)"
            )
            column_names[f"{selected_category_key}_count"] = f"{clean_name} CPAP Cases"
            column_names[f"{selected_category_key}_total"] = (
                f"{clean_name} Total Newborns"
            )

        download_df = download_df.rename(columns=column_names)

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = (
            selected_category.replace(" ", "_")
            .replace("–", "-")
            .replace("+", "plus")
            .lower()
        )
        download_csv_button(
            download_df,
            f"cpap_coverage_{clean_filename}_{timestamp}.csv",
            "📥 Download CSV",
            f"Download CPAP coverage data for {selected_category} as CSV",
        )

        # Show a preview of the data
        st.write(f"Data Preview for {selected_category} (All Columns):")
        if selected_category == "All Categories":
            # Show subset for All Categories preview
            preview_cols = ["Period"]
            for i, (category_key, category_info) in enumerate(
                BIRTH_WEIGHT_CATEGORIES.items()
            ):
                if i < 3:  # Show only first 3 categories in preview
                    clean_name = clean_category_name(category_info["name"])
                    preview_cols.extend(
                        [
                            f"{clean_name} CPAP Rate (%)",
                            f"{clean_name} CPAP Cases",
                            f"{clean_name} Total Newborns",
                        ]
                    )
            preview_df = download_df[preview_cols].head(10)
            st.dataframe(preview_df, use_container_width=True)
            if len(BIRTH_WEIGHT_CATEGORIES) > 3:
                st.info(
                    f"Note: Showing first 3 of {len(BIRTH_WEIGHT_CATEGORIES)} categories in preview. All {len(BIRTH_WEIGHT_CATEGORIES)} categories are included in the download."
                )
        else:
            # Show all columns for single category
            st.dataframe(download_df.head(10), use_container_width=True)


# ---------------- LEGACY FUNCTION FOR BACKWARD COMPATIBILITY ----------------
def render_cpap_trend_chart(
    df,
    period_col="period_display",
    title="CPAP Coverage Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
):
    """Render CPAP coverage trend chart - LEGACY FUNCTION (calls general CPAP)"""
    # Call the new single plot function for backward compatibility
    render_cpap_general_trend_chart(
        df, period_col, title, bg_color, text_color, facility_uids
    )


# ---------------- FACILITY COMPARISON FUNCTIONS (UPDATED) ----------------
def render_birth_weight_facility_comparison(
    df,
    period_col="period_display",
    title="Birth Weight Distribution - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for birth weight - FIXED"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # For facility comparison, show bar chart for selected period
    periods = sorted(df[period_col].unique()) if not df.empty else []
    if not periods:
        st.warning("⚠️ No data available for comparison.")
        return

    selected_period = st.selectbox(
        "Select Period for Comparison:",
        options=periods,
        key=f"bw_facility_period_{str(facility_uids)}",
    )

    period_df = df[df[period_col] == selected_period]

    # Compute data for each facility
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
        if not facility_period_df.empty:
            bw_data = compute_birth_weight_kpi(facility_period_df, [facility_uid])

            row_data = {
                "Facility": facility_name,
                "Total with Birth Weight": bw_data["total_with_birth_weight"],
            }

            # Add each category count
            for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                row_data[category_info["name"]] = bw_data["bw_category_counts"].get(
                    category_key, 0
                )

            facility_data.append(row_data)

    if not facility_data:
        st.warning("⚠️ No data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create bar chart
    fig = go.Figure()

    # Add bars for each BW category
    for category_info in BIRTH_WEIGHT_CATEGORIES.values():
        category_name = category_info["name"]
        if category_name in facility_df.columns:
            fig.add_trace(
                go.Bar(
                    x=facility_df["Facility"],
                    y=facility_df[category_name],
                    name=category_name,
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.0f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{title} - {selected_period}",
        height=400,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander("📥 Download Facility Comparison Data", expanded=False):
        download_df = facility_df.copy()

        # Clean column names for download
        column_names = {
            "Facility": "Facility",
            "Total with Birth Weight": "Total Newborns with Birth Weight",
        }
        for category_info in BIRTH_WEIGHT_CATEGORIES.values():
            clean_name = clean_category_name(category_info["name"])
            column_names[category_info["name"]] = f"{clean_name} Newborns"

        download_df = download_df.rename(columns=column_names)

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        download_csv_button(
            download_df,
            f"birth_weight_facility_comparison_{selected_period}_{timestamp}.csv",
            "📥 Download CSV",
            "Download birth weight facility comparison data as CSV",
        )

        # Show a preview of the data
        st.write(f"Data Preview for {selected_period} (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


def render_kmc_facility_comparison(
    df,
    period_col="period_display",
    title="KMC Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for KMC coverage - FIXED WITH CATEGORY SELECTION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # For KMC, focus on one category at a time for comparison
    category_options = [info["name"] for info in BIRTH_WEIGHT_CATEGORIES.values()]
    selected_category = st.selectbox(
        "Select Birth Weight Category to Compare:",
        options=category_options,
        index=3,  # Default to 2000-2499g
        key=f"kmc_category_facility_{str(facility_uids)}",
    )

    # Find the category key
    category_key = None
    for key, info in BIRTH_WEIGHT_CATEGORIES.items():
        if info["name"] == selected_category:
            category_key = key
            break

    if not category_key:
        st.error("Selected category not found.")
        return

    # Get all periods
    periods = sorted(df[period_col].unique()) if not df.empty else []
    if not periods:
        st.warning("⚠️ No data available for comparison.")
        return

    selected_period = st.selectbox(
        "Select Period for Comparison:",
        options=periods,
        key=f"kmc_facility_period_{str(facility_uids)}",
    )

    period_df = df[df[period_col] == selected_period]

    # Compute KMC rates for each facility
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
        if not facility_period_df.empty:
            kmc_data = compute_kmc_coverage_kpi(facility_period_df, [facility_uid])

            rate = kmc_data["kmc_rates_by_category"].get(category_key, 0)
            count = kmc_data["kmc_counts_by_category"].get(category_key, 0)
            total = kmc_data["kmc_total_by_category"].get(category_key, 0)

            facility_data.append(
                {
                    "Facility": facility_name,
                    "KMC Rate (%)": rate,
                    "KMC Cases": count,
                    "Total Newborns": total,  # CHANGED FROM "Total Eligible"
                }
            )

    if not facility_data:
        st.warning("⚠️ No KMC data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create bar chart
    fig = px.bar(
        facility_df,
        x="Facility",
        y="KMC Rate (%)",
        title=f"{selected_category} KMC Coverage - {selected_period}",
        height=400,
        color="Facility",
        hover_data=["KMC Cases", "Total Newborns"],  # CHANGED
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>KMC Rate: %{y:.1f}%<br>KMC Cases: %{customdata[0]}<br>Total Newborns: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack(
            (facility_df["KMC Cases"], facility_df["Total Newborns"])  # CHANGED
        ),
    )

    fig.update_layout(
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander("📥 Download Facility Comparison Data", expanded=False):
        download_df = facility_df.copy()

        # Clean filename
        clean_category = clean_category_name(selected_category)

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        download_csv_button(
            download_df,
            f"kmc_coverage_{clean_category.replace(' ', '_').lower()}_facility_comparison_{selected_period}_{timestamp}.csv",
            "📥 Download CSV",
            f"Download KMC coverage facility comparison data for {selected_category} as CSV",
        )

        # Show a preview of the data
        st.write(
            f"Data Preview for {selected_category} - {selected_period} (All Columns):"
        )
        st.dataframe(download_df.head(10), use_container_width=True)


def render_cpap_facility_comparison(
    df,
    period_col="period_display",
    title="CPAP Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison for CPAP coverage - UPDATED (REMOVED PROPHYLACTIC)"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.warning("⚠️ No facilities selected for comparison.")
        return

    # Select CPAP type to compare (REMOVED PROPHYLACTIC OPTION)
    cpap_type = st.selectbox(
        "Select CPAP Type to Compare:",
        options=[
            "General CPAP",
            "CPAP for RDS",
            "CPAP by Birth Weight",
        ],
        key=f"cpap_type_facility_{str(facility_uids)}",
    )

    periods = sorted(df[period_col].unique()) if not df.empty else []
    if not periods:
        st.warning("⚠️ No data available for comparison.")
        return

    selected_period = st.selectbox(
        "Select Period for Comparison:",
        options=periods,
        key=f"cpap_facility_period_{str(facility_uids)}",
    )

    period_df = df[df[period_col] == selected_period]

    # Compute data for each facility
    facility_data = []
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
        if not facility_period_df.empty:
            if cpap_type == "General CPAP":
                cpap_data = compute_cpap_general_kpi(facility_period_df, [facility_uid])
                rate = cpap_data.get("cpap_general_rate", 0)
                count = cpap_data.get("cpap_general_count", 0)
                total = cpap_data.get("total_admitted", 0)
                rate_label = "General CPAP Rate (%)"
                count_label = "General CPAP Cases"
                total_label = "Total Admitted Newborns"
            elif cpap_type == "CPAP for RDS":
                cpap_data = compute_cpap_for_rds_kpi(facility_period_df, [facility_uid])
                rate = cpap_data.get("cpap_rate", 0)
                count = cpap_data.get("cpap_count", 0)
                total = cpap_data.get("total_rds", 0)
                rate_label = "CPAP for RDS Rate (%)"
                count_label = "CPAP for RDS Cases"
                total_label = "Total RDS Newborns"
            else:  # CPAP by Birth Weight
                # For CPAP by birth weight, we need to select a category
                category_options = [
                    info["name"] for info in BIRTH_WEIGHT_CATEGORIES.values()
                ]
                selected_category = st.selectbox(
                    "Select Birth Weight Category:",
                    options=category_options,
                    index=3,
                    key=f"cpap_category_facility_{str(facility_uids)}",
                )

                # Find the category key
                category_key = None
                for key, info in BIRTH_WEIGHT_CATEGORIES.items():
                    if info["name"] == selected_category:
                        category_key = key
                        break

                if category_key:
                    cpap_data = compute_cpap_coverage_by_weight_kpi(
                        facility_period_df, [facility_uid]
                    )
                    rate = cpap_data["cpap_rates_by_category"].get(category_key, 0)
                    count = cpap_data["cpap_counts_by_category"].get(category_key, 0)
                    total = cpap_data["cpap_total_by_category"].get(category_key, 0)
                    rate_label = (
                        f"{clean_category_name(selected_category)} CPAP Rate (%)"
                    )
                    count_label = f"{clean_category_name(selected_category)} CPAP Cases"
                    total_label = (
                        f"{clean_category_name(selected_category)} Total Newborns"
                    )
                else:
                    rate = 0
                    count = 0
                    total = 0
                    rate_label = "Rate (%)"
                    count_label = "Cases"
                    total_label = "Total"

            facility_data.append(
                {
                    "Facility": facility_name,
                    rate_label: rate,
                    count_label: count,
                    total_label: total,
                }
            )

    if not facility_data:
        st.warning(f"⚠️ No {cpap_type} data available for facility comparison.")
        return

    facility_df = pd.DataFrame(facility_data)

    # Create bar chart
    fig = px.bar(
        facility_df,
        x="Facility",
        y=rate_label,
        title=f"{cpap_type} - {selected_period}",
        height=400,
        color="Facility",
        hover_data=[count_label, total_label],
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f}%<br>Cases: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack(
            (facility_df[count_label], facility_df[total_label])
        ),
    )

    fig.update_layout(
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander("📥 Download Facility Comparison Data", expanded=False):
        download_df = facility_df.copy()

        # Clean filename
        clean_cpap_type = cpap_type.replace(" ", "_").lower()

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        download_csv_button(
            download_df,
            f"cpap_{clean_cpap_type}_facility_comparison_{selected_period}_{timestamp}.csv",
            "📥 Download CSV",
            f"Download {cpap_type} facility comparison data as CSV",
        )

        # Show a preview of the data
        st.write(f"Data Preview for {cpap_type} - {selected_period} (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


# ---------------- REGION COMPARISON FUNCTIONS (UPDATED) ----------------
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
    """Render region comparison for birth weight"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Select period for comparison
    periods = sorted(df[period_col].unique()) if not df.empty else []
    if not periods:
        st.warning("⚠️ No data available for comparison.")
        return

    selected_period = st.selectbox(
        "Select Period for Comparison:",
        options=periods,
        key=f"bw_region_period_{str(region_names)}",
    )

    period_df = df[df[period_col] == selected_period]

    # Compute data for each region
    region_data = []
    for region_name in region_names:
        facility_uids = facilities_by_region.get(region_name, [])
        if facility_uids:
            region_period_df = period_df[period_df["orgUnit"].isin(facility_uids)]
            if not region_period_df.empty:
                bw_data = compute_birth_weight_kpi(region_period_df, facility_uids)

                row_data = {
                    "Region": region_name,
                    "Total with Birth Weight": bw_data["total_with_birth_weight"],
                }

                # Add each category count
                for category_key, category_info in BIRTH_WEIGHT_CATEGORIES.items():
                    row_data[category_info["name"]] = bw_data["bw_category_counts"].get(
                        category_key, 0
                    )

                region_data.append(row_data)

    if not region_data:
        st.warning("⚠️ No data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create bar chart
    fig = go.Figure()

    # Add bars for each BW category
    for category_info in BIRTH_WEIGHT_CATEGORIES.values():
        category_name = category_info["name"]
        if category_name in region_df.columns:
            fig.add_trace(
                go.Bar(
                    x=region_df["Region"],
                    y=region_df[category_name],
                    name=category_name,
                    hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.0f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{title} - {selected_period}",
        height=400,
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander("📥 Download Region Comparison Data", expanded=False):
        download_df = region_df.copy()

        # Clean column names for download
        column_names = {
            "Region": "Region",
            "Total with Birth Weight": "Total Newborns with Birth Weight",
        }
        for category_info in BIRTH_WEIGHT_CATEGORIES.values():
            clean_name = clean_category_name(category_info["name"])
            column_names[category_info["name"]] = f"{clean_name} Newborns"

        download_df = download_df.rename(columns=column_names)

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        download_csv_button(
            download_df,
            f"birth_weight_region_comparison_{selected_period}_{timestamp}.csv",
            "📥 Download CSV",
            "Download birth weight region comparison data as CSV",
        )

        # Show a preview of the data
        st.write(f"Data Preview for {selected_period} (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


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
    """Render region comparison for KMC coverage"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Select birth weight category
    category_options = [info["name"] for info in BIRTH_WEIGHT_CATEGORIES.values()]
    selected_category = st.selectbox(
        "Select Birth Weight Category to Compare:",
        options=category_options,
        index=3,
        key=f"kmc_category_region_{str(region_names)}",
    )

    # Find the category key
    category_key = None
    for key, info in BIRTH_WEIGHT_CATEGORIES.items():
        if info["name"] == selected_category:
            category_key = key
            break

    if not category_key:
        st.error("Selected category not found.")
        return

    # Select period
    periods = sorted(df[period_col].unique()) if not df.empty else []
    if not periods:
        st.warning("⚠️ No data available for comparison.")
        return

    selected_period = st.selectbox(
        "Select Period for Comparison:",
        options=periods,
        key=f"kmc_region_period_{str(region_names)}",
    )

    period_df = df[df[period_col] == selected_period]

    # Compute KMC rates for each region
    region_data = []
    for region_name in region_names:
        facility_uids = facilities_by_region.get(region_name, [])
        if facility_uids:
            region_period_df = period_df[period_df["orgUnit"].isin(facility_uids)]
            if not region_period_df.empty:
                kmc_data = compute_kmc_coverage_kpi(region_period_df, facility_uids)

                rate = kmc_data["kmc_rates_by_category"].get(category_key, 0)
                count = kmc_data["kmc_counts_by_category"].get(category_key, 0)
                total = kmc_data["kmc_total_by_category"].get(category_key, 0)

                region_data.append(
                    {
                        "Region": region_name,
                        "KMC Rate (%)": rate,
                        "KMC Cases": count,
                        "Total Newborns": total,  # CHANGED FROM "Total Eligible"
                    }
                )

    if not region_data:
        st.warning("⚠️ No KMC data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create bar chart
    fig = px.bar(
        region_df,
        x="Region",
        y="KMC Rate (%)",
        title=f"{selected_category} KMC Coverage - {selected_period}",
        height=400,
        color="Region",
        hover_data=["KMC Cases", "Total Newborns"],  # CHANGED
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>KMC Rate: %{y:.1f}%<br>KMC Cases: %{customdata[0]}<br>Total Newborns: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack(
            (region_df["KMC Cases"], region_df["Total Newborns"])  # CHANGED
        ),
    )

    fig.update_layout(
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander("📥 Download Region Comparison Data", expanded=False):
        download_df = region_df.copy()

        # Clean filename
        clean_category = clean_category_name(selected_category)

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        download_csv_button(
            download_df,
            f"kmc_coverage_{clean_category.replace(' ', '_').lower()}_region_comparison_{selected_period}_{timestamp}.csv",
            "📥 Download CSV",
            f"Download KMC coverage region comparison data for {selected_category} as CSV",
        )

        # Show a preview of the data
        st.write(
            f"Data Preview for {selected_category} - {selected_period} (All Columns):"
        )
        st.dataframe(download_df.head(10), use_container_width=True)


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
    """Render region comparison for CPAP coverage - UPDATED (REMOVED PROPHYLACTIC)"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not region_mapping or not facilities_by_region:
        st.warning("⚠️ No regions available for comparison.")
        return

    # Select CPAP type (REMOVED PROPHYLACTIC OPTION)
    cpap_type = st.selectbox(
        "Select CPAP Type to Compare:",
        options=[
            "General CPAP",
            "CPAP for RDS",
            "CPAP by Birth Weight",
        ],
        key=f"cpap_type_region_{str(region_names)}",
    )

    # Select period
    periods = sorted(df[period_col].unique()) if not df.empty else []
    if not periods:
        st.warning("⚠️ No data available for comparison.")
        return

    selected_period = st.selectbox(
        "Select Period for Comparison:",
        options=periods,
        key=f"cpap_region_period_{str(region_names)}",
    )

    period_df = df[df[period_col] == selected_period]

    # Compute data for each region
    region_data = []
    for region_name in region_names:
        facility_uids = facilities_by_region.get(region_name, [])
        if facility_uids:
            region_period_df = period_df[period_df["orgUnit"].isin(facility_uids)]
            if not region_period_df.empty:
                if cpap_type == "General CPAP":
                    cpap_data = compute_cpap_general_kpi(
                        region_period_df, facility_uids
                    )
                    rate = cpap_data.get("cpap_general_rate", 0)
                    count = cpap_data.get("cpap_general_count", 0)
                    total = cpap_data.get("total_admitted", 0)
                    rate_label = "General CPAP Rate (%)"
                    count_label = "General CPAP Cases"
                    total_label = "Total Admitted Newborns"
                elif cpap_type == "CPAP for RDS":
                    cpap_data = compute_cpap_for_rds_kpi(
                        region_period_df, facility_uids
                    )
                    rate = cpap_data.get("cpap_rate", 0)
                    count = cpap_data.get("cpap_count", 0)
                    total = cpap_data.get("total_rds", 0)
                    rate_label = "CPAP for RDS Rate (%)"
                    count_label = "CPAP for RDS Cases"
                    total_label = "Total RDS Newborns"
                else:  # CPAP by Birth Weight
                    # For CPAP by birth weight, we need to select a category
                    category_options = [
                        info["name"] for info in BIRTH_WEIGHT_CATEGORIES.values()
                    ]
                    selected_category = st.selectbox(
                        "Select Birth Weight Category:",
                        options=category_options,
                        index=3,
                        key=f"cpap_category_region_{str(region_names)}",
                    )

                    # Find the category key
                    category_key = None
                    for key, info in BIRTH_WEIGHT_CATEGORIES.items():
                        if info["name"] == selected_category:
                            category_key = key
                            break

                    if category_key:
                        cpap_data = compute_cpap_coverage_by_weight_kpi(
                            region_period_df, facility_uids
                        )
                        rate = cpap_data["cpap_rates_by_category"].get(category_key, 0)
                        count = cpap_data["cpap_counts_by_category"].get(
                            category_key, 0
                        )
                        total = cpap_data["cpap_total_by_category"].get(category_key, 0)
                        rate_label = (
                            f"{clean_category_name(selected_category)} CPAP Rate (%)"
                        )
                        count_label = (
                            f"{clean_category_name(selected_category)} CPAP Cases"
                        )
                        total_label = (
                            f"{clean_category_name(selected_category)} Total Newborns"
                        )
                    else:
                        rate = 0
                        count = 0
                        total = 0
                        rate_label = "Rate (%)"
                        count_label = "Cases"
                        total_label = "Total"

                region_data.append(
                    {
                        "Region": region_name,
                        rate_label: rate,
                        count_label: count,
                        total_label: total,
                    }
                )

    if not region_data:
        st.warning(f"⚠️ No {cpap_type} data available for region comparison.")
        return

    region_df = pd.DataFrame(region_data)

    # Create bar chart
    fig = px.bar(
        region_df,
        x="Region",
        y=rate_label,
        title=f"{cpap_type} - {selected_period}",
        height=400,
        color="Region",
        hover_data=[count_label, total_label],
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Coverage: %{y:.1f}%<br>Cases: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack((region_df[count_label], region_df[total_label])),
    )

    fig.update_layout(
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
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # ADD DOWNLOAD BUTTON
    with st.expander("📥 Download Region Comparison Data", expanded=False):
        download_df = region_df.copy()

        # Clean filename
        clean_cpap_type = cpap_type.replace(" ", "_").lower()

        # Use helper function for download
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        download_csv_button(
            download_df,
            f"cpap_{clean_cpap_type}_region_comparison_{selected_period}_{timestamp}.csv",
            "📥 Download CSV",
            f"Download {cpap_type} region comparison data as CSV",
        )

        # Show a preview of the data
        st.write(f"Data Preview for {cpap_type} - {selected_period} (All Columns):")
        st.dataframe(download_df.head(10), use_container_width=True)


# ---------------- Export all functions ----------------
__all__ = [
    # Cache functions
    "get_cache_key_simplified",
    "clear_cache_simplified",
    # Helper functions
    "download_csv_button",
    "clean_category_name",
    # Computation functions
    "compute_birth_weight_by_category",
    "compute_total_with_birth_weight",
    "compute_birth_weight_kpi",
    "compute_kmc_by_weight_category",
    "compute_kmc_coverage_kpi",
    # CPAP computation functions (REMOVED PROPHYLACTIC)
    "compute_cpap_for_rds_kpi",
    "compute_cpap_general_kpi",
    "compute_cpap_by_weight_category",
    "compute_cpap_coverage_by_weight_kpi",
    # Chart functions
    "render_birth_weight_trend_chart",
    "render_kmc_coverage_trend_chart",
    # Individual CPAP chart functions (REMOVED PROPHYLACTIC)
    "render_cpap_general_trend_chart",
    "render_cpap_rds_trend_chart",
    "render_cpap_by_weight_trend_chart",
    # Legacy function for backward compatibility
    "render_cpap_trend_chart",
    # Facility comparison functions
    "render_birth_weight_facility_comparison",
    "render_kmc_facility_comparison",
    "render_cpap_facility_comparison",
    # Region comparison functions
    "render_birth_weight_region_comparison",
    "render_kmc_region_comparison",
    "render_cpap_region_comparison",
    # Constants
    "BIRTH_WEIGHT_CATEGORIES",
    "BIRTH_WEIGHT_COL",
    "KMC_ADMINISTERED_COL",
    "KMC_YES_CODE",
    "CPAP_ADMINISTERED_COL",
    "CPAP_YES_CODE",
    "RDS_CODE",
]
