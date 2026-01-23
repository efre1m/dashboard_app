import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt
import plotly.express as px
import hashlib

# Import only utility functions from kpi_utils
from utils.kpi_utils import (
    auto_text_color,
    format_period_month_year,
    compute_total_deliveries,
    get_relevant_date_column_for_kpi,
    render_trend_chart,
    render_facility_comparison_chart,
    render_region_comparison_chart,
)

# ---------------- Caching Setup ----------------
if "antipartum_compl_cache" not in st.session_state:
    st.session_state.antipartum_compl_cache = {}


def get_antipartum_compl_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Antepartum Complications computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_antipartum_compl_cache():
    """Clear the Antepartum Complications cache"""
    st.session_state.antipartum_compl_cache = {}


# Antepartum Complications KPI Configuration
COMPLICATION_COL = "obstetric_complications_diagnosis"
# Codes to count: 1 to 6 (Any complication except None 0)
COMPLICATION_CODES = ["1", "2", "3", "4", "5", "6"]

# Mapping for distribution chart - UPDATED MAPPING
COMPLICATION_LABELS = {
    "1": "HDP",
    "2": "PROM",
    "3": "GDM",
    "4": "APH",
    "5": "Amniotic fluid abnormalities",
    "6": "Others (specify)",
}


def compute_antipartum_compl_count(df, facility_uids=None):
    """
    Count Antepartum Complications occurrences - any code from 1 to 6
    Handles multi-select string format like "1,2,3"
    """
    cache_key = get_antipartum_compl_cache_key(df, facility_uids, "antipartum_compl_count")

    if cache_key in st.session_state.antipartum_compl_cache:
        return st.session_state.antipartum_compl_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if COMPLICATION_COL not in filtered_df.columns:
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string and handle multiple codes
            df_copy["complication_clean"] = df_copy[COMPLICATION_COL].astype(str)

            # Check for codes 1-6 in the string (standalone or in comma-separated list)
            # Regex pattern matching any digit from 1-6 as whole words
            pattern = r"(^|[,;\s])([1-6])([,;\s]|$)"
            
            mask = df_copy["complication_clean"].str.contains(
                pattern, regex=True, na=False
            )

            result = int(mask.sum())

    st.session_state.antipartum_compl_cache[cache_key] = result
    return result


def compute_complication_distribution(df, facility_uids=None):
    """
    Compute distribution of complications
    Returns: Dictionary with counts for each complication type
    """
    if df is None or df.empty:
        return {name: 0 for name in COMPLICATION_LABELS.values()}

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if COMPLICATION_COL not in filtered_df.columns:
        return {name: 0 for name in COMPLICATION_LABELS.values()}

    # Initialize counts
    named_counts = {name: 0 for name in COMPLICATION_LABELS.values()}

    df_copy = filtered_df.copy()
    df_copy["complication_clean"] = df_copy[COMPLICATION_COL].astype(str)

    # Process each value
    for value in df_copy["complication_clean"]:
        if pd.isna(value) or value == "nan":
            continue

        value_str = str(value).strip()

        # Skip empty values
        if not value_str or value_str.lower() in ["n/a", "nan", "null"]:
            continue

        # Split by comma and count each code
        codes = [code.strip() for code in value_str.split(",")]

        for code in codes:
            if code in COMPLICATION_LABELS:
                condition_name = COMPLICATION_LABELS[code]
                named_counts[condition_name] += 1

    # Add total count
    total = sum(named_counts.values())
    named_counts["Total Complications"] = total

    return named_counts


def compute_antipartum_compl_rate(df, facility_uids=None):
    """
    Compute Antepartum Complications Rate
    Returns: (rate, complication_cases, total_deliveries)
    """
    cache_key = get_antipartum_compl_cache_key(df, facility_uids, "antipartum_compl_rate")

    if cache_key in st.session_state.antipartum_compl_cache:
        return st.session_state.antipartum_compl_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for this KPI
        date_column = get_relevant_date_column_for_kpi(
            "Antepartum Complications Rate (%)"
        )

        # Count cases
        cases = compute_antipartum_compl_count(df, facility_uids)

        # Get total deliveries
        total_deliveries = compute_total_deliveries(df, facility_uids, date_column)

        # Calculate rate
        rate = (cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (rate, cases, total_deliveries)

    st.session_state.antipartum_compl_cache[cache_key] = result
    return result


def compute_antipartum_compl_kpi(df, facility_uids=None):
    """
    Compute KPI data for dashboard
    """
    rate, cases, total = compute_antipartum_compl_rate(df, facility_uids)

    # Get distribution
    distribution = compute_complication_distribution(df, facility_uids)

    return {
        "antipartum_compl_rate": float(rate),
        "antipartum_compl_cases": int(cases),
        "total_deliveries": int(total),
        "complication_distribution": distribution,
    }


def get_numerator_denominator_for_antipartum_compl(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for the KPI with date range filtering
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column
    date_column = get_relevant_date_column_for_kpi("Antepartum Complications Rate (%)")

    # Filter by date range
    if date_column in filtered_df.columns:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
                filtered_df = filtered_df[
                    (filtered_df[date_column] >= start_dt) & 
                    (filtered_df[date_column] < end_dt)
                ].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    rate, cases, total = compute_antipartum_compl_rate(filtered_df, facility_uids)
    return (cases, total, rate)


# ---------------- Chart Functions ----------------
def render_antipartum_compl_trend_chart(*args, **kwargs):
    return render_trend_chart(*args, **kwargs)


def render_antipartum_compl_facility_comparison_chart(*args, **kwargs):
    return render_facility_comparison_chart(*args, **kwargs)


def render_antipartum_compl_region_comparison_chart(*args, **kwargs):
    return render_region_comparison_chart(*args, **kwargs)


def render_complication_type_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """
    Render a pie chart showing distribution of complication types
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    distribution = compute_complication_distribution(df, facility_uids)

    if distribution["Total Complications"] == 0:
        st.info("⚠️ No complication data available for distribution.")
        return

    pie_data = []
    for name, count in distribution.items():
        if name != "Total Complications" and count > 0:
            pie_data.append({"Complication": name, "Count": count})

    if not pie_data:
        st.info("⚠️ No complication data available.")
        return

    pie_df = pd.DataFrame(pie_data).sort_values("Count", ascending=False)
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

    st.markdown('<div style="text-align: center; font-weight: bold;">Distribution of Antepartum Complication Types</div>', unsafe_allow_html=True)
    
    fig = px.pie(
        pie_df,
        values="Count",
        names="Complication",
        hover_data=["Percentage"],
        height=400,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    st.plotly_chart(fig, use_container_width=True)
