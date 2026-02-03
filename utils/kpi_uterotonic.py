import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import datetime as dt
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
if "uterotonic_cache" not in st.session_state:
    st.session_state.uterotonic_cache = {}


def get_uterotonic_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Uterotonic computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_uterotonic_cache():
    """Clear the Uterotonic cache"""
    st.session_state.uterotonic_cache = {}


# Uterotonic KPI Configuration
UTEROTONIC_COL = "uterotonics_given_delivery_summary"
UTEROTONIC_CODES = {
    "1",  # Ergometrine
    "2",  # Oxytocin
    "3",  # Misoprostol
}

# Date column for uterotonic KPI
UTEROTONIC_DATE_COL = "enrollment_date"


def compute_uterotonic_count(df, facility_uids=None):
    """
    Count Uterotonic administration occurrences - VECTORIZED for performance
    """
    cache_key = get_uterotonic_cache_key(df, facility_uids, "uterotonic_count")

    if cache_key in st.session_state.uterotonic_cache:
        return st.session_state.uterotonic_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if UTEROTONIC_COL not in filtered_df.columns:
            result = 0
        else:
            # Check for codes 1, 2, or 3 in the string (standalone or in comma-separated list)
            # Regex pattern matching any of 1, 2, 3 as whole words
            pattern = r"(^|[,;\s])([123])([,;\s]|$)"
            
            mask = filtered_df[UTEROTONIC_COL].astype(str).str.contains(
                pattern, regex=True, na=False
            )
            
            # Additional check for enrollment date if required by this specific KPI logic
            if UTEROTONIC_DATE_COL in filtered_df.columns:
                date_mask = filtered_df[UTEROTONIC_DATE_COL].notna()
                mask = mask & date_mask

            result = int(mask.sum())

    st.session_state.uterotonic_cache[cache_key] = result
    return result

    st.session_state.uterotonic_cache[cache_key] = result
    return result


def compute_uterotonic_by_type(df, facility_uids=None):
    """
    Compute distribution of uterotonic types - VECTORIZED for performance
    """
    if df is None or df.empty:
        return {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if UTEROTONIC_COL not in filtered_df.columns:
        return {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    # Filter out rows without uterotonic date
    if UTEROTONIC_DATE_COL in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[UTEROTONIC_DATE_COL].notna()].copy()

    # Initialize counts
    counts = {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    # Vectorized computation using pandas split and explode
    s = filtered_df[UTEROTONIC_COL].astype(str).str.split(",")
    exploded = s.explode().str.strip()
    dist = exploded.value_counts()

    mapping = {"1": "Ergometrine", "2": "Oxytocin", "3": "Misoprostol"}
    
    total = 0
    for code, count in dist.items():
        if code in mapping:
            name = mapping[code]
            c_val = int(count)
            counts[name] = c_val
            total += c_val

    counts["total"] = total
    return counts


def compute_uterotonic_rate(df, facility_uids=None, date_range_filters=None):
    """
    Compute Uterotonic Administration Rate from patient-level data
    Returns: (rate, uterotonic_cases, total_deliveries)
    """
    cache_key = get_uterotonic_cache_key(df, facility_uids, "uterotonic_rate")

    if cache_key in st.session_state.uterotonic_cache:
        return st.session_state.uterotonic_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Filter data by facility if specified
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Apply date range filtering for uterotonic date
        if date_range_filters and UTEROTONIC_DATE_COL in filtered_df.columns:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                # Convert to datetime
                filtered_df[UTEROTONIC_DATE_COL] = pd.to_datetime(
                    filtered_df[UTEROTONIC_DATE_COL], errors="coerce"
                )

                # Filter by date range
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                filtered_df = filtered_df[
                    (filtered_df[UTEROTONIC_DATE_COL] >= start_dt)
                    & (filtered_df[UTEROTONIC_DATE_COL] < end_dt)
                ].copy()

        # Count Uterotonic cases from filtered data
        uterotonic_cases = compute_uterotonic_count(filtered_df, facility_uids)

        # Get total deliveries using the date column specific to uterotonic KPI
        date_column = get_relevant_date_column_for_kpi(
            "Delivered women who received uterotonic (%)"
        )
        total_deliveries = compute_total_deliveries(
            filtered_df, facility_uids, date_column
        )

        # Calculate rate
        rate = (
            (uterotonic_cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        )
        result = (rate, uterotonic_cases, total_deliveries)

    st.session_state.uterotonic_cache[cache_key] = result
    return result


def compute_uterotonic_kpi(df, facility_uids=None):
    """
    Compute Uterotonic KPI data from patient-level data
    This is the function your dashboard is calling
    """
    # Get date range filters from session state if available
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    rate, uterotonic_cases, total_deliveries = compute_uterotonic_rate(
        df, facility_uids, date_range_filters
    )

    # Get type distribution
    type_counts = compute_uterotonic_by_type(df, facility_uids)

    return {
        "uterotonic_rate": float(rate),
        "uterotonic_cases": int(uterotonic_cases),
        "total_deliveries": int(total_deliveries),
        "uterotonic_types": type_counts,
    }


def get_numerator_denominator_for_uterotonic(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Uterotonic KPI from patient-level data
    WITH DATE RANGE FILTERING
    Returns: (numerator, denominator, rate)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Apply date range filtering on the uterotonic-specific date column
    if date_range_filters and UTEROTONIC_DATE_COL in filtered_df.columns:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            # Convert to datetime
            filtered_df[UTEROTONIC_DATE_COL] = pd.to_datetime(
                filtered_df[UTEROTONIC_DATE_COL], errors="coerce"
            )

            # Filter by date range
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            filtered_df = filtered_df[
                (filtered_df[UTEROTONIC_DATE_COL] >= start_dt)
                & (filtered_df[UTEROTONIC_DATE_COL] < end_dt)
            ].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    # Compute Uterotonic rate on filtered data
    rate, uterotonic_cases, total_deliveries = compute_uterotonic_rate(
        filtered_df, facility_uids, date_range_filters
    )

    return (uterotonic_cases, total_deliveries, rate)


# ---------------- Chart Functions WITH TABLES ----------------
def render_uterotonic_trend_chart(*args, **kwargs):
    """Render trend chart for Uterotonic using standard utility"""
    return render_trend_chart(*args, **kwargs)


def render_uterotonic_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for Uterotonic using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)


def render_uterotonic_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for Uterotonic using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


def render_uterotonic_type_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of uterotonic types from patient-level data"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute uterotonic type distribution
    type_data = compute_uterotonic_by_type(df, facility_uids)

    if type_data["total"] == 0:
        st.info("⚠️ No data available for uterotonic type distribution.")
        return

    # Prepare data for visualization
    pie_data = {
        "Type": ["Oxytocin", "Ergometrine", "Misoprostol"],
        "Count": [
            type_data["Oxytocin"],
            type_data["Ergometrine"],
            type_data["Misoprostol"],
        ],
    }

    pie_df = pd.DataFrame(pie_data)

    # Calculate percentages
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

    # Add chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Pie Chart", "Donut Chart"],
        index=0,
        key=f"uterotonic_chart_type_{str(facility_uids)}",
    )

    # Add CSS for better pie chart layout
    st.markdown(
        """
    <style>
    .pie-chart-container {
        margin-top: -30px;
        margin-bottom: 20px;
    }
    .pie-chart-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create chart
    if chart_type == "Pie Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            color="Type",
            color_discrete_map={
                "Oxytocin": "#ff7f0e",
                "Ergometrine": "#1f77b4",
                "Misoprostol": "#2ca02c",
            },
        )
    else:  # Donut Chart
        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            hole=0.4,
            color="Type",
            color_discrete_map={
                "Oxytocin": "#ff7f0e",
                "Ergometrine": "#1f77b4",
                "Misoprostol": "#2ca02c",
            },
        )

    # Calculate if we should use inside text for small slices
    total_count = pie_df["Count"].sum()
    use_inside_text = any((pie_df["Count"] / total_count) < 0.05)

    if use_inside_text:
        # For small slices, put text inside with white background
        fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textfont=dict(size=10),
            insidetextfont=dict(color="white", size=9),
            outsidetextfont=dict(size=9),
        )
    else:
        # For normal slices, use outside text
        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textfont=dict(size=10),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.3,
            font=dict(size=10),
            itemwidth=30,
        ),
        margin=dict(l=0, r=150, t=20, b=20),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    # Ensure no "undefined" placeholder
    fig.update_layout(title=None)
    fig.layout.pop("title", None)

    # Use container to control layout
    with st.container():
        st.markdown(
            '<div class="pie-chart-title">Distribution of Uterotonic Types</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="uterotonic_distribution_chart")

    # Show summary table
    st.subheader("📋 Uterotonic Type Summary")
    summary_df = pie_df.copy()
    summary_df.insert(0, "No", range(1, len(summary_df) + 1))

    styled_table = (
        summary_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"})
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Add download button for CSV
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="uterotonic_type_distribution.csv",
        mime="text/csv",
    )


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_uterotonic_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Uterotonic trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    SAME AS ASSISTED FUNCTION
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Uterotonic (enrollment_date)
    date_column = get_relevant_date_column_for_kpi(kpi_name)

    # Check if the SPECIFIC date column exists
    if date_column not in filtered_df.columns:
        # Try to use enrollment_date as fallback
        if "enrollment_date" in filtered_df.columns:
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
