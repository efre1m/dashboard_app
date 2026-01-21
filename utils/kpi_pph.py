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
if "pph_cache" not in st.session_state:
    st.session_state.pph_cache = {}


def get_pph_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for PPH computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_pph_cache():
    """Clear the PPH cache"""
    st.session_state.pph_cache = {}


# PPH KPI Configuration
PPH_CONDITION_COL = "obstetric_condition_at_delivery_delivery_summary"
PPH_CODE = "3"

# Obstetric condition codes mapping
OBSTETRIC_CONDITION_CODES = {
    "1": "Severe pre-eclampsia/eclampsia",
    "2": "Antepartum hemorrhage",
    "3": "Postpartum hemorrhage",
    "4": "Sepsis/severe systemic infection",
    "5": "Obstructed labor",
    "6": "Ruptured uterus",
    "7": "Severe anemia",
    "8": "Malaria with severe anemia",
    "9": "HIV-related conditions",
    "10": "Other complications",
}


def compute_pph_count(df, facility_uids=None):
    """
    Count PPH occurrences - "3" alone or in combinations (e.g., "2,3", "3,5")
    EXACT SAME METHOD PATTERN AS compute_csection_count in kpi_utils.py
    """
    cache_key = get_pph_cache_key(df, facility_uids, "pph_count")

    if cache_key in st.session_state.pph_cache:
        return st.session_state.pph_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if PPH_CONDITION_COL not in filtered_df.columns:
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string and handle multiple codes (e.g., "2,3", "3,5,10")
            df_copy["obstetric_condition_clean"] = df_copy[PPH_CONDITION_COL].astype(
                str
            )

            # Check for "3" in the string (standalone or in comma-separated list)
            # Pattern matches: "3" alone, "3," at start, ",3" in middle, ",3," anywhere, or "3" at end
            pph_mask = df_copy["obstetric_condition_clean"].str.contains(
                r"(^|[,;\s])3([,;\s]|$)", regex=True, na=False
            )

            result = int(pph_mask.sum())

    st.session_state.pph_cache[cache_key] = result
    return result


def compute_obstetric_condition_distribution(df, facility_uids=None):
    """
    Compute distribution of obstetric conditions
    Returns: Dictionary with counts for each obstetric condition
    Handles comma-separated values like "2,3" - each code counts separately
    """
    if df is None or df.empty:
        return {name: 0 for name in OBSTETRIC_CONDITION_CODES.values()}

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if PPH_CONDITION_COL not in filtered_df.columns:
        return {name: 0 for name in OBSTETRIC_CONDITION_CODES.values()}

    # Initialize counts
    counts = {code: 0 for code in OBSTETRIC_CONDITION_CODES.keys()}
    named_counts = {name: 0 for name in OBSTETRIC_CONDITION_CODES.values()}

    df_copy = filtered_df.copy()
    df_copy["obstetric_condition_clean"] = df_copy[PPH_CONDITION_COL].astype(str)

    # Process each value
    for value in df_copy["obstetric_condition_clean"]:
        if pd.isna(value) or value == "nan":
            continue

        value_str = str(value).strip()

        # Skip empty values
        if not value_str or value_str.lower() in ["n/a", "nan", "null"]:
            continue

        # Split by comma and count each code
        codes = [code.strip() for code in value_str.split(",")]

        for code in codes:
            if code in OBSTETRIC_CONDITION_CODES:
                counts[code] += 1
                condition_name = OBSTETRIC_CONDITION_CODES[code]
                named_counts[condition_name] += 1

    # Add total count
    total = sum(counts.values())
    named_counts["Total Complications"] = total

    return named_counts


def compute_pph_rate(df, facility_uids=None):
    """
    Compute PPH Rate - EXACT SAME PATTERN AS compute_csection_rate in kpi_utils.py
    Returns: (rate, pph_cases, total_deliveries)
    """
    cache_key = get_pph_cache_key(df, facility_uids, "pph_rate")

    if cache_key in st.session_state.pph_cache:
        return st.session_state.pph_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for PPH (enrollment_date)
        date_column = get_relevant_date_column_for_kpi(
            "Postpartum Hemorrhage (PPH) Rate (%)"
        )

        # Count PPH cases (value "3" in obstetric condition)
        pph_cases = compute_pph_count(df, facility_uids)

        # Get total deliveries - USING SAME LOGIC AS C-SECTION
        total_deliveries = compute_total_deliveries(df, facility_uids, date_column)

        # Calculate rate
        rate = (pph_cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (rate, pph_cases, total_deliveries)

    st.session_state.pph_cache[cache_key] = result
    return result


def compute_pph_kpi(df, facility_uids=None):
    """
    Compute PPH KPI data - SAME STRUCTURE AS compute_csection_rate
    """
    rate, pph_cases, total_deliveries = compute_pph_rate(df, facility_uids)

    # Get obstetric condition distribution
    condition_distribution = compute_obstetric_condition_distribution(df, facility_uids)

    return {
        "pph_rate": float(rate),
        "pph_cases": int(pph_cases),
        "total_deliveries": int(total_deliveries),
        "obstetric_conditions": condition_distribution,
    }


def get_numerator_denominator_for_pph(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for PPH KPI - SAME PATTERN AS C-SECTION
    WITH DATE RANGE FILTERING
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for PPH (enrollment_date)
    date_column = get_relevant_date_column_for_kpi(
        "Postpartum Hemorrhage (PPH) Rate (%)"
    )

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

    # Compute PPH rate on date-filtered data
    rate, pph_cases, total_deliveries = compute_pph_rate(filtered_df, facility_uids)

    return (pph_cases, total_deliveries, rate)


# ---------------- Pie Chart Function ----------------
def render_obstetric_condition_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """
    Render a pie chart showing distribution of obstetric conditions
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute obstetric condition distribution
    condition_data = compute_obstetric_condition_distribution(df, facility_uids)

    if condition_data["Total Complications"] == 0:
        st.info("⚠️ No data available for obstetric condition distribution.")
        return

    # Prepare data for visualization (exclude "Total Complications" from pie chart)
    pie_data = []
    for condition_name, count in condition_data.items():
        if condition_name != "Total Complications" and count > 0:
            pie_data.append({"Condition": condition_name, "Count": count})

    if not pie_data:
        st.info("⚠️ No obstetric condition data available.")
        return

    pie_df = pd.DataFrame(pie_data)

    # Sort by count (descending)
    pie_df = pie_df.sort_values("Count", ascending=False)

    # Calculate percentages
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

    # Add chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Pie Chart", "Donut Chart", "Bar Chart"],
        index=0,
        key=f"obstetric_chart_type_{str(facility_uids)}",
    )

    # Add CSS for better layout
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
            names="Condition",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            color="Condition",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    elif chart_type == "Donut Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Condition",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            hole=0.4,
            color="Condition",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    else:  # Bar Chart
        fig = px.bar(
            pie_df,
            x="Condition",
            y="Count",
            color="Condition",
            title="Obstetric Condition Distribution",
            height=500,
            color_discrete_sequence=px.colors.qualitative.Set3,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Obstetric Condition",
            yaxis_title="Count",
            xaxis_tickangle=-45,
            showlegend=False,
        )

    if chart_type in ["Pie Chart", "Donut Chart"]:
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
    if chart_type in ["Pie Chart", "Donut Chart"]:
        fig.update_layout(title=None)
        fig.layout.pop("title", None)

    # Use container to control layout
    with st.container():
        if chart_type in ["Pie Chart", "Donut Chart"]:
            st.markdown(
                '<div class="pie-chart-title">Distribution of Obstetric Conditions</div>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 Obstetric Condition Summary")

    # Create summary dataframe with all conditions
    summary_data = []
    for condition_name, count in condition_data.items():
        if condition_name != "Total Complications":
            percentage = (
                (count / condition_data["Total Complications"] * 100)
                if condition_data["Total Complications"] > 0
                else 0
            )
            summary_data.append(
                {"Condition": condition_name, "Count": count, "Percentage": percentage}
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df[summary_df["Count"] > 0].sort_values(
        "Count", ascending=False
    )

    if not summary_df.empty:
        summary_df.insert(0, "No", range(1, len(summary_df) + 1))

        styled_table = (
            summary_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"})
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_table.to_html(), unsafe_allow_html=True)

        # Add total row
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Complications", f"{condition_data['Total Complications']:,.0f}"
            )
        with col2:
            st.metric(
                "PPH Cases", f"{condition_data.get('Postpartum hemorrhage', 0):,.0f}"
            )

        # Add download button for CSV
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="obstetric_condition_distribution.csv",
            mime="text/csv",
        )
    else:
        st.info("No obstetric condition data available.")


# ---------------- Chart Functions WITH TABLES ----------------
def render_pph_trend_chart(*args, **kwargs):
    """Render trend chart for PPH using standard utility"""
    return render_trend_chart(*args, **kwargs)


def render_pph_facility_comparison_chart(*args, **kwargs):
    """Render facility comparison chart for PPH using standard utility"""
    return render_facility_comparison_chart(*args, **kwargs)


def render_pph_region_comparison_chart(*args, **kwargs):
    """Render region comparison chart for PPH using standard utility"""
    return render_region_comparison_chart(*args, **kwargs)


# ---------------- Additional Helper Functions ----------------
def prepare_pph_data_for_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for PPH trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for PPH (enrollment_date)
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
