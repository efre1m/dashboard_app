import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- Hypothermia KPI Constants ----------------
TEMPERATURE_ON_ADMISSION_UID = "gZi9y12E9i7"  # Temperature on admission (°C)
DATE_OF_ADMISSION_UID = "UOmhJkyAK6h"  # Date of Admission
HYPOTHERMIA_THRESHOLD = 36.5  # in degree Celcius


# ---------------- Hypothermia KPI Computation Functions ----------------
def compute_hypothermia_numerator(df, facility_uids=None):
    """
    Compute numerator for hypothermia KPI: Count of newborns with temperature < 36.5°C on admission

    Formula: Count where Temperature on admission < 36.5
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for temperature on admission events and convert to numeric
    temp_events = df[
        (df["dataElement_uid"] == TEMPERATURE_ON_ADMISSION_UID) & df["value"].notna()
    ].copy()

    if temp_events.empty:
        return 0

    # Convert temperature values to numeric, coercing errors to NaN
    temp_events["temp_value"] = pd.to_numeric(temp_events["value"], errors="coerce")

    # Count newborns with temperature < 36.5°C
    hypothermia_cases = temp_events[temp_events["temp_value"] < HYPOTHERMIA_THRESHOLD][
        "tei_id"
    ].nunique()

    return hypothermia_cases


def compute_hypothermia_denominator(df, facility_uids=None):
    """
    Compute denominator for hypothermia KPI: Total count of newborns admitted

    Formula: Count where Date of Admission is present
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count newborns with admission date (each date corresponds to a single newborn admission)
    admitted_newborns = df[
        (df["dataElement_uid"] == DATE_OF_ADMISSION_UID) & df["value"].notna()
    ]["tei_id"].nunique()

    return admitted_newborns


def compute_hypothermia_kpi(df, facility_uids=None):
    """
    Compute hypothermia KPI for the given dataframe

    Formula: Hypothermia on Admission Rate (%) =
             (Newborns with temperature < 36.5°C on admission) ÷ (Total newborns admitted) × 100

    Returns:
        Dictionary with hypothermia metrics
    """
    # Handle case where df might be None or not a DataFrame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "hypothermia_rate": 0.0,
            "hypothermia_count": 0,
            "total_admissions": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        # Handle both single facility UID and list of UIDs
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count hypothermia cases (numerator)
    hypothermia_count = compute_hypothermia_numerator(df, facility_uids)

    # Count total admissions (denominator)
    total_admissions = compute_hypothermia_denominator(df, facility_uids)

    # Calculate hypothermia rate
    hypothermia_rate = (
        (hypothermia_count / total_admissions * 100) if total_admissions > 0 else 0.0
    )

    return {
        "hypothermia_rate": float(hypothermia_rate),
        "hypothermia_count": int(hypothermia_count),
        "total_admissions": int(total_admissions),
    }


def compute_hypothermia_trend_data(df, period_col="period_display", facility_uids=None):
    """
    Compute hypothermia trend data by period

    Returns:
        DataFrame with columns: period_display, total_admissions, hypothermia_count, hypothermia_rate
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Ensure period column exists
    if period_col not in df.columns:
        return pd.DataFrame()

    trend_data = []

    for period in df[period_col].unique():
        period_df = df[df[period_col] == period]
        hypothermia_data = compute_hypothermia_kpi(period_df, facility_uids)

        trend_data.append(
            {
                period_col: period,
                "total_admissions": hypothermia_data["total_admissions"],
                "hypothermia_count": hypothermia_data["hypothermia_count"],
                "hypothermia_rate": hypothermia_data["hypothermia_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- Hypothermia Chart Functions ----------------
def render_hypothermia_trend_chart(
    df,
    period_col="period_display",
    value_col="hypothermia_rate",
    title="Hypothermia on Admission Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="Hypothermia Cases",
    denominator_name="Total Admissions",
    facility_uids=None,
):
    """Render a LINE CHART ONLY for hypothermia rate trend"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Create line chart
    fig = px.line(
        df,
        x=period_col,
        y=value_col,
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
    )

    # Update traces for line chart - SIMPLIFIED HOVER
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Hypothermia Rate: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Hypothermia on Admission (%)",
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

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Show trend indicator
    if len(df) > 1:
        last_value = df[value_col].iloc[-1]
        prev_value = df[value_col].iloc[-2]
        trend_symbol = (
            "▲"
            if last_value > prev_value
            else ("▼" if last_value < prev_value else "–")
        )
        trend_class = (
            "trend-up"
            if last_value > prev_value
            else ("trend-down" if last_value < prev_value else "trend-neutral")
        )
        st.markdown(
            f'<p style="font-size:1.2rem;font-weight:600;">Latest Value: {last_value:.2f}% <span class="{trend_class}">{trend_symbol}</span></p>',
            unsafe_allow_html=True,
        )

    # Summary table
    st.subheader(f"📋 {title} Summary Table")
    summary_df = df.copy().reset_index(drop=True)
    summary_df = summary_df[[period_col, numerator_name, denominator_name, value_col]]

    # Calculate overall value
    total_numerator = summary_df[numerator_name].sum()
    total_denominator = summary_df[denominator_name].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = pd.DataFrame(
        {
            period_col: [f"Overall {title}"],
            numerator_name: [total_numerator],
            denominator_name: [total_denominator],
            value_col: [overall_value],
        }
    )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Format table
    styled_table = (
        summary_table.style.format(
            {
                value_col: "{:.1f}%",
                numerator_name: "{:,.0f}",
                denominator_name: "{:,.0f}",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_summary.csv",
        mime="text/csv",
    )


def render_hypothermia_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="hypothermia_rate",
    title="Hypothermia on Admission - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="Hypothermia Cases",
    denominator_name="Total Admissions",
):
    """Render facility comparison with BOTH LINE AND BAR CHART options for hypothermia rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if (
        not facility_names
        or not facility_uids
        or len(facility_names) != len(facility_uids)
    ):
        st.info("⚠️ No facilities selected for comparison.")
        return

    # Create mapping
    facility_uid_to_name = dict(zip(facility_uids, facility_names))

    # Chart options with radio button
    chart_options = ["Line Chart", "Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_facility_comparison_{str(facility_uids)}",
    )

    if chart_type == "Line Chart":
        # Compute time series data for line chart - process each facility separately
        time_series_data = []

        # Get all unique periods from the data
        all_periods = sorted(df["period_display"].unique())

        for period_display in all_periods:
            period_df = df[df["period_display"] == period_display]

            for facility_uid in facility_uids:
                # Filter data for this specific facility and period
                facility_period_df = period_df[period_df["orgUnit"] == facility_uid]

                if not facility_period_df.empty:
                    # Compute KPI for this specific facility and period
                    hypothermia_data = compute_hypothermia_kpi(
                        facility_period_df, [facility_uid]
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": hypothermia_data["hypothermia_rate"],
                        }
                    )

        if not time_series_data:
            st.info("⚠️ No time series data available for facility comparison.")
            return

        time_series_df = pd.DataFrame(time_series_data)

        # Create line chart
        fig = px.line(
            time_series_df,
            x="period_display",
            y="value",
            color="Facility",
            markers=True,
            title=title,
            height=500,
            category_orders={"period_display": all_periods},
        )

        # Update traces - SIMPLIFIED HOVER: Only show facility name and rate
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<extra></extra>",
        )

    else:  # Bar Chart
        # For bar chart, compute overall values for each facility across all periods
        bar_data = []
        for facility_uid in facility_uids:
            # Use ALL data for this facility (not filtered by period)
            facility_df = df[df["orgUnit"] == facility_uid]
            if not facility_df.empty:
                hypothermia_data = compute_hypothermia_kpi(facility_df, [facility_uid])
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": hypothermia_data["hypothermia_rate"],
                    }
                )

        if not bar_data:
            st.info("⚠️ No data available for bar chart.")
            return

        bar_df = pd.DataFrame(bar_data)

        # Create bar chart
        fig = px.bar(
            bar_df,
            x="Facility",
            y="value",
            title=title,
            height=500,
            color="Facility",
        )

        # Update traces for bar chart - SIMPLIFIED HOVER: Only show facility name and rate
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Hypothermia Rate: %{y:.2f}%<extra></extra>",
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="Hypothermia on Admission (%)",
        xaxis=dict(
            type="category",
            tickangle=-45 if chart_type == "Line Chart" else 0,
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
        showlegend=True,
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Facility comparison table
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            hypothermia_data = compute_hypothermia_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Hypothermia Cases": hypothermia_data["hypothermia_count"],
                    "Total Admissions": hypothermia_data["total_admissions"],
                    "Hypothermia Rate (%)": hypothermia_data["hypothermia_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Hypothermia Cases"].sum()
    total_denominator = facility_table_df["Total Admissions"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "Hypothermia Cases": total_numerator,
        "Total Admissions": total_denominator,
        "Hypothermia Rate (%)": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Hypothermia Cases": "{:,.0f}",
                "Total Admissions": "{:,.0f}",
                "Hypothermia Rate (%)": "{:.2f}%",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = facility_table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hypothermia_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_hypothermia_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="hypothermia_rate",
    title="Hypothermia on Admission - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="Hypothermia Cases",
    denominator_name="Total Admissions",
):
    """Render region comparison with BOTH LINE AND BAR CHART options for hypothermia rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # Chart options with radio button
    chart_options = ["Line Chart", "Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_region_comparison_{str(region_names)}",
    )

    if chart_type == "Line Chart":
        # Compute time series data for line chart - process each region separately
        time_series_data = []

        # Get all unique periods from the data
        all_periods = sorted(df["period_display"].unique())

        for period_display in all_periods:
            period_df = df[df["period_display"] == period_display]

            for region_name in region_names:
                region_facility_uids = [
                    uid for _, uid in facilities_by_region.get(region_name, [])
                ]

                if region_facility_uids:
                    # Filter data for this specific region and period
                    region_period_df = period_df[
                        period_df["orgUnit"].isin(region_facility_uids)
                    ]

                    if not region_period_df.empty:
                        # Compute KPI for this specific region and period
                        hypothermia_data = compute_hypothermia_kpi(
                            region_period_df, region_facility_uids
                        )
                        time_series_data.append(
                            {
                                "period_display": period_display,
                                "Region": region_name,
                                "value": hypothermia_data["hypothermia_rate"],
                            }
                        )

        if not time_series_data:
            st.info("⚠️ No time series data available for region comparison.")
            return

        time_series_df = pd.DataFrame(time_series_data)

        # Create line chart
        fig = px.line(
            time_series_df,
            x="period_display",
            y="value",
            color="Region",
            markers=True,
            title=title,
            height=500,
            category_orders={"period_display": all_periods},
        )

        # Update traces - SIMPLIFIED HOVER: Only show region name and rate
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<extra></extra>",
        )

    else:  # Bar Chart
        # For bar chart, compute overall values for each region across all periods
        bar_data = []
        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

            if region_facility_uids:
                # Use ALL data for this region (not filtered by period)
                region_df = df[df["orgUnit"].isin(region_facility_uids)]

                if not region_df.empty:
                    hypothermia_data = compute_hypothermia_kpi(
                        region_df, region_facility_uids
                    )
                    bar_data.append(
                        {
                            "Region": region_name,
                            "value": hypothermia_data["hypothermia_rate"],
                        }
                    )

        if not bar_data:
            st.info("⚠️ No data available for bar chart.")
            return

        bar_df = pd.DataFrame(bar_data)

        # Create bar chart
        fig = px.bar(
            bar_df,
            x="Region",
            y="value",
            title=title,
            height=500,
            color="Region",
        )

        # Update traces for bar chart - SIMPLIFIED HOVER: Only show region name and rate
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Hypothermia Rate: %{y:.2f}%<extra></extra>",
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="Hypothermia on Admission (%)",
        xaxis=dict(
            type="category",
            tickangle=-45 if chart_type == "Line Chart" else 0,
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
        showlegend=True,
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Region comparison table
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]

        if region_facility_uids:
            region_df = df[df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                hypothermia_data = compute_hypothermia_kpi(
                    region_df, region_facility_uids
                )
                region_table_data.append(
                    {
                        "Region Name": region_name,
                        "Hypothermia Cases": hypothermia_data["hypothermia_count"],
                        "Total Admissions": hypothermia_data["total_admissions"],
                        "Hypothermia Rate (%)": hypothermia_data["hypothermia_rate"],
                    }
                )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Hypothermia Cases"].sum()
    total_denominator = region_table_df["Total Admissions"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "Hypothermia Cases": total_numerator,
        "Total Admissions": total_denominator,
        "Hypothermia Rate (%)": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Hypothermia Cases": "{:,.0f}",
                "Total Admissions": "{:,.0f}",
                "Hypothermia Rate (%)": "{:.2f}%",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = region_table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hypothermia_rate_region_comparison.csv",
        mime="text/csv",
    )
