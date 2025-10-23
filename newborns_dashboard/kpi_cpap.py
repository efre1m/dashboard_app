import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- CPAP KPI Constants ----------------
FIRST_REASON_ADMISSION_UID = "T30GbTiVgFR"  # First Reason for Admission
SECOND_REASON_ADMISSION_UID = "OpHw2X58x5i"  # Second Reason for Admission
THIRD_REASON_ADMISSION_UID = "gJH6PkYI6IV"  # Third Reason for Admission
CPAP_ADMINISTERED_UID = "wlHEf9FdmJM"  # CPAP Administered data element

RDS_VALUE = "5"  # RDS diagnosis code
CPAP_YES_VALUE = "1"  # CPAP Administered = Yes


# ---------------- CPAP KPI Computation Functions ----------------
def compute_cpap_numerator(df, facility_uids=None):
    """
    Compute numerator for CPAP KPI: Count of newborns with RDS who received CPAP

    Formula: Count where:
        - RDS = Yes (in any of the three admission reason fields)
        - CPAP Administered = 1 (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # First, identify newborns with RDS diagnosis
    rds_newborns = df[
        (
            (df["dataElement_uid"] == FIRST_REASON_ADMISSION_UID)
            | (df["dataElement_uid"] == SECOND_REASON_ADMISSION_UID)
            | (df["dataElement_uid"] == THIRD_REASON_ADMISSION_UID)
        )
        & (df["value"] == RDS_VALUE)
    ]["tei_id"].unique()

    if len(rds_newborns) == 0:
        return 0

    # Filter for RDS newborns and check CPAP administration
    rds_df = df[df["tei_id"].isin(rds_newborns)]

    # Count newborns with CPAP administered = Yes
    cpap_cases = rds_df[
        (rds_df["dataElement_uid"] == CPAP_ADMINISTERED_UID)
        & (rds_df["value"] == CPAP_YES_VALUE)
    ]["tei_id"].nunique()

    return cpap_cases


def compute_cpap_denominator(df, facility_uids=None):
    """
    Compute denominator for CPAP KPI: Total count of newborns with RDS diagnosis

    Formula: Count where RDS = Yes in any of the three admission reason fields
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count newborns with RDS diagnosis (any of the three admission reasons)
    rds_newborns = df[
        (
            (df["dataElement_uid"] == FIRST_REASON_ADMISSION_UID)
            | (df["dataElement_uid"] == SECOND_REASON_ADMISSION_UID)
            | (df["dataElement_uid"] == THIRD_REASON_ADMISSION_UID)
        )
        & (df["value"] == RDS_VALUE)
    ]["tei_id"].nunique()

    return rds_newborns


def compute_cpap_kpi(df, facility_uids=None):
    """
    Compute CPAP KPI for the given dataframe

    Formula: CPAP Coverage for RDS (%) =
             (RDS newborns who received CPAP) ÷ (Total RDS newborns) × 100

    Returns:
        Dictionary with CPAP metrics
    """
    # Handle case where df might be None or not a DataFrame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "cpap_rate": 0.0,
            "cpap_count": 0,
            "total_rds": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        # Handle both single facility UID and list of UIDs
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count CPAP cases (numerator)
    cpap_count = compute_cpap_numerator(df, facility_uids)

    # Count total RDS newborns (denominator)
    total_rds = compute_cpap_denominator(df, facility_uids)

    # Calculate CPAP rate
    cpap_rate = (cpap_count / total_rds * 100) if total_rds > 0 else 0.0

    return {
        "cpap_rate": float(cpap_rate),
        "cpap_count": int(cpap_count),
        "total_rds": int(total_rds),
    }


def compute_cpap_trend_data(df, period_col="period_display", facility_uids=None):
    """
    Compute CPAP trend data by period

    Returns:
        DataFrame with columns: period_display, total_rds, cpap_count, cpap_rate
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
        cpap_data = compute_cpap_kpi(period_df, facility_uids)

        trend_data.append(
            {
                period_col: period,
                "total_rds": cpap_data["total_rds"],
                "cpap_count": cpap_data["cpap_count"],
                "cpap_rate": cpap_data["cpap_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- CPAP Chart Functions ----------------
def render_cpap_trend_chart(
    df,
    period_col="period_display",
    value_col="cpap_rate",
    title="CPAP Coverage for RDS Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="CPAP Cases",
    denominator_name="Total RDS Newborns",
    facility_uids=None,
):
    """Render a LINE CHART ONLY for CPAP coverage rate trend"""
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

    # Update traces for line chart
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>CPAP Coverage: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage for RDS (%)",
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


def render_cpap_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="cpap_rate",
    title="CPAP Coverage for RDS - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="CPAP Cases",
    denominator_name="Total RDS Newborns",
):
    """Render facility comparison LINE CHART ONLY for CPAP coverage"""
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
    filtered_df = df[df["orgUnit"].isin(facility_uids)].copy()

    if filtered_df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Compute time series data for line chart
    time_series_data = []
    all_periods = (
        filtered_df[["period_display", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")
    )
    period_order = all_periods["period_display"].tolist()

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
            if not facility_period_df.empty:
                cpap_data = compute_cpap_kpi(facility_period_df, [facility_uid])
                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": cpap_data["cpap_rate"],
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
        category_orders={"period_display": period_order},
    )

    # Update traces
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<extra></extra>",
    )

    # Layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage for RDS (%)",
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
            cpap_data = compute_cpap_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "CPAP Cases": cpap_data["cpap_count"],
                    "Total RDS Newborns": cpap_data["total_rds"],
                    "CPAP Rate (%)": cpap_data["cpap_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["CPAP Cases"].sum()
    total_denominator = facility_table_df["Total RDS Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "CPAP Cases": total_numerator,
        "Total RDS Newborns": total_denominator,
        "CPAP Rate (%)": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "CPAP Cases": "{:,.0f}",
                "Total RDS Newborns": "{:,.0f}",
                "CPAP Rate (%)": "{:.2f}%",
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
        file_name="cpap_coverage_facility_comparison.csv",
        mime="text/csv",
    )


def render_cpap_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="cpap_rate",
    title="CPAP Coverage for RDS - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="CPAP Cases",
    denominator_name="Total RDS Newborns",
):
    """Render region comparison LINE CHART ONLY for CPAP coverage"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # Get all facility UIDs for selected regions
    all_facility_uids = []
    for region_name in region_names:
        facility_uids = [uid for _, uid in facilities_by_region.get(region_name, [])]
        all_facility_uids.extend(facility_uids)

    filtered_df = df[df["orgUnit"].isin(all_facility_uids)].copy()

    if filtered_df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Compute time series data for line chart
    time_series_data = []
    all_periods = (
        filtered_df[["period_display", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")
    )
    period_order = all_periods["period_display"].tolist()

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_period_df = period_df[
                period_df["orgUnit"].isin(region_facility_uids)
            ]

            if not region_period_df.empty:
                cpap_data = compute_cpap_kpi(region_period_df, region_facility_uids)
                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Region": region_name,
                        "value": cpap_data["cpap_rate"],
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
        category_orders={"period_display": period_order},
    )

    # Update traces
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<extra></extra>",
    )

    # Layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="CPAP Coverage for RDS (%)",
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
        region_df = df[df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            cpap_data = compute_cpap_kpi(region_df, region_facility_uids)
            region_table_data.append(
                {
                    "Region Name": region_name,
                    "CPAP Cases": cpap_data["cpap_count"],
                    "Total RDS Newborns": cpap_data["total_rds"],
                    "CPAP Rate (%)": cpap_data["cpap_rate"],
                }
            )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["CPAP Cases"].sum()
    total_denominator = region_table_df["Total RDS Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "CPAP Cases": total_numerator,
        "Total RDS Newborns": total_denominator,
        "CPAP Rate (%)": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "CPAP Cases": "{:,.0f}",
                "Total RDS Newborns": "{:,.0f}",
                "CPAP Rate (%)": "{:.2f}%",
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
        file_name="cpap_coverage_region_comparison.csv",
        mime="text/csv",
    )
