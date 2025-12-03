# kpi_kmc_1000_2499.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- KMC KPI Constants ----------------
WEIGHT_ON_ADMISSION_UID = "yxWUMt3sCil"  # Weight on admission
KMC_ADMINISTERED_UID = "QK7Fi6OwtDC"  # KMC Administered data element
KMC_YES_VALUE = "1"  # KMC Administered = Yes


# ---------------- KMC 1000-1999g KPI Computation Functions ----------------
def compute_kmc_1000_1999_numerator(df, facility_uids=None):
    """
    Compute numerator for KMC 1000-1999g KPI: Count of newborns (1000-1999g) who received KMC

    Formula: Count where:
        - Weight on admission between 1000-1999 g
        - KMC Administered = 1 (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # First, identify newborns in weight range (1000-1999g)
    target_newborns = df[
        (df["dataElement_uid"] == WEIGHT_ON_ADMISSION_UID)
        & df["value"].notna()
        & (pd.to_numeric(df["value"], errors="coerce") >= 1000)
        & (pd.to_numeric(df["value"], errors="coerce") <= 1999)
    ]["tei_id"].unique()

    if len(target_newborns) == 0:
        return 0

    # Filter for target newborns and check KMC administration
    target_df = df[df["tei_id"].isin(target_newborns)]

    # Count newborns with KMC administered = Yes
    kmc_cases = target_df[
        (target_df["dataElement_uid"] == KMC_ADMINISTERED_UID)
        & (target_df["value"] == KMC_YES_VALUE)
    ]["tei_id"].nunique()

    return kmc_cases


def compute_kmc_1000_1999_denominator(df, facility_uids=None):
    """
    Compute denominator for KMC 1000-1999g KPI: Total count of newborns (1000-1999g)

    Formula: Count where Weight on admission between 1000-1999 g
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count newborns in weight range (1000-1999g)
    target_newborns = df[
        (df["dataElement_uid"] == WEIGHT_ON_ADMISSION_UID)
        & df["value"].notna()
        & (pd.to_numeric(df["value"], errors="coerce") >= 1000)
        & (pd.to_numeric(df["value"], errors="coerce") <= 1999)
    ]["tei_id"].nunique()

    return target_newborns


def compute_kmc_1000_1999_kpi(df, facility_uids=None):
    """
    Compute KMC 1000-1999g KPI for the given dataframe

    Formula: KMC Coverage 1000-1999g (%) =
             (Newborns 1000-1999g who received KMC) ÷ (Total newborns 1000-1999g) × 100

    Returns:
        Dictionary with KMC metrics
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "kmc_rate": 0.0,
            "kmc_count": 0,
            "total_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count KMC cases (numerator)
    kmc_count = compute_kmc_1000_1999_numerator(df, facility_uids)

    # Count total newborns in weight range (denominator)
    total_newborns = compute_kmc_1000_1999_denominator(df, facility_uids)

    # Calculate KMC rate
    kmc_rate = (kmc_count / total_newborns * 100) if total_newborns > 0 else 0.0

    return {
        "kmc_rate": float(kmc_rate),
        "kmc_count": int(kmc_count),
        "total_newborns": int(total_newborns),
    }


def compute_kmc_1000_1999_trend_data(
    df, period_col="period_display", facility_uids=None
):
    """
    Compute KMC 1000-1999g trend data by period

    Returns:
        DataFrame with columns: period_display, total_newborns, kmc_count, kmc_rate
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
        kmc_data = compute_kmc_1000_1999_kpi(period_df, facility_uids)

        trend_data.append(
            {
                period_col: period,
                "total_newborns": kmc_data["total_newborns"],
                "kmc_count": kmc_data["kmc_count"],
                "kmc_rate": kmc_data["kmc_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- KMC 2000-2499g KPI Computation Functions ----------------
def compute_kmc_2000_2499_numerator(df, facility_uids=None):
    """
    Compute numerator for KMC 2000-2499g KPI: Count of newborns (2000-2499g) who received KMC

    Formula: Count where:
        - Weight on admission between 2000-2499 g
        - KMC Administered = 1 (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # First, identify newborns in weight range (2000-2499g)
    target_newborns = df[
        (df["dataElement_uid"] == WEIGHT_ON_ADMISSION_UID)
        & df["value"].notna()
        & (pd.to_numeric(df["value"], errors="coerce") >= 2000)
        & (pd.to_numeric(df["value"], errors="coerce") <= 2499)
    ]["tei_id"].unique()

    if len(target_newborns) == 0:
        return 0

    # Filter for target newborns and check KMC administration
    target_df = df[df["tei_id"].isin(target_newborns)]

    # Count newborns with KMC administered = Yes
    kmc_cases = target_df[
        (target_df["dataElement_uid"] == KMC_ADMINISTERED_UID)
        & (target_df["value"] == KMC_YES_VALUE)
    ]["tei_id"].nunique()

    return kmc_cases


def compute_kmc_2000_2499_denominator(df, facility_uids=None):
    """
    Compute denominator for KMC 2000-2499g KPI: Total count of newborns (2000-2499g)

    Formula: Count where Weight on admission between 2000-2499 g
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count newborns in weight range (2000-2499g)
    target_newborns = df[
        (df["dataElement_uid"] == WEIGHT_ON_ADMISSION_UID)
        & df["value"].notna()
        & (pd.to_numeric(df["value"], errors="coerce") >= 2000)
        & (pd.to_numeric(df["value"], errors="coerce") <= 2499)
    ]["tei_id"].nunique()

    return target_newborns


def compute_kmc_2000_2499_kpi(df, facility_uids=None):
    """
    Compute KMC 2000-2499g KPI for the given dataframe

    Formula: KMC Coverage 2000-2499g (%) =
             (Newborns 2000-2499g who received KMC) ÷ (Total newborns 2000-2499g) × 100

    Returns:
        Dictionary with KMC metrics
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "kmc_rate": 0.0,
            "kmc_count": 0,
            "total_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count KMC cases (numerator)
    kmc_count = compute_kmc_2000_2499_numerator(df, facility_uids)

    # Count total newborns in weight range (denominator)
    total_newborns = compute_kmc_2000_2499_denominator(df, facility_uids)

    # Calculate KMC rate
    kmc_rate = (kmc_count / total_newborns * 100) if total_newborns > 0 else 0.0

    return {
        "kmc_rate": float(kmc_rate),
        "kmc_count": int(kmc_count),
        "total_newborns": int(total_newborns),
    }


def compute_kmc_2000_2499_trend_data(
    df, period_col="period_display", facility_uids=None
):
    """
    Compute KMC 2000-2499g trend data by period

    Returns:
        DataFrame with columns: period_display, total_newborns, kmc_count, kmc_rate
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
        kmc_data = compute_kmc_2000_2499_kpi(period_df, facility_uids)

        trend_data.append(
            {
                period_col: period,
                "total_newborns": kmc_data["total_newborns"],
                "kmc_count": kmc_data["kmc_count"],
                "kmc_rate": kmc_data["kmc_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- COMBINED CHART FUNCTIONS ----------------
def render_kmc_both_ranges_trend_chart(
    df,
    period_col="period_display",
    title="KMC Coverage by Birth Weight Range",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """
    Render a combined line chart showing BOTH KMC 1000-1999g and 2000-2499g trends
    WITH BOTH CATEGORIES SHOWN SIDE-BY-SIDE IN TABLE
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Filter by facilities if specified
    filtered_df = df.copy()
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)]

    if filtered_df.empty:
        st.info("⚠️ No data available for the selected facilities.")
        return

    # Get unique periods sorted
    periods = sorted(filtered_df[period_col].unique())

    # Prepare data for both ranges
    trend_data = []
    table_data = []

    for period in periods:
        period_df = filtered_df[filtered_df[period_col] == period]

        # Compute KMC 1000-1999g
        kmc_1000_1999_data = compute_kmc_1000_1999_kpi(period_df)

        # Compute KMC 2000-2499g
        kmc_2000_2499_data = compute_kmc_2000_2499_kpi(period_df)

        # For chart (both ranges separately)
        trend_data.append(
            {
                period_col: period,
                "Range": "1000-1999g",
                "KMC Rate (%)": kmc_1000_1999_data["kmc_rate"],
                "KMC Cases": kmc_1000_1999_data["kmc_count"],
                "Total Newborns": kmc_1000_1999_data["total_newborns"],
            }
        )

        trend_data.append(
            {
                period_col: period,
                "Range": "2000-2499g",
                "KMC Rate (%)": kmc_2000_2499_data["kmc_rate"],
                "KMC Cases": kmc_2000_2499_data["kmc_count"],
                "Total Newborns": kmc_2000_2499_data["total_newborns"],
            }
        )

        # For table (both ranges side-by-side)
        table_data.append(
            {
                "Period": period,
                # 1000-1999g columns
                "1000-1999g KMC Cases": kmc_1000_1999_data["kmc_count"],
                "1000-1999g Total Newborns": kmc_1000_1999_data["total_newborns"],
                "1000-1999g KMC Rate (%)": kmc_1000_1999_data["kmc_rate"],
                # 2000-2499g columns
                "2000-2499g KMC Cases": kmc_2000_2499_data["kmc_count"],
                "2000-2499g Total Newborns": kmc_2000_2499_data["total_newborns"],
                "2000-2499g KMC Rate (%)": kmc_2000_2499_data["kmc_rate"],
            }
        )

    trend_df = pd.DataFrame(trend_data)

    if trend_df.empty:
        st.info("⚠️ No trend data available.")
        return

    # Create line chart with both ranges
    fig = px.line(
        trend_df,
        x=period_col,
        y="KMC Rate (%)",
        color="Range",
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
        category_orders={period_col: periods},
    )

    # Customize line styles: solid for 1000-1999g, dashed for 2000-2499g
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>Cases: %{customdata[0]}/%{customdata[1]}<extra></extra>",
        customdata=trend_df[["KMC Cases", "Total Newborns"]].values,
    )

    # Set dashed line for 2000-2499g range
    for i, trace in enumerate(fig.data):
        if trace.name == "2000-2499g":
            trace.line.dash = "dash"

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="KMC Coverage (%)",
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
            title="Birth Weight Range", yanchor="top", y=0.99, xanchor="right", x=0.99
        ),
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Show summary table for both ranges
    st.subheader("📋 KMC Coverage Trend by Period")

    if not table_data:
        st.info("⚠️ No data available for trend table.")
        return

    table_df = pd.DataFrame(table_data)

    if not table_df.empty:
        # Calculate overall totals by SUMMING period totals (NOT recalculating from entire dataset)
        overall_1000_1999_kmc_cases = table_df["1000-1999g KMC Cases"].sum()
        overall_1000_1999_total = table_df["1000-1999g Total Newborns"].sum()
        overall_1000_1999_rate = (
            (overall_1000_1999_kmc_cases / overall_1000_1999_total * 100)
            if overall_1000_1999_total > 0
            else 0.0
        )

        overall_2000_2499_kmc_cases = table_df["2000-2499g KMC Cases"].sum()
        overall_2000_2499_total = table_df["2000-2499g Total Newborns"].sum()
        overall_2000_2499_rate = (
            (overall_2000_2499_kmc_cases / overall_2000_2499_total * 100)
            if overall_2000_2499_total > 0
            else 0.0
        )

        # Add overall row
        overall_row = {
            "Period": "Overall",
            "1000-1999g KMC Cases": overall_1000_1999_kmc_cases,
            "1000-1999g Total Newborns": overall_1000_1999_total,
            "1000-1999g KMC Rate (%)": overall_1000_1999_rate,
            "2000-2499g KMC Cases": overall_2000_2499_kmc_cases,
            "2000-2499g Total Newborns": overall_2000_2499_total,
            "2000-2499g KMC Rate (%)": overall_2000_2499_rate,
        }

        table_df = pd.concat([table_df, pd.DataFrame([overall_row])], ignore_index=True)
        table_df.insert(0, "No", range(1, len(table_df) + 1))

        # Format table
        styled_table = (
            table_df.style.format(
                {
                    "1000-1999g KMC Cases": "{:,.0f}",
                    "1000-1999g Total Newborns": "{:,.0f}",
                    "1000-1999g KMC Rate (%)": "{:.2f}%",
                    "2000-2499g KMC Cases": "{:,.0f}",
                    "2000-2499g Total Newborns": "{:,.0f}",
                    "2000-2499g KMC Rate (%)": "{:.2f}%",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_table.to_html(), unsafe_allow_html=True)

        # Add a note about the calculation method
        st.caption(
            "📝 **Note**: Overall totals are calculated by summing period totals."
        )

        # Download button
        csv = table_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="kmc_coverage_by_weight_range_trend.csv",
            mime="text/csv",
        )
    else:
        st.info("⚠️ No data available for trend table.")


def render_kmc_both_ranges_facility_comparison_chart(
    df,
    period_col="period_display",
    title="KMC Coverage by Birth Weight Range - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """
    Render facility comparison chart showing BOTH KMC 1000-1999g and 2000-2499g
    WITH BOTH CATEGORIES SHOWN SIDE-BY-SIDE IN TABLE
    """
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

    # Prepare data for all facilities and both ranges for chart
    chart_data = []
    table_data = []

    for facility_uid, facility_name in facility_uid_to_name.items():
        facility_df = filtered_df[filtered_df["orgUnit"] == facility_uid]

        if not facility_df.empty:
            # Compute both ranges for this facility
            kmc_1000_1999_data = compute_kmc_1000_1999_kpi(facility_df)
            kmc_2000_2499_data = compute_kmc_2000_2499_kpi(facility_df)

            # For chart
            chart_data.append(
                {
                    "Facility": facility_name,
                    "Range": "1000-1999g",
                    "KMC Rate (%)": kmc_1000_1999_data["kmc_rate"],
                    "KMC Cases": kmc_1000_1999_data["kmc_count"],
                    "Total Newborns": kmc_1000_1999_data["total_newborns"],
                }
            )

            chart_data.append(
                {
                    "Facility": facility_name,
                    "Range": "2000-2499g",
                    "KMC Rate (%)": kmc_2000_2499_data["kmc_rate"],
                    "KMC Cases": kmc_2000_2499_data["kmc_count"],
                    "Total Newborns": kmc_2000_2499_data["total_newborns"],
                }
            )

            # For table
            table_data.append(
                {
                    "Facility Name": facility_name,
                    # 1000-1999g columns
                    "1000-1999g KMC Rate (%)": kmc_1000_1999_data["kmc_rate"],
                    "1000-1999g KMC Cases": kmc_1000_1999_data["kmc_count"],
                    "1000-1999g Total Newborns": kmc_1000_1999_data["total_newborns"],
                    # 2000-2499g columns
                    "2000-2499g KMC Rate (%)": kmc_2000_2499_data["kmc_rate"],
                    "2000-2499g KMC Cases": kmc_2000_2499_data["kmc_count"],
                    "2000-2499g Total Newborns": kmc_2000_2499_data["total_newborns"],
                }
            )

    if not chart_data:
        st.info("⚠️ No comparison data available.")
        return

    chart_df = pd.DataFrame(chart_data)

    # Create grouped bar chart
    fig = px.bar(
        chart_df,
        x="Facility",
        y="KMC Rate (%)",
        color="Range",
        barmode="group",
        title=title,
        height=500,
        text_auto=".1f",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Facility",
        yaxis_title="KMC Coverage (%)",
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
            title="Birth Weight Range", yanchor="top", y=0.99, xanchor="right", x=0.99
        ),
    )

    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        hovertemplate="<b>%{x} - %{data.name}</b><br>Rate: %{y:.1f}%<br>Cases: %{customdata[0]}/%{customdata[1]}<extra></extra>",
        customdata=chart_df[["KMC Cases", "Total Newborns"]].values,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Facility comparison table
    st.subheader("📋 Facility Comparison Summary")

    if not table_data:
        st.info("⚠️ No facility summary data available.")
        return

    table_df = pd.DataFrame(table_data)

    if not table_df.empty:
        # Calculate overall totals by SUMMING facility totals (NOT recalculating from entire dataset)
        overall_1000_1999_kmc_cases = table_df["1000-1999g KMC Cases"].sum()
        overall_1000_1999_total = table_df["1000-1999g Total Newborns"].sum()
        overall_1000_1999_rate = (
            (overall_1000_1999_kmc_cases / overall_1000_1999_total * 100)
            if overall_1000_1999_total > 0
            else 0.0
        )

        overall_2000_2499_kmc_cases = table_df["2000-2499g KMC Cases"].sum()
        overall_2000_2499_total = table_df["2000-2499g Total Newborns"].sum()
        overall_2000_2499_rate = (
            (overall_2000_2499_kmc_cases / overall_2000_2499_total * 100)
            if overall_2000_2499_total > 0
            else 0.0
        )

        # Add overall row
        overall_row = {
            "Facility Name": "Overall",
            "1000-1999g KMC Rate (%)": overall_1000_1999_rate,
            "1000-1999g KMC Cases": overall_1000_1999_kmc_cases,
            "1000-1999g Total Newborns": overall_1000_1999_total,
            "2000-2499g KMC Rate (%)": overall_2000_2499_rate,
            "2000-2499g KMC Cases": overall_2000_2499_kmc_cases,
            "2000-2499g Total Newborns": overall_2000_2499_total,
        }

        table_df = pd.concat([table_df, pd.DataFrame([overall_row])], ignore_index=True)
        table_df.insert(0, "No", range(1, len(table_df) + 1))

        # Format table
        styled_table = (
            table_df.style.format(
                {
                    "1000-1999g KMC Rate (%)": "{:.2f}%",
                    "1000-1999g KMC Cases": "{:,.0f}",
                    "1000-1999g Total Newborns": "{:,.0f}",
                    "2000-2499g KMC Rate (%)": "{:.2f}%",
                    "2000-2499g KMC Cases": "{:,.0f}",
                    "2000-2499g Total Newborns": "{:,.0f}",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_table.to_html(), unsafe_allow_html=True)

        # Add a note about the calculation method
        st.caption(
            "📝 **Note**: Overall totals are calculated by summing facility totals."
        )

        # Download button
        csv = table_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="kmc_coverage_by_weight_range_facility_comparison.csv",
            mime="text/csv",
        )


def render_kmc_both_ranges_region_comparison_chart(
    df,
    period_col="period_display",
    title="KMC Coverage by Birth Weight Range - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
):
    """
    Render region comparison chart showing BOTH KMC 1000-1999g and 2000-2499g
    WITH BOTH CATEGORIES SHOWN SIDE-BY-SIDE IN TABLE
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # Prepare data for all regions for chart and table
    chart_data = []
    table_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]

        if region_facility_uids:
            region_df = df[df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                # Compute both ranges for this region
                kmc_1000_1999_data = compute_kmc_1000_1999_kpi(region_df)
                kmc_2000_2499_data = compute_kmc_2000_2499_kpi(region_df)

                # For chart
                chart_data.append(
                    {
                        "Region": region_name,
                        "Range": "1000-1999g",
                        "KMC Rate (%)": kmc_1000_1999_data["kmc_rate"],
                        "KMC Cases": kmc_1000_1999_data["kmc_count"],
                        "Total Newborns": kmc_1000_1999_data["total_newborns"],
                    }
                )

                chart_data.append(
                    {
                        "Region": region_name,
                        "Range": "2000-2499g",
                        "KMC Rate (%)": kmc_2000_2499_data["kmc_rate"],
                        "KMC Cases": kmc_2000_2499_data["kmc_count"],
                        "Total Newborns": kmc_2000_2499_data["total_newborns"],
                    }
                )

                # For table
                table_data.append(
                    {
                        "Region Name": region_name,
                        # 1000-1999g columns
                        "1000-1999g KMC Rate (%)": kmc_1000_1999_data["kmc_rate"],
                        "1000-1999g KMC Cases": kmc_1000_1999_data["kmc_count"],
                        "1000-1999g Total Newborns": kmc_1000_1999_data[
                            "total_newborns"
                        ],
                        # 2000-2499g columns
                        "2000-2499g KMC Rate (%)": kmc_2000_2499_data["kmc_rate"],
                        "2000-2499g KMC Cases": kmc_2000_2499_data["kmc_count"],
                        "2000-2499g Total Newborns": kmc_2000_2499_data[
                            "total_newborns"
                        ],
                    }
                )

    if not chart_data:
        st.info("⚠️ No comparison data available.")
        return

    chart_df = pd.DataFrame(chart_data)

    # Create grouped bar chart
    fig = px.bar(
        chart_df,
        x="Region",
        y="KMC Rate (%)",
        color="Range",
        barmode="group",
        title=title,
        height=500,
        text_auto=".1f",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Region",
        yaxis_title="KMC Coverage (%)",
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
            title="Birth Weight Range", yanchor="top", y=0.99, xanchor="right", x=0.99
        ),
    )

    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        hovertemplate="<b>%{x} - %{data.name}</b><br>Rate: %{y:.1f}%<br>Cases: %{customdata[0]}/%{customdata[1]}<extra></extra>",
        customdata=chart_df[["KMC Cases", "Total Newborns"]].values,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Region comparison table
    st.subheader("📋 Region Comparison Summary")

    if not table_data:
        st.info("⚠️ No region summary data available.")
        return

    table_df = pd.DataFrame(table_data)

    if not table_df.empty:
        # Calculate overall totals by SUMMING region totals (NOT recalculating from entire dataset)
        overall_1000_1999_kmc_cases = table_df["1000-1999g KMC Cases"].sum()
        overall_1000_1999_total = table_df["1000-1999g Total Newborns"].sum()
        overall_1000_1999_rate = (
            (overall_1000_1999_kmc_cases / overall_1000_1999_total * 100)
            if overall_1000_1999_total > 0
            else 0.0
        )

        overall_2000_2499_kmc_cases = table_df["2000-2499g KMC Cases"].sum()
        overall_2000_2499_total = table_df["2000-2499g Total Newborns"].sum()
        overall_2000_2499_rate = (
            (overall_2000_2499_kmc_cases / overall_2000_2499_total * 100)
            if overall_2000_2499_total > 0
            else 0.0
        )

        # Add overall row
        overall_row = {
            "Region Name": "Overall",
            "1000-1999g KMC Rate (%)": overall_1000_1999_rate,
            "1000-1999g KMC Cases": overall_1000_1999_kmc_cases,
            "1000-1999g Total Newborns": overall_1000_1999_total,
            "2000-2499g KMC Rate (%)": overall_2000_2499_rate,
            "2000-2499g KMC Cases": overall_2000_2499_kmc_cases,
            "2000-2499g Total Newborns": overall_2000_2499_total,
        }

        table_df = pd.concat([table_df, pd.DataFrame([overall_row])], ignore_index=True)
        table_df.insert(0, "No", range(1, len(table_df) + 1))

        # Format table
        styled_table = (
            table_df.style.format(
                {
                    "1000-1999g KMC Rate (%)": "{:.2f}%",
                    "1000-1999g KMC Cases": "{:,.0f}",
                    "1000-1999g Total Newborns": "{:,.0f}",
                    "2000-2499g KMC Rate (%)": "{:.2f}%",
                    "2000-2499g KMC Cases": "{:,.0f}",
                    "2000-2499g Total Newborns": "{:,.0f}",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_table.to_html(), unsafe_allow_html=True)

        # Add a note about the calculation method
        st.caption(
            "📝 **Note**: Overall totals are calculated by summing region totals."
        )

        # Download button
        csv = table_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="kmc_coverage_by_weight_range_region_comparison.csv",
            mime="text/csv",
        )
