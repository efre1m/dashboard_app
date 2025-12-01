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
BIRTHWEIGHT_GRAMS_UID = (
    "yxWUMt3sCil"  # Birthweight in grams (same as weight on admission)
)

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


def compute_cpap_general_numerator(df, facility_uids=None):
    """
    Compute numerator for General CPAP KPI: Count of all newborns who received CPAP

    Formula: Count where CPAP Administered = 1 (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count all newborns with CPAP administered = Yes
    cpap_cases = df[
        (df["dataElement_uid"] == CPAP_ADMINISTERED_UID)
        & (df["value"] == CPAP_YES_VALUE)
    ]["tei_id"].nunique()

    return cpap_cases


def compute_cpap_prophylactic_numerator(df, facility_uids=None):
    """
    Compute numerator for Prophylactic CPAP KPI: Count of newborns with birthweight 1000-2499g who received CPAP

    Formula: Count where:
        - Birthweight between 1000-2499g
        - CPAP Administered = 1 (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # First, identify newborns with birthweight 1000-2499g
    birthweight_newborns = df[
        (df["dataElement_uid"] == BIRTHWEIGHT_GRAMS_UID) & df["value"].notna()
    ].copy()

    if birthweight_newborns.empty:
        return 0

    # Convert birthweight to numeric
    birthweight_newborns["weight_value"] = pd.to_numeric(
        birthweight_newborns["value"], errors="coerce"
    )

    # Filter for birthweight 1000-2499g
    target_weight_newborns = birthweight_newborns[
        (birthweight_newborns["weight_value"] >= 1000)
        & (birthweight_newborns["weight_value"] <= 2499)
    ]["tei_id"].unique()

    if len(target_weight_newborns) == 0:
        return 0

    # Filter for target weight newborns and check CPAP administration
    target_df = df[df["tei_id"].isin(target_weight_newborns)]

    # Count newborns with CPAP administered = Yes
    cpap_cases = target_df[
        (target_df["dataElement_uid"] == CPAP_ADMINISTERED_UID)
        & (target_df["value"] == CPAP_YES_VALUE)
    ]["tei_id"].nunique()

    return cpap_cases


def compute_cpap_prophylactic_denominator(df, facility_uids=None):
    """
    Compute denominator for Prophylactic CPAP KPI: Total count of newborns with birthweight 1000-2499g
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count newborns with birthweight 1000-2499g
    birthweight_newborns = df[
        (df["dataElement_uid"] == BIRTHWEIGHT_GRAMS_UID) & df["value"].notna()
    ].copy()

    if birthweight_newborns.empty:
        return 0

    # Convert birthweight to numeric
    birthweight_newborns["weight_value"] = pd.to_numeric(
        birthweight_newborns["value"], errors="coerce"
    )

    # Count newborns with birthweight 1000-2499g
    target_weight_newborns = birthweight_newborns[
        (birthweight_newborns["weight_value"] >= 1000)
        & (birthweight_newborns["weight_value"] <= 2499)
    ]["tei_id"].nunique()

    return target_weight_newborns


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


def compute_cpap_general_kpi(df, facility_uids=None, tei_df=None):
    """
    Compute General CPAP KPI: % of all admissions that received CPAP

    Formula: CPAP Coverage (%) =
             (All newborns who received CPAP) ÷ (Total admitted newborns) × 100
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "cpap_general_rate": 0.0,
            "cpap_general_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count all CPAP cases (numerator)
    cpap_general_count = compute_cpap_general_numerator(df, facility_uids)

    # Count total admitted newborns (denominator) - same as hypothermia style
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate general CPAP rate
    cpap_general_rate = (
        (cpap_general_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "cpap_general_rate": float(cpap_general_rate),
        "cpap_general_count": int(cpap_general_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_cpap_prophylactic_kpi(df, facility_uids=None):
    """
    Compute Prophylactic CPAP KPI: % of newborns with birthweight 1000-2499g who received CPAP

    Formula: Prophylactic CPAP Coverage (%) =
             (Newborns with birthweight 1000-2499g who received CPAP) ÷
             (Total newborns with birthweight 1000-2499g) × 100
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "cpap_prophylactic_rate": 0.0,
            "cpap_prophylactic_count": 0,
            "total_target_weight": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count prophylactic CPAP cases (numerator)
    cpap_prophylactic_count = compute_cpap_prophylactic_numerator(df, facility_uids)

    # Count total newborns with target birthweight (denominator)
    total_target_weight = compute_cpap_prophylactic_denominator(df, facility_uids)

    # Calculate prophylactic CPAP rate
    cpap_prophylactic_rate = (
        (cpap_prophylactic_count / total_target_weight * 100)
        if total_target_weight > 0
        else 0.0
    )

    return {
        "cpap_prophylactic_rate": float(cpap_prophylactic_rate),
        "cpap_prophylactic_count": int(cpap_prophylactic_count),
        "total_target_weight": int(total_target_weight),
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


def compute_cpap_general_trend_data(
    df, period_col="period_display", facility_uids=None, tei_df=None
):
    """
    Compute General CPAP trend data by period - USING HYPOTHERMIA-STYLE DENOMINATOR TRACKING
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

    # ✅ USE HYPOTHERMIA-STYLE DENOMINATOR: Track counted newborns across periods
    counted_newborns = set()

    # ✅ FIX: Sort periods chronologically using period_sort if available
    if "period_sort" in df.columns:
        # Use period_sort for proper chronological ordering
        periods_sorted = sorted(
            df[["period_display", "period_sort"]].drop_duplicates().itertuples(),
            key=lambda x: x.period_sort,
        )
        periods = [p.period_display for p in periods_sorted]
    else:
        # Fallback: try to sort period_display as dates
        try:
            periods = sorted(
                df[period_col].unique(),
                key=lambda x: pd.to_datetime(x, errors="coerce"),
            )
        except:
            periods = sorted(df[period_col].unique())

    for period in periods:
        period_df = df[df[period_col] == period]
        period_display = (
            period_df["period_display"].iloc[0] if not period_df.empty else period
        )

        # ✅ HYPOTHERMIA-STYLE DENOMINATOR: Get newborns in this period who haven't been counted yet
        period_newborns = set(period_df["tei_id"].unique())
        new_newborns = period_newborns - counted_newborns

        if new_newborns:
            # Filter to only new newborns in this period for denominator
            new_newborns_df = period_df[period_df["tei_id"].isin(new_newborns)]

            # ✅ KEEP ORIGINAL NUMERATOR: Count ALL CPAP events in the FULL period data
            cpap_mask = (period_df["dataElement_uid"] == CPAP_ADMINISTERED_UID) & (
                period_df["value"] == CPAP_YES_VALUE
            )
            cpap_general_count = period_df[cpap_mask][
                "tei_id"
            ].nunique()  # Count unique newborns with CPAP

            # ✅ HYPOTHERMIA-STYLE DENOMINATOR: Count only new newborns for this period
            total_admitted_newborns = len(new_newborns)

            # ✅ Update counted newborns for next period
            counted_newborns.update(new_newborns)
        else:
            # No new newborns in this period
            cpap_general_count = 0
            total_admitted_newborns = 0

        # Calculate general CPAP rate
        cpap_general_rate = (
            (cpap_general_count / total_admitted_newborns * 100)
            if total_admitted_newborns > 0
            else 0.0
        )

        trend_data.append(
            {
                period_col: period_display,
                "cpap_general_count": int(cpap_general_count),
                "total_admitted_newborns": int(total_admitted_newborns),
                "cpap_general_rate": float(cpap_general_rate),
            }
        )

    return pd.DataFrame(trend_data)


def compute_cpap_prophylactic_trend_data(
    df, period_col="period_display", facility_uids=None
):
    """
    Compute Prophylactic CPAP trend data by period
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
        cpap_data = compute_cpap_prophylactic_kpi(period_df, facility_uids)

        trend_data.append(
            {
                period_col: period,
                "total_target_weight": cpap_data["total_target_weight"],
                "cpap_prophylactic_count": cpap_data["cpap_prophylactic_count"],
                "cpap_prophylactic_rate": cpap_data["cpap_prophylactic_rate"],
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


def render_cpap_general_trend_chart(
    df,
    period_col="period_display",
    title="General CPAP Coverage Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
    tei_df=None,
):
    """Render a LINE CHART ONLY for general CPAP coverage rate trend WITH CHRONOLOGICAL ORDERING"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Compute trend data with proper period-based counting
    trend_df = compute_cpap_general_trend_data(df, period_col, facility_uids, tei_df)

    if trend_df.empty:
        st.info("⚠️ No general CPAP data available for the selected period.")
        return

    trend_df = trend_df.copy()
    trend_df["cpap_general_rate"] = pd.to_numeric(
        trend_df["cpap_general_rate"], errors="coerce"
    ).fillna(0)

    # ✅ FIX: Ensure chronological ordering in the chart
    if "period_sort" in df.columns:
        period_sort_mapping = df[["period_display", "period_sort"]].drop_duplicates()
        trend_df = trend_df.merge(period_sort_mapping, on="period_display", how="left")
        trend_df = trend_df.sort_values("period_sort")
        period_order = trend_df["period_display"].tolist()
    else:
        try:
            trend_df["period_datetime"] = pd.to_datetime(
                trend_df[period_col], errors="coerce"
            )
            trend_df = trend_df.sort_values("period_datetime")
            period_order = trend_df[period_col].tolist()
        except:
            period_order = sorted(trend_df[period_col].unique())

    # Create line chart with proper ordering
    fig = px.line(
        trend_df,
        x=period_col,
        y="cpap_general_rate",
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
        category_orders={period_col: period_order},
    )

    # Update traces for line chart
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>General CPAP Coverage: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="General CPAP Coverage (%)",
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
    if len(trend_df) > 1:
        last_value = trend_df["cpap_general_rate"].iloc[-1]
        prev_value = trend_df["cpap_general_rate"].iloc[-2]
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
    st.subheader("📋 General CPAP Trend Summary Table")
    summary_df = trend_df.copy().reset_index(drop=True)

    # Calculate overall value - SUM of numerators and denominators
    total_numerator = summary_df["cpap_general_count"].sum()
    total_denominator = summary_df["total_admitted_newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = pd.DataFrame(
        {
            period_col: ["Overall"],
            "cpap_general_count": [total_numerator],
            "total_admitted_newborns": [total_denominator],
            "cpap_general_rate": [overall_value],
        }
    )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)

    # Create display dataframe with proper column names
    display_columns = {
        period_col: "Period",
        "cpap_general_count": "CPAP Cases",
        "total_admitted_newborns": "Total Admitted Newborns",
        "cpap_general_rate": "General CPAP Rate (%)",
    }

    # Rename columns for display
    summary_table_display = summary_table.rename(columns=display_columns)

    # Reorder columns
    column_order = [
        "Period",
        "CPAP Cases",
        "Total Admitted Newborns",
        "General CPAP Rate (%)",
    ]
    summary_table_display = summary_table_display[column_order]

    summary_table_display.insert(0, "No", range(1, len(summary_table_display) + 1))

    # Format table
    styled_table = (
        summary_table_display.style.format(
            {
                "General CPAP Rate (%)": "{:.1f}%",
                "CPAP Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = summary_table_display.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="general_cpap_trend_summary.csv",
        mime="text/csv",
    )


def render_cpap_prophylactic_trend_chart(
    df,
    period_col="period_display",
    value_col="cpap_prophylactic_rate",
    title="Prophylactic CPAP Coverage Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="Prophylactic CPAP Cases",
    denominator_name="Total Newborns (1000-2499g)",
    facility_uids=None,
):
    """Render a LINE CHART ONLY for prophylactic CPAP coverage rate trend"""
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
        hovertemplate="<b>%{x}</b><br>Prophylactic CPAP Coverage: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Prophylactic CPAP Coverage (%)",
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


def render_cpap_general_facility_comparison_chart(
    df,
    period_col="period_display",
    title="General CPAP Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    tei_df=None,
):
    """Render facility comparison with BOTH LINE AND BAR CHART options for general CPAP rate"""
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

    # ✅ FIX: Generate unique key for radio button
    import hashlib

    key_suffix = hashlib.md5(str(facility_uids).encode()).hexdigest()[:8]

    # Chart options with radio button
    chart_options = ["Line Chart", "Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_facility_comparison_general_cpap_{key_suffix}",
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
                    # Compute KPI for this specific facility and period with proper counting
                    cpap_data = compute_cpap_general_kpi(
                        facility_period_df, [facility_uid], tei_df
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": cpap_data["cpap_general_rate"],
                            "cpap_general_count": cpap_data["cpap_general_count"],
                            "total_admitted_newborns": cpap_data[
                                "total_admitted_newborns"
                            ],
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

        # Update traces
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
                cpap_data = compute_cpap_general_kpi(
                    facility_df, [facility_uid], tei_df
                )
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": cpap_data["cpap_general_rate"],
                        "cpap_general_count": cpap_data["cpap_general_count"],
                        "total_admitted_newborns": cpap_data["total_admitted_newborns"],
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

        # Update traces for bar chart
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>General CPAP Rate: %{y:.2f}%<extra></extra>",
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="General CPAP Coverage (%)",
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

    # Facility comparison table - PROPER COLUMN NAMES
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            cpap_data = compute_cpap_general_kpi(facility_df, [facility_uid], tei_df)
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "CPAP Cases": cpap_data["cpap_general_count"],
                    "Total Admitted Newborns": cpap_data["total_admitted_newborns"],
                    "General CPAP Rate": cpap_data["cpap_general_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["CPAP Cases"].sum()
    total_denominator = facility_table_df["Total Admitted Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "CPAP Cases": total_numerator,
        "Total Admitted Newborns": total_denominator,
        "General CPAP Rate": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Facility Name",
        "CPAP Cases",
        "Total Admitted Newborns",
        "General CPAP Rate",
    ]
    facility_table_df = facility_table_df[column_order]

    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "CPAP Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
                "General CPAP Rate": "{:.2f}%",
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
        file_name="general_cpap_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_cpap_prophylactic_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="cpap_prophylactic_rate",
    title="Prophylactic CPAP Coverage - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="Prophylactic CPAP Cases",
    denominator_name="Total Newborns (1000-2499g)",
):
    """Render facility comparison LINE CHART ONLY for prophylactic CPAP coverage"""
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
                cpap_data = compute_cpap_prophylactic_kpi(
                    facility_period_df, [facility_uid]
                )
                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": cpap_data["cpap_prophylactic_rate"],
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
        yaxis_title="Prophylactic CPAP Coverage (%)",
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
            cpap_data = compute_cpap_prophylactic_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Prophylactic CPAP Cases": cpap_data["cpap_prophylactic_count"],
                    "Total Newborns (1000-2499g)": cpap_data["total_target_weight"],
                    "Prophylactic CPAP Rate (%)": cpap_data["cpap_prophylactic_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Prophylactic CPAP Cases"].sum()
    total_denominator = facility_table_df["Total Newborns (1000-2499g)"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "Prophylactic CPAP Cases": total_numerator,
        "Total Newborns (1000-2499g)": total_denominator,
        "Prophylactic CPAP Rate (%)": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Prophylactic CPAP Cases": "{:,.0f}",
                "Total Newborns (1000-2499g)": "{:,.0f}",
                "Prophylactic CPAP Rate (%)": "{:.2f}%",
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
        file_name="prophylactic_cpap_coverage_facility_comparison.csv",
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


def render_cpap_general_region_comparison_chart(
    df,
    period_col="period_display",
    title="General CPAP Coverage - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    facilities_by_region=None,
    tei_df=None,
):
    """Render region comparison with BOTH LINE AND BAR CHART options for general CPAP rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # ✅ FIX: Generate unique key for radio button
    import hashlib

    key_suffix = hashlib.md5(str(region_names).encode()).hexdigest()[:8]

    # Chart options with radio button
    chart_options = ["Line Chart", "Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_region_comparison_general_cpap_{key_suffix}",
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
                        # Compute KPI for this specific region and period with proper counting
                        cpap_data = compute_cpap_general_kpi(
                            region_period_df, region_facility_uids, tei_df
                        )
                        time_series_data.append(
                            {
                                "period_display": period_display,
                                "Region": region_name,
                                "value": cpap_data["cpap_general_rate"],
                                "cpap_general_count": cpap_data["cpap_general_count"],
                                "total_admitted_newborns": cpap_data[
                                    "total_admitted_newborns"
                                ],
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

        # Update traces
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
                    cpap_data = compute_cpap_general_kpi(
                        region_df, region_facility_uids, tei_df
                    )
                    bar_data.append(
                        {
                            "Region": region_name,
                            "value": cpap_data["cpap_general_rate"],
                            "cpap_general_count": cpap_data["cpap_general_count"],
                            "total_admitted_newborns": cpap_data[
                                "total_admitted_newborns"
                            ],
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

        # Update traces for bar chart
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>General CPAP Rate: %{y:.2f}%<extra></extra>",
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="General CPAP Coverage (%)",
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

    # Region comparison table - PROPER COLUMN NAMES
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]

        if region_facility_uids:
            region_df = df[df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                cpap_data = compute_cpap_general_kpi(
                    region_df, region_facility_uids, tei_df
                )
                region_table_data.append(
                    {
                        "Region Name": region_name,
                        "CPAP Cases": cpap_data["cpap_general_count"],
                        "Total Admitted Newborns": cpap_data["total_admitted_newborns"],
                        "General CPAP Rate": cpap_data["cpap_general_rate"],
                    }
                )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["CPAP Cases"].sum()
    total_denominator = region_table_df["Total Admitted Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "CPAP Cases": total_numerator,
        "Total Admitted Newborns": total_denominator,
        "General CPAP Rate": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Region Name",
        "CPAP Cases",
        "Total Admitted Newborns",
        "General CPAP Rate",
    ]
    region_table_df = region_table_df[column_order]

    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "CPAP Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
                "General CPAP Rate": "{:.2f}%",
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
        file_name="general_cpap_rate_region_comparison.csv",
        mime="text/csv",
    )


def render_cpap_prophylactic_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="cpap_prophylactic_rate",
    title="Prophylactic CPAP Coverage - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="Prophylactic CPAP Cases",
    denominator_name="Total Newborns (1000-2499g)",
):
    """Render region comparison LINE CHART ONLY for prophylactic CPAP coverage"""
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
                cpap_data = compute_cpap_prophylactic_kpi(
                    region_period_df, region_facility_uids
                )
                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Region": region_name,
                        "value": cpap_data["cpap_prophylactic_rate"],
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
        yaxis_title="Prophylactic CPAP Coverage (%)",
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
            cpap_data = compute_cpap_prophylactic_kpi(region_df, region_facility_uids)
            region_table_data.append(
                {
                    "Region Name": region_name,
                    "Prophylactic CPAP Cases": cpap_data["cpap_prophylactic_count"],
                    "Total Newborns (1000-2499g)": cpap_data["total_target_weight"],
                    "Prophylactic CPAP Rate (%)": cpap_data["cpap_prophylactic_rate"],
                }
            )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Prophylactic CPAP Cases"].sum()
    total_denominator = region_table_df["Total Newborns (1000-2499g)"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "Prophylactic CPAP Cases": total_numerator,
        "Total Newborns (1000-2499g)": total_denominator,
        "Prophylactic CPAP Rate (%)": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Prophylactic CPAP Cases": "{:,.0f}",
                "Total Newborns (1000-2499g)": "{:,.0f}",
                "Prophylactic CPAP Rate (%)": "{:.2f}%",
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
        file_name="prophylactic_cpap_coverage_region_comparison.csv",
        mime="text/csv",
    )
