import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color

# ---------------- Inborn KPI Constants ----------------
BIRTH_LOCATION_UID = "aK5txmRYpVX"  # birth location
INBORN_CODE = "1"  # inborn code value

# ---------------- Antibiotics KPI Constants ----------------
SUBCATEGORIES_INFECTION_UID = "wn0tHaHcceW"  # sub categories of infection
ANTIBIOTICS_ADMINISTERED_UID = "sxtsEDilKZd"  # were antibiotics administered
PROBABLE_SEPSIS_CODE = "1"  # probable sepsis code value
YES_CODE = "1"  # yes code value


# ---------------- Inborn KPI Computation Functions ----------------
def compute_inborn_numerator(df, facility_uids=None):
    """
    Compute numerator for inborn KPI: Count of newborns with birth location = '1' (inborn)
    Uses vectorized operations for optimization
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Vectorized filtering for birth location events
    birth_location_mask = (df["dataElement_uid"] == BIRTH_LOCATION_UID) & df[
        "value"
    ].notna()

    birth_location_events = df[birth_location_mask]

    if birth_location_events.empty:
        return 0

    # Vectorized filtering for inborn cases
    inborn_mask = birth_location_events["value"] == INBORN_CODE

    # Count unique newborns with birth location = '1' (inborn)
    inborn_cases = birth_location_events[inborn_mask]["tei_id"].nunique()

    return inborn_cases


def compute_inborn_kpi(df, facility_uids=None, tei_df=None):
    """
    Compute inborn KPI for the given dataframe
    Uses unique TEI count from the filtered events dataframe for denominator

    Formula: % Inborn Babies =
             (Newborns with birth location = '1' during period) ÷
             (Total admitted newborns in period) × 100
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "inborn_rate": 0.0,
            "inborn_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count inborn cases (numerator) - filters for birth location events
    inborn_count = compute_inborn_numerator(df, facility_uids)

    # Count unique TEIs in THIS PERIOD only (not total) - same as hypothermia
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate inborn rate
    inborn_rate = (
        (inborn_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "inborn_rate": float(inborn_rate),
        "inborn_count": int(inborn_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_inborn_trend_data(
    df, period_col="period_display", facility_uids=None, tei_df=None
):
    """
    Compute inborn trend data by period - USE HYPOTHERMIA-STYLE DENOMINATOR TRACKING
    WITH CHRONOLOGICAL ORDERING
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

            # ✅ KEEP ORIGINAL NUMERATOR: Count ALL inborn events in the FULL period data
            birth_location_mask = (
                period_df["dataElement_uid"] == BIRTH_LOCATION_UID
            ) & (period_df["value"] == INBORN_CODE)
            inborn_count = birth_location_mask.sum()  # Count ALL events in period

            # ✅ HYPOTHERMIA-STYLE DENOMINATOR: Count only new newborns for this period
            total_admitted_newborns = len(new_newborns)

            # ✅ Update counted newborns for next period
            counted_newborns.update(new_newborns)
        else:
            # No new newborns in this period
            inborn_count = 0
            total_admitted_newborns = 0

        # Calculate inborn rate
        inborn_rate = (
            (inborn_count / total_admitted_newborns * 100)
            if total_admitted_newborns > 0
            else 0.0
        )

        trend_data.append(
            {
                period_col: period_display,
                "inborn_count": int(inborn_count),
                "total_admitted_newborns": int(total_admitted_newborns),
                "inborn_rate": float(inborn_rate),
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- Antibiotics KPI Computation Functions ----------------
def compute_antibiotics_numerator(df, facility_uids=None):
    """
    Compute numerator for antibiotics KPI:
    Count of babies with both:
    1. SubCategoriesofInfection = '1' (Probable Sepsis) AND
    2. WereAntibioticsAdministered = '1' (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for babies with Probable Sepsis
    sepsis_mask = (df["dataElement_uid"] == SUBCATEGORIES_INFECTION_UID) & (
        df["value"] == PROBABLE_SEPSIS_CODE
    )

    sepsis_events = df[sepsis_mask]

    if sepsis_events.empty:
        return 0

    # Count unique TEIs with antibiotics = Yes
    antibiotics_count = 0
    sepsis_teis = sepsis_events["tei_id"].unique()

    for tei in sepsis_teis:
        tei_df = df[df["tei_id"] == tei]
        antibiotics_events = tei_df[
            (tei_df["dataElement_uid"] == ANTIBIOTICS_ADMINISTERED_UID)
            & (tei_df["value"] == YES_CODE)
        ]
        if not antibiotics_events.empty:
            antibiotics_count += 1

    return antibiotics_count


def compute_antibiotics_kpi(df, facility_uids=None, tei_df=None):
    """
    Compute antibiotics KPI for the given dataframe

    Formula: % Antibiotics for babies with clinical sepsis =
             (Babies with Probable Sepsis AND received antibiotics) ÷
             (Total babies with Probable Sepsis) × 100
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "antibiotics_rate": 0.0,
            "antibiotics_count": 0,
            "probable_sepsis_count": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count babies with Probable Sepsis (denominator)
    sepsis_mask = (df["dataElement_uid"] == SUBCATEGORIES_INFECTION_UID) & (
        df["value"] == PROBABLE_SEPSIS_CODE
    )

    sepsis_events = df[sepsis_mask]
    probable_sepsis_count = sepsis_events["tei_id"].nunique()

    # Count antibiotics cases (numerator)
    antibiotics_count = compute_antibiotics_numerator(df, facility_uids)

    # Calculate antibiotics rate
    antibiotics_rate = (
        (antibiotics_count / probable_sepsis_count * 100)
        if probable_sepsis_count > 0
        else 0.0
    )

    return {
        "antibiotics_rate": float(antibiotics_rate),
        "antibiotics_count": int(antibiotics_count),
        "probable_sepsis_count": int(probable_sepsis_count),
    }


def compute_antibiotics_trend_data(
    df, period_col="period_display", facility_uids=None, tei_df=None
):
    """
    Compute antibiotics trend data by period
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

    # Sort periods chronologically
    if "period_sort" in df.columns:
        periods_sorted = sorted(
            df[["period_display", "period_sort"]].drop_duplicates().itertuples(),
            key=lambda x: x.period_sort,
        )
        periods = [p.period_display for p in periods_sorted]
    else:
        try:
            periods = sorted(
                df[period_col].unique(),
                key=lambda x: pd.to_datetime(x, errors="coerce"),
            )
        except:
            periods = sorted(df[period_col].unique())

    for period in periods:
        period_df = df[df[period_col] == period]

        # Compute KPI for this period
        period_kpi = compute_antibiotics_kpi(period_df, facility_uids, tei_df)

        trend_data.append(
            {
                period_col: period,
                "antibiotics_count": period_kpi["antibiotics_count"],
                "probable_sepsis_count": period_kpi["probable_sepsis_count"],
                "antibiotics_rate": period_kpi["antibiotics_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- Inborn Chart Functions ----------------
def render_inborn_trend_chart(
    df,
    period_col="period_display",
    title="Inborn Babies Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
    tei_df=None,
):
    """Render a LINE CHART ONLY for inborn rate trend WITH CHRONOLOGICAL ORDERING"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Compute trend data with proper period-based counting
    trend_df = compute_inborn_trend_data(df, period_col, facility_uids, tei_df)

    if trend_df.empty:
        st.info("⚠️ No inborn data available for the selected period.")
        return

    trend_df = trend_df.copy()
    trend_df["inborn_rate"] = pd.to_numeric(
        trend_df["inborn_rate"], errors="coerce"
    ).fillna(0)

    # ✅ FIX: Ensure chronological ordering in the chart
    # Use period_sort if available, otherwise try to sort period_display as dates
    if "period_sort" in df.columns:
        # Merge period_sort back to trend_df for proper ordering
        period_sort_mapping = df[["period_display", "period_sort"]].drop_duplicates()
        trend_df = trend_df.merge(period_sort_mapping, on="period_display", how="left")
        trend_df = trend_df.sort_values("period_sort")
        period_order = trend_df["period_display"].tolist()
    else:
        # Fallback: try to sort period_display as dates
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
        y="inborn_rate",
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
        category_orders={period_col: period_order},  # ✅ Ensure chronological order
    )

    # Update traces for line chart
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Inborn Rate: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Inborn Babies (%)",
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
        last_value = trend_df["inborn_rate"].iloc[-1]
        prev_value = trend_df["inborn_rate"].iloc[-2]
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

    # Summary table - PROPER COLUMN NAMES
    st.subheader("📋 Inborn Trend Summary Table")
    summary_df = trend_df.copy().reset_index(drop=True)

    # Calculate overall value - SUM of numerators and denominators
    total_numerator = summary_df["inborn_count"].sum()
    total_denominator = summary_df["total_admitted_newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = pd.DataFrame(
        {
            period_col: ["Overall"],
            "inborn_count": [total_numerator],
            "total_admitted_newborns": [total_denominator],
            "inborn_rate": [overall_value],
        }
    )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)

    # Create display dataframe with proper column names
    display_columns = {
        period_col: "Period",
        "inborn_count": "Inborn Cases",
        "total_admitted_newborns": "Total Admitted Newborns",
        "inborn_rate": "Inborn Rate (%)",
    }

    # Rename columns for display
    summary_table_display = summary_table.rename(columns=display_columns)

    # Reorder columns
    column_order = [
        "Period",
        "Inborn Cases",
        "Total Admitted Newborns",
        "Inborn Rate (%)",
    ]
    summary_table_display = summary_table_display[column_order]

    summary_table_display.insert(0, "No", range(1, len(summary_table_display) + 1))

    # Format table
    styled_table = (
        summary_table_display.style.format(
            {
                "Inborn Rate (%)": "{:.1f}%",
                "Inborn Cases": "{:,.0f}",
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
        file_name="inborn_trend_summary.csv",
        mime="text/csv",
    )


def render_inborn_facility_comparison_chart(
    df,
    period_col="period_display",
    title="Inborn Babies - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    tei_df=None,
):
    """Render facility comparison LINE CHART for inborn rate"""
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
                facility_kpi = compute_inborn_kpi(
                    facility_period_df, [facility_uid], tei_df
                )

                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": facility_kpi["inborn_rate"],
                        "inborn_count": facility_kpi["inborn_count"],
                        "total_admitted_newborns": facility_kpi[
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

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Inborn Babies (%)",
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

    # Facility comparison table - PROPER COLUMN NAMES
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            inborn_data = compute_inborn_kpi(facility_df, [facility_uid], tei_df)
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Inborn Cases": inborn_data["inborn_count"],
                    "Total Admitted Newborns": inborn_data["total_admitted_newborns"],
                    "Inborn Rate": inborn_data["inborn_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Inborn Cases"].sum()
    total_denominator = facility_table_df["Total Admitted Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "Inborn Cases": total_numerator,
        "Total Admitted Newborns": total_denominator,
        "Inborn Rate": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Facility Name",
        "Inborn Cases",
        "Total Admitted Newborns",
        "Inborn Rate",
    ]
    facility_table_df = facility_table_df[column_order]

    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Inborn Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
                "Inborn Rate": "{:.2f}%",
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
        file_name="inborn_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_inborn_region_comparison_chart(
    df,
    period_col="period_display",
    title="Inborn Babies - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    facilities_by_region=None,
    tei_df=None,
):
    """Render region comparison LINE CHART for inborn rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

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
                    region_kpi = compute_inborn_kpi(
                        region_period_df, region_facility_uids, tei_df
                    )

                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": region_kpi["inborn_rate"],
                            "inborn_count": region_kpi["inborn_count"],
                            "total_admitted_newborns": region_kpi[
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

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Inborn Babies (%)",
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
                inborn_data = compute_inborn_kpi(
                    region_df, region_facility_uids, tei_df
                )
                region_table_data.append(
                    {
                        "Region Name": region_name,
                        "Inborn Cases": inborn_data["inborn_count"],
                        "Total Admitted Newborns": inborn_data[
                            "total_admitted_newborns"
                        ],
                        "Inborn Rate": inborn_data["inborn_rate"],
                    }
                )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Inborn Cases"].sum()
    total_denominator = region_table_df["Total Admitted Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "Inborn Cases": total_numerator,
        "Total Admitted Newborns": total_denominator,
        "Inborn Rate": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Region Name",
        "Inborn Cases",
        "Total Admitted Newborns",
        "Inborn Rate",
    ]
    region_table_df = region_table_df[column_order]

    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Inborn Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
                "Inborn Rate": "{:.2f}%",
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
        file_name="inborn_rate_region_comparison.csv",
        mime="text/csv",
    )


# ---------------- Antibiotics Chart Functions ----------------
def render_antibiotics_trend_chart(
    df,
    period_col="period_display",
    title="Antibiotics for Clinical Sepsis Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
    tei_df=None,
):
    """Render a LINE CHART ONLY for antibiotics rate trend"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Compute trend data
    trend_df = compute_antibiotics_trend_data(df, period_col, facility_uids, tei_df)

    if trend_df.empty:
        st.info("⚠️ No antibiotics data available for the selected period.")
        return

    trend_df = trend_df.copy()
    trend_df["antibiotics_rate"] = pd.to_numeric(
        trend_df["antibiotics_rate"], errors="coerce"
    ).fillna(0)

    # Ensure chronological ordering
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

    # Create line chart
    fig = px.line(
        trend_df,
        x=period_col,
        y="antibiotics_rate",
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
        hovertemplate="<b>%{x}</b><br>Antibiotics Rate: %{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Antibiotics for Clinical Sepsis (%)",
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
        last_value = trend_df["antibiotics_rate"].iloc[-1]
        prev_value = trend_df["antibiotics_rate"].iloc[-2]
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
    st.subheader("📋 Antibiotics Trend Summary Table")
    summary_df = trend_df.copy().reset_index(drop=True)

    # Calculate overall value
    total_numerator = summary_df["antibiotics_count"].sum()
    total_denominator = summary_df["probable_sepsis_count"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = pd.DataFrame(
        {
            period_col: ["Overall"],
            "antibiotics_count": [total_numerator],
            "probable_sepsis_count": [total_denominator],
            "antibiotics_rate": [overall_value],
        }
    )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)

    # Create display dataframe with proper column names
    display_columns = {
        period_col: "Period",
        "antibiotics_count": "Antibiotics Cases",
        "probable_sepsis_count": "Probable Sepsis Cases",
        "antibiotics_rate": "Antibiotics Rate (%)",
    }

    summary_table_display = summary_table.rename(columns=display_columns)

    # Reorder columns
    column_order = [
        "Period",
        "Antibiotics Cases",
        "Probable Sepsis Cases",
        "Antibiotics Rate (%)",
    ]
    summary_table_display = summary_table_display[column_order]

    summary_table_display.insert(0, "No", range(1, len(summary_table_display) + 1))

    # Format table
    styled_table = (
        summary_table_display.style.format(
            {
                "Antibiotics Rate (%)": "{:.1f}%",
                "Antibiotics Cases": "{:,.0f}",
                "Probable Sepsis Cases": "{:,.0f}",
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
        file_name="antibiotics_trend_summary.csv",
        mime="text/csv",
    )


def render_antibiotics_facility_comparison_chart(
    df,
    period_col="period_display",
    title="Antibiotics for Clinical Sepsis - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    tei_df=None,
):
    """Render facility comparison LINE CHART for antibiotics rate"""
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

    # Compute time series data for line chart
    time_series_data = []
    all_periods = sorted(df["period_display"].unique())

    for period_display in all_periods:
        period_df = df[df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_period_df = period_df[period_df["orgUnit"] == facility_uid]

            if not facility_period_df.empty:
                facility_kpi = compute_antibiotics_kpi(
                    facility_period_df, [facility_uid], tei_df
                )

                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": facility_kpi["antibiotics_rate"],
                        "antibiotics_count": facility_kpi["antibiotics_count"],
                        "probable_sepsis_count": facility_kpi["probable_sepsis_count"],
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

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Antibiotics for Clinical Sepsis (%)",
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
            facility_kpi = compute_antibiotics_kpi(facility_df, [facility_uid], tei_df)
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Antibiotics Cases": facility_kpi["antibiotics_count"],
                    "Probable Sepsis Cases": facility_kpi["probable_sepsis_count"],
                    "Antibiotics Rate": facility_kpi["antibiotics_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Antibiotics Cases"].sum()
    total_denominator = facility_table_df["Probable Sepsis Cases"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "Antibiotics Cases": total_numerator,
        "Probable Sepsis Cases": total_denominator,
        "Antibiotics Rate": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Facility Name",
        "Antibiotics Cases",
        "Probable Sepsis Cases",
        "Antibiotics Rate",
    ]
    facility_table_df = facility_table_df[column_order]

    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Antibiotics Cases": "{:,.0f}",
                "Probable Sepsis Cases": "{:,.0f}",
                "Antibiotics Rate": "{:.2f}%",
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
        file_name="antibiotics_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_antibiotics_region_comparison_chart(
    df,
    period_col="period_display",
    title="Antibiotics for Clinical Sepsis - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    facilities_by_region=None,
    tei_df=None,
):
    """Render region comparison LINE CHART for antibiotics rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # Compute time series data for line chart
    time_series_data = []
    all_periods = sorted(df["period_display"].unique())

    for period_display in all_periods:
        period_df = df[df["period_display"] == period_display]

        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

            if region_facility_uids:
                region_period_df = period_df[
                    period_df["orgUnit"].isin(region_facility_uids)
                ]

                if not region_period_df.empty:
                    region_kpi = compute_antibiotics_kpi(
                        region_period_df, region_facility_uids, tei_df
                    )

                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": region_kpi["antibiotics_rate"],
                            "antibiotics_count": region_kpi["antibiotics_count"],
                            "probable_sepsis_count": region_kpi[
                                "probable_sepsis_count"
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

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Antibiotics for Clinical Sepsis (%)",
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

        if region_facility_uids:
            region_df = df[df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                region_kpi = compute_antibiotics_kpi(
                    region_df, region_facility_uids, tei_df
                )
                region_table_data.append(
                    {
                        "Region Name": region_name,
                        "Antibiotics Cases": region_kpi["antibiotics_count"],
                        "Probable Sepsis Cases": region_kpi["probable_sepsis_count"],
                        "Antibiotics Rate": region_kpi["antibiotics_rate"],
                    }
                )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Antibiotics Cases"].sum()
    total_denominator = region_table_df["Probable Sepsis Cases"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "Antibiotics Cases": total_numerator,
        "Probable Sepsis Cases": total_denominator,
        "Antibiotics Rate": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Region Name",
        "Antibiotics Cases",
        "Probable Sepsis Cases",
        "Antibiotics Rate",
    ]
    region_table_df = region_table_df[column_order]

    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Antibiotics Cases": "{:,.0f}",
                "Probable Sepsis Cases": "{:,.0f}",
                "Antibiotics Rate": "{:.2f}%",
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
        file_name="antibiotics_rate_region_comparison.csv",
        mime="text/csv",
    )
