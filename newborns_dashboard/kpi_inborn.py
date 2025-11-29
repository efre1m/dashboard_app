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

    for period in sorted(df[period_col].unique()):
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
    """Render a LINE CHART ONLY for inborn rate trend"""
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

    # Create line chart
    fig = px.line(
        trend_df,
        x=period_col,
        y="inborn_rate",
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
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
    """Render facility comparison with BOTH LINE AND BAR CHART options for inborn rate"""
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
        key=f"chart_type_facility_comparison_inborn_{key_suffix}",
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
                    # ✅ FIX: Use direct event counting for numerator in trend
                    birth_location_mask = (
                        facility_period_df["dataElement_uid"] == BIRTH_LOCATION_UID
                    ) & (facility_period_df["value"] == INBORN_CODE)
                    inborn_count = birth_location_mask.sum()

                    total_admitted_newborns = facility_period_df["tei_id"].nunique()

                    inborn_rate = (
                        (inborn_count / total_admitted_newborns * 100)
                        if total_admitted_newborns > 0
                        else 0.0
                    )

                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": inborn_rate,
                            "inborn_count": inborn_count,
                            "total_admitted_newborns": total_admitted_newborns,
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
                inborn_data = compute_inborn_kpi(facility_df, [facility_uid], tei_df)
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": inborn_data["inborn_rate"],
                        "inborn_count": inborn_data["inborn_count"],
                        "total_admitted_newborns": inborn_data[
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
            x="Facility",
            y="value",
            title=title,
            height=500,
            color="Facility",
        )

        # Update traces for bar chart
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Inborn Rate: %{y:.2f}%<extra></extra>",
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="Inborn Babies (%)",
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
    """Render region comparison with BOTH LINE AND BAR CHART options for inborn rate"""
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
        key=f"chart_type_region_comparison_inborn_{key_suffix}",
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
                        # ✅ FIX: Use direct event counting for numerator in trend
                        birth_location_mask = (
                            region_period_df["dataElement_uid"] == BIRTH_LOCATION_UID
                        ) & (region_period_df["value"] == INBORN_CODE)
                        inborn_count = birth_location_mask.sum()

                        total_admitted_newborns = region_period_df["tei_id"].nunique()

                        inborn_rate = (
                            (inborn_count / total_admitted_newborns * 100)
                            if total_admitted_newborns > 0
                            else 0.0
                        )

                        time_series_data.append(
                            {
                                "period_display": period_display,
                                "Region": region_name,
                                "value": inborn_rate,
                                "inborn_count": inborn_count,
                                "total_admitted_newborns": total_admitted_newborns,
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
                    inborn_data = compute_inborn_kpi(
                        region_df, region_facility_uids, tei_df
                    )
                    bar_data.append(
                        {
                            "Region": region_name,
                            "value": inborn_data["inborn_rate"],
                            "inborn_count": inborn_data["inborn_count"],
                            "total_admitted_newborns": inborn_data[
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
            hovertemplate="<b>%{x}</b><br>Inborn Rate: %{y:.2f}%<extra></extra>",
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="Inborn Babies (%)",
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
