import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- Hypothermia KPI Constants ----------------
TEMPERATURE_ON_ADMISSION_UID = "gZi9y12E9i7"  # Temperature on admission (°C)
HYPOTHERMIA_THRESHOLD = 36.5  # in degree Celcius


# ---------------- Hypothermia KPI Computation Functions ----------------
def compute_hypothermia_numerator(df, facility_uids=None):
    """
    Compute numerator for hypothermia KPI: Count of newborns with temperature < 36.5°C on admission
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

    # Count unique newborns with temperature < 36.5°C
    hypothermia_cases = temp_events[temp_events["temp_value"] < HYPOTHERMIA_THRESHOLD][
        "tei_id"
    ].nunique()

    return hypothermia_cases


def compute_hypothermia_kpi(df, facility_uids=None, tei_df=None):
    """
    Compute hypothermia KPI for the given dataframe
    Uses unique TEI count from the filtered events dataframe for denominator

    Formula: Hypothermia on Admission Rate (%) =
             (Newborns with temperature < 36.5°C during period) ÷
             (Total admitted newborns in period) × 100
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "hypothermia_rate": 0.0,
            "hypothermia_count": 0,
            "total_admitted_newborns": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count hypothermia cases (numerator) - filters for temperature events
    hypothermia_count = compute_hypothermia_numerator(df, facility_uids)

    # ✅ FIX: Count unique TEIs in THIS PERIOD only (not total)
    total_admitted_newborns = df["tei_id"].nunique()

    # Calculate hypothermia rate
    hypothermia_rate = (
        (hypothermia_count / total_admitted_newborns * 100)
        if total_admitted_newborns > 0
        else 0.0
    )

    return {
        "hypothermia_rate": float(hypothermia_rate),
        "hypothermia_count": int(hypothermia_count),
        "total_admitted_newborns": int(total_admitted_newborns),
    }


def compute_hypothermia_trend_data(
    df, period_col="period_display", facility_uids=None, tei_df=None
):
    """
    Compute hypothermia trend data by period - COUNT UNIQUE NEWBORNS PER PERIOD
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

    # ✅ FIX: Track counted newborns across periods to prevent double-counting
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

        # ✅ FIX: Get newborns in this period who haven't been counted yet
        period_newborns = set(period_df["tei_id"].unique())
        new_newborns = period_newborns - counted_newborns

        if new_newborns:
            # Filter to only new newborns in this period
            new_newborns_df = period_df[period_df["tei_id"].isin(new_newborns)]

            # Compute KPI for only the new newborns in this period
            hypothermia_data = compute_hypothermia_kpi(
                new_newborns_df, facility_uids, tei_df
            )

            # ✅ Update counted newborns for next period
            counted_newborns.update(new_newborns)
        else:
            # No new newborns in this period
            hypothermia_data = {
                "hypothermia_rate": 0.0,
                "hypothermia_count": 0,
                "total_admitted_newborns": 0,
            }

        trend_data.append(
            {
                period_col: period_display,
                "hypothermia_count": hypothermia_data["hypothermia_count"],
                "total_admitted_newborns": hypothermia_data["total_admitted_newborns"],
                "hypothermia_rate": hypothermia_data["hypothermia_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- Hypothermia Chart Functions ----------------
def render_hypothermia_trend_chart(
    df,
    period_col="period_display",
    title="Hypothermia on Admission Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
    tei_df=None,
):
    """Render a LINE CHART ONLY for hypothermia rate trend WITH CHRONOLOGICAL ORDERING"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Compute trend data with proper period-based counting
    trend_df = compute_hypothermia_trend_data(df, period_col, facility_uids, tei_df)

    if trend_df.empty:
        st.info("⚠️ No hypothermia data available for the selected period.")
        return

    trend_df = trend_df.copy()
    trend_df["hypothermia_rate"] = pd.to_numeric(
        trend_df["hypothermia_rate"], errors="coerce"
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
        y="hypothermia_rate",
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
    if len(trend_df) > 1:
        last_value = trend_df["hypothermia_rate"].iloc[-1]
        prev_value = trend_df["hypothermia_rate"].iloc[-2]
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
    st.subheader("📋 Hypothermia Trend Summary Table")
    summary_df = trend_df.copy().reset_index(drop=True)

    # Calculate overall value - SUM of numerators and denominators
    total_numerator = summary_df["hypothermia_count"].sum()
    total_denominator = summary_df["total_admitted_newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = pd.DataFrame(
        {
            period_col: ["Overall"],
            "hypothermia_count": [total_numerator],
            "total_admitted_newborns": [total_denominator],
            "hypothermia_rate": [overall_value],
        }
    )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)

    # Create display dataframe with proper column names
    display_columns = {
        period_col: "Period",
        "hypothermia_count": "Hypothermia Cases",
        "total_admitted_newborns": "Total Admitted Newborns",
        "hypothermia_rate": "Hypothermia Rate (%)",
    }

    # Rename columns for display
    summary_table_display = summary_table.rename(columns=display_columns)

    # Reorder columns
    column_order = [
        "Period",
        "Hypothermia Cases",
        "Total Admitted Newborns",
        "Hypothermia Rate (%)",
    ]
    summary_table_display = summary_table_display[column_order]

    summary_table_display.insert(0, "No", range(1, len(summary_table_display) + 1))

    # Format table
    styled_table = (
        summary_table_display.style.format(
            {
                "Hypothermia Rate (%)": "{:.1f}%",
                "Hypothermia Cases": "{:,.0f}",
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
        file_name="hypothermia_trend_summary.csv",
        mime="text/csv",
    )


def render_hypothermia_facility_comparison_chart(
    df,
    period_col="period_display",
    title="Hypothermia on Admission - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    tei_df=None,
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
        key=f"chart_type_facility_comparison_{key_suffix}",  # ✅ Unique key
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
                    hypothermia_data = compute_hypothermia_kpi(
                        facility_period_df, [facility_uid], tei_df
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": hypothermia_data["hypothermia_rate"],
                            "hypothermia_count": hypothermia_data["hypothermia_count"],
                            "total_admitted_newborns": hypothermia_data[
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
                hypothermia_data = compute_hypothermia_kpi(
                    facility_df, [facility_uid], tei_df
                )
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": hypothermia_data["hypothermia_rate"],
                        "hypothermia_count": hypothermia_data["hypothermia_count"],
                        "total_admitted_newborns": hypothermia_data[
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

    # Facility comparison table - PROPER COLUMN NAMES
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            hypothermia_data = compute_hypothermia_kpi(
                facility_df, [facility_uid], tei_df
            )
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Hypothermia Cases": hypothermia_data["hypothermia_count"],
                    "Total Admitted Newborns": hypothermia_data[
                        "total_admitted_newborns"
                    ],
                    "Hypothermia Rate": hypothermia_data["hypothermia_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Hypothermia Cases"].sum()
    total_denominator = facility_table_df["Total Admitted Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "Hypothermia Cases": total_numerator,
        "Total Admitted Newborns": total_denominator,
        "Hypothermia Rate": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Facility Name",
        "Hypothermia Cases",
        "Total Admitted Newborns",
        "Hypothermia Rate",
    ]
    facility_table_df = facility_table_df[column_order]

    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Hypothermia Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
                "Hypothermia Rate": "{:.2f}%",
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
    title="Hypothermia on Admission - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    facilities_by_region=None,
    tei_df=None,
):
    """Render region comparison with BOTH LINE AND BAR CHART options for hypothermia rate"""
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
        key=f"chart_type_region_comparison_{key_suffix}",  # ✅ Unique key
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
                        hypothermia_data = compute_hypothermia_kpi(
                            region_period_df, region_facility_uids, tei_df
                        )
                        time_series_data.append(
                            {
                                "period_display": period_display,
                                "Region": region_name,
                                "value": hypothermia_data["hypothermia_rate"],
                                "hypothermia_count": hypothermia_data[
                                    "hypothermia_count"
                                ],
                                "total_admitted_newborns": hypothermia_data[
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
                    hypothermia_data = compute_hypothermia_kpi(
                        region_df, region_facility_uids, tei_df
                    )
                    bar_data.append(
                        {
                            "Region": region_name,
                            "value": hypothermia_data["hypothermia_rate"],
                            "hypothermia_count": hypothermia_data["hypothermia_count"],
                            "total_admitted_newborns": hypothermia_data[
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
                hypothermia_data = compute_hypothermia_kpi(
                    region_df, region_facility_uids, tei_df
                )
                region_table_data.append(
                    {
                        "Region Name": region_name,
                        "Hypothermia Cases": hypothermia_data["hypothermia_count"],
                        "Total Admitted Newborns": hypothermia_data[
                            "total_admitted_newborns"
                        ],
                        "Hypothermia Rate": hypothermia_data["hypothermia_rate"],
                    }
                )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Hypothermia Cases"].sum()
    total_denominator = region_table_df["Total Admitted Newborns"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "Hypothermia Cases": total_numerator,
        "Total Admitted Newborns": total_denominator,
        "Hypothermia Rate": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Reorder columns
    column_order = [
        "Region Name",
        "Hypothermia Cases",
        "Total Admitted Newborns",
        "Hypothermia Rate",
    ]
    region_table_df = region_table_df[column_order]

    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Hypothermia Cases": "{:,.0f}",
                "Total Admitted Newborns": "{:,.0f}",
                "Hypothermia Rate": "{:.2f}%",
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
