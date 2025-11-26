import pandas as pd
import plotly.express as px
import streamlit as st
from utils.kpi_utils import compute_total_deliveries

# Assisted Delivery KPI Configuration
ASSISTED_DELIVERY_STAGE_UID = "bwk9bBfYcsD"
INSTRUMENTAL_DELIVERY_UID = "K8BCYRU1TUP"
INSTRUMENTAL_YES_CODE = "1"


def compute_assisted_count(df, facility_uids=None):
    """Count Assisted Delivery occurrences - COUNT OCCURRENCES of instrumental delivery = 1"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter to only include instrumental delivery stage events
    instrumental_df = df[df["programStage_uid"] == ASSISTED_DELIVERY_STAGE_UID]

    if instrumental_df.empty:
        return 0

    # Count assisted deliveries using vectorized operations - COUNT OCCURRENCES
    assisted_count = len(
        instrumental_df[
            (instrumental_df["dataElement_uid"] == INSTRUMENTAL_DELIVERY_UID)
            & (instrumental_df["value"] == INSTRUMENTAL_YES_CODE)
        ]
    )

    return assisted_count


def compute_assisted_delivery_kpi(df, facility_uids=None):
    """
    Optimized computation for Assisted Delivery Rate
    Uses vectorized operations instead of loops
    """
    if df is None or df.empty:
        return {
            "assisted_delivery_rate": 0.0,
            "assisted_deliveries": 0,
            "total_deliveries": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count assisted deliveries - COUNT OCCURRENCES
    assisted_deliveries = compute_assisted_count(df, facility_uids)

    # Get total deliveries (using the same logic as other KPIs)
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Calculate rate
    rate = (
        (assisted_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0
    )

    return {
        "assisted_delivery_rate": float(rate),
        "assisted_deliveries": int(assisted_deliveries),
        "total_deliveries": int(total_deliveries),
    }


def render_assisted_trend_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names=None,
    numerator_name="Assisted Deliveries",
    denominator_name="Total Deliveries",
    facility_uids=None,
):
    """Render trend chart for Assisted Delivery with same styling as C-section and SVD"""
    if text_color is None:
        text_color = "#000000" if bg_color == "#FFFFFF" else "#FFFFFF"

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = "period_display"

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Create the same chart options as C-section and SVD
    chart_options = ["Line", "Bar", "Gauge"]

    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_assisted_{str(facility_uids)}",
    ).lower()

    # Handle gauge chart type (same as C-section and SVD)
    if chart_type == "gauge":
        # Compute overall KPI value from the original filtered data
        original_filtered_events = st.session_state.get(
            "filtered_events", pd.DataFrame()
        )

        if not original_filtered_events.empty:
            # Compute the KPI for the entire date range
            kpi_data = compute_assisted_delivery_kpi(
                original_filtered_events, facility_uids
            )
            gauge_value = kpi_data["assisted_delivery_rate"]
        else:
            gauge_value = df[value_col].mean() if not df.empty else 0

        # Use same gauge rendering as C-section and SVD
        from utils.kpi_utils import render_gauge_chart

        render_gauge_chart(
            gauge_value,
            f"{title} (Overall)",
            bg_color,
            text_color,
            min_val=0,
            max_val=100,
            reverse_colors=False,
        )
        return

    # Create custom hover text with numerator and denominator if available
    hover_data = {}
    if numerator_name in df.columns and denominator_name in df.columns:
        hover_data = {numerator_name: True, denominator_name: True}

    # Create chart based on selected type (same as C-section and SVD)
    if chart_type == "line":
        fig = px.line(
            df,
            x=x_axis_col,
            y=value_col,
            markers=True,
            line_shape="linear",
            title=title,
            height=400,
            hover_data=hover_data,
        )
    elif chart_type == "bar":
        fig = px.bar(
            df,
            x=x_axis_col,
            y=value_col,
            title=title,
            height=400,
            hover_data=hover_data,
        )
    else:
        fig = px.line(
            df,
            x=x_axis_col,
            y=value_col,
            markers=True,
            line_shape="linear",
            title=title,
            height=400,
            hover_data=hover_data,
        )

    # Apply same styling as C-section and SVD
    if chart_type in ["line", "area"]:
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate=f"<b>%{{x}}</b><br>Assisted Delivery Rate: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
        )
    elif chart_type == "bar":
        fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>Assisted Delivery Rate: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>"
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Assisted Delivery Rate (%)",
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

    # Show trend analysis (same as C-section and SVD)
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

    # Enhanced summary table (same as C-section and SVD)
    st.subheader(f"📋 {title} Summary Table")
    summary_df = df.copy().reset_index(drop=True)

    # Keep only relevant columns
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        summary_df = summary_df[
            [x_axis_col, numerator_name, denominator_name, value_col]
        ]
    else:
        summary_df = summary_df[[x_axis_col, value_col]]

    # Calculate overall value using same formula as individual periods
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        total_numerator = summary_df[numerator_name].sum()
        total_denominator = summary_df[denominator_name].sum()
        overall_value = (
            (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        )

        overall_row = pd.DataFrame(
            {
                x_axis_col: [f"Overall {title}"],
                numerator_name: [total_numerator],
                denominator_name: [total_denominator],
                value_col: [overall_value],
            }
        )
    else:
        overall_value = summary_df[value_col].mean() if not summary_df.empty else 0
        overall_row = pd.DataFrame(
            {x_axis_col: [f"Overall {title}"], value_col: [overall_value]}
        )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Format table with same styling as C-section and SVD
    if (
        numerator_name in summary_table.columns
        and denominator_name in summary_table.columns
    ):
        styled_table = (
            summary_table.style.format(
                {
                    value_col: "{:.2f}",
                    numerator_name: "{:,.0f}",
                    denominator_name: "{:,.0f}",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )
    else:
        styled_table = (
            summary_table.style.format({value_col: "{:.2f}"})
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"assisted_delivery_trend.csv",
        mime="text/csv",
    )


def render_assisted_facility_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names,
    facility_uids,
    numerator_name="Assisted Deliveries",
    denominator_name="Total Deliveries",
):
    """Render facility comparison chart with same logic as C-section and SVD"""
    if text_color is None:
        text_color = "#000000" if bg_color == "#FFFFFF" else "#FFFFFF"

    # Create facility mapping
    facility_uid_to_name = dict(zip(facility_uids, facility_names))

    # Filter to selected facilities
    filtered_df = df[df["orgUnit"].isin(facility_uids)].copy()

    if filtered_df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Get chronological period order
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    # Prepare comparison data using same approach as C-section and SVD
    comparison_data = []

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_df = period_df[period_df["orgUnit"] == facility_uid]
            if not facility_df.empty:
                kpi_value = compute_assisted_delivery_kpi(facility_df, [facility_uid])

                comparison_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": kpi_value["assisted_delivery_rate"],
                        "Assisted Deliveries": kpi_value["assisted_deliveries"],
                        "Total Deliveries": kpi_value["total_deliveries"],
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Create comparison line chart with same styling
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=f"{title} - Facility Comparison",
        height=500,
        category_orders={"period_display": period_order},
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Assisted Delivery Rate (%)",
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
            title="Facilities",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Enhanced facility comparison table (same as C-section and SVD)
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if facility_df.empty:
            continue

        kpi_data = compute_assisted_delivery_kpi(facility_df, [facility_uid])

        facility_table_data.append(
            {
                "Facility Name": facility_name,
                "Assisted Deliveries": kpi_data["assisted_deliveries"],
                "Total Deliveries": kpi_data["total_deliveries"],
                "Assisted Delivery Rate (%)": kpi_data["assisted_delivery_rate"],
            }
        )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall using same aggregation logic as C-section and SVD
    total_assisted = facility_table_df["Assisted Deliveries"].sum()
    total_deliveries = facility_table_df["Total Deliveries"].sum()
    overall_rate = (
        (total_assisted / total_deliveries * 100) if total_deliveries > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "Assisted Deliveries": total_assisted,
        "Total Deliveries": total_deliveries,
        "Assisted Delivery Rate (%)": overall_rate,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Add row numbering and formatting
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format with same styling as C-section and SVD
    styled_table = (
        facility_table_df.style.format(
            {
                "Assisted Deliveries": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "Assisted Delivery Rate (%)": "{:.2f}",
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
        file_name="assisted_delivery_facility_comparison.csv",
        mime="text/csv",
    )


def render_assisted_region_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    region_names,
    region_mapping,
    facilities_by_region,
    numerator_name="Assisted Deliveries",
    denominator_name="Total Deliveries",
):
    """Render region comparison chart with same logic as C-section and SVD"""
    if text_color is None:
        text_color = "#000000" if bg_color == "#FFFFFF" else "#FFFFFF"

    # Get all facility UIDs for selected regions
    all_facility_uids = []
    for region_name in region_names:
        facility_uids = [uid for _, uid in facilities_by_region.get(region_name, [])]
        all_facility_uids.extend(facility_uids)

    # Filter to regions
    filtered_df = df[df["orgUnit"].isin(all_facility_uids)].copy()

    if filtered_df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Get chronological period order
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    # Prepare comparison data
    comparison_data = []

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_df = period_df[period_df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                kpi_value = compute_assisted_delivery_kpi(
                    region_df, region_facility_uids
                )

                comparison_data.append(
                    {
                        "period_display": period_display,
                        "Region": region_name,
                        "value": kpi_value["assisted_delivery_rate"],
                        "Assisted Deliveries": kpi_value["assisted_deliveries"],
                        "Total Deliveries": kpi_value["total_deliveries"],
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for region comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Create region comparison line chart with same styling
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=f"{title} - Region Comparison",
        height=500,
        category_orders={"period_display": period_order},
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Assisted Delivery Rate (%)",
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
            title="Regions",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Enhanced region comparison table (same as C-section and SVD)
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        facility_uids = [uid for _, uid in facilities_by_region.get(region_name, [])]
        region_df = df[df["orgUnit"].isin(facility_uids)]

        if region_df.empty:
            continue

        kpi_data = compute_assisted_delivery_kpi(region_df, facility_uids)

        region_table_data.append(
            {
                "Region Name": region_name,
                "Assisted Deliveries": kpi_data["assisted_deliveries"],
                "Total Deliveries": kpi_data["total_deliveries"],
                "Assisted Delivery Rate (%)": kpi_data["assisted_delivery_rate"],
            }
        )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall using same aggregation logic
    total_assisted = region_table_df["Assisted Deliveries"].sum()
    total_deliveries = region_table_df["Total Deliveries"].sum()
    overall_rate = (
        (total_assisted / total_deliveries * 100) if total_deliveries > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "Assisted Deliveries": total_assisted,
        "Total Deliveries": total_deliveries,
        "Assisted Delivery Rate (%)": overall_rate,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Add row numbering and formatting
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format with same styling as C-section and SVD
    styled_table = (
        region_table_df.style.format(
            {
                "Assisted Deliveries": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "Assisted Delivery Rate (%)": "{:.2f}",
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
        file_name="assisted_delivery_region_comparison.csv",
        mime="text/csv",
    )
