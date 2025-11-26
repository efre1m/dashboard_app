import pandas as pd
import plotly.express as px
import streamlit as st
from utils.kpi_utils import compute_total_deliveries

# SVD KPI Configuration - Using same UIDs and logic as C-section
DELIVERY_TYPE_UID = "lphtwP2ViZU"  # Same as C-section
SVD_CODE = "1"  # Normal Vaginal Delivery code


def compute_svd_count(df, facility_uids=None):
    """Count SVD occurrences (not unique patients) - EXACT same logic as C-section"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for numerator calculation
    if "has_actual_event" in df.columns:
        actual_events_df = df[df["has_actual_event"] == True]
    else:
        actual_events_df = df

    # Count occurrences (not unique patients) - EXACT same as C-section
    svd_count = len(
        actual_events_df[
            (actual_events_df["dataElement_uid"] == DELIVERY_TYPE_UID)
            & (actual_events_df["value"] == SVD_CODE)
        ]
    )

    return svd_count


def compute_svd_kpi(df, facility_uids=None):
    """
    Compute SVD (Normal Vaginal Delivery) Rate using EXACT same logic as C-section
    """
    if df is None or df.empty:
        return {"svd_rate": 0.0, "svd_deliveries": 0, "total_deliveries": 0}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter out placeholder events for calculation
    if "has_actual_event" in df.columns:
        actual_events_df = df[df["has_actual_event"] == True]
    else:
        actual_events_df = df

    if actual_events_df.empty:
        return {"svd_rate": 0.0, "svd_deliveries": 0, "total_deliveries": 0}

    # Count SVD deliveries using EXACT same logic as C-section
    svd_deliveries = compute_svd_count(df, facility_uids)

    # Get total deliveries using EXACT same logic as C-section
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Calculate rate using same formula as C-section
    rate = (svd_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0

    return {
        "svd_rate": float(rate),
        "svd_deliveries": int(svd_deliveries),
        "total_deliveries": int(total_deliveries),  # Consistent naming with C-section
    }


def render_svd_trend_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names=None,
    numerator_name="SVD Deliveries",
    denominator_name="Total Deliveries",  # Changed from "Total Admissions" to match C-section
    facility_uids=None,
):
    """Render trend chart for SVD with same styling as C-section"""
    if text_color is None:
        text_color = "#000000" if bg_color == "#FFFFFF" else "#FFFFFF"

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = "period_display"

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Create the same chart options as C-section
    chart_options = ["Line", "Bar", "Gauge"]

    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_svd_{str(facility_uids)}",
    ).lower()

    # Handle gauge chart type (same as C-section)
    if chart_type == "gauge":
        # Compute overall KPI value from the original filtered data
        original_filtered_events = st.session_state.get(
            "filtered_events", pd.DataFrame()
        )

        if not original_filtered_events.empty:
            # Compute the KPI for the entire date range
            kpi_data = compute_svd_kpi(original_filtered_events, facility_uids)
            gauge_value = kpi_data["svd_rate"]
        else:
            gauge_value = df[value_col].mean() if not df.empty else 0

        # Use same gauge rendering as C-section
        from utils.kpi_utils import render_gauge_chart

        render_gauge_chart(
            gauge_value,
            f"{title} (Overall)",
            bg_color,
            text_color,
            min_val=0,
            max_val=100,
            reverse_colors=False,  # Higher SVD rate is good, same as C-section
        )
        return

    # Create custom hover text with numerator and denominator if available
    hover_data = {}
    if numerator_name in df.columns and denominator_name in df.columns:
        hover_data = {numerator_name: True, denominator_name: True}

    # Create chart based on selected type (same as C-section)
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

    # Apply same styling as C-section
    if chart_type in ["line", "area"]:
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate=f"<b>%{{x}}</b><br>SVD Rate: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
        )
    elif chart_type == "bar":
        fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>SVD Rate: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>"
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="SVD Rate (%)",
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

    # Show trend analysis (same as C-section)
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

    # Enhanced summary table (same as C-section)
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

    # Format table with same styling as C-section
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
        file_name=f"svd_delivery_trend.csv",
        mime="text/csv",
    )


def render_svd_facility_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names,
    facility_uids,
    numerator_name="SVD Deliveries",
    denominator_name="Total Deliveries",  # Consistent naming
):
    """Render facility comparison chart with same logic as C-section"""
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

    # Prepare comparison data using same approach as C-section
    comparison_data = []

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_df = period_df[period_df["orgUnit"] == facility_uid]
            if not facility_df.empty:
                kpi_value = compute_svd_kpi(facility_df, [facility_uid])

                comparison_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": kpi_value["svd_rate"],
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
        yaxis_title="SVD Rate (%)",
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

    # Enhanced facility comparison table (same as C-section)
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if facility_df.empty:
            continue

        kpi_data = compute_svd_kpi(facility_df, [facility_uid])

        facility_table_data.append(
            {
                "Facility Name": facility_name,
                "SVD Deliveries": kpi_data["svd_deliveries"],
                "Total Deliveries": kpi_data["total_deliveries"],
                "SVD Rate (%)": kpi_data["svd_rate"],
            }
        )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall using same aggregation logic as C-section
    total_svd = facility_table_df["SVD Deliveries"].sum()
    total_deliveries = facility_table_df["Total Deliveries"].sum()
    overall_rate = (total_svd / total_deliveries * 100) if total_deliveries > 0 else 0

    overall_row = {
        "Facility Name": "Overall",
        "SVD Deliveries": total_svd,
        "Total Deliveries": total_deliveries,
        "SVD Rate (%)": overall_rate,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Add row numbering and formatting
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format with same styling as C-section
    styled_table = (
        facility_table_df.style.format(
            {
                "SVD Deliveries": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "SVD Rate (%)": "{:.2f}",
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
        file_name="svd_delivery_facility_comparison.csv",
        mime="text/csv",
    )


def render_svd_region_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    region_names,
    region_mapping,
    facilities_by_region,
    numerator_name="SVD Deliveries",
    denominator_name="Total Deliveries",  # Consistent naming
):
    """Render region comparison chart with same logic as C-section"""
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
                kpi_value = compute_svd_kpi(region_df, region_facility_uids)

                comparison_data.append(
                    {
                        "period_display": period_display,
                        "Region": region_name,
                        "value": kpi_value["svd_rate"],
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
        yaxis_title="SVD Rate (%)",
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

    # Enhanced region comparison table (same as C-section)
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        facility_uids = [uid for _, uid in facilities_by_region.get(region_name, [])]
        region_df = df[df["orgUnit"].isin(facility_uids)]

        if region_df.empty:
            continue

        kpi_data = compute_svd_kpi(region_df, facility_uids)

        region_table_data.append(
            {
                "Region Name": region_name,
                "SVD Deliveries": kpi_data["svd_deliveries"],
                "Total Deliveries": kpi_data["total_deliveries"],
                "SVD Rate (%)": kpi_data["svd_rate"],
            }
        )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall using same aggregation logic
    total_svd = region_table_df["SVD Deliveries"].sum()
    total_deliveries = region_table_df["Total Deliveries"].sum()
    overall_rate = (total_svd / total_deliveries * 100) if total_deliveries > 0 else 0

    overall_row = {
        "Region Name": "Overall",
        "SVD Deliveries": total_svd,
        "Total Deliveries": total_deliveries,
        "SVD Rate (%)": overall_rate,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )

    # Add row numbering and formatting
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format with same styling as C-section
    styled_table = (
        region_table_df.style.format(
            {
                "SVD Deliveries": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "SVD Rate (%)": "{:.2f}",
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
        file_name="svd_delivery_region_comparison.csv",
        mime="text/csv",
    )
