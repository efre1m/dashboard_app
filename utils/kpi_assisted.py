import pandas as pd
import plotly.express as px
import streamlit as st
from utils.kpi_utils import compute_total_deliveries

# Assisted Delivery KPI Configuration
ASSISTED_DELIVERY_STAGE_UID = "bwk9bBfYcsD"
INSTRUMENTAL_DELIVERY_UID = "K8BCYRU1TUP"
INSTRUMENTAL_YES_CODE = "1"


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

    # Filter to only include instrumental delivery stage events - FIXED: use programStage_uid
    instrumental_df = df[df["programStage_uid"] == ASSISTED_DELIVERY_STAGE_UID]

    if instrumental_df.empty:
        return {
            "assisted_delivery_rate": 0.0,
            "assisted_deliveries": 0,
            "total_deliveries": 0,
        }

    # Count assisted deliveries using vectorized operations
    assisted_deliveries = instrumental_df[
        (instrumental_df["dataElement_uid"] == INSTRUMENTAL_DELIVERY_UID)
        & (instrumental_df["value"] == INSTRUMENTAL_YES_CODE)
    ]["tei_id"].nunique()

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
    """Render a simple line chart for Assisted Delivery trend"""
    if text_color is None:
        text_color = "#000000" if bg_color == "#FFFFFF" else "#FFFFFF"

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = "period_display"

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Create simple line chart
    fig = px.line(
        df,
        x=x_axis_col,
        y=value_col,
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}%<extra></extra>",
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

    # Simple table (not styled)
    st.subheader(f"📋 {title} Summary")

    summary_df = df.copy().reset_index(drop=True)

    # Keep only relevant columns
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        summary_df = summary_df[
            [x_axis_col, numerator_name, denominator_name, value_col]
        ]
    else:
        summary_df = summary_df[[x_axis_col, value_col]]

    # Calculate overall value
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        total_numerator = summary_df[numerator_name].sum()
        total_denominator = summary_df[denominator_name].sum()
        overall_value = (
            (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        )

        overall_row = pd.DataFrame(
            {
                x_axis_col: ["Overall"],
                numerator_name: [total_numerator],
                denominator_name: [total_denominator],
                value_col: [overall_value],
            }
        )
    else:
        overall_value = summary_df[value_col].mean() if not summary_df.empty else 0
        overall_row = pd.DataFrame(
            {x_axis_col: ["Overall"], value_col: [overall_value]}
        )

    summary_table = pd.concat([summary_df, overall_row], ignore_index=True)

    # Add simple row numbering
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Display simple table without styling
    st.dataframe(summary_table, use_container_width=True)

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
    """Render a simple facility comparison line chart"""
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

    # Prepare comparison data using vectorized operations
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
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Create comparison line chart
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

    # Simple facility comparison table
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

    # Calculate overall
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

    # Add simple row numbering
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Display simple table
    st.dataframe(facility_table_df, use_container_width=True)

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
    """Render a simple region comparison line chart"""
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
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for region comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Create region comparison line chart
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

    # Simple region comparison table
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

    # Calculate overall
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

    # Add simple row numbering
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Display simple table
    st.dataframe(region_table_df, use_container_width=True)

    # Download button
    csv = region_table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="assisted_delivery_region_comparison.csv",
        mime="text/csv",
    )
