import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Import the same function used by other KPIs
from utils.kpi_utils import compute_total_deliveries, auto_text_color


# ---------------- PPH KPI Constants ----------------
PPH_CONDITION_UID = "CJiTafFo0TS"
PPH_CODE = "3"  # Postpartum Hemorrhage (PPH) option code

# Option Set Values for reference:
# 0: None
# 1: Obstructed
# 2: Eclampsia
# 3: Postpartum Hemorrhage (PPH) ← use for numerator
# 4: Antepartum Hemorrhage (APH)
# 5: PROM / Sepsis
# 6: Ruptured Uterus
# 7: Prolonged Labor
# 8: Repaired Uterus
# 9: Hysterectomy
# 10: Other specify


# ---------------- PPH KPI Computation Functions ----------------
def compute_pph_kpi(df, facility_uids=None):
    """
    Compute PPH KPI for the given dataframe

    Formula: PPH Rate (%) = (Count of deliveries where condition = "PPH") ÷ (Total Deliveries) × 100

    Returns:
        Dictionary with PPH metrics
    """
    if df is None or df.empty:
        return {"pph_rate": 0.0, "pph_count": 0, "total_deliveries": 0}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Use the EXACT SAME total deliveries calculation as other KPIs
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Filter for PPH cases specifically (where obstetric condition = PPH)
    pph_cases = df[
        (df["dataElement_uid"] == PPH_CONDITION_UID)
        & (df["value"] == PPH_CODE)
        & df["value"].notna()
    ]
    pph_count = pph_cases["tei_id"].nunique()

    # Calculate PPH rate
    pph_rate = (pph_count / total_deliveries * 100) if total_deliveries > 0 else 0.0

    return {
        "pph_rate": float(pph_rate),
        "pph_count": int(pph_count),
        "total_deliveries": int(total_deliveries),
    }


def compute_obstetric_condition_distribution(df, facility_uids=None):
    """
    Compute distribution of all obstetric conditions

    Returns:
        DataFrame with columns: condition, count, percentage
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for delivery summary events with obstetric condition data
    delivery_events = df[
        (df["dataElement_uid"] == PPH_CONDITION_UID) & df["value"].notna()
    ]

    if delivery_events.empty:
        return pd.DataFrame()

    # Count occurrences of each condition
    condition_counts = delivery_events["value"].value_counts().reset_index()
    condition_counts.columns = ["condition_code", "count"]

    # Map condition codes to human-readable names
    condition_mapping = {
        "0": "None",
        "1": "Obstructed",
        "2": "Eclampsia",
        "3": "Postpartum Hemorrhage (PPH)",
        "4": "Antepartum Hemorrhage (APH)",
        "5": "PROM / Sepsis",
        "6": "Ruptured Uterus",
        "7": "Prolonged Labor",
        "8": "Repaired Uterus",
        "9": "Hysterectomy",
        "10": "Other specify",
    }

    condition_counts["condition"] = condition_counts["condition_code"].map(
        condition_mapping
    )

    # Calculate percentages
    total_count = condition_counts["count"].sum()
    condition_counts["percentage"] = (
        (condition_counts["count"] / total_count * 100) if total_count > 0 else 0
    )

    return condition_counts[["condition", "count", "percentage"]]


# ---------------- PPH Chart Functions ----------------
def render_pph_trend_chart(
    df,
    period_col="period_display",
    value_col="pph_rate",
    title="PPH Rate Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="pph_count",
    denominator_name="total_deliveries",
    facility_uids=None,
):
    """Render a trend chart for PPH rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Chart options
    chart_options = ["Line", "Bar"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_{title}_{str(facility_uids)}",
    ).lower()

    # Create hover data
    hover_data = {}
    if numerator_name in df.columns and denominator_name in df.columns:
        hover_data = {numerator_name: True, denominator_name: True}

    # Create chart
    if chart_type == "line":
        fig = px.line(
            df,
            x=period_col,
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
            x=period_col,
            y=value_col,
            title=title,
            height=400,
            hover_data=hover_data,
        )
    else:
        fig = px.line(
            df,
            x=period_col,
            y=value_col,
            markers=True,
            line_shape="linear",
            title=title,
            height=400,
            hover_data=hover_data,
        )

    # Update traces
    if chart_type in ["line"]:
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
        )
    elif chart_type == "bar":
        fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>"
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="PPH Rate (%)",
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

    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        summary_df = summary_df[
            [period_col, numerator_name, denominator_name, value_col]
        ]
    else:
        summary_df = summary_df[[period_col, value_col]]

    # Calculate overall value
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
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
    else:
        overall_value = summary_df[value_col].mean() if not summary_df.empty else 0
        overall_row = pd.DataFrame(
            {period_col: [f"Overall {title}"], value_col: [overall_value]}
        )
        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)

    # Add row numbering
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Format table
    if (
        numerator_name in summary_table.columns
        and denominator_name in summary_table.columns
    ):
        styled_table = (
            summary_table.style.format(
                {
                    value_col: "{:.2f}%",
                    numerator_name: "{:,.0f}",
                    denominator_name: "{:,.0f}",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )
    else:
        styled_table = (
            summary_table.style.format({value_col: "{:.2f}%"})
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


def render_pph_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="pph_rate",
    title="PPH Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
):
    """Render a comparison chart showing each facility's PPH performance with heatmap option"""
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

    # Prepare comparison data
    comparison_data = []
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_df = period_df[period_df["orgUnit"] == facility_uid]
            if not facility_df.empty:
                pph_data = compute_pph_kpi(facility_df, [facility_uid])
                comparison_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": pph_data["pph_rate"],
                        "pph_count": pph_data["pph_count"],
                        "total_deliveries": pph_data["total_deliveries"],
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Chart options - Line Chart vs Heatmap
    chart_options = ["Line Chart", "Heatmap"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_facility_comparison_{str(facility_uids)}",
    )

    if chart_type == "Heatmap":
        # Call the heatmap function with facility label
        render_pph_heatmap(
            df,
            facility_names,
            period_order,
            bg_color,
            text_color,
            entity_label="Facility",
        )

        # Generate heatmap CSV for download
        heatmap_data = []
        for facility_name in facility_names:
            for period in period_order:
                period_data = comparison_df[
                    (comparison_df["Facility"] == facility_name)
                    & (comparison_df["period_display"] == period)
                ]
                if not period_data.empty:
                    pph_rate = period_data["value"].iloc[0]
                    pph_count = period_data["pph_count"].iloc[0]
                    total_deliveries = period_data["total_deliveries"].iloc[0]
                else:
                    pph_rate = 0.0
                    pph_count = 0
                    total_deliveries = 0

                heatmap_data.append(
                    {
                        "Facility": facility_name,
                        "Period": period,
                        "PPH Rate": pph_rate,
                        "PPH Cases": pph_count,
                        "Total Deliveries": total_deliveries,
                    }
                )

        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot(
            index="Facility", columns="Period", values="PPH Rate"
        )

        # Download button for heatmap CSV
        csv = pivot_df.to_csv()
        st.download_button(
            label="Download Heatmap CSV",
            data=csv,
            file_name="pph_rate_facility_heatmap.csv",
            mime="text/csv",
            key="facility_heatmap_download",
        )

        return  # Exit early for heatmap view

    # Line chart view (original functionality)
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=title,
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
        yaxis_title="PPH Rate (%)",
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

    # Facility comparison table
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            pph_data = compute_pph_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "PPH Cases": pph_data["pph_count"],
                    "Total Deliveries": pph_data["total_deliveries"],
                    "PPH Rate (%)": pph_data["pph_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["PPH Cases"].sum()
    total_denominator = facility_table_df["Total Deliveries"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "PPH Cases": total_numerator,
        "Total Deliveries": total_denominator,
        "PPH Rate (%)": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "PPH Cases": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "PPH Rate (%)": "{:.2f}%",
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
        file_name="pph_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_pph_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="pph_rate",
    title="PPH Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
):
    """Render a comparison chart showing each region's PPH performance with heatmap option"""
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

    # Prepare comparison data
    comparison_data = []
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    for period_display in period_order:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_df = period_df[period_df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                pph_data = compute_pph_kpi(region_df, region_facility_uids)
                comparison_data.append(
                    {
                        "period_display": period_display,
                        "Region": region_name,
                        "value": pph_data["pph_rate"],
                        "pph_count": pph_data["pph_count"],
                        "total_deliveries": pph_data["total_deliveries"],
                    }
                )

    if not comparison_data:
        st.info("⚠️ No data available for region comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Chart options - Line Chart vs Heatmap
    chart_options = ["Line Chart", "Heatmap"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_region_comparison_{str(region_names)}",
    )

    if chart_type == "Heatmap":
        # Call the heatmap function with region label
        render_pph_heatmap(
            df, region_names, period_order, bg_color, text_color, entity_label="Region"
        )

        # Generate heatmap CSV for download
        heatmap_data = []
        for region_name in region_names:
            for period in period_order:
                period_data = comparison_df[
                    (comparison_df["Region"] == region_name)
                    & (comparison_df["period_display"] == period)
                ]
                if not period_data.empty:
                    pph_rate = period_data["value"].iloc[0]
                    pph_count = period_data["pph_count"].iloc[0]
                    total_deliveries = period_data["total_deliveries"].iloc[0]
                else:
                    pph_rate = 0.0
                    pph_count = 0
                    total_deliveries = 0

                heatmap_data.append(
                    {
                        "Region": region_name,
                        "Period": period,
                        "PPH Rate": pph_rate,
                        "PPH Cases": pph_count,
                        "Total Deliveries": total_deliveries,
                    }
                )

        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot(index="Region", columns="Period", values="PPH Rate")

        # Download button for heatmap CSV
        csv = pivot_df.to_csv()
        st.download_button(
            label="Download Heatmap CSV",
            data=csv,
            file_name="pph_rate_region_heatmap.csv",
            mime="text/csv",
            key="region_heatmap_download",
        )

        return  # Exit early for heatmap view

    # Line chart view (original functionality)
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=title,
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
        yaxis_title="PPH Rate (%)",
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

    # Region comparison table
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        region_df = df[df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            pph_data = compute_pph_kpi(region_df, region_facility_uids)
            region_table_data.append(
                {
                    "Region Name": region_name,
                    "PPH Cases": pph_data["pph_count"],
                    "Total Deliveries": pph_data["total_deliveries"],
                    "PPH Rate (%)": pph_data["pph_rate"],
                }
            )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["PPH Cases"].sum()
    total_denominator = region_table_df["Total Deliveries"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "PPH Cases": total_numerator,
        "Total Deliveries": total_denominator,
        "PPH Rate (%)": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "PPH Cases": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "PPH Rate (%)": "{:.2f}%",
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
        file_name="pph_rate_region_comparison.csv",
        mime="text/csv",
    )


def render_pph_heatmap(
    df, entity_names, period_order, bg_color, text_color, entity_label="Facility"
):
    """Render a professionally styled heatmap of PPH rates by entity (facility/region) and period"""
    if df is None or df.empty:
        st.info("⚠️ No data available for heatmap.")
        return

    # Extract and sort periods chronologically using the same mechanism as other graphs
    if "period_sort" in df.columns:
        # Use the period_sort column for proper chronological ordering
        period_df = df[["period_display", "period_sort"]].drop_duplicates()
        period_df = period_df.sort_values("period_sort")
        chronological_periods = period_df["period_display"].tolist()
    else:
        # Fallback: try to parse periods as dates for sorting
        try:
            chronological_periods = sorted(
                period_order,
                key=lambda x: pd.to_datetime(
                    x.split()[0] + " " + x.split()[1] if len(x.split()) == 2 else x
                ),
            )
        except:
            # Final fallback: use the original order
            chronological_periods = period_order

    # Prepare data for heatmap - include ALL entities and periods, even with zero values
    heatmap_data = []

    for entity_name in entity_names:
        # For regions, we need to filter by all facilities in that region
        if entity_label == "Region":
            # This would need access to the facilities_by_region mapping
            # For simplicity, we'll assume the data is already filtered by region
            entity_df = (
                df[df["region"] == entity_name]
                if "region" in df.columns
                else pd.DataFrame()
            )
        else:
            # For facilities, filter by facility UID
            entity_df = df[df["orgUnit"] == entity_name]

        for period in chronological_periods:
            if not entity_df.empty:
                period_df = entity_df[entity_df["period_display"] == period]
                if not period_df.empty:
                    if entity_label == "Region":
                        # For regions, we need to compute the KPI for all facilities in the region
                        region_facility_uids = [
                            facility_uid
                            for facility_uid in period_df["orgUnit"].unique()
                        ]
                        pph_data = compute_pph_kpi(period_df, region_facility_uids)
                    else:
                        # For facilities, compute KPI for the single facility
                        pph_data = compute_pph_kpi(period_df, [entity_name])

                    pph_rate = pph_data["pph_rate"]
                    pph_count = pph_data["pph_count"]
                    total_deliveries = pph_data["total_deliveries"]
                else:
                    # No data for this entity in this period
                    pph_rate = 0.0
                    pph_count = 0
                    total_deliveries = 0
            else:
                # No data for this entity at all
                pph_rate = 0.0
                pph_count = 0
                total_deliveries = 0

            heatmap_data.append(
                {
                    entity_label: entity_name,
                    "Period": period,
                    "PPH Rate": pph_rate,
                    "PPH Cases": pph_count,
                    "Total Deliveries": total_deliveries,
                }
            )

    if not heatmap_data:
        st.info("⚠️ No data available for heatmap.")
        return

    heatmap_df = pd.DataFrame(heatmap_data)

    # Pivot for heatmap format with chronological ordering
    pivot_df = heatmap_df.pivot(index=entity_label, columns="Period", values="PPH Rate")

    # Ensure columns are in chronological order
    pivot_df = pivot_df[chronological_periods]

    # Fill any remaining NaN values with 0
    pivot_df = pivot_df.fillna(0)

    # Create heatmap with enhanced styling and larger font sizes
    fig, ax = plt.subplots(figsize=(24, 18))  # Increased size for better visibility

    # Use a custom color palette for good/medium/bad performance
    # Good (green): 0-2%, Medium (yellow): 2-5%, Bad (red): 5%+
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]  # Green, Yellow, Red
    cmap = LinearSegmentedColormap.from_list("pph_cmap", colors, N=100)

    # Create the heatmap with enhanced styling and larger fonts
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": "PPH Rate (%)", "shrink": 0.8, "pad": 0.02, "aspect": 30},
        linewidths=2,  # Thicker cell borders for better separation
        linecolor="white",  # White border lines
        annot_kws={
            "size": 30,  # Increased from 10 to 30 for PPH values
            "weight": "bold",
            "color": "white" if np.mean(pivot_df.values) > 2.0 else "black",
        },
        square=False,  # Rectangular cells for better readability
    )

    # Enhanced title and labels with larger fonts
    ax.set_title(
        f"PPH RATE HEATMAP - {entity_label.upper()} PERFORMANCE OVERVIEW",
        fontsize=42,  # Increased from 38 to 42
        fontweight="bold",
        pad=30,  # Increased padding
        color=text_color,
    )
    ax.set_xlabel(
        "PERIOD",
        fontsize=36,  # Increased from 38 to 36 (slightly smaller than title)
        fontweight="bold",
        color=text_color,
        labelpad=25,  # Increased padding
    )
    ax.set_ylabel(
        entity_label.upper(),
        fontsize=36,  # Increased from 38 to 36
        fontweight="bold",
        color=text_color,
        labelpad=25,  # Increased padding
    )

    # Rotate x-axis and y-axis labels with larger fonts
    plt.xticks(
        rotation=45,
        ha="right",
        color=text_color,
        fontsize=30,  # Increased from 15 to 20 for period labels
    )
    plt.yticks(
        rotation=0,
        color=text_color,
        fontsize=30,  # Increased from 30 to 22 (more balanced size)
    )

    # Set background and text colors
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor("#f8f9fa")  # Light gray background for the heatmap area

    # Colorbar styling with larger fonts
    cbar = ax.collections[0].colorbar
    cbar.set_label(
        "PPH Rate (%)",
        fontweight="bold",
        color=text_color,
        fontsize=24,  # Increased from 20 to 24
    )
    cbar.ax.tick_params(
        colors=text_color, labelsize=16  # Increased colorbar tick labels
    )
    cbar.outline.set_edgecolor(text_color)

    # Add grid-like appearance with thicker borders
    ax.grid(False)  # Disable default grid
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(3)  # Thicker border around the entire heatmap

    # Adjust layout to prevent cutting off
    plt.tight_layout()

    # Display the heatmap
    st.pyplot(fig)

    # Filtering options
    st.markdown("### 🔍 Filter Data")

    threshold = st.slider(
        f"Show {entity_label.lower()}s with PPH Rate above:",
        min_value=0.0,
        max_value=float(max(10.0, heatmap_df["PPH Rate"].max() * 1.2)),
        value=2.0,
        step=0.1,
        help=f"Filter to show only {entity_label.lower()}s with PPH rates above this threshold",
    )

    # Display filtered data with increased width
    filtered_data = heatmap_df[heatmap_df["PPH Rate"] >= threshold]
    if not filtered_data.empty:
        st.info(f"📋 Showing {len(filtered_data)} entries with PPH Rate ≥ {threshold}%")

        # Create a styled table for filtered results with increased width
        display_df = filtered_data.pivot(
            index=entity_label, columns="Period", values="PPH Rate"
        )

        # Ensure chronological ordering in the display table too
        display_df = display_df[chronological_periods]

        # Use st.dataframe with custom width
        st.dataframe(
            display_df.style.background_gradient(cmap=cmap, axis=None)
            .format("{:.2f}%")
            .set_properties(
                **{
                    "border": "2px solid #ddd",
                    "text-align": "center",
                    "font-weight": "bold",
                    "min-width": "100px",
                    "font-size": "14px",  # Added larger font for table
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#2c3e50"),
                            ("color", "white"),
                            ("font-weight", "bold"),
                            ("min-width", "120px"),
                            ("font-size", "16px"),  # Increased from 12px
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("min-width", "100px"),
                            ("font-size", "14px"),  # Increased from 11px
                        ],
                    },
                ]
            ),
            width=1200,
            height=400,
        )
    else:
        st.warning(f"No {entity_label.lower()}s found with PPH Rate ≥ {threshold}%")

    # Additional insights
    st.markdown("### 💡 Insights")

    high_pph_entities = heatmap_df[heatmap_df["PPH Rate"] > 5.0][entity_label].unique()
    if len(high_pph_entities) > 0:
        st.warning(
            f"🚨 **Attention Needed**: {len(high_pph_entities)} {entity_label.lower()}s have PPH rates above 5%: "
            f"{', '.join(high_pph_entities[:5])}{'...' if len(high_pph_entities) > 5 else ''}"
        )

    zero_pph_entities = heatmap_df[heatmap_df["PPH Rate"] == 0.0][entity_label].unique()
    if len(zero_pph_entities) > 0:
        st.success(
            f"✅ **Excellent Performance**: {len(zero_pph_entities)} {entity_label.lower()}s have 0% PPH rates: "
            f"{', '.join(zero_pph_entities[:5])}{'...' if len(zero_pph_entities) > 5 else ''}"
        )


def render_obstetric_condition_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of obstetric conditions"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute condition distribution
    condition_df = compute_obstetric_condition_distribution(df, facility_uids)

    if condition_df.empty:
        st.info("⚠️ No data available for obstetric condition distribution.")
        return

    # Add CSS for better pie chart layout
    st.markdown(
        """
    <style>
    .pie-chart-container {
        margin-top: -30px;
        margin-bottom: 20px;
    }
    .pie-chart-title {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Add chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Pie Chart", "Donut Chart"],
        index=0,
        key="obstetric_chart_type",
    )

    # Create chart with reduced size
    if chart_type == "Pie Chart":
        fig = px.pie(
            condition_df,
            values="count",
            names="condition",
            hover_data=["percentage"],
            labels={"count": "Count", "percentage": "Percentage"},
            height=500,  # Slightly increased height
        )
    else:  # Donut Chart
        fig = px.pie(
            condition_df,
            values="count",
            names="condition",
            hover_data=["percentage"],
            labels={"count": "Count", "percentage": "Percentage"},
            height=500,
            hole=0.4,
        )

    # Calculate if we should use inside text for small slices
    total_count = condition_df["count"].sum()
    use_inside_text = any((condition_df["count"] / total_count) < 0.05)

    if use_inside_text:
        # For small slices, put text inside with white background
        fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textfont=dict(size=10),
            insidetextfont=dict(color="white", size=9),
            outsidetextfont=dict(size=9),
            pull=[
                0.05 if cond == "Postpartum Hemorrhage (PPH)" else 0
                for cond in condition_df["condition"]
            ],
        )
    else:
        # For normal slices, use outside text
        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textfont=dict(size=10),
            pull=[
                0.05 if cond == "Postpartum Hemorrhage (PPH)" else 0
                for cond in condition_df["condition"]
            ],
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        height=500,  # Increased height
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.3,
            font=dict(size=10),
            itemwidth=30,
        ),
        margin=dict(l=0, r=150, t=20, b=20),  # Increased top and bottom margins
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    # Ensure no "undefined" placeholder
    fig.update_layout(title=None)
    fig.layout.pop("title", None)

    # Use container to control layout
    with st.container():
        st.markdown(
            '<div class="pie-chart-title">Distribution of Obstetric Conditions at Delivery</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 Obstetric Condition Summary")
    condition_df = condition_df.copy()
    condition_df.insert(0, "No", range(1, len(condition_df) + 1))

    styled_table = (
        condition_df.style.format({"count": "{:,.0f}", "percentage": "{:.2f}%"})
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Add download button for CSV
    csv = condition_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="obstetric_condition_distribution.csv",
        mime="text/csv",
    )
