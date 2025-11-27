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
    Compute PPH KPI for the given dataframe.

    Formula:
        PPH Rate (%) = (Number of women with Obstetric condition at delivery including code '3')
        ÷ (Total Deliveries) × 100

    Handles multi-code (comma or semicolon-separated) option values such as "3,5,6".
    """
    if df is None or df.empty:
        return {"pph_rate": 0.0, "pph_count": 0, "total_deliveries": 0}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Compute total deliveries
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Filter rows for the correct data element
    cond_df = df[df["dataElement_uid"] == PPH_CONDITION_UID].copy()

    # Safety check: if no such data element exists
    if cond_df.empty or "tei_id" not in cond_df.columns:
        return {
            "pph_rate": 0.0,
            "pph_count": 0,
            "total_deliveries": int(total_deliveries),
        }

    # Ensure 'value' is string
    cond_df["value"] = cond_df["value"].astype(str)

    # Mark rows containing PPH code in multi-text field
    cond_df["has_pph"] = cond_df["value"].apply(
        lambda v: any(
            part.strip() == PPH_CODE
            for part in v.replace(";", ",").split(",")
            if part.strip()
        )
    )

    # Count unique TEIs that have PPH
    pph_cases = cond_df[cond_df["has_pph"]]
    pph_count = pph_cases["tei_id"].nunique()

    # Compute rate
    pph_rate = (pph_count / total_deliveries * 100) if total_deliveries > 0 else 0.0

    return {
        "pph_rate": float(pph_rate),
        "pph_count": int(pph_count),
        "total_deliveries": int(total_deliveries),
    }


def compute_pph_numerator(df, facility_uids=None):
    """
    Compute only the PPH numerator count (number of PPH cases)
    Used in trend calculations where denominator is tracked separately
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter rows for the correct data element
    cond_df = df[df["dataElement_uid"] == PPH_CONDITION_UID].copy()

    # Safety check: if no such data element exists
    if cond_df.empty or "tei_id" not in cond_df.columns:
        return 0

    # Ensure 'value' is string
    cond_df["value"] = cond_df["value"].astype(str)

    # Mark rows containing PPH code in multi-text field
    cond_df["has_pph"] = cond_df["value"].apply(
        lambda v: any(
            part.strip() == PPH_CODE
            for part in v.replace(";", ",").split(",")
            if part.strip()
        )
    )

    # Count unique TEIs that have PPH
    pph_cases = cond_df[cond_df["has_pph"]]
    pph_count = pph_cases["tei_id"].nunique()

    return int(pph_count)


def compute_obstetric_condition_distribution(df, facility_uids=None):
    """
    Compute distribution of all obstetric complications
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter only rows for obstetric conditions
    delivery_events = df[
        (df["dataElement_uid"] == PPH_CONDITION_UID) & df["value"].notna()
    ]

    if delivery_events.empty:
        return pd.DataFrame()

    # ---- EXPAND MULTI-CODE VALUES ----
    expanded_rows = []

    # Only accept these valid codes
    valid_codes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

    for _, row in delivery_events.iterrows():
        value_str = str(row["value"]).strip()

        # Skip empty values, "nan", "null"
        if (
            not value_str
            or value_str.lower() in ["nan", "null", ""]
            or value_str.isspace()
        ):
            continue

        # Split by comma or semicolon
        codes = [
            c.strip()
            for c in value_str.replace(";", ",").split(",")
            if c.strip() and c.strip() in valid_codes  # Only keep valid codes
        ]

        for code in codes:
            new_row = row.copy()
            new_row["condition_code"] = code
            expanded_rows.append(new_row)

    if not expanded_rows:
        return pd.DataFrame()

    expanded_df = pd.DataFrame(expanded_rows)

    # ---- COUNT OCCURRENCES ----
    condition_counts = expanded_df["condition_code"].value_counts().reset_index()
    condition_counts.columns = ["condition_code", "count"]

    # ---- MAP CODES TO READABLE NAMES ----
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

    # ---- CALCULATE PERCENTAGES ----
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
    """Render facility comparison chart for PPH rate"""
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

    # Prepare comparison data
    comparison_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            pph_data = compute_pph_kpi(facility_df, [facility_uid])
            comparison_data.append(
                {
                    "Facility": facility_name,
                    "value": pph_data["pph_rate"],
                    "pph_count": pph_data["pph_count"],
                    "total_deliveries": pph_data["total_deliveries"],
                }
            )

    if not comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Chart options
    chart_options = ["Bar Chart", "Line Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_facility_comparison_{str(facility_uids)}",
    )

    # Create chart
    if chart_type == "Line Chart":
        # For line chart, we need time series data
        time_series_data = []
        all_periods = (
            df[["period_display", "period_sort"]]
            .drop_duplicates()
            .sort_values("period_sort")
        )
        period_order = all_periods["period_display"].tolist()

        for period_display in period_order:
            period_df = df[df["period_display"] == period_display]

            for facility_uid in facility_uids:
                facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
                if not facility_period_df.empty:
                    pph_data = compute_pph_kpi(facility_period_df, [facility_uid])
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": pph_data["pph_rate"],
                            "pph_count": pph_data["pph_count"],
                            "total_deliveries": pph_data["total_deliveries"],
                        }
                    )

        if not time_series_data:
            st.info("⚠️ No time series data available for line chart.")
            return

        time_series_df = pd.DataFrame(time_series_data)

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

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>PPH Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (time_series_df["pph_count"], time_series_df["total_deliveries"])
            ),
        )
    else:  # Bar Chart
        fig = px.bar(
            comparison_df,
            x="Facility",
            y="value",
            title=title,
            height=500,
            color="Facility",
            hover_data=["pph_count", "total_deliveries"],
        )

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>PPH Rate: %{y:.2f}%<br>PPH Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (comparison_df["pph_count"], comparison_df["total_deliveries"])
            ),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="PPH Rate (%)",
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
    """Render region comparison chart for PPH rate"""
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

    # Prepare comparison data
    comparison_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        region_df = df[df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            pph_data = compute_pph_kpi(region_df, region_facility_uids)
            comparison_data.append(
                {
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

    # Chart options
    chart_options = ["Bar Chart", "Line Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_region_comparison_{str(region_names)}",
    )

    # Create chart
    if chart_type == "Line Chart":
        # For line chart, we need time series data
        time_series_data = []
        all_periods = (
            df[["period_display", "period_sort"]]
            .drop_duplicates()
            .sort_values("period_sort")
        )
        period_order = all_periods["period_display"].tolist()

        for period_display in period_order:
            period_df = df[df["period_display"] == period_display]

            for region_name in region_names:
                region_facility_uids = [
                    uid for _, uid in facilities_by_region.get(region_name, [])
                ]
                region_period_df = period_df[
                    period_df["orgUnit"].isin(region_facility_uids)
                ]

                if not region_period_df.empty:
                    pph_data = compute_pph_kpi(region_period_df, region_facility_uids)
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": pph_data["pph_rate"],
                            "pph_count": pph_data["pph_count"],
                            "total_deliveries": pph_data["total_deliveries"],
                        }
                    )

        if not time_series_data:
            st.info("⚠️ No time series data available for line chart.")
            return

        time_series_df = pd.DataFrame(time_series_data)

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

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>PPH Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (time_series_df["pph_count"], time_series_df["total_deliveries"])
            ),
        )
    else:  # Bar Chart
        fig = px.bar(
            comparison_df,
            x="Region",
            y="value",
            title=title,
            height=500,
            color="Region",
            hover_data=["pph_count", "total_deliveries"],
        )

        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>PPH Rate: %{y:.2f}%<br>PPH Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (comparison_df["pph_count"], comparison_df["total_deliveries"])
            ),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="PPH Rate (%)",
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
