# kpi_newborn_bw.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color

# ---------------- Newborn Birth Weight KPI Constants ----------------
BIRTH_WEIGHT_UID = (
    "yxWUMt3sCil"  # Data Element: Birth Weight (grams) - SAME AS MATERNAL
)

# Newborn birth weight categories (INCLUDES ALL CATEGORIES - not just LBW)
NEWBORN_BW_CATEGORIES = {
    "lt_1000": {"name": "<1000 g", "min": 0, "max": 999},
    "1000_1499": {"name": "1000–1499 g", "min": 1000, "max": 1499},
    "1500_1999": {"name": "1500–1999 g", "min": 1500, "max": 1999},
    "2000_2499": {"name": "2000–2499 g", "min": 2000, "max": 2499},
    "2500_4000": {"name": "2500–4000 g", "min": 2500, "max": 4000},
    "gt_4000": {"name": "4001+ g", "min": 4001, "max": 9999},  # Reasonable upper limit
}


# ---------------- Newborn Birth Weight KPI Computation Functions ----------------
def compute_newborn_bw_numerator(df, category_key, facility_uids=None):
    """Compute numerator for specific birth weight category"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for birth weight events with valid numeric values
    birth_weight_events = df[
        (df["dataElement_uid"] == BIRTH_WEIGHT_UID) & df["value"].notna()
    ].copy()

    if birth_weight_events.empty:
        return 0

    # Convert values to numeric, coercing errors to NaN
    birth_weight_events["weight_numeric"] = pd.to_numeric(
        birth_weight_events["value"], errors="coerce"
    )

    # Get category range
    category_info = NEWBORN_BW_CATEGORIES[category_key]

    # Filter out NaN values and count births in category range
    category_cases = birth_weight_events[
        (birth_weight_events["weight_numeric"] >= category_info["min"])
        & (birth_weight_events["weight_numeric"] <= category_info["max"])
        & (birth_weight_events["weight_numeric"] > 0)  # Exclude negative/zero weights
    ]

    return category_cases["tei_id"].nunique()


def compute_newborn_bw_by_category(df, facility_uids=None):
    """Compute distribution of birth weights by all categories"""
    if df is None or df.empty:
        return {category: 0 for category in NEWBORN_BW_CATEGORIES.keys()}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for birth weight events with valid numeric values
    birth_weight_events = df[
        (df["dataElement_uid"] == BIRTH_WEIGHT_UID) & df["value"].notna()
    ].copy()

    if birth_weight_events.empty:
        return {category: 0 for category in NEWBORN_BW_CATEGORIES.keys()}

    # Convert values to numeric
    birth_weight_events["weight_numeric"] = pd.to_numeric(
        birth_weight_events["value"], errors="coerce"
    )

    # Filter out NaN and invalid weights
    valid_weights = birth_weight_events[
        (birth_weight_events["weight_numeric"].notna())
        & (birth_weight_events["weight_numeric"] > 0)
    ]

    if valid_weights.empty:
        return {category: 0 for category in NEWBORN_BW_CATEGORIES.keys()}

    # Count occurrences in each category
    result = {}
    for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
        count = valid_weights[
            (valid_weights["weight_numeric"] >= category_info["min"])
            & (valid_weights["weight_numeric"] <= category_info["max"])
        ]["tei_id"].nunique()
        result[category_key] = int(count)

    return result


def compute_newborn_bw_denominator(df, facility_uids=None):
    """Compute denominator for newborn BW KPI: Count of all admitted newborns"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # For newborn dashboard, denominator is total unique newborns (TEIs)
    # This matches the inborn/NMR denominator approach
    return df["tei_id"].nunique()


def compute_newborn_bw_kpi(df, facility_uids=None):
    """Compute newborn birth weight KPI for the given dataframe"""
    if df is None or df.empty:
        return {
            "bw_categories": {category: 0 for category in NEWBORN_BW_CATEGORIES.keys()},
            "category_rates": {
                category: 0.0 for category in NEWBORN_BW_CATEGORIES.keys()
            },
            "total_admissions": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if isinstance(facility_uids, str):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count total admissions (denominator) - same as inborn/NMR
    total_admissions = compute_newborn_bw_denominator(df, facility_uids)

    # Get distribution of BW categories
    bw_categories = compute_newborn_bw_by_category(df, facility_uids)

    # Calculate rates for each category
    category_rates = {}
    for category_key, count in bw_categories.items():
        category_rates[category_key] = (
            (count / total_admissions * 100) if total_admissions > 0 else 0.0
        )

    return {
        "bw_categories": bw_categories,
        "category_rates": category_rates,
        "total_admissions": int(total_admissions),
    }


# ---------------- Newborn Birth Weight Line Chart Functions ----------------
def render_newborn_bw_trend_chart(
    df,
    period_col="period_display",
    title="Newborn Birth Weight Distribution Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render a trend line chart for newborn birth weight distribution"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Process periods in chronological order
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

    # Compute trend data with chronological ordering
    trend_data = []
    counted_newborns = set()  # Track newborns across periods to prevent double-counting

    for period in periods:
        period_df = df[df[period_col] == period]
        period_display = (
            period_df["period_display"].iloc[0] if not period_df.empty else period
        )

        # Get newborns in this period who haven't been counted yet
        period_newborns = set(period_df["tei_id"].unique())
        new_newborns = period_newborns - counted_newborns

        if new_newborns:
            # Filter to only new newborns in this period for denominator
            new_newborns_df = period_df[period_df["tei_id"].isin(new_newborns)]

            # Compute BW distribution for this period
            bw_data = compute_newborn_bw_kpi(
                period_df, facility_uids
            )  # Use full period data for numerator

            # But use new newborns count for denominator
            total_admissions = len(new_newborns)

            # Update counted newborns
            counted_newborns.update(new_newborns)
        else:
            # No new newborns in this period
            bw_data = compute_newborn_bw_kpi(
                pd.DataFrame(), facility_uids
            )  # Empty data
            total_admissions = 0

        # Calculate rates based on actual period denominator
        period_rates = {}
        for category_key in NEWBORN_BW_CATEGORIES.keys():
            count = bw_data["bw_categories"].get(category_key, 0)
            period_rates[category_key] = (
                (count / total_admissions * 100) if total_admissions > 0 else 0.0
            )

        # Create period row
        period_row = {
            period_col: period_display,
            "total_admissions": total_admissions,
        }

        # Add each category rate and count
        for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
            period_row[f"{category_key}_rate"] = period_rates[category_key]
            period_row[f"{category_key}_count"] = bw_data["bw_categories"].get(
                category_key, 0
            )

        trend_data.append(period_row)

    if not trend_data:
        st.info("⚠️ No birth weight data available for the selected period.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create line chart with multiple lines
    fig = go.Figure()

    # Colors for each line
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    # Add lines for each BW category
    for i, (category_key, category_info) in enumerate(NEWBORN_BW_CATEGORIES.items()):
        rate_col = f"{category_key}_rate"

        if rate_col in trend_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trend_df[period_col],
                    y=trend_df[rate_col],
                    mode="lines+markers",
                    name=f"{category_info['name']}",
                    line=dict(width=2, color=colors[i]),
                    marker=dict(size=5),
                    hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']}: %{{y:.2f}}%<br>Count: %{{customdata[0]}}<br>Total Admissions: %{{customdata[1]}}<extra></extra>",
                    customdata=np.column_stack(
                        (
                            trend_df.get(f"{category_key}_count", 0),
                            trend_df["total_admissions"],
                        )
                    ),
                )
            )

    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="Period",
        yaxis_title="Distribution (%)",
        showlegend=True,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
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

    # Summary table
    st.subheader(f"📋 {title} Summary Table")

    # Prepare summary data
    summary_data = []
    for period_idx, period_row in trend_df.iterrows():
        period_data = {
            "Period": period_row[period_col],
            "Total Admissions": period_row["total_admissions"],
        }

        # Add each category's count and rate
        for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
            count_col = f"{category_key}_count"
            rate_col = f"{category_key}_rate"

            period_data[f"{category_info['name']} Count"] = period_row.get(count_col, 0)
            period_data[f"{category_info['name']} Rate (%)"] = period_row.get(
                rate_col, 0
            )

        summary_data.append(period_data)

    summary_df = pd.DataFrame(summary_data)

    # Add overall row
    total_admissions = summary_df["Total Admissions"].sum()
    overall_row = {
        "Period": "Overall",
        "Total Admissions": total_admissions,
    }

    for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
        total_category_count = summary_df[f"{category_info['name']} Count"].sum()
        overall_category_rate = (
            (total_category_count / total_admissions * 100)
            if total_admissions > 0
            else 0
        )

        overall_row[f"{category_info['name']} Count"] = total_category_count
        overall_row[f"{category_info['name']} Rate (%)"] = overall_category_rate

    overall_df = pd.DataFrame([overall_row])
    summary_table = pd.concat([summary_df, overall_df], ignore_index=True)
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Format table
    format_dict = {"Total Admissions": "{:,.0f}"}
    for category_info in NEWBORN_BW_CATEGORIES.values():
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"
        format_dict[f"{category_info['name']} Rate (%)"] = "{:.2f}%"

    styled_table = (
        summary_table.style.format(format_dict)
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="newborn_birth_weight_trend.csv",
        mime="text/csv",
    )


def render_newborn_bw_facility_comparison_chart(
    df,
    period_col="period_display",
    title="Newborn Birth Weight Distribution - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
):
    """Render facility comparison line chart for newborn birth weight distribution"""
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

    # For line chart, focus on one category at a time
    selected_category = st.selectbox(
        "Select Birth Weight Category to Compare:",
        options=[info["name"] for info in NEWBORN_BW_CATEGORIES.values()],
        index=4,  # Default to 2500-4000g (normal range)
        key=f"line_category_{str(facility_uids)}",
    )

    # Find the category key for the selected category
    category_key = None
    for key, info in NEWBORN_BW_CATEGORIES.items():
        if info["name"] == selected_category:
            category_key = key
            break

    if not category_key:
        st.error("Selected category not found.")
        return

    # Compute time series data
    time_series_data = []
    all_periods = sorted(df["period_display"].unique())

    for period_display in all_periods:
        period_df = df[df["period_display"] == period_display]

        for facility_uid in facility_uids:
            facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
            if not facility_period_df.empty:
                bw_data = compute_newborn_bw_kpi(facility_period_df, [facility_uid])
                category_rate = bw_data["category_rates"].get(category_key, 0)

                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": category_rate,
                        "count": bw_data["bw_categories"].get(category_key, 0),
                        "total_admissions": bw_data["total_admissions"],
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
        title=f"{selected_category} - Facility Comparison",
        height=500,
        category_orders={"period_display": all_periods},
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>Count: %{customdata[0]}<br>Total Admissions: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack(
            (time_series_df["count"], time_series_df["total_admissions"])
        ),
    )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
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

    # Facility comparison table
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            bw_data = compute_newborn_bw_kpi(facility_df, [facility_uid])

            row_data = {
                "Facility Name": facility_name,
                "Total Admissions": bw_data["total_admissions"],
            }

            # Add each category count and rate
            for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
                row_data[f"{category_info['name']} Count"] = bw_data["bw_categories"][
                    category_key
                ]
                row_data[f"{category_info['name']} Rate (%)"] = bw_data[
                    "category_rates"
                ][category_key]

            facility_table_data.append(row_data)

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_admissions = facility_table_df["Total Admissions"].sum()
    overall_row = {
        "Facility Name": "Overall",
        "Total Admissions": total_admissions,
    }

    for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
        total_category_count = facility_table_df[f"{category_info['name']} Count"].sum()
        overall_category_rate = (
            (total_category_count / total_admissions * 100)
            if total_admissions > 0
            else 0
        )

        overall_row[f"{category_info['name']} Count"] = total_category_count
        overall_row[f"{category_info['name']} Rate (%)"] = overall_category_rate

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    format_dict = {"Total Admissions": "{:,.0f}"}
    for category_info in NEWBORN_BW_CATEGORIES.values():
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"
        format_dict[f"{category_info['name']} Rate (%)"] = "{:.2f}%"

    styled_table = (
        facility_table_df.style.format(format_dict)
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = facility_table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="newborn_birth_weight_facility_comparison.csv",
        mime="text/csv",
    )


def render_newborn_bw_region_comparison_chart(
    df,
    period_col="period_display",
    title="Newborn Birth Weight Distribution - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    facilities_by_region=None,
):
    """Render region comparison line chart for newborn birth weight distribution"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # For line chart, focus on one category at a time
    selected_category = st.selectbox(
        "Select Birth Weight Category to Compare:",
        options=[info["name"] for info in NEWBORN_BW_CATEGORIES.values()],
        index=4,  # Default to 2500-4000g (normal range)
        key=f"line_category_region_{str(region_names)}",
    )

    # Find the category key for the selected category
    category_key = None
    for key, info in NEWBORN_BW_CATEGORIES.items():
        if info["name"] == selected_category:
            category_key = key
            break

    if not category_key:
        st.error("Selected category not found.")
        return

    # Compute time series data
    time_series_data = []
    all_periods = sorted(df["period_display"].unique())

    for period_display in all_periods:
        period_df = df[df["period_display"] == period_display]

        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_period_df = period_df[
                period_df["orgUnit"].isin(region_facility_uids)
            ]

            if not region_period_df.empty:
                bw_data = compute_newborn_bw_kpi(region_period_df, region_facility_uids)
                category_rate = bw_data["category_rates"].get(category_key, 0)

                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Region": region_name,
                        "value": category_rate,
                        "count": bw_data["bw_categories"].get(category_key, 0),
                        "total_admissions": bw_data["total_admissions"],
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
        title=f"{selected_category} - Region Comparison",
        height=500,
        category_orders={"period_display": all_periods},
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>Count: %{customdata[0]}<br>Total Admissions: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack(
            (time_series_df["count"], time_series_df["total_admissions"])
        ),
    )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
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

    # Region comparison table
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        region_df = df[df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            bw_data = compute_newborn_bw_kpi(region_df, region_facility_uids)

            row_data = {
                "Region Name": region_name,
                "Total Admissions": bw_data["total_admissions"],
            }

            # Add each category count and rate
            for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
                row_data[f"{category_info['name']} Count"] = bw_data["bw_categories"][
                    category_key
                ]
                row_data[f"{category_info['name']} Rate (%)"] = bw_data[
                    "category_rates"
                ][category_key]

            region_table_data.append(row_data)

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_admissions = region_table_df["Total Admissions"].sum()
    overall_row = {
        "Region Name": "Overall",
        "Total Admissions": total_admissions,
    }

    for category_key, category_info in NEWBORN_BW_CATEGORIES.items():
        total_category_count = region_table_df[f"{category_info['name']} Count"].sum()
        overall_category_rate = (
            (total_category_count / total_admissions * 100)
            if total_admissions > 0
            else 0
        )

        overall_row[f"{category_info['name']} Count"] = total_category_count
        overall_row[f"{category_info['name']} Rate (%)"] = overall_category_rate

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    format_dict = {"Total Admissions": "{:,.0f}"}
    for category_info in NEWBORN_BW_CATEGORIES.values():
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"
        format_dict[f"{category_info['name']} Rate (%)"] = "{:.2f}%"

    styled_table = (
        region_table_df.style.format(format_dict)
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = region_table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="newborn_birth_weight_region_comparison.csv",
        mime="text/csv",
    )
