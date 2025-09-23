import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- LBW KPI Constants ----------------
BIRTH_WEIGHT_UID = "QUlJEvzGcQK"  # Data Element: Birth Weight (grams)

# LBW categories and their weight ranges (ONLY LBW CATEGORIES - exclude normal weight)
LBW_CATEGORIES = {
    "lt_1000": {"name": "<1000 g", "min": 0, "max": 999},
    "1000_1499": {"name": "1000–1499 g", "min": 1000, "max": 1499},
    "1500_1999": {"name": "1500–1999 g", "min": 1500, "max": 1999},
    "2000_2499": {"name": "2000–2499 g", "min": 2000, "max": 2499},
}

# Overall LBW threshold
LBW_THRESHOLD = 2500  # grams


# ---------------- LBW KPI Computation Functions ----------------
def compute_lbw_numerator(df, facility_uids=None):
    """Compute numerator for LBW KPI: Count of live births <2,500 g"""
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

    # Filter out NaN values and count births < 2500g
    lbw_cases = birth_weight_events[
        (birth_weight_events["weight_numeric"] < LBW_THRESHOLD)
        & (birth_weight_events["weight_numeric"] > 0)  # Exclude negative/zero weights
    ]

    return lbw_cases["tei_id"].nunique()


def compute_lbw_by_category(df, facility_uids=None):
    """Compute distribution of birth weights by LBW categories"""
    if df is None or df.empty:
        return {category: 0 for category in LBW_CATEGORIES.keys()}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for birth weight events with valid numeric values
    birth_weight_events = df[
        (df["dataElement_uid"] == BIRTH_WEIGHT_UID) & df["value"].notna()
    ].copy()

    if birth_weight_events.empty:
        return {category: 0 for category in LBW_CATEGORIES.keys()}

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
        return {category: 0 for category in LBW_CATEGORIES.keys()}

    # Count occurrences in each category
    result = {}
    for category_key, category_info in LBW_CATEGORIES.items():
        count = valid_weights[
            (valid_weights["weight_numeric"] >= category_info["min"])
            & (valid_weights["weight_numeric"] <= category_info["max"])
        ]["tei_id"].nunique()
        result[category_key] = int(count)

    return result


def compute_lbw_denominator(df, facility_uids=None):
    """Compute denominator for LBW KPI: Count of all live births that were weighed"""
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

    # Convert values to numeric and filter valid weights
    birth_weight_events["weight_numeric"] = pd.to_numeric(
        birth_weight_events["value"], errors="coerce"
    )

    valid_weights = birth_weight_events[
        (birth_weight_events["weight_numeric"].notna())
        & (birth_weight_events["weight_numeric"] > 0)
    ]

    return valid_weights["tei_id"].nunique()


def compute_lbw_kpi(df, facility_uids=None):
    """Compute LBW KPI for the given dataframe"""
    if df is None or df.empty:
        return {
            "lbw_rate": 0.0,
            "lbw_count": 0,
            "total_weighed": 0,
            "lbw_categories": {category: 0 for category in LBW_CATEGORIES.keys()},
            "category_rates": {category: 0.0 for category in LBW_CATEGORIES.keys()},
        }

    # Filter by facilities if specified
    if facility_uids:
        if isinstance(facility_uids, str):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count LBW cases (numerator)
    lbw_count = compute_lbw_numerator(df, facility_uids)

    # Count total weighed births (denominator)
    total_weighed = compute_lbw_denominator(df, facility_uids)

    # Get distribution of LBW categories
    lbw_categories = compute_lbw_by_category(df, facility_uids)

    # Calculate LBW rate
    lbw_rate = (lbw_count / total_weighed * 100) if total_weighed > 0 else 0.0

    # Calculate rates for each category (LBW rate for each weight category)
    category_rates = {}
    for category_key, count in lbw_categories.items():
        category_rates[category_key] = (
            (count / total_weighed * 100) if total_weighed > 0 else 0.0
        )

    return {
        "lbw_rate": float(lbw_rate),
        "lbw_count": int(lbw_count),
        "total_weighed": int(total_weighed),
        "lbw_categories": lbw_categories,
        "category_rates": category_rates,
    }


# ---------------- LBW Chart Functions ----------------
def render_lbw_trend_chart(
    df,
    period_col="period_display",
    value_col="lbw_rate",
    title="Low Birth Weight Rate Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="LBW Cases",
    denominator_name="Total Weighed Births",
    facility_uids=None,
):
    """Render a trend chart for low birth weight rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Chart options
    chart_options = ["Line Chart", "Stacked Horizontal Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_{title}_{str(facility_uids)}",
    ).lower()

    # Create chart based on selected type
    if chart_type == "line chart":
        # FIX 1: Show five lines - one for each LBW category and one for overall
        fig = go.Figure()

        # Colors for each line
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]

        # Add line for overall LBW rate
        fig.add_trace(
            go.Scatter(
                x=df[period_col],
                y=df[value_col],
                mode="lines+markers",
                name="Overall LBW Rate (<2500g)",
                line=dict(width=3, color=colors[4]),
                marker=dict(size=7),
                hovertemplate="<b>%{x}</b><br>Overall LBW Rate: %{y:.2f}%<br>LBW Cases: %{customdata[0]}<br>Total Weighed: %{customdata[1]}<extra></extra>",
                customdata=np.column_stack((df[numerator_name], df[denominator_name])),
            )
        )

        # Add lines for each LBW category
        for i, (category_key, category_info) in enumerate(LBW_CATEGORIES.items()):
            rate_col = f"{category_key}_rate"
            count_col = f"{category_key}_count"

            if rate_col in df.columns and count_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df[period_col],
                        y=df[rate_col],
                        mode="lines+markers",
                        name=f"{category_info['name']} LBW Rate",
                        line=dict(width=2, color=colors[i]),
                        marker=dict(size=5),
                        hovertemplate=f"<b>%{{x}}</b><br>{category_info['name']} LBW Rate: %{{y:.2f}}%<br>Count: %{{customdata[0]}}<br>Total Weighed: %{{customdata[1]}}<extra></extra>",
                        customdata=np.column_stack(
                            (df[count_col], df[denominator_name])
                        ),
                    )
                )

        fig.update_layout(
            title=title,
            height=400,
            xaxis_title="Period",
            yaxis_title="LBW Rate (%)",
            showlegend=True,
        )

    elif chart_type == "stacked horizontal bar chart":
        # FIX 2: Stacked bar chart showing LBW rate contribution by category
        lbw_category_cols = [f"{category}_rate" for category in LBW_CATEGORIES.keys()]

        if all(col in df.columns for col in lbw_category_cols):
            fig = go.Figure()

            # Colors for LBW categories
            colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]

            for i, (category_key, category_info) in enumerate(LBW_CATEGORIES.items()):
                fig.add_trace(
                    go.Bar(
                        name=f"{category_info['name']} LBW Rate",
                        y=df[period_col],
                        x=df[f"{category_key}_rate"],
                        orientation="h",
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f"<b>%{{y}}</b><br>{category_info['name']} LBW Rate: %{{x:.2f}}%<br>Count: %{{customdata[0]}}<br>Total Weighed: %{{customdata[1]}}<extra></extra>",
                        customdata=np.column_stack(
                            (df[f"{category_key}_count"], df[denominator_name])
                        ),
                    )
                )

            # Calculate total LBW rate for each period
            total_lbw_rates = df[lbw_category_cols].sum(axis=1)

            # Set x-axis range to 0-100% with proper intervals
            max_rate = 100

            fig.update_layout(
                barmode="stack",
                title="LBW Rate by Weight Category (%)",
                height=400,
                xaxis_title="LBW Rate (%)",
                yaxis_title="Period",
                xaxis=dict(
                    range=[0, 100],
                    tickmode="array",
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0%", "25%", "50%", "75%", "100%"],
                    tickformat=".0f",
                    showgrid=True,
                    gridcolor="rgba(128,128,128,0.2)",
                ),
                bargap=0.3,
            )

            fig.update_traces(width=0.6)
        else:
            # Fall back to simple bar chart
            fig = px.bar(
                df,
                x=period_col,
                y=value_col,
                title=title,
                height=400,
            )
            fig.update_traces(
                hovertemplate=f"<b>%{{x}}</b><br>LBW Rate: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
                customdata=np.column_stack((df[numerator_name], df[denominator_name])),
            )

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

    if chart_type == "line chart":
        fig.update_layout(yaxis_tickformat=".2f")

    st.plotly_chart(fig, use_container_width=True)

    # Show trend indicator (only for line chart)
    if len(df) > 1 and chart_type == "line chart":
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
            f'<p style="font-size:1.2rem;font-weight:600;">Latest LBW Rate: {last_value:.2f}% <span class="{trend_class}">{trend_symbol}</span></p>',
            unsafe_allow_html=True,
        )

    # Summary table
    st.subheader(f"📋 {title} Summary Table")

    # Main KPI Overview
    st.markdown("**Main KPI Overview**")
    main_summary_df = df.copy().reset_index(drop=True)
    main_summary_df = main_summary_df[
        [period_col, numerator_name, denominator_name, value_col]
    ]

    # Calculate overall value
    total_numerator = main_summary_df[numerator_name].sum()
    total_denominator = main_summary_df[denominator_name].sum()
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

    main_summary_table = pd.concat([main_summary_df, overall_row], ignore_index=True)
    main_summary_table.insert(0, "No", range(1, len(main_summary_table) + 1))

    # Format main table
    styled_main_table = (
        main_summary_table.style.format(
            {
                value_col: "{:.1f}%",
                numerator_name: "{:,.0f}",
                denominator_name: "{:,.0f}",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_main_table.to_html(), unsafe_allow_html=True)

    # FIX 3: Category Breakdown Table - Show LBW rates for each category
    lbw_category_rate_cols = [f"{category}_rate" for category in LBW_CATEGORIES.keys()]
    lbw_category_count_cols = [
        f"{category}_count" for category in LBW_CATEGORIES.keys()
    ]

    if all(
        col in df.columns for col in lbw_category_rate_cols + lbw_category_count_cols
    ):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**LBW Rate by Weight Category**")

        category_summary_df = df.copy().reset_index(drop=True)
        category_columns = (
            [period_col, denominator_name]
            + lbw_category_rate_cols
            + lbw_category_count_cols
        )
        category_summary_df = category_summary_df[category_columns]

        # Calculate overall values for category table
        total_counts = {}
        for category in LBW_CATEGORIES.keys():
            total_counts[category] = category_summary_df[f"{category}_count"].sum()

        overall_category_row = {
            period_col: f"Overall {title}",
            denominator_name: total_denominator,
        }

        for category in LBW_CATEGORIES.keys():
            # Calculate average LBW rate for overall
            avg_rate = (
                (total_counts[category] / total_denominator * 100)
                if total_denominator > 0
                else 0
            )
            overall_category_row[f"{category}_rate"] = avg_rate
            overall_category_row[f"{category}_count"] = total_counts[category]

        overall_category_df = pd.DataFrame([overall_category_row])
        category_summary_table = pd.concat(
            [category_summary_df, overall_category_df], ignore_index=True
        )
        category_summary_table.insert(
            0, "No", range(1, len(category_summary_table) + 1)
        )

        # Rename columns for display
        display_columns = {
            "period_display": "Period",
            "Total Weighed Births": "Total<br>Weighed",
        }

        for category_key, category_info in LBW_CATEGORIES.items():
            display_columns[f"{category_key}_rate"] = (
                f"{category_info['name']}<br>LBW Rate (%)"
            )
            display_columns[f"{category_key}_count"] = (
                f"{category_info['name']}<br>Count"
            )

        category_summary_table = category_summary_table.rename(columns=display_columns)

        # Format category table
        format_dict = {"Total<br>Weighed": "{:,.0f}"}

        for category_info in LBW_CATEGORIES.values():
            format_dict[f"{category_info['name']}<br>LBW Rate (%)"] = "{:.2f}%"
            format_dict[f"{category_info['name']}<br>Count"] = "{:,.0f}"

        styled_category_table = (
            category_summary_table.style.format(format_dict)
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_category_table.to_html(), unsafe_allow_html=True)

    # Download button
    combined_table = main_summary_table
    if "category_summary_table" in locals():
        combined_table = pd.concat([main_summary_table, category_summary_table], axis=1)

    csv = combined_table.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_summary.csv",
        mime="text/csv",
    )


def render_lbw_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="lbw_rate",
    title="Low Birth Weight Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="LBW Cases",
    denominator_name="Total Weighed Births",
):
    """Render a comparison chart showing each facility's LBW performance"""
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

    # Prepare comparison data - FIX: Ensure lbw_count is properly calculated
    comparison_data = []
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    for facility_uid in facility_uids:
        facility_df = filtered_df[filtered_df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            lbw_data = compute_lbw_kpi(facility_df, [facility_uid])
            # FIX: Use actual calculated lbw_count
            actual_lbw_count = lbw_data["lbw_count"]
            comparison_data.append(
                {
                    "Facility": facility_uid_to_name[facility_uid],
                    "value": lbw_data["lbw_rate"],
                    "lbw_count": actual_lbw_count,  # This should now show correct values
                    "total_weighed": lbw_data["total_weighed"],
                    **{
                        f"{category}_rate": lbw_data["category_rates"][category]
                        for category in LBW_CATEGORIES.keys()
                    },
                    **{
                        f"{category}_count": lbw_data["lbw_categories"][category]
                        for category in LBW_CATEGORIES.keys()
                    },
                }
            )

    if not comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Chart options
    chart_options = ["Line Chart", "Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_facility_comparison_{str(facility_uids)}",
    )

    # Create chart
    if chart_type == "Line Chart":
        time_series_data = []
        for period_display in period_order:
            period_df = filtered_df[filtered_df["period_display"] == period_display]

            for facility_uid in facility_uids:
                facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
                if not facility_period_df.empty:
                    lbw_data = compute_lbw_kpi(facility_period_df, [facility_uid])
                    actual_lbw_count = lbw_data["lbw_count"]  # FIX: Get actual count
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": lbw_data["lbw_rate"],
                            "lbw_count": actual_lbw_count,  # This should fix hover issue
                            "total_weighed": lbw_data["total_weighed"],
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

        # FIX: Use correct custom data for hover
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>LBW Cases: %{customdata[0]}<br>Total Weighed: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (time_series_df["lbw_count"], time_series_df["total_weighed"])
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
            hover_data=["lbw_count", "total_weighed"],
        )

        # FIX: Use correct custom data
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>LBW Rate: %{y:.2f}%<br>LBW Cases: %{customdata[0]}<br>Total Weighed: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (comparison_df["lbw_count"], comparison_df["total_weighed"])
            ),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="LBW Rate (%)",
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

    # FIX: Facility comparison table - Only show LBW categories
    st.subheader("📋 Facility Comparison Summary")
    facility_table_data = []

    for facility_name, facility_uid in zip(facility_names, facility_uids):
        facility_df = df[df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            lbw_data = compute_lbw_kpi(facility_df, [facility_uid])
            row_data = {
                "Facility Name": facility_name,
                "LBW Cases": lbw_data["lbw_count"],
                "Total Weighed": lbw_data["total_weighed"],
                "LBW Rate (%)": lbw_data["lbw_rate"],
            }
            # Add LBW rates for each weight category
            for category_key, category_info in LBW_CATEGORIES.items():
                row_data[f"{category_info['name']} LBW Rate (%)"] = lbw_data[
                    "category_rates"
                ][category_key]
                row_data[f"{category_info['name']} Count"] = lbw_data["lbw_categories"][
                    category_key
                ]

            facility_table_data.append(row_data)

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["LBW Cases"].sum()
    total_denominator = facility_table_df["Total Weighed"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "LBW Cases": total_numerator,
        "Total Weighed": total_denominator,
        "LBW Rate (%)": overall_value,
    }

    # Add category totals (only LBW categories)
    for category_key, category_info in LBW_CATEGORIES.items():
        overall_row[f"{category_info['name']} LBW Rate (%)"] = (
            (
                facility_table_df[f"{category_info['name']} Count"].sum()
                / total_denominator
                * 100
            )
            if total_denominator > 0
            else 0
        )
        overall_row[f"{category_info['name']} Count"] = facility_table_df[
            f"{category_info['name']} Count"
        ].sum()

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    format_dict = {
        "LBW Cases": "{:,.0f}",
        "Total Weighed": "{:,.0f}",
        "LBW Rate (%)": "{:.2f}%",
    }

    for category_info in LBW_CATEGORIES.values():
        format_dict[f"{category_info['name']} LBW Rate (%)"] = "{:.2f}%"
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"

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
        file_name="lbw_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_lbw_category_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of LBW categories"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute LBW category distribution
    category_data = compute_lbw_by_category(df, facility_uids)
    total_weighed = compute_lbw_denominator(df, facility_uids)

    if total_weighed == 0:
        st.info("⚠️ No data available for LBW category distribution.")
        return

    # Prepare data for visualization - Only LBW categories
    pie_data = []
    for category_key, category_info in LBW_CATEGORIES.items():
        pie_data.append(
            {
                "Category": category_info["name"],
                "Count": category_data[category_key],
                "Percentage": (
                    (category_data[category_key] / total_weighed * 100)
                    if total_weighed > 0
                    else 0
                ),
            }
        )

    pie_df = pd.DataFrame(pie_data)

    # Chart type selection
    st.markdown("### 📊 Chart Type Selection")
    chart_type = st.selectbox(
        "Choose how to display the LBW category distribution:",
        options=["Pie Chart", "Donut Chart"],
        index=0,
        key=f"lbw_chart_type_{str(facility_uids)}",
    )

    st.markdown("---")

    # Create chart
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]

    if chart_type == "Pie Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Category",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            color="Category",
            color_discrete_sequence=colors,
        )
    else:  # Donut Chart
        fig = px.pie(
            pie_df,
            values="Count",
            names="Category",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            hole=0.4,
            color="Category",
            color_discrete_sequence=colors,
        )

    # Configure text display
    total_count = pie_df["Count"].sum()
    use_inside_text = any((pie_df["Count"] / total_count) < 0.05)

    if use_inside_text:
        fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textfont=dict(size=10),
            insidetextfont=dict(color="white", size=9),
            outsidetextfont=dict(size=9),
        )
    else:
        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textfont=dict(size=10),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        height=500,
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
        margin=dict(l=0, r=150, t=20, b=20),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    # Display chart
    with st.container():
        st.markdown(
            '<div style="font-size: 16px; font-weight: bold; margin-bottom: 10px; text-align: center;">Distribution of LBW Weight Categories</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 LBW Category Summary")
    summary_df = pie_df.copy()
    summary_df.insert(0, "No", range(1, len(summary_df) + 1))

    styled_table = (
        summary_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"})
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="lbw_category_distribution.csv",
        mime="text/csv",
    )


def render_lbw_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="lbw_rate",
    title="Low Birth Weight Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="LBW Cases",
    denominator_name="Total Weighed Births",
):
    """Render a comparison chart showing each region's LBW performance"""
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

    # Prepare comparison data - FIX: Ensure lbw_count is properly calculated
    comparison_data = []
    all_periods = filtered_df[["period_display", "period_sort"]].drop_duplicates()
    all_periods = all_periods.sort_values("period_sort")
    period_order = all_periods["period_display"].tolist()

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        region_df = filtered_df[filtered_df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            lbw_data = compute_lbw_kpi(region_df, region_facility_uids)
            # FIX: Use actual calculated lbw_count
            actual_lbw_count = lbw_data["lbw_count"]
            comparison_data.append(
                {
                    "Region": region_name,
                    "value": lbw_data["lbw_rate"],
                    "lbw_count": actual_lbw_count,  # This should now show correct values
                    "total_weighed": lbw_data["total_weighed"],
                    **{
                        f"{category}_rate": lbw_data["category_rates"][category]
                        for category in LBW_CATEGORIES.keys()
                    },
                    **{
                        f"{category}_count": lbw_data["lbw_categories"][category]
                        for category in LBW_CATEGORIES.keys()
                    },
                }
            )

    if not comparison_data:
        st.info("⚠️ No data available for region comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Chart options and creation
    chart_options = ["Line Chart", "Bar Chart"]
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_region_comparison_{str(region_names)}",
    )

    # Create chart
    if chart_type == "Line Chart":
        time_series_data = []
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
                    lbw_data = compute_lbw_kpi(region_period_df, region_facility_uids)
                    actual_lbw_count = lbw_data["lbw_count"]  # FIX: Get actual count
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": lbw_data["lbw_rate"],
                            "lbw_count": actual_lbw_count,  # This should fix hover issue
                            "total_weighed": lbw_data["total_weighed"],
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
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>LBW Cases: %{customdata[0]}<br>Total Weighed: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (time_series_df["lbw_count"], time_series_df["total_weighed"])
            ),
        )
    else:
        fig = px.bar(
            comparison_df,
            x="Region",
            y="value",
            title=title,
            height=500,
            color="Region",
            hover_data=["lbw_count", "total_weighed"],
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>LBW Rate: %{y:.2f}%<br>LBW Cases: %{customdata[0]}<br>Total Weighed: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (comparison_df["lbw_count"], comparison_df["total_weighed"])
            ),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="LBW Rate (%)",
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

    # FIX: Region comparison table - Only show LBW categories
    st.subheader("📋 Region Comparison Summary")
    region_table_data = []

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        region_df = df[df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            lbw_data = compute_lbw_kpi(region_df, region_facility_uids)
            row_data = {
                "Region Name": region_name,
                "LBW Cases": lbw_data["lbw_count"],
                "Total Weighed": lbw_data["total_weighed"],
                "LBW Rate (%)": lbw_data["lbw_rate"],
            }
            # Add LBW rates for each weight category
            for category_key, category_info in LBW_CATEGORIES.items():
                row_data[f"{category_info['name']} LBW Rate (%)"] = lbw_data[
                    "category_rates"
                ][category_key]
                row_data[f"{category_info['name']} Count"] = lbw_data["lbw_categories"][
                    category_key
                ]

            region_table_data.append(row_data)

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["LBW Cases"].sum()
    total_denominator = region_table_df["Total Weighed"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "LBW Cases": total_numerator,
        "Total Weighed": total_denominator,
        "LBW Rate (%)": overall_value,
    }

    # Add category totals (only LBW categories)
    for category_key, category_info in LBW_CATEGORIES.items():
        overall_row[f"{category_info['name']} LBW Rate (%)"] = (
            (
                region_table_df[f"{category_info['name']} Count"].sum()
                / total_denominator
                * 100
            )
            if total_denominator > 0
            else 0
        )
        overall_row[f"{category_info['name']} Count"] = region_table_df[
            f"{category_info['name']} Count"
        ].sum()

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    format_dict = {
        "LBW Cases": "{:,.0f}",
        "Total Weighed": "{:,.0f}",
        "LBW Rate (%)": "{:.2f}%",
    }

    for category_info in LBW_CATEGORIES.values():
        format_dict[f"{category_info['name']} LBW Rate (%)"] = "{:.2f}%"
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"

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
        file_name="lbw_rate_region_comparison.csv",
        mime="text/csv",
    )
