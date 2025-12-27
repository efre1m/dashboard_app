import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- LBW KPI Constants ----------------
# Updated column names for patient-level data
BIRTH_WEIGHT_COL = (
    "newborn_birth_weight_delivery_summary"  # Updated to match transformed dataset
)

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
    """Compute numerator for LBW KPI: Count of live births <2,500 g - UPDATED FOR PATIENT-LEVEL DATA"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)]

    if BIRTH_WEIGHT_COL not in filtered_df.columns:
        return 0

    # Filter rows with birth weight data
    birth_weight_data = filtered_df[filtered_df[BIRTH_WEIGHT_COL].notna()].copy()

    if birth_weight_data.empty:
        return 0

    # Convert birth weight to numeric
    birth_weight_data["weight_numeric"] = pd.to_numeric(
        birth_weight_data[BIRTH_WEIGHT_COL], errors="coerce"
    )

    # Filter out NaN values and count births < 2500g
    lbw_cases = birth_weight_data[
        (birth_weight_data["weight_numeric"] < LBW_THRESHOLD)
        & (birth_weight_data["weight_numeric"] > 0)  # Exclude negative/zero weights
    ]

    return len(lbw_cases)


def compute_lbw_by_category(df, facility_uids=None):
    """Compute distribution of birth weights by LBW categories - UPDATED FOR PATIENT-LEVEL DATA"""
    if df is None or df.empty:
        return {category: 0 for category in LBW_CATEGORIES.keys()}

    # Filter by facilities if specified
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)]

    if BIRTH_WEIGHT_COL not in filtered_df.columns:
        return {category: 0 for category in LBW_CATEGORIES.keys()}

    # Filter rows with birth weight data
    birth_weight_data = filtered_df[filtered_df[BIRTH_WEIGHT_COL].notna()].copy()

    if birth_weight_data.empty:
        return {category: 0 for category in LBW_CATEGORIES.keys()}

    # Convert birth weight to numeric
    birth_weight_data["weight_numeric"] = pd.to_numeric(
        birth_weight_data[BIRTH_WEIGHT_COL], errors="coerce"
    )

    # Filter out NaN and invalid weights
    valid_weights = birth_weight_data[
        (birth_weight_data["weight_numeric"].notna())
        & (birth_weight_data["weight_numeric"] > 0)
    ]

    if valid_weights.empty:
        return {category: 0 for category in LBW_CATEGORIES.keys()}

    # Count occurrences in each category
    result = {}
    for category_key, category_info in LBW_CATEGORIES.items():
        count = len(
            valid_weights[
                (valid_weights["weight_numeric"] >= category_info["min"])
                & (valid_weights["weight_numeric"] <= category_info["max"])
            ]
        )
        result[category_key] = int(count)

    return result


def compute_lbw_denominator(df, facility_uids=None):
    """Compute denominator for LBW KPI: Count of all live births that were weighed - UPDATED FOR PATIENT-LEVEL DATA"""
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)]

    if BIRTH_WEIGHT_COL not in filtered_df.columns:
        return 0

    # Filter rows with birth weight data
    birth_weight_data = filtered_df[filtered_df[BIRTH_WEIGHT_COL].notna()].copy()

    if birth_weight_data.empty:
        return 0

    # Convert birth weight to numeric and filter valid weights
    birth_weight_data["weight_numeric"] = pd.to_numeric(
        birth_weight_data[BIRTH_WEIGHT_COL], errors="coerce"
    )

    valid_weights = birth_weight_data[
        (birth_weight_data["weight_numeric"].notna())
        & (birth_weight_data["weight_numeric"] > 0)
    ]

    return len(valid_weights)


def compute_lbw_kpi(df, facility_uids=None):
    """Compute LBW KPI for the given dataframe - UPDATED FOR PATIENT-LEVEL DATA"""
    if df is None or df.empty:
        return {
            "lbw_rate": 0.0,
            "lbw_count": 0,
            "total_weighed": 0,
            "lbw_categories": {category: 0 for category in LBW_CATEGORIES.keys()},
            "category_rates": {category: 0.0 for category in LBW_CATEGORIES.keys()},
        }

    # Filter by facilities if specified
    filtered_df = df.copy()
    if facility_uids:
        if isinstance(facility_uids, str):
            facility_uids = [facility_uids]
        if "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)]

    # Count LBW cases (numerator)
    lbw_count = compute_lbw_numerator(filtered_df, facility_uids)

    # Count total weighed births (denominator)
    total_weighed = compute_lbw_denominator(filtered_df, facility_uids)

    # Get distribution of LBW categories
    lbw_categories = compute_lbw_by_category(filtered_df, facility_uids)

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


# ---------------- Helper Functions for Trend Analysis ----------------
def get_numerator_denominator_for_lbw(df, facility_uids=None, date_range_filters=None):
    """Get numerator and denominator for LBW KPI with date filtering - UPDATED FOR PATIENT-LEVEL DATA"""
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()

    # Apply facility filter
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Apply date filtering if provided
    if date_range_filters:
        # Get the delivery date column for LBW KPI
        from utils.kpi_utils import get_relevant_date_column_for_kpi

        date_column = get_relevant_date_column_for_kpi(
            "Low Birth Weight (LBW) Rate (%)"
        )

        if date_column and date_column in filtered_df.columns:
            # Convert to datetime
            filtered_df[date_column] = pd.to_datetime(
                filtered_df[date_column], errors="coerce"
            )

            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                filtered_df = filtered_df[
                    (filtered_df[date_column] >= start_dt)
                    & (filtered_df[date_column] < end_dt)
                ].copy()

    # Compute KPI
    kpi_data = compute_lbw_kpi(filtered_df, facility_uids)

    return (
        kpi_data.get("lbw_count", 0),
        kpi_data.get("total_weighed", 1),
        kpi_data.get("lbw_rate", 0.0),
    )


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
    """Render a trend chart for low birth weight rate - UPDATED FOR PATIENT-LEVEL DATA"""
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

    # FIX: Ensure all required category columns exist
    required_category_columns = []
    for category_key in LBW_CATEGORIES.keys():
        rate_col = f"{category_key}_rate"
        count_col = f"{category_key}_count"
        required_category_columns.extend([rate_col, count_col])

        # Add missing columns with default values
        if rate_col not in df.columns:
            df[rate_col] = 0.0
        if count_col not in df.columns:
            df[count_col] = 0

    # Create chart based on selected type
    if chart_type == "line chart":
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
        # FIXED: Stacked bar chart with corrected hover template and dynamic x-axis
        fig = go.Figure()

        # Colors for LBW categories
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]

        # Calculate the maximum overall LBW rate for dynamic x-axis range
        max_overall_rate = df[value_col].max()
        # Set x-axis range with some padding, capped at 100%
        xaxis_max = min(max_overall_rate * 1.1, 100)

        # Calculate proportional contributions of each category to the overall LBW rate
        # This ensures the total bar length equals the overall LBW rate
        for period_idx, period_row in df.iterrows():
            overall_lbw_rate = period_row[value_col]
            total_category_rates = sum(
                [
                    period_row.get(f"{category_key}_rate", 0)
                    for category_key in LBW_CATEGORIES.keys()
                ]
            )

            # If there are no LBW cases or overall rate is 0, all segments should be 0
            if overall_lbw_rate == 0 or total_category_rates == 0:
                # Add zero-width bars for all categories
                for category_key in LBW_CATEGORIES.keys():
                    category_info = LBW_CATEGORIES[category_key]
                    category_rate = period_row.get(f"{category_key}_rate", 0)
                    category_count = period_row.get(f"{category_key}_count", 0)

                    fig.add_trace(
                        go.Bar(
                            name=category_info["name"],
                            y=[period_row[period_col]],
                            x=[0],
                            orientation="h",
                            marker_color=colors[
                                list(LBW_CATEGORIES.keys()).index(category_key)
                                % len(colors)
                            ],
                            showlegend=(
                                period_idx == 0
                            ),  # Only show legend for first period
                            # FIXED: Use actual category rate in hover template
                            hovertemplate=(
                                f"<b>{period_row[period_col]}</b><br>"
                                f"{category_info['name']}: {category_rate:.2f}%<br>"
                                f"Count: {category_count}<br>"
                                f"Total Weighed: {period_row[denominator_name]}"
                                "<extra></extra>"
                            ),
                        )
                    )
            else:
                # Calculate proportional contributions
                cumulative_width = 0
                for i, (category_key, category_info) in enumerate(
                    LBW_CATEGORIES.items()
                ):
                    category_rate = period_row.get(f"{category_key}_rate", 0)
                    category_count = period_row.get(f"{category_key}_count", 0)

                    # Calculate proportional width of this category segment
                    if total_category_rates > 0:
                        proportional_width = (
                            category_rate / total_category_rates
                        ) * overall_lbw_rate
                    else:
                        proportional_width = 0

                    fig.add_trace(
                        go.Bar(
                            name=category_info["name"],
                            y=[period_row[period_col]],
                            x=[proportional_width],
                            orientation="h",
                            marker_color=colors[i % len(colors)],
                            showlegend=(
                                period_idx == 0
                            ),  # Only show legend for first period
                            # FIXED: Use actual category rate in hover template
                            hovertemplate=(
                                f"<b>{period_row[period_col]}</b><br>"
                                f"{category_info['name']}: {category_rate:.2f}%<br>"
                                f"Count: {category_count}<br>"
                                f"Total Weighed: {period_row[denominator_name]}"
                                "<extra></extra>"
                            ),
                            base=[cumulative_width],  # Stack on previous segments
                        )
                    )
                    cumulative_width += proportional_width

        # Set dynamic x-axis range with better tick spacing
        fig.update_layout(
            barmode="stack",
            title="LBW Rate by Weight Category (%) - Proportional to Overall Rate",
            height=400,
            xaxis_title="LBW Rate (%)",
            yaxis_title="Period",
            xaxis=dict(
                range=[0, xaxis_max],
                tickmode="linear",
                tick0=0,
                dtick=10,  # Show ticks every 10% for better readability
                showgrid=True,
                gridcolor="rgba(128,128,128,0.2)",
            ),
            bargap=0.3,
            showlegend=True,
        )

        fig.update_traces(width=0.6)

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        title=dict(
            text=(
                title
                if chart_type == "line chart"
                else "LBW Rate by Weight Category (%) - Proportional to Overall Rate"
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            type="category" if chart_type == "line chart" else "linear",
            tickangle=-45 if chart_type == "line chart" else 0,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            type="linear" if chart_type == "line chart" else "category",
            rangemode="tozero" if chart_type == "line chart" else None,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True if chart_type == "line chart" else False,
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

    # Summary table - FIX: Ensure all required columns exist
    st.subheader(f"📋 {title} Summary Table")

    # Main KPI Overview
    st.markdown("**Main KPI Overview**")
    main_summary_df = df.copy().reset_index(drop=True)

    # Ensure all required columns exist
    table_columns = [period_col, numerator_name, denominator_name, value_col]
    for col in table_columns:
        if col not in main_summary_df.columns:
            main_summary_df[col] = 0  # Add missing column with default value

    main_summary_df = main_summary_df[table_columns]

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
                value_col: "{:.2f}%",
                numerator_name: "{:,.0f}",
                denominator_name: "{:,.0f}",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_main_table.to_html(), unsafe_allow_html=True)

    # Category Breakdown Table - FIX: Show Count first then Rate for each category
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**LBW Rate by Weight Category**")

    category_summary_data = []
    for period_idx, period_row in df.iterrows():
        period_data = {
            "Period": period_row[period_col],
            "Total Weighed": period_row[denominator_name],
            "Overall LBW Rate (%)": period_row[value_col],
        }

        # Add each category's COUNT first then RATE (like uterotonic KPI)
        for category_key, category_info in LBW_CATEGORIES.items():
            rate_col = f"{category_key}_rate"
            count_col = f"{category_key}_count"

            # Add count first, then rate
            period_data[f"{category_info['name']} Count"] = period_row.get(count_col, 0)
            period_data[f"{category_info['name']} Rate (%)"] = period_row.get(
                rate_col, 0
            )

        category_summary_data.append(period_data)

    category_summary_df = pd.DataFrame(category_summary_data)

    # Add overall row
    overall_category_row = {
        "Period": f"Overall {title}",
        "Total Weighed": total_denominator,
        "Overall LBW Rate (%)": overall_value,
    }
    for category_key, category_info in LBW_CATEGORIES.items():
        total_category_count = category_summary_df[
            f"{category_info['name']} Count"
        ].sum()
        overall_category_rate = (
            (total_category_count / total_denominator * 100)
            if total_denominator > 0
            else 0
        )

        # Add count first, then rate
        overall_category_row[f"{category_info['name']} Count"] = total_category_count
        overall_category_row[f"{category_info['name']} Rate (%)"] = (
            overall_category_rate
        )

    overall_category_df = pd.DataFrame([overall_category_row])
    category_summary_table = pd.concat(
        [category_summary_df, overall_category_df], ignore_index=True
    )
    category_summary_table.insert(0, "No", range(1, len(category_summary_table) + 1))

    # Format category table - ensure Count comes before Rate in formatting
    format_dict = {"Total Weighed": "{:,.0f}", "Overall LBW Rate (%)": "{:.2f}%"}
    for category_info in LBW_CATEGORIES.values():
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"
        format_dict[f"{category_info['name']} Rate (%)"] = "{:.2f}%"

    styled_category_table = (
        category_summary_table.style.format(format_dict)
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_category_table.to_html(), unsafe_allow_html=True)

    # Download button
    combined_table = pd.concat(
        [main_summary_table, category_summary_table.iloc[:, 2:]], axis=1
    )
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
    """SIMPLIFIED VERSION: Render facility comparison without numerator/denominator in hover - UPDATED FOR PATIENT-LEVEL DATA"""
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
        # For line chart, compute time series data
        time_series_data = []
        all_periods = (
            filtered_df[["period_display", "period_sort"]]
            .drop_duplicates()
            .sort_values("period_sort")
        )
        period_order = all_periods["period_display"].tolist()

        for period_display in period_order:
            period_df = filtered_df[filtered_df["period_display"] == period_display]

            for facility_uid in facility_uids:
                facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
                if not facility_period_df.empty:
                    lbw_data = compute_lbw_kpi(facility_period_df, [facility_uid])
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": lbw_data["lbw_rate"],
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

        # SIMPLE HOVER: Only show rate, no numerator/denominator
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<extra></extra>",
        )

    else:  # Bar Chart
        # For bar chart, compute overall values
        bar_data = []
        for facility_uid in facility_uids:
            facility_df = filtered_df[filtered_df["orgUnit"] == facility_uid]
            if not facility_df.empty:
                lbw_data = compute_lbw_kpi(facility_df, [facility_uid])
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": lbw_data["lbw_rate"],
                        "lbw_count": lbw_data["lbw_count"],
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

        if not bar_data:
            st.info("⚠️ No data available for bar chart.")
            return

        bar_df = pd.DataFrame(bar_data)

        fig = px.bar(
            bar_df, x="Facility", y="value", title=title, height=500, color="Facility"
        )

        # SIMPLE HOVER: Only show rate, no numerator/denominator
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>LBW Rate: %{y:.2f}%<extra></extra>"
        )

    # Common layout updates
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
        showlegend=True,
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Facility comparison table (shows all details including numerator/denominator)
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
            # Add LBW Count first then Rate for each weight category
            for category_key, category_info in LBW_CATEGORIES.items():
                row_data[f"{category_info['name']} Count"] = lbw_data["lbw_categories"][
                    category_key
                ]
                row_data[f"{category_info['name']} LBW Rate (%)"] = lbw_data[
                    "category_rates"
                ][category_key]

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

    # Add category totals (only LBW categories) - Count first then Rate
    for category_key, category_info in LBW_CATEGORIES.items():
        total_category_count = facility_table_df[f"{category_info['name']} Count"].sum()
        overall_row[f"{category_info['name']} Count"] = total_category_count
        overall_row[f"{category_info['name']} LBW Rate (%)"] = (
            (total_category_count / total_denominator * 100)
            if total_denominator > 0
            else 0
        )

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table - Count first then Rate
    format_dict = {
        "LBW Cases": "{:,.0f}",
        "Total Weighed": "{:,.0f}",
        "LBW Rate (%)": "{:.2f}%",
    }

    for category_info in LBW_CATEGORIES.values():
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"
        format_dict[f"{category_info['name']} LBW Rate (%)"] = "{:.2f}%"

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
    """SIMPLIFIED VERSION: Render region comparison without numerator/denominator in hover - UPDATED FOR PATIENT-LEVEL DATA"""
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
        # For line chart, compute time series data
        time_series_data = []
        all_periods = (
            filtered_df[["period_display", "period_sort"]]
            .drop_duplicates()
            .sort_values("period_sort")
        )
        period_order = all_periods["period_display"].tolist()

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
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": lbw_data["lbw_rate"],
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

        # SIMPLE HOVER: Only show rate, no numerator/denominator
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<extra></extra>",
        )

    else:  # Bar Chart
        # For bar chart, compute overall values
        bar_data = []
        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]
            region_df = filtered_df[filtered_df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                lbw_data = compute_lbw_kpi(region_df, region_facility_uids)
                bar_data.append(
                    {
                        "Region": region_name,
                        "value": lbw_data["lbw_rate"],
                        "lbw_count": lbw_data["lbw_count"],
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

        if not bar_data:
            st.info("⚠️ No data available for bar chart.")
            return

        bar_df = pd.DataFrame(bar_data)

        fig = px.bar(
            bar_df, x="Region", y="value", title=title, height=500, color="Region"
        )

        # SIMPLE HOVER: Only show rate, no numerator/denominator
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>LBW Rate: %{y:.2f}%<extra></extra>"
        )

    # Common layout updates
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
        showlegend=True,
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Region comparison table (shows all details including numerator/denominator)
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
            # Add LBW Count first then Rate for each weight category
            for category_key, category_info in LBW_CATEGORIES.items():
                row_data[f"{category_info['name']} Count"] = lbw_data["lbw_categories"][
                    category_key
                ]
                row_data[f"{category_info['name']} LBW Rate (%)"] = lbw_data[
                    "category_rates"
                ][category_key]

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

    # Add category totals (only LBW categories) - Count first then Rate
    for category_key, category_info in LBW_CATEGORIES.items():
        total_category_count = region_table_df[f"{category_info['name']} Count"].sum()
        overall_row[f"{category_info['name']} Count"] = total_category_count
        overall_row[f"{category_info['name']} LBW Rate (%)"] = (
            (total_category_count / total_denominator * 100)
            if total_denominator > 0
            else 0
        )

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table - Count first then Rate
    format_dict = {
        "LBW Cases": "{:,.0f}",
        "Total Weighed": "{:,.0f}",
        "LBW Rate (%)": "{:.2f}%",
    }

    for category_info in LBW_CATEGORIES.values():
        format_dict[f"{category_info['name']} Count"] = "{:,.0f}"
        format_dict[f"{category_info['name']} LBW Rate (%)"] = "{:.2f}%"

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


def render_lbw_category_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of LBW categories - UPDATED FOR PATIENT-LEVEL DATA"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute LBW category distribution
    category_data = compute_lbw_by_category(df, facility_uids)
    total_weighed = compute_lbw_denominator(df, facility_uids)
    total_lbw_cases = compute_lbw_numerator(df, facility_uids)

    if total_weighed == 0:
        st.info("⚠️ No data available for LBW category distribution.")
        return

    # Prepare data for visualization - Only LBW categories
    pie_data = []
    for category_key, category_info in LBW_CATEGORIES.items():
        category_count = category_data[category_key]
        distribution_percentage = (
            (category_count / total_lbw_cases * 100) if total_lbw_cases > 0 else 0
        )

        pie_data.append(
            {
                "Category": category_info["name"],
                "Count": category_count,
                "Distribution Percentage": distribution_percentage,
            }
        )

    pie_df = pd.DataFrame(pie_data)

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

    # Chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Pie Chart", "Donut Chart"],
        index=0,
        key=f"lbw_chart_type_{str(facility_uids)}",
    )

    # Create chart
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]

    if chart_type == "Pie Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Category",
            hover_data=["Distribution Percentage"],
            labels={"Count": "Count", "Distribution Percentage": "Distribution %"},
            height=500,
        )
    else:  # Donut Chart
        fig = px.pie(
            pie_df,
            values="Count",
            names="Category",
            hover_data=["Distribution Percentage"],
            labels={"Count": "Count", "Distribution Percentage": "Distribution %"},
            height=500,
            hole=0.4,
        )

    # Calculate if we should use inside text for small slices
    total_count = pie_df["Count"].sum()
    use_inside_text = any((pie_df["Count"] / total_count) < 0.05)

    if use_inside_text:
        fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Distribution: %{customdata[0]:.2f}%<extra></extra>",
            customdata=pie_df["Distribution Percentage"],
            textfont=dict(size=10),
            insidetextfont=dict(color="white", size=9),
            outsidetextfont=dict(size=9),
        )
    else:
        fig.update_traces(
            textinfo="percent+label",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Distribution: %{customdata[0]:.2f}%<extra></extra>",
            customdata=pie_df["Distribution Percentage"],
            textfont=dict(size=10),
        )

    # FIX: COMPLETELY REMOVE ANY TITLE FROM THE LAYOUT
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

    # CRITICAL FIX: Remove any trace of title completely
    fig.update_layout(title=None)  # Explicitly set title to None
    fig.layout.pop("title", None)  # Remove title from layout completely

    # Also check and remove any annotations that might contain "undefined"
    if hasattr(fig.layout, "annotations") and fig.layout.annotations:
        fig.layout.annotations = []

    # Use container to control layout
    with st.container():
        st.markdown(
            '<div class="pie-chart-title">Distribution of LBW Weight Categories</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 LBW Category Distribution Summary")
    summary_df = pie_df.copy()
    summary_df.insert(0, "No", range(1, len(summary_df) + 1))

    # Add total row
    total_row = {
        "No": len(summary_df) + 1,
        "Category": "Total LBW Cases",
        "Count": total_lbw_cases,
        "Distribution Percentage": 100.0 if total_lbw_cases > 0 else 0.0,
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    styled_table = (
        summary_df.style.format(
            {"Count": "{:,.0f}", "Distribution Percentage": "{:.2f}%"}
        )
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
