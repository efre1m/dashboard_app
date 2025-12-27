import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt
import plotly.express as px
import hashlib

# Import only utility functions from kpi_utils
from utils.kpi_utils import (
    auto_text_color,
    format_period_month_year,
    compute_total_deliveries,  # Use the same total deliveries function
    get_relevant_date_column_for_kpi,
)

# ---------------- Caching Setup ----------------
if "pph_cache" not in st.session_state:
    st.session_state.pph_cache = {}


def get_pph_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for PPH computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_pph_cache():
    """Clear the PPH cache"""
    st.session_state.pph_cache = {}


# PPH KPI Configuration
PPH_CONDITION_COL = "obstetric_condition_at_delivery_delivery_summary"
PPH_CODE = "3"

# Obstetric condition codes mapping
OBSTETRIC_CONDITION_CODES = {
    "1": "Severe pre-eclampsia/eclampsia",
    "2": "Antepartum hemorrhage",
    "3": "Postpartum hemorrhage",
    "4": "Sepsis/severe systemic infection",
    "5": "Obstructed labor",
    "6": "Ruptured uterus",
    "7": "Severe anemia",
    "8": "Malaria with severe anemia",
    "9": "HIV-related conditions",
    "10": "Other complications",
}


def compute_pph_count(df, facility_uids=None):
    """
    Count PPH occurrences - "3" alone or in combinations (e.g., "2,3", "3,5")
    EXACT SAME METHOD PATTERN AS compute_csection_count in kpi_utils.py
    """
    cache_key = get_pph_cache_key(df, facility_uids, "pph_count")

    if cache_key in st.session_state.pph_cache:
        return st.session_state.pph_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if PPH_CONDITION_COL not in filtered_df.columns:
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string and handle multiple codes (e.g., "2,3", "3,5,10")
            df_copy["obstetric_condition_clean"] = df_copy[PPH_CONDITION_COL].astype(
                str
            )

            # Check for "3" in the string (standalone or in comma-separated list)
            # Pattern matches: "3" alone, "3," at start, ",3" in middle, ",3," anywhere, or "3" at end
            pph_mask = df_copy["obstetric_condition_clean"].str.contains(
                r"(^|[,;\s])3([,;\s]|$)", regex=True, na=False
            )

            result = int(pph_mask.sum())

    st.session_state.pph_cache[cache_key] = result
    return result


def compute_obstetric_condition_distribution(df, facility_uids=None):
    """
    Compute distribution of obstetric conditions
    Returns: Dictionary with counts for each obstetric condition
    Handles comma-separated values like "2,3" - each code counts separately
    """
    if df is None or df.empty:
        return {name: 0 for name in OBSTETRIC_CONDITION_CODES.values()}

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if PPH_CONDITION_COL not in filtered_df.columns:
        return {name: 0 for name in OBSTETRIC_CONDITION_CODES.values()}

    # Initialize counts
    counts = {code: 0 for code in OBSTETRIC_CONDITION_CODES.keys()}
    named_counts = {name: 0 for name in OBSTETRIC_CONDITION_CODES.values()}

    df_copy = filtered_df.copy()
    df_copy["obstetric_condition_clean"] = df_copy[PPH_CONDITION_COL].astype(str)

    # Process each value
    for value in df_copy["obstetric_condition_clean"]:
        if pd.isna(value) or value == "nan":
            continue

        value_str = str(value).strip()

        # Skip empty values
        if not value_str or value_str.lower() in ["n/a", "nan", "null"]:
            continue

        # Split by comma and count each code
        codes = [code.strip() for code in value_str.split(",")]

        for code in codes:
            if code in OBSTETRIC_CONDITION_CODES:
                counts[code] += 1
                condition_name = OBSTETRIC_CONDITION_CODES[code]
                named_counts[condition_name] += 1

    # Add total count
    total = sum(counts.values())
    named_counts["Total Complications"] = total

    return named_counts


def compute_pph_rate(df, facility_uids=None):
    """
    Compute PPH Rate - EXACT SAME PATTERN AS compute_csection_rate in kpi_utils.py
    Returns: (rate, pph_cases, total_deliveries)
    """
    cache_key = get_pph_cache_key(df, facility_uids, "pph_rate")

    if cache_key in st.session_state.pph_cache:
        return st.session_state.pph_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for PPH (same as delivery summary)
        date_column = get_relevant_date_column_for_kpi(
            "Postpartum Hemorrhage (PPH) Rate (%)"
        )

        # Count PPH cases (value "3" in obstetric condition)
        pph_cases = compute_pph_count(df, facility_uids)

        # Get total deliveries - USING SAME LOGIC AS C-SECTION
        total_deliveries = compute_total_deliveries(df, facility_uids, date_column)

        # Calculate rate
        rate = (pph_cases / total_deliveries * 100) if total_deliveries > 0 else 0.0
        result = (rate, pph_cases, total_deliveries)

    st.session_state.pph_cache[cache_key] = result
    return result


def compute_pph_kpi(df, facility_uids=None):
    """
    Compute PPH KPI data - SAME STRUCTURE AS compute_csection_rate
    """
    rate, pph_cases, total_deliveries = compute_pph_rate(df, facility_uids)

    # Get obstetric condition distribution
    condition_distribution = compute_obstetric_condition_distribution(df, facility_uids)

    return {
        "pph_rate": float(rate),
        "pph_cases": int(pph_cases),
        "total_deliveries": int(total_deliveries),
        "obstetric_conditions": condition_distribution,
    }


def get_numerator_denominator_for_pph(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for PPH KPI - SAME PATTERN AS C-SECTION
    WITH DATE RANGE FILTERING
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for PPH (same as delivery summary)
    date_column = get_relevant_date_column_for_kpi(
        "Postpartum Hemorrhage (PPH) Rate (%)"
    )

    # IMPORTANT: Filter to only include rows that have this specific date
    if date_column in filtered_df.columns:
        # Convert to datetime and filter out rows without this date
        filtered_df[date_column] = pd.to_datetime(
            filtered_df[date_column], errors="coerce"
        )
        filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

        # Apply date range filtering if provided
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                filtered_df = filtered_df[
                    (filtered_df[date_column] >= start_dt)
                    & (filtered_df[date_column] < end_dt)
                ].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    # Compute PPH rate on date-filtered data
    rate, pph_cases, total_deliveries = compute_pph_rate(filtered_df, facility_uids)

    return (pph_cases, total_deliveries, rate)


# ---------------- Pie Chart Function ----------------
def render_obstetric_condition_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """
    Render a pie chart showing distribution of obstetric conditions
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute obstetric condition distribution
    condition_data = compute_obstetric_condition_distribution(df, facility_uids)

    if condition_data["Total Complications"] == 0:
        st.info("⚠️ No data available for obstetric condition distribution.")
        return

    # Prepare data for visualization (exclude "Total Complications" from pie chart)
    pie_data = []
    for condition_name, count in condition_data.items():
        if condition_name != "Total Complications" and count > 0:
            pie_data.append({"Condition": condition_name, "Count": count})

    if not pie_data:
        st.info("⚠️ No obstetric condition data available.")
        return

    pie_df = pd.DataFrame(pie_data)

    # Sort by count (descending)
    pie_df = pie_df.sort_values("Count", ascending=False)

    # Calculate percentages
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

    # Add chart type selection
    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Pie Chart", "Donut Chart", "Bar Chart"],
        index=0,
        key=f"obstetric_chart_type_{str(facility_uids)}",
    )

    # Add CSS for better layout
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

    # Create chart
    if chart_type == "Pie Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Condition",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            color="Condition",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    elif chart_type == "Donut Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Condition",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            hole=0.4,
            color="Condition",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
    else:  # Bar Chart
        fig = px.bar(
            pie_df,
            x="Condition",
            y="Count",
            color="Condition",
            title="Obstetric Condition Distribution",
            height=500,
            color_discrete_sequence=px.colors.qualitative.Set3,
            text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Obstetric Condition",
            yaxis_title="Count",
            xaxis_tickangle=-45,
            showlegend=False,
        )

    if chart_type in ["Pie Chart", "Donut Chart"]:
        # Calculate if we should use inside text for small slices
        total_count = pie_df["Count"].sum()
        use_inside_text = any((pie_df["Count"] / total_count) < 0.05)

        if use_inside_text:
            # For small slices, put text inside with white background
            fig.update_traces(
                textinfo="percent+label",
                textposition="inside",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                textfont=dict(size=10),
                insidetextfont=dict(color="white", size=9),
                outsidetextfont=dict(size=9),
            )
        else:
            # For normal slices, use outside text
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

    # Ensure no "undefined" placeholder
    if chart_type in ["Pie Chart", "Donut Chart"]:
        fig.update_layout(title=None)
        fig.layout.pop("title", None)

    # Use container to control layout
    with st.container():
        if chart_type in ["Pie Chart", "Donut Chart"]:
            st.markdown(
                '<div class="pie-chart-title">Distribution of Obstetric Conditions</div>',
                unsafe_allow_html=True,
            )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 Obstetric Condition Summary")

    # Create summary dataframe with all conditions
    summary_data = []
    for condition_name, count in condition_data.items():
        if condition_name != "Total Complications":
            percentage = (
                (count / condition_data["Total Complications"] * 100)
                if condition_data["Total Complications"] > 0
                else 0
            )
            summary_data.append(
                {"Condition": condition_name, "Count": count, "Percentage": percentage}
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df[summary_df["Count"] > 0].sort_values(
        "Count", ascending=False
    )

    if not summary_df.empty:
        summary_df.insert(0, "No", range(1, len(summary_df) + 1))

        styled_table = (
            summary_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"})
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_table.to_html(), unsafe_allow_html=True)

        # Add total row
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Complications", f"{condition_data['Total Complications']:,.0f}"
            )
        with col2:
            st.metric(
                "PPH Cases", f"{condition_data.get('Postpartum hemorrhage', 0):,.0f}"
            )

        # Add download button for CSV
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="obstetric_condition_distribution.csv",
            mime="text/csv",
        )
    else:
        st.info("No obstetric condition data available.")


# ---------------- Chart Functions WITH TABLES ----------------
def render_pph_trend_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Postpartum Hemorrhage (PPH) Rate Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
    facility_uids=None,
):
    """Render trend chart for PPH - EXACT SAME AS render_svd_trend_chart but with PPH titles"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    chart_options = ["Line", "Bar", "Area"]

    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_pph_{str(facility_uids)}",
    ).lower()

    if "numerator" in df.columns and "denominator" in df.columns:
        df[numerator_name] = df["numerator"]
        df[denominator_name] = df["denominator"]
        hover_columns = [numerator_name, denominator_name]
        use_hover_data = True
    else:
        hover_columns = []
        use_hover_data = False

    try:
        if chart_type == "line":
            fig = px.line(
                df,
                x=x_axis_col,
                y=value_col,
                markers=True,
                line_shape="linear",
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=7),
            )
        elif chart_type == "bar":
            fig = px.bar(
                df,
                x=x_axis_col,
                y=value_col,
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
            )
        elif chart_type == "area":
            fig = px.area(
                df,
                x=x_axis_col,
                y=value_col,
                title=title,
                height=400,
                hover_data=hover_columns if use_hover_data else None,
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
                hover_data=hover_columns if use_hover_data else None,
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=7),
            )
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        fig = px.line(
            df,
            x=x_axis_col,
            y=value_col,
            markers=True,
            title=title,
            height=400,
        )

    is_categorical = (
        not all(isinstance(x, (dt.date, dt.datetime)) for x in df[period_col])
        if not df.empty
        else True
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title=period_col,
        yaxis_title=value_col,
        xaxis=dict(
            type="category" if is_categorical else None,
            tickangle=-45 if is_categorical else 0,
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

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Data Table")

    # Create a clean display dataframe
    display_df = df.copy()

    # Select columns to show in table
    table_columns = [x_axis_col, value_col]

    # Add numerator and denominator if available
    if "numerator" in display_df.columns and "denominator" in display_df.columns:
        display_df[numerator_name] = display_df["numerator"]
        display_df[denominator_name] = display_df["denominator"]
        table_columns.extend([numerator_name, denominator_name])

    # Format the dataframe for display
    display_df = display_df[table_columns].copy()

    # Format numbers
    display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:.2f}%")

    if numerator_name in display_df.columns:
        display_df[numerator_name] = display_df[numerator_name].apply(
            lambda x: f"{x:,.0f}"
        )
    if denominator_name in display_df.columns:
        display_df[denominator_name] = display_df[denominator_name].apply(
            lambda x: f"{x:,.0f}"
        )

    # Add Overall/Total row
    if "numerator" in df.columns and "denominator" in df.columns:
        total_numerator = df["numerator"].sum()
        total_denominator = df["denominator"].sum()
        overall_value = (
            (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        )
    else:
        overall_value = df[value_col].mean() if not df.empty else 0
        total_numerator = df[value_col].sum() if not df.empty else 0
        total_denominator = len(df)

    # Create overall row with consistent date format
    overall_row = {
        x_axis_col: "Overall",
        value_col: f"{overall_value:.2f}%",
    }

    if numerator_name in display_df.columns:
        overall_row[numerator_name] = f"{total_numerator:,.0f}"
    if denominator_name in display_df.columns:
        overall_row[denominator_name] = f"{total_denominator:,.0f}"

    # Convert to DataFrame and append
    overall_df = pd.DataFrame([overall_row])
    display_df = pd.concat([display_df, overall_df], ignore_index=True)

    # Display the table
    st.dataframe(display_df, use_container_width=True)

    # Add summary statistics
    if len(df) > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📈 Latest Value", f"{df[value_col].iloc[-1]:.2f}%")

        with col2:
            st.metric("📊 Average", f"{df[value_col].mean():.2f}%")

        with col3:
            # Calculate trend
            last_value = df[value_col].iloc[-1]
            prev_value = df[value_col].iloc[-2]
            trend_change = last_value - prev_value
            trend_symbol = (
                "▲" if trend_change > 0 else ("▼" if trend_change < 0 else "–")
            )
            st.metric("📈 Trend from Previous", f"{trend_change:.2f}% {trend_symbol}")

    # Download button
    summary_df = df.copy().reset_index(drop=True)

    if "numerator" in summary_df.columns and "denominator" in summary_df.columns:
        summary_df = summary_df[
            [x_axis_col, "numerator", "denominator", value_col]
        ].copy()

        # Format period column
        if x_axis_col in summary_df.columns:
            summary_df[x_axis_col] = summary_df[x_axis_col].apply(
                format_period_month_year
            )

        summary_df = summary_df.rename(
            columns={
                "numerator": numerator_name,
                "denominator": denominator_name,
                value_col: "PPH Rate (%)",
            }
        )

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
                "PPH Rate (%)": [overall_value],
            }
        )

        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    else:
        summary_df = summary_df[[x_axis_col, value_col]].copy()

        # Format period column
        if x_axis_col in summary_df.columns:
            summary_df[x_axis_col] = summary_df[x_axis_col].apply(
                format_period_month_year
            )

        summary_df = summary_df.rename(columns={value_col: "PPH Rate (%)"})
        summary_table = summary_df.copy()

        overall_value = (
            summary_table["PPH Rate (%)"].mean() if not summary_table.empty else 0
        )
        overall_row = pd.DataFrame(
            {x_axis_col: ["Overall"], "PPH Rate (%)": [overall_value]}
        )
        summary_table = pd.concat([summary_table, overall_row], ignore_index=True)

    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name="pph_rate_trend_data.csv",
        mime="text/csv",
        help="Download the exact data shown in the chart",
    )


def render_pph_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Postpartum Hemorrhage (PPH) Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
):
    """Render facility comparison chart - EXACT SAME AS render_svd_facility_comparison_chart"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # STANDARDIZE COLUMN NAMES - UPDATED TO MATCH YOUR DATA STRUCTURE
    if "orgUnit" not in df.columns:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["orgunit", "facility_uid", "facility_id", "uid", "ou"]:
                df = df.rename(columns={col: "orgUnit"})

    # Check for facility name column - LOOK FOR orgUnit_name FIRST
    if "orgUnit_name" in df.columns:
        df = df.rename(columns={"orgUnit_name": "Facility"})
    elif "Facility" not in df.columns:
        # Try to find other facility name columns
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["facility_name", "facility", "name", "display_name"]:
                df = df.rename(columns={col: "Facility"})
                break

    if "orgUnit" not in df.columns or "Facility" not in df.columns:
        st.error(
            f"❌ Facility identifier columns not found in the data. Cannot perform facility comparison.\n"
            f"Available columns: {list(df.columns)}"
        )
        return

    if df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Create a mapping from orgUnit to facility name
    facility_mapping = {}
    for _, row in df.iterrows():
        if pd.notna(row["orgUnit"]) and pd.notna(row["Facility"]):
            facility_mapping[str(row["orgUnit"])] = str(row["Facility"])

    # If we have facility_names parameter, update the mapping
    if facility_names and len(facility_names) == len(facility_uids):
        for uid, name in zip(facility_uids, facility_names):
            facility_mapping[str(uid)] = name

    # Prepare comparison data
    comparison_data = []

    # Get unique periods in order
    if "period_sort" in df.columns:
        unique_periods = df[["period_display", "period_sort"]].drop_duplicates()
        unique_periods = unique_periods.sort_values("period_sort")
        period_order = unique_periods["period_display"].tolist()
    else:
        # Try to sort by month-year
        try:
            period_order = sorted(
                df["period_display"].unique().tolist(),
                key=lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if "-" in x
                    else x
                ),
            )
        except:
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order]

    # Prepare data for each facility and period
    for facility_uid, facility_name in facility_mapping.items():
        facility_df = df[df["orgUnit"] == facility_uid].copy()

        if facility_df.empty:
            continue

        # Group by period for this facility
        for period_display, period_group in facility_df.groupby("period_display"):
            if not period_group.empty:
                # Get the first row for this facility/period combination
                row = period_group.iloc[0]
                formatted_period = format_period_month_year(period_display)

                # Skip if both numerator and denominator are 0
                numerator_val = row.get("numerator", 0)
                denominator_val = row.get("denominator", 1)

                if numerator_val == 0 and denominator_val == 0:
                    continue  # Skip this period for this facility

                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Facility": facility_name,
                        "value": row.get(value_col, 0) if value_col in row else 0,
                        "numerator": numerator_val,
                        "denominator": denominator_val,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: dt.datetime.strptime(x, "%b-%y")
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: dt.datetime.strptime(x, "%b-%y"),
        )
    except:
        pass

    # Filter out facilities that have no data (all periods with 0 numerator and denominator)
    facilities_with_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        # Check if facility has any non-zero data
        if not (
            facility_data["numerator"].sum() == 0
            and facility_data["denominator"].sum() == 0
        ):
            facilities_with_data.append(facility_name)

    # Filter comparison_df to only include facilities with data
    comparison_df = comparison_df[
        comparison_df["Facility"].isin(facilities_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all facilities have zero data).")
        return

    # Create the chart
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=f"{title} - Facility Comparison",
        height=500,
        category_orders={"period_display": period_order},
        hover_data=["numerator", "denominator"],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Facility: %{{fullData.name}}<br>"
            f"{title}: %{{y:.2f}}<br>"
            f"{numerator_name}: %{{customdata[0]}}<br>"
            f"{denominator_name}: %{{customdata[1]}}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=title,
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

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Facility Comparison Data")

    # Create pivot table for better display with Overall row
    pivot_data = []

    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            total_numerator = facility_data["numerator"].sum()
            total_denominator = facility_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )

            pivot_data.append(
                {
                    "Facility": facility_name,
                    numerator_name: f"{total_numerator:,.0f}",
                    denominator_name: f"{total_denominator:,.0f}",
                    "Overall Value": f"{overall_value:.2f}%",
                }
            )

    # Add Overall row for all facilities
    if pivot_data:
        all_numerators = comparison_df["numerator"].sum()
        all_denominators = comparison_df["denominator"].sum()
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )

        pivot_data.append(
            {
                "Facility": "Overall",
                numerator_name: f"{all_numerators:,.0f}",
                denominator_name: f"{all_denominators:,.0f}",
                "Overall Value": f"{grand_overall:.2f}%",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Keep download functionality
    csv_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            total_numerator = facility_data["numerator"].sum()
            total_denominator = facility_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )
            csv_data.append(
                {
                    "Facility": facility_name,
                    numerator_name: total_numerator,
                    denominator_name: total_denominator,
                    title: f"{overall_value:.2f}%",
                }
            )

    # Add overall row to CSV
    if csv_data:
        all_numerators = sum(item[numerator_name] for item in csv_data)
        all_denominators = sum(item[denominator_name] for item in csv_data)
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )
        csv_data.append(
            {
                "Facility": "Overall",
                numerator_name: all_numerators,
                denominator_name: all_denominators,
                title: f"{grand_overall:.2f}%",
            }
        )

        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Overall Comparison Data",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_facility_summary.csv",
            mime="text/csv",
            help="Download overall summary data for facility comparison",
        )


def render_pph_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Postpartum Hemorrhage (PPH) Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
):
    """Render region comparison chart - EXACT SAME AS render_svd_region_comparison_chart"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if "Region" not in df.columns:
        st.error(
            f"❌ Region column not found in the data. Cannot perform region comparison.\n"
            f"Available columns: {list(df.columns)}"
        )
        return

    if df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Prepare comparison data
    comparison_data = []

    # Get unique periods in order
    if "period_sort" in df.columns:
        unique_periods = df[["period_display", "period_sort"]].drop_duplicates()
        unique_periods = unique_periods.sort_values("period_sort")
        period_order = unique_periods["period_display"].tolist()
    else:
        try:
            period_order = sorted(
                df["period_display"].unique().tolist(),
                key=lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if "-" in x
                    else x
                ),
            )
        except:
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order]

    # Prepare data for each region and period
    for region_name in df["Region"].unique():
        region_df = df[df["Region"] == region_name].copy()

        if region_df.empty:
            continue

        # Group by period for this region
        for period_display, period_group in region_df.groupby("period_display"):
            if not period_group.empty:
                # Get aggregated values for this region/period
                avg_value = (
                    period_group[value_col].mean()
                    if value_col in period_group.columns
                    else 0
                )
                total_numerator = period_group["numerator"].sum()
                total_denominator = (
                    period_group["denominator"].sum()
                    if period_group["denominator"].sum() > 0
                    else 1
                )

                # Skip if both numerator and denominator are 0
                if total_numerator == 0 and total_denominator == 0:
                    continue

                formatted_period = format_period_month_year(period_display)
                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Region": region_name,
                        "value": avg_value,
                        "numerator": total_numerator,
                        "denominator": total_denominator,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available for regions.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: dt.datetime.strptime(x, "%b-%y")
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: dt.datetime.strptime(x, "%b-%y"),
        )
    except:
        pass

    # Filter out regions that have no data (all periods with 0 numerator and denominator)
    regions_with_data = []
    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        # Check if region has any non-zero data
        if not (
            region_data["numerator"].sum() == 0
            and region_data["denominator"].sum() == 0
        ):
            regions_with_data.append(region_name)

    # Filter comparison_df to only include regions with data
    comparison_df = comparison_df[
        comparison_df["Region"].isin(regions_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all regions have zero data).")
        return

    # Create the chart
    fig = px.line(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=f"{title} - Region Comparison",
        height=500,
        category_orders={"period_display": period_order},
        hover_data=["numerator", "denominator"],
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Region: %{{fullData.name}}<br>"
            f"{title}: %{{y:.2f}}<br>"
            f"{numerator_name}: %{{customdata[0]}}<br>"
            f"{denominator_name}: %{{customdata[1]}}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=title,
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

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Region Comparison Data")

    # Create pivot table for better display with Overall row
    pivot_data = []

    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        if not region_data.empty:
            total_numerator = region_data["numerator"].sum()
            total_denominator = region_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )

            pivot_data.append(
                {
                    "Region": region_name,
                    numerator_name: f"{total_numerator:,.0f}",
                    denominator_name: f"{total_denominator:,.0f}",
                    "Overall Value": f"{overall_value:.2f}%",
                }
            )

    # Add Overall row for all regions
    if pivot_data:
        all_numerators = comparison_df["numerator"].sum()
        all_denominators = comparison_df["denominator"].sum()
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )

        pivot_data.append(
            {
                "Region": "Overall",
                numerator_name: f"{all_numerators:,.0f}",
                denominator_name: f"{all_denominators:,.0f}",
                "Overall Value": f"{grand_overall:.2f}%",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Keep download functionality
    csv_data = []
    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        if not region_data.empty:
            total_numerator = region_data["numerator"].sum()
            total_denominator = region_data["denominator"].sum()
            overall_value = (
                (total_numerator / total_denominator * 100)
                if total_denominator > 0
                else 0
            )
            csv_data.append(
                {
                    "Region": region_name,
                    numerator_name: total_numerator,
                    denominator_name: total_denominator,
                    title: f"{overall_value:.2f}%",
                }
            )

    # Add overall row to CSV
    if csv_data:
        all_numerators = sum(item[numerator_name] for item in csv_data)
        all_denominators = sum(item[denominator_name] for item in csv_data)
        grand_overall = (
            (all_numerators / all_denominators * 100) if all_denominators > 0 else 0
        )
        csv_data.append(
            {
                "Region": "Overall",
                numerator_name: all_numerators,
                denominator_name: all_denominators,
                title: f"{grand_overall:.2f}%",
            }
        )

        csv_df = pd.DataFrame(csv_data)
        csv = csv_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Overall Comparison Data",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_')}_region_summary.csv",
            mime="text/csv",
            help="Download overall summary data for region comparison",
        )


# ---------------- Additional Helper Functions ----------------
def prepare_pph_data_for_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for PPH trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for PPH (same as delivery summary)
    date_column = get_relevant_date_column_for_kpi(kpi_name)

    # Check if the SPECIFIC date column exists
    if date_column not in filtered_df.columns:
        # Try to use event_date as fallback
        if "event_date" in filtered_df.columns:
            date_column = "event_date"
            st.warning(
                f"⚠️ KPI-specific date column not found for {kpi_name}, using 'event_date' instead"
            )
        else:
            st.warning(
                f"⚠️ Required date column '{date_column}' not found for {kpi_name}"
            )
            return pd.DataFrame(), date_column

    # Create result dataframe
    result_df = filtered_df.copy()

    # Convert to datetime
    result_df["event_date"] = pd.to_datetime(result_df[date_column], errors="coerce")

    # CRITICAL: Apply date range filtering
    if date_range_filters:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")

        if start_date and end_date:
            # Convert to datetime for comparison
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # Include end date

            # Filter by date range
            result_df = result_df[
                (result_df["event_date"] >= start_dt)
                & (result_df["event_date"] < end_dt)
            ].copy()

    # Filter out rows without valid dates
    result_df = result_df[result_df["event_date"].notna()].copy()

    if result_df.empty:
        st.info(f"⚠️ No data with valid dates in '{date_column}' for {kpi_name}")
        return pd.DataFrame(), date_column

    # Get period label
    period_label = st.session_state.get("period_label", "Monthly")
    if "filters" in st.session_state and "period_label" in st.session_state.filters:
        period_label = st.session_state.filters["period_label"]

    # Create period columns using time_filter utility
    from utils.time_filter import assign_period

    result_df = assign_period(result_df, "event_date", period_label)

    # Filter by facility if needed
    if facility_uids and "orgUnit" in result_df.columns:
        result_df = result_df[result_df["orgUnit"].isin(facility_uids)].copy()

    return result_df, date_column
