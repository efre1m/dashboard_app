import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import datetime as dt
import hashlib

# Import only utility functions from kpi_utils
from utils.kpi_utils import (
    auto_text_color,
    format_period_month_year,
    compute_total_deliveries,  # Use the same total deliveries function
    get_relevant_date_column_for_kpi,
    DELIVERY_MODE_COL,  # Same column as C-section
    DELIVERY_DATE_COL,  # Same date column as C-section
)

# ---------------- Caching Setup ----------------
if "svd_cache" not in st.session_state:
    st.session_state.svd_cache = {}


def get_svd_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for SVD computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_svd_cache():
    """Clear the SVD cache"""
    st.session_state.svd_cache = {}


# SVD KPI Configuration
SVD_CODE = "1"  # Normal Vaginal Delivery code


def compute_svd_count(df, facility_uids=None):
    """
    Count SVD occurrences - EXACT SAME METHOD AS compute_csection_count in kpi_utils.py
    but checking for code 1 instead of 2
    """
    cache_key = get_svd_cache_key(df, facility_uids, "svd_count")

    if cache_key in st.session_state.svd_cache:
        return st.session_state.svd_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        if DELIVERY_MODE_COL not in filtered_df.columns:
            result = 0
        else:
            df_copy = filtered_df.copy()

            # Convert to string first, then extract numeric part - SAME AS C-SECTION
            df_copy["delivery_mode_clean"] = df_copy[DELIVERY_MODE_COL].astype(str)
            df_copy["delivery_mode_numeric"] = pd.to_numeric(
                df_copy["delivery_mode_clean"].str.split(".").str[0], errors="coerce"
            )

            # Count SVD (value = 1) - ONLY DIFFERENCE FROM C-SECTION
            svd_mask = df_copy["delivery_mode_numeric"] == 1
            result = int(svd_mask.sum())

    st.session_state.svd_cache[cache_key] = result
    return result


def compute_svd_rate(df, facility_uids=None):
    """
    Compute SVD Rate - EXACT SAME PATTERN AS compute_csection_rate in kpi_utils.py
    Returns: (rate, svd_deliveries, total_deliveries)
    """
    cache_key = get_svd_cache_key(df, facility_uids, "svd_rate")

    if cache_key in st.session_state.svd_cache:
        return st.session_state.svd_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        # Get date column for SVD (same as C-section)
        date_column = get_relevant_date_column_for_kpi(
            "Normal Vaginal Delivery (SVD) Rate (%)"
        )

        # Count SVD deliveries
        svd_deliveries = compute_svd_count(df, facility_uids)

        # Get total deliveries - USING SAME LOGIC AS C-SECTION
        total_deliveries = compute_total_deliveries(df, facility_uids, date_column)

        # Calculate rate
        rate = (
            (svd_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0
        )
        result = (rate, svd_deliveries, total_deliveries)

    st.session_state.svd_cache[cache_key] = result
    return result


def compute_svd_kpi(df, facility_uids=None):
    """
    Compute SVD KPI data - SAME STRUCTURE AS compute_csection_rate
    """
    rate, svd_deliveries, total_deliveries = compute_svd_rate(df, facility_uids)

    return {
        "svd_rate": float(rate),
        "svd_deliveries": int(svd_deliveries),
        "total_deliveries": int(total_deliveries),
    }


def get_numerator_denominator_for_svd(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for SVD KPI - SAME PATTERN AS C-SECTION
    WITH DATE RANGE FILTERING
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for SVD (same as C-section)
    date_column = get_relevant_date_column_for_kpi(
        "Normal Vaginal Delivery (SVD) Rate (%)"
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

    # Compute SVD rate on date-filtered data
    rate, svd_deliveries, total_deliveries = compute_svd_rate(
        filtered_df, facility_uids
    )

    return (svd_deliveries, total_deliveries, rate)


# ---------------- Chart Functions WITH TABLES ----------------
def render_svd_trend_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Normal Vaginal Delivery Rate Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="SVD Deliveries",
    denominator_name="Total Deliveries",
    facility_uids=None,
):
    """Render trend chart for SVD - FULL IMPLEMENTATION"""
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
        key=f"chart_type_svd_{str(facility_uids)}",
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
                value_col: "SVD Rate (%)",
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
                "SVD Rate (%)": [overall_value],
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

        summary_df = summary_df.rename(columns={value_col: "SVD Rate (%)"})
        summary_table = summary_df.copy()

        overall_value = (
            summary_table["SVD Rate (%)"].mean() if not summary_table.empty else 0
        )
        overall_row = pd.DataFrame(
            {x_axis_col: ["Overall"], "SVD Rate (%)": [overall_value]}
        )
        summary_table = pd.concat([summary_table, overall_row], ignore_index=True)

    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name="svd_rate_trend_data.csv",
        mime="text/csv",
        help="Download the exact data shown in the chart",
    )


def render_svd_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Normal Vaginal Delivery Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="SVD Deliveries",
    denominator_name="Total Deliveries",
):
    """Render facility comparison chart - FULL IMPLEMENTATION"""
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


def render_svd_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Normal Vaginal Delivery Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="SVD Deliveries",
    denominator_name="Total Deliveries",
):
    """Render region comparison chart - FULL IMPLEMENTATION"""
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
def prepare_svd_data_for_trend_chart(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for SVD trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for SVD (same as C-section)
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
