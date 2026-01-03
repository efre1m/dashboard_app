import logging
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
    compute_total_deliveries,  # Import for consistency
    get_relevant_date_column_for_kpi,
)

# ---------------- Caching Setup ----------------
if "admitted_mothers_cache" not in st.session_state:
    st.session_state.admitted_mothers_cache = {}


def get_admitted_mothers_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Admitted Mothers computations"""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_admitted_mothers_cache():
    """Clear the Admitted Mothers cache"""
    st.session_state.admitted_mothers_cache = {}


# ---------------- KPI Computation Functions ----------------
def compute_admitted_mothers_count(df, facility_uids=None):
    """
    Count Admitted Mothers occurrences - SAME METHOD AS compute_csection_count in kpi_utils.py
    Counts unique TEI IDs with enrollment dates
    """
    cache_key = get_admitted_mothers_cache_key(
        df, facility_uids, "admitted_mothers_count"
    )

    if cache_key in st.session_state.admitted_mothers_cache:
        return st.session_state.admitted_mothers_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # Use enrollment date directly for this KPI
        date_column = "enrollment_date"

        # Check if enrollment date column exists
        if date_column not in filtered_df.columns:
            result = 0
        else:
            # Filter to only include rows that have enrollment dates
            filtered_df[date_column] = pd.to_datetime(
                filtered_df[date_column], errors="coerce"
            )
            filtered_df = filtered_df[filtered_df[date_column].notna()].copy()

            # Count unique TEI IDs with enrollment dates
            if "tei_id" in filtered_df.columns:
                result = filtered_df["tei_id"].dropna().nunique()
            else:
                result = len(filtered_df)

    st.session_state.admitted_mothers_cache[cache_key] = result
    return result


def compute_admitted_mothers_rate(df, facility_uids=None):
    """
    For Admitted Mothers, rate is just the count (since it's not a percentage)
    Returns: (count, count, 1) to match the pattern
    """
    cache_key = get_admitted_mothers_cache_key(
        df, facility_uids, "admitted_mothers_rate"
    )

    if cache_key in st.session_state.admitted_mothers_cache:
        return st.session_state.admitted_mothers_cache[cache_key]

    if df is None or df.empty:
        result = (0, 0, 0.0)  # (count, denominator, value)
    else:
        # Get date column for Admitted Mothers
        date_column = get_relevant_date_column_for_kpi("Admitted Mothers")

        # Count admitted mothers
        admitted_mothers = compute_admitted_mothers_count(df, facility_uids)

        # For Admitted Mothers, we just return the count as the value
        result = (admitted_mothers, 1, float(admitted_mothers))

    st.session_state.admitted_mothers_cache[cache_key] = result
    return result


def compute_admitted_mothers_kpi(df, facility_uids=None):
    """
    Compute Admitted Mothers KPI data
    This is the function your dashboard is calling
    """
    count, denominator, value = compute_admitted_mothers_rate(df, facility_uids)

    return {
        "admitted_mothers_count": int(count),
        "admitted_mothers_value": float(value),
        "admitted_mothers_denominator": int(denominator),
    }


def get_numerator_denominator_for_admitted_mothers(
    df, facility_uids=None, date_range_filters=None
):
    """
    Get numerator and denominator for Admitted Mothers KPI
    WITH DATE RANGE FILTERING
    Returns: (numerator, denominator, value)
    For Admitted Mothers: numerator = count, denominator = 1, value = count
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Admitted Mothers
    date_column = get_relevant_date_column_for_kpi("Admitted Mothers")

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

    # Compute Admitted Mothers count on date-filtered data
    count, denominator, value = compute_admitted_mothers_rate(
        filtered_df, facility_uids
    )

    return (count, denominator, value)


# ---------------- Chart Functions WITH TABLES ----------------
# In kpi_admitted_mothers.py - FIXED VERSION WITH UNIQUE KEYS


def render_admitted_mothers_trend_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Admitted Mothers Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    value_name="Admitted Mothers",
    facility_uids=None,
):
    """Render trend chart for Admitted Mothers - FOLLOWING SAME PATTERN AS ASSISTED DELIVERY"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # =========== FIX: Sort periods chronologically ===========
    # Check if we have a period_sort column
    if "period_sort" in df.columns:
        # Sort by period_sort
        df = df.sort_values("period_sort")
        logging.info(f"📅 Sorted by period_sort: {df[period_col].tolist()}")
    else:
        # Try to sort by month-year format
        try:
            # Convert period_display to datetime for sorting
            df["sort_key"] = df[period_col].apply(
                lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if isinstance(x, str) and "-" in x
                    else x
                )
            )
            df = df.sort_values("sort_key")
            df = df.drop(columns=["sort_key"])
            logging.info(f"📅 Sorted by datetime: {df[period_col].tolist()}")
        except Exception as e:
            # Fallback: sort alphabetically
            df = df.sort_values(period_col)
            logging.warning(
                f"⚠️ Could not sort chronologically: {e}. Sorted alphabetically: {df[period_col].tolist()}"
            )
    # =========== END FIX ===========

    # For Admitted Mothers, we use bar chart by default (counts)
    chart_type = "Bar"  # Force bar chart for counts

    try:
        # Use bar chart for counts
        fig = px.bar(
            df,
            x=x_axis_col,
            y=value_col,
            title=title,
            height=400,
            hover_data=[value_col],
            text=value_col,  # Add value labels on bars
            # =========== FIX: Ensure category order ===========
            category_orders={x_axis_col: df[x_axis_col].tolist()},
        )

        # Format text on bars
        fig.update_traces(
            texttemplate="%{text:.0f}",
            textposition="outside",
            hovertemplate=(
                f"<b>%{{x}}</b><br>" f"{value_name}: %{{y:,.0f}}<br>" f"<extra></extra>"
            ),
        )
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        fig = px.bar(
            df,
            x=x_axis_col,
            y=value_col,
            title=title,
            height=400,
            # =========== FIX: Ensure category order ===========
            category_orders={x_axis_col: df[x_axis_col].tolist()},
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
        xaxis_title="Period (Month-Year)",
        yaxis_title=value_name,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            # =========== FIX: Force category order ===========
            categoryorder="array",
            categoryarray=df[x_axis_col].tolist(),
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    # Format y-axis as integers with commas (not percentages)
    fig.update_layout(yaxis_tickformat=",")

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Data Table")

    # Create a clean display dataframe
    display_df = df.copy()

    # Select columns to show in table
    table_columns = [x_axis_col, value_col]

    # Format the dataframe for display
    display_df = display_df[table_columns].copy()

    # Format numbers with commas (not percentages)
    display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:,.0f}")

    # Add Overall/Total row
    total_value = df[value_col].sum() if not df.empty else 0

    # Create overall row
    overall_row = {
        x_axis_col: "Overall",
        value_col: f"{total_value:,.0f}",
    }

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
            st.metric("📈 Latest Count", f"{df[value_col].iloc[-1]:,.0f}")

        with col2:
            avg_value = df[value_col].mean()
            st.metric("📊 Average per Period", f"{avg_value:,.1f}")

        with col3:
            # Calculate trend
            last_value = df[value_col].iloc[-1]
            prev_value = df[value_col].iloc[-2] if len(df) > 1 else 0
            trend_change = last_value - prev_value
            trend_symbol = (
                "▲" if trend_change > 0 else ("▼" if trend_change < 0 else "–")
            )
            st.metric("📈 Trend from Previous", f"{trend_change:,.0f} {trend_symbol}")

    # Download button - FIXED WITH UNIQUE KEY
    summary_df = df.copy().reset_index(drop=True)

    summary_df = summary_df[[x_axis_col, value_col]].copy()

    # Format period column
    if x_axis_col in summary_df.columns:
        summary_df[x_axis_col] = summary_df[x_axis_col].apply(format_period_month_year)

    summary_df = summary_df.rename(columns={value_col: "Admitted Mothers Count"})
    summary_table = summary_df.copy()

    overall_row = pd.DataFrame(
        {x_axis_col: ["Overall"], "Admitted Mothers Count": [total_value]}
    )
    summary_table = pd.concat([summary_table, overall_row], ignore_index=True)

    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    csv = summary_table.to_csv(index=False)

    # FIX: Generate unique key for maternal download button
    import time

    unique_key = f"maternal_admitted_trend_{int(time.time())}_{hash(str(df))}"

    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name="admitted_mothers_trend_data.csv",
        mime="text/csv",
        help="Download the exact data shown in the chart",
        key=unique_key,  # UNIQUE KEY for maternal
    )


def render_admitted_mothers_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Admitted Mothers - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    value_name="Admitted Mothers",
):
    """Render facility comparison chart for Admitted Mothers - UPDATED FIXED VERSION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Ensure we have required columns
    if "orgUnit" not in df.columns:
        st.error(f"❌ 'orgUnit' column not found in comparison data")
        st.write("Available columns:", list(df.columns))
        return

    # Create facility mapping
    facility_mapping = {}
    if facility_names and facility_uids and len(facility_names) == len(facility_uids):
        for uid, name in zip(facility_uids, facility_names):
            facility_mapping[str(uid)] = name
    elif "Facility" in df.columns:
        # Extract mapping from data
        for _, row in df.iterrows():
            if "orgUnit" in row and pd.notna(row["orgUnit"]):
                facility_mapping[str(row["orgUnit"])] = row.get(
                    "Facility", str(row["orgUnit"])
                )
    else:
        # Create simple mapping from UIDs
        unique_orgunits = df["orgUnit"].dropna().unique()
        for uid in unique_orgunits:
            facility_mapping[str(uid)] = f"Facility {str(uid)[:8]}"

    if not facility_mapping:
        st.info("⚠️ No facility mapping available for comparison.")
        return

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
                    if isinstance(x, str) and "-" in x
                    else x
                ),
            )
        except:
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order if p is not None]

    # Prepare data for each facility and period
    for facility_uid, facility_name in facility_mapping.items():
        facility_df = df[df["orgUnit"] == facility_uid].copy()

        if facility_df.empty:
            # Skip facilities with no data
            continue

        # Group by period for this facility
        for period_display, period_group in facility_df.groupby("period_display"):
            if not period_group.empty:
                # Sum values for this facility/period
                total_value = (
                    period_group[value_col].sum()
                    if value_col in period_group.columns
                    else 0
                )

                # For Admitted Mothers, we want to show the count even if it's 0
                # But skip if there's no data at all for this period
                if len(period_group) == 0:
                    continue

                formatted_period = format_period_month_year(period_display)

                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Facility": facility_name,
                        "value": total_value,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available (all facilities have zero data).")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: (
                dt.datetime.strptime(x, "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            )
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: (
                dt.datetime.strptime(x, "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            ),
        )
    except:
        # Sort alphabetically as fallback
        comparison_df = comparison_df.sort_values(["Facility", "period_display"])
        period_order = sorted(comparison_df["period_display"].unique().tolist())

    # Filter out facilities that have no data
    facilities_with_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            facilities_with_data.append(facility_name)

    # Filter comparison_df to only include facilities with data
    comparison_df = comparison_df[
        comparison_df["Facility"].isin(facilities_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all facilities have zero data).")
        return

    # Create the chart - Use bar chart for counts
    fig = px.bar(
        comparison_df,
        x="period_display",
        y="value",
        color="Facility",
        title=f"{title} - Facility Comparison",
        height=500,
        category_orders={"period_display": period_order},
        barmode="group",
        text="value",  # Show values on bars
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Facility: %{{fullData.name}}<br>"
            f"{value_name}: %{{y:,.0f}}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=value_name,
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

    # Format y-axis as integers with commas
    fig.update_layout(yaxis_tickformat=",")

    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Facility Comparison Data")

    # Create pivot table for better display with Overall row
    pivot_data = []

    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not facility_data.empty:
            total_count = facility_data["value"].sum()

            pivot_data.append(
                {
                    "Facility": facility_name,
                    value_name: f"{total_count:,.0f}",
                }
            )

    # Add Overall row for all facilities
    if pivot_data:
        all_counts = comparison_df["value"].sum()

        pivot_data.append(
            {
                "Facility": "Overall",
                value_name: f"{all_counts:,.0f}",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Download button - FIXED WITH UNIQUE KEY
    csv = comparison_df.to_csv(index=False)

    # FIX: Generate unique key for maternal download button
    import time

    unique_key = f"maternal_admitted_facility_{int(time.time())}_{hash(str(df))}"

    st.download_button(
        label="📥 Download Facility Comparison Data",
        data=csv,
        file_name="admitted_mothers_facility_comparison.csv",
        mime="text/csv",
        help="Download the facility comparison data",
        key=unique_key,  # UNIQUE KEY for maternal
    )


def render_admitted_mothers_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Admitted Mothers - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    value_name="Admitted Mothers",
):
    """Render region comparison chart for Admitted Mothers - UPDATED FIXED VERSION"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Check if we have Region column
    if "Region" not in df.columns:
        st.error(f"❌ 'Region' column not found in comparison data")
        st.write("Available columns:", list(df.columns))

        # Try to infer region from facilities_by_region mapping
        if facilities_by_region and "orgUnit" in df.columns:
            st.write("⚠️ Attempting to infer region from facility UIDs...")
            # Create region mapping from facility UIDs
            reverse_mapping = {}
            for region_name, facilities in facilities_by_region.items():
                for facility_name, facility_uid in facilities:
                    reverse_mapping[facility_uid] = region_name

            # Add Region column based on mapping
            df["Region"] = df["orgUnit"].map(
                lambda x: reverse_mapping.get(x, "Unknown")
            )

            if df["Region"].isna().all():
                st.error("❌ Could not map facility UIDs to regions")
                return
        else:
            return

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
                    if isinstance(x, str) and "-" in x
                    else x
                ),
            )
        except Exception as e:
            st.write(f"⚠️ Could not sort periods: {e}")
            period_order = sorted(df["period_display"].unique().tolist())

    # Format periods to proper month-year format
    period_order = [format_period_month_year(p) for p in period_order if p is not None]

    # Prepare data for each region and period
    for region_name in df["Region"].dropna().unique():
        region_df = df[df["Region"] == region_name].copy()

        if region_df.empty:
            st.write(f"⚠️ No data for region: {region_name}")
            continue

        # Group by period for this region
        for period_display, period_group in region_df.groupby("period_display"):
            if not period_group.empty:
                # Sum values for this region/period
                total_value = (
                    period_group[value_col].sum()
                    if value_col in period_group.columns
                    else 0
                )

                # For Admitted Mothers, we want to show the count even if it's 0
                # But skip if there's no data at all for this period
                if len(period_group) == 0:
                    continue

                formatted_period = format_period_month_year(period_display)

                comparison_data.append(
                    {
                        "period_display": formatted_period,
                        "Region": region_name,
                        "value": total_value,
                    }
                )

    if not comparison_data:
        st.info("⚠️ No comparison data available for regions.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Sort periods properly for display
    try:
        comparison_df["period_sort"] = comparison_df["period_display"].apply(
            lambda x: (
                dt.datetime.strptime(x, "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            )
        )
        comparison_df = comparison_df.sort_values("period_sort")
        period_order = sorted(
            comparison_df["period_display"].unique().tolist(),
            key=lambda x: (
                dt.datetime.strptime(x, "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            ),
        )
    except Exception as e:
        st.write(f"⚠️ Could not sort periods chronologically: {e}")
        # Sort alphabetically as fallback
        comparison_df = comparison_df.sort_values(["Region", "period_display"])
        period_order = sorted(comparison_df["period_display"].unique().tolist())

    # Filter out regions that have no data (all periods with 0)
    regions_with_data = []
    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        # For Admitted Mothers, we want to show even if value is 0, as long as there's data
        if not region_data.empty:
            regions_with_data.append(region_name)

    # Filter comparison_df to only include regions with data
    comparison_df = comparison_df[
        comparison_df["Region"].isin(regions_with_data)
    ].copy()

    if comparison_df.empty:
        st.info("⚠️ No valid comparison data available (all regions have zero data).")
        return

    # Create the chart - Use bar chart for counts (since it's not a percentage)
    fig = px.bar(
        comparison_df,
        x="period_display",
        y="value",
        color="Region",
        title=f"{title} - Region Comparison",
        height=500,
        category_orders={"period_display": period_order},
        barmode="group",
        text="value",  # Show values on bars
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=(
            f"<b>%{{x}}</b><br>"
            f"Region: %{{fullData.name}}<br>"
            f"{value_name}: %{{y:,.0f}}<extra></extra>"
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=value_name,
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

    # Format y-axis as integers with commas (not percentages)
    fig.update_layout(yaxis_tickformat=",")

    st.plotly_chart(fig, use_container_width=True)

    # =========== DISPLAY TABLE BELOW GRAPH ===========
    st.markdown("---")
    st.subheader("📋 Region Comparison Data")

    # Create pivot table for better display with Overall row
    pivot_data = []

    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        if not region_data.empty:
            total_count = region_data["value"].sum()

            pivot_data.append(
                {
                    "Region": region_name,
                    value_name: f"{total_count:,.0f}",
                }
            )

    # Add Overall row for all regions
    if pivot_data:
        all_counts = comparison_df["value"].sum()

        pivot_data.append(
            {
                "Region": "Overall",
                value_name: f"{all_counts:,.0f}",
            }
        )

        pivot_df = pd.DataFrame(pivot_data)
        st.dataframe(pivot_df, use_container_width=True)

    # Download button - FIXED WITH UNIQUE KEY
    csv = comparison_df.to_csv(index=False)

    # FIX: Generate unique key for maternal download button
    import time

    unique_key = f"maternal_admitted_region_{int(time.time())}_{hash(str(df))}"

    st.download_button(
        label="📥 Download Region Comparison Data",
        data=csv,
        file_name="admitted_mothers_region_comparison.csv",
        mime="text/csv",
        help="Download the region comparison data",
        key=unique_key,  # UNIQUE KEY for maternal
    )


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_admitted_mothers_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Admitted Mothers trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
    SAME AS ASSISTED DELIVERY FUNCTION
    """
    if df.empty:
        return pd.DataFrame(), None

    filtered_df = df.copy()

    # Filter by facility UIDs if provided
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for Admitted Mothers
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


# ---------------- Main Entry Point ----------------
def get_admitted_mothers_kpi_data(df, facility_uids=None):
    """
    Main function to get Admitted Mothers KPI data for dashboard
    """
    # Get date range filters from session state if available
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # Compute KPI
    count, denominator, value = get_numerator_denominator_for_admitted_mothers(
        df, facility_uids, date_range_filters
    )

    return {
        "admitted_mothers_count": int(count),
        "admitted_mothers_value": float(value),
    }
