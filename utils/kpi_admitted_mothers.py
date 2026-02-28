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
    get_attractive_hover_template,  # FIXED: Added missing import
    get_comparison_hover_template,
    get_current_period_label,
    format_period_for_download,
    _build_next_month_forecast_payload,
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
    **kwargs,
):
    """Render trend chart for Admitted Mothers - Specialized for Counts (Bar Chart)"""
    # Use numerator_name if provided in kwargs
    final_name = kwargs.get("numerator_name", value_name)

    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    x_axis_col = period_col

    df = df.reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Sort periods chronologically
    if "period_sort" in df.columns:
        df = df.sort_values("period_sort")
    else:
        try:
            df["sort_key"] = df[period_col].apply(
                lambda x: (
                    dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                    if isinstance(x, str) and "-" in x
                    else x
                )
            )
            df = df.sort_values("sort_key")
            df = df.drop(columns=["sort_key"])
        except Exception:
            df = df.sort_values(period_col)

    forecast_payload = None
    if str(get_current_period_label()).lower() == "monthly":
        forecast_payload = _build_next_month_forecast_payload(
            df,
            x_axis_col,
            value_col,
            forecast_min_points=4,
        )
        if forecast_payload:
            forecast_payload["forecast_y"] = max(
                0.0, float(forecast_payload.get("forecast_y", 0.0))
            )

    plot_df = df[[x_axis_col, value_col]].copy()
    plot_df["Series"] = "Actual"
    category_order = df[x_axis_col].tolist()
    if forecast_payload:
        next_period = forecast_payload["next_x"]
        forecast_value = float(np.round(forecast_payload["forecast_y"], 0))
        plot_df = pd.concat(
            [
                plot_df,
                pd.DataFrame(
                    [
                        {
                            x_axis_col: next_period,
                            value_col: forecast_value,
                            "Series": "Forecast Next Month",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        if next_period not in category_order:
            category_order.append(next_period)

    # Create chart
    try:
        fig = px.bar(
            plot_df,
            x=x_axis_col,
            y=value_col,
            color="Series",
            color_discrete_map={
                "Actual": "#1f77b4",
                "Forecast Next Month": "#f39c12",
            },
            title=title,
            height=400,
            text=value_col,
            category_orders={x_axis_col: category_order},
        )

        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,.0f}<extra></extra>",
        )
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        fig = px.bar(
            plot_df,
            x=x_axis_col,
            y=value_col,
            title=title,
            height=400,
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=final_name,
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            categoryorder="array",
            categoryarray=category_order,
        ),
        yaxis=dict(
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
            tickformat=",",
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key=f"admitted_mothers_chart_{kwargs.get('key_suffix', '')}")
    if forecast_payload:
        delta = forecast_payload["forecast_y"] - forecast_payload["last_y"]
        direction = "Increase" if delta > 0 else ("Decrease" if delta < 0 else "No Change")
        st.caption(
            f"Forecast (next month): {forecast_payload['forecast_y']:,.0f} "
            f"({direction} vs latest actual)."
        )

    # Table below graph
    st.markdown("---")
    st.subheader("📋 Data Table")
    display_df = df[[x_axis_col, value_col]].copy()
    display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:,.0f}")
    total_value = df[value_col].sum() if not df.empty else 0
    overall_df = pd.DataFrame({x_axis_col: ["Overall"], value_col: [f"{total_value:,.0f}"]})
    display_df = pd.concat([display_df, overall_df], ignore_index=True)
    st.dataframe(display_df, use_container_width=True)

    # Add summary statistics
    if len(df) > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📈 Latest Count", f"{df[value_col].iloc[-1]:,.0f}")
        with col2:
            st.metric("📊 Average per Period", f"{df[value_col].mean():,.1f}")
        with col3:
            change = df[value_col].iloc[-1] - df[value_col].iloc[-2]
            symbol = "▲" if change > 0 else ("▼" if change < 0 else "–")
            st.metric("📈 Trend from Previous", f"{change:,.0f} {symbol}")

    # Download button
    import time
    unique_key = f"maternal_admitted_trend_{int(time.time())}_{hash(str(df))}"
    period_label = get_current_period_label()
    summary_df = df[[x_axis_col, value_col]].copy()
    summary_df[x_axis_col] = summary_df[x_axis_col].apply(
        lambda p: format_period_for_download(p, period_label)
    )
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name="admitted_mothers_trend_data.csv",
        mime="text/csv",
        key=f"{unique_key}_{kwargs.get('key_suffix', '')}",
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
    suppress_plot=False,
    **kwargs,
):
    """Render facility comparison chart for Admitted Mothers - Specialized for Counts (Bar Chart)"""
    # Use numerator_name if provided in kwargs
    final_name = kwargs.get("numerator_name", value_name)

    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty:
        st.info("⚠️ No data available for facility comparison.")
        return

    # Sort periods properly
    try:
        period_order = sorted(
            df[period_col].unique().tolist(),
            key=lambda x: (
                dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            ),
        )
    except:
        period_order = sorted(df[period_col].unique().tolist())

    # Create the chart - Use bar chart for comparison as requested
    fig = px.bar(
        df,
        x=period_col,
        y=value_col,
        color="Facility",
        title=title,
        height=500,
        category_orders={period_col: period_order},
        barmode="group",
        text=value_col,
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=get_comparison_hover_template(
            "Facility", final_name, "", "", is_count=True
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        xaxis_title="Period (Month-Year)",
        yaxis_title=final_name,
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if not suppress_plot:
        st.plotly_chart(fig, use_container_width=True, key=f"admitted_mothers_chart_{kwargs.get('key_suffix', '')}")
    else:
        st.info(f"💡 Showing comparison table only for **{title}**.")

    # Table
    st.markdown("---")
    st.subheader("📋 Facility Comparison Data")
    pivot_data = []
    for f_name in df["Facility"].unique():
        f_count = df[df["Facility"] == f_name][value_col].sum()
        pivot_data.append({"Facility": f_name, f"Total {final_name}": f"{f_count:,.0f}"})
    
    pivot_data.append({"Facility": "Overall", f"Total {final_name}": f"{df[value_col].sum():,.0f}"})
    st.dataframe(pd.DataFrame(pivot_data), use_container_width=True)

    import time
    key_suffix = kwargs.get("key_suffix", "")
    unique_key = f"maternal_admitted_facility_{int(time.time())}_{key_suffix}_{hash(str(df))}"
    period_label = get_current_period_label()
    csv_df = df.copy()
    csv_df["Time Period"] = csv_df[period_col].apply(
        lambda p: format_period_for_download(p, period_label)
    )
    csv_df = csv_df[["Facility", "Time Period", value_col]].rename(
        columns={value_col: f"{final_name} Count"}
    )
    st.download_button(
        label="📥 Download Facility Comparison Data",
        data=csv_df.to_csv(index=False),
        file_name="admitted_mothers_facility_comparison.csv",
        mime="text/csv",
        key=unique_key,
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
    suppress_plot=False,
    **kwargs,
):
    """Render region comparison chart for Admitted Mothers - Specialized for Counts (Bar Chart)"""
    # Use numerator_name if provided in kwargs
    final_name = kwargs.get("numerator_name", value_name)

    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Ensure clean index
    df = df.reset_index(drop=True)

    if df is None or df.empty:
        st.info("⚠️ No data available for region comparison.")
        return

    # Sort periods properly
    try:
        period_order = sorted(
            df[period_col].unique().tolist(),
            key=lambda x: (
                dt.datetime.strptime(format_period_month_year(x), "%b-%y")
                if isinstance(x, str) and "-" in x
                else x
            ),
        )
    except:
        period_order = sorted(df[period_col].unique().tolist())

    # Create the chart - Use bar chart for counts
    fig = px.bar(
        df,
        x=period_col,
        y=value_col,
        color="Region",
        title=title,
        height=500,
        category_orders={period_col: period_order},
        barmode="group",
        text=value_col,
    )

    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside",
        hovertemplate=get_comparison_hover_template(
            "Region", final_name, "", "", is_count=True
        ),
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        yaxis_title=final_name,
        yaxis_tickformat=",",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if not suppress_plot:
        st.plotly_chart(fig, use_container_width=True, key=f"admitted_mothers_chart_{kwargs.get('key_suffix', '')}")
    else:
        st.info(f"💡 Showing comparison table only for **{title}**.")

    # Table
    st.markdown("---")
    st.subheader("📋 Region Comparison Data")
    pivot_data = []
    for r_name in df["Region"].unique():
        r_count = df[df["Region"] == r_name][value_col].sum()
        pivot_data.append({"Region": r_name, f"Total {final_name}": f"{r_count:,.0f}"})
    
    pivot_data.append({"Region": "Overall", f"Total {final_name}": f"{df[value_col].sum():,.0f}"})
    st.dataframe(pd.DataFrame(pivot_data), use_container_width=True)

    import time
    key_suffix = kwargs.get("key_suffix", "")
    unique_key = f"maternal_admitted_region_{int(time.time())}_{key_suffix}_{hash(str(df))}"
    period_label = get_current_period_label()
    csv_df = df.copy()
    csv_df["Time Period"] = csv_df[period_col].apply(
        lambda p: format_period_for_download(p, period_label)
    )
    csv_df = csv_df[["Region", "Time Period", value_col]].rename(
        columns={value_col: f"{final_name} Count"}
    )
    st.download_button(
        label="📥 Download Region Comparison Data",
        data=csv_df.to_csv(index=False),
        file_name="admitted_mothers_region_comparison.csv",
        mime="text/csv",
        key=unique_key,
    )


# ---------------- Additional Helper Functions ----------------
def prepare_data_for_admitted_mothers_trend(
    df, kpi_name, facility_uids=None, date_range_filters=None
):
    """
    Prepare patient-level data for Admitted Mothers trend chart
    Returns: DataFrame filtered by KPI-specific dates AND date range AND the date column used
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
    result_df["enrollment_date"] = pd.to_datetime(result_df[date_column], errors="coerce")

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
                (result_df["enrollment_date"] >= start_dt)
                & (result_df["enrollment_date"] < end_dt)
            ].copy()

    # Filter out rows without valid dates
    result_df = result_df[result_df["enrollment_date"].notna()].copy()

    if result_df.empty:
        st.info(f"⚠️ No data with valid dates in '{date_column}' for {kpi_name}")
        return pd.DataFrame(), date_column

    # Get period label
    period_label = st.session_state.get("period_label", "Monthly")
    if "filters" in st.session_state and "period_label" in st.session_state.filters:
        period_label = st.session_state.filters["period_label"]

    # Create period columns using time_filter utility
    from utils.time_filter import assign_period

    result_df = assign_period(result_df, "enrollment_date", period_label)

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
