import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from matplotlib.colors import LinearSegmentedColormap

# Import the same functions used by other KPIs
from utils.kpi_utils import (
    compute_total_deliveries,
    auto_text_color,
    get_cache_key,
    format_period_month_year,
)

# ---------------- PPH KPI Constants ----------------
PPH_CONDITION_COL = "obstetric_condition_at_admission_delivery_summary"
PPH_CODE = "3"  # Postpartum Hemorrhage (PPH) option code
DELIVERY_DATE_COL = "event_date_delivery_summary"  # Program stage specific date

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
def compute_pph_count(df, facility_uids=None):
    """Count PPH occurrences from patient-level data"""
    cache_key = get_cache_key(df, facility_uids, "pph_count")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = 0
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        actual_events_df = filtered_df.copy()

        if PPH_CONDITION_COL not in actual_events_df.columns:
            result = 0
        else:
            # Handle different data types
            pph_series = actual_events_df[PPH_CONDITION_COL].dropna()

            if pph_series.dtype in [np.float64, np.int64]:
                pph_codes = pph_series.astype(int).astype(str)
            else:
                pph_codes = pph_series.astype(str)

            # Function to check if PPH code (3) is in the value (handles multi-code values)
            def contains_pph(value):
                if pd.isna(value):
                    return False
                value_str = str(value)
                # Split by common separators
                codes = [
                    c.strip()
                    for c in value_str.replace(";", ",").split(",")
                    if c.strip()
                ]
                return "3" in codes

            # Apply the check
            pph_mask = pph_codes.apply(contains_pph)
            result = int(pph_mask.sum())

    st.session_state.kpi_cache[cache_key] = result
    return result


def compute_pph_kpi(df, facility_uids=None):
    """
    Compute PPH KPI for patient-level data
    Returns: pph_rate, pph_count, total_deliveries
    """
    cache_key = get_cache_key(df, facility_uids, "pph_rate")

    if cache_key in st.session_state.kpi_cache:
        return st.session_state.kpi_cache[cache_key]

    if df is None or df.empty:
        result = (0.0, 0, 0)
    else:
        filtered_df = df.copy()
        if facility_uids and "orgUnit" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

        # IMPORTANT: Filter to only include rows with delivery summary date
        if DELIVERY_DATE_COL in filtered_df.columns:
            filtered_df[DELIVERY_DATE_COL] = pd.to_datetime(
                filtered_df[DELIVERY_DATE_COL], errors="coerce"
            )
            filtered_df = filtered_df[filtered_df[DELIVERY_DATE_COL].notna()].copy()

        if filtered_df.empty:
            result = (0.0, 0, 0)
        else:
            # Get PPH cases
            pph_count = compute_pph_count(filtered_df, facility_uids)

            # Get total deliveries - using date-filtered data
            total_deliveries = compute_total_deliveries(
                filtered_df, facility_uids, DELIVERY_DATE_COL
            )

            # Calculate rate
            rate = (pph_count / total_deliveries * 100) if total_deliveries > 0 else 0.0
            result = (float(rate), int(pph_count), int(total_deliveries))

    st.session_state.kpi_cache[cache_key] = result
    return result


def get_numerator_denominator_for_pph(df, facility_uids=None, date_range_filters=None):
    """
    Get numerator and denominator for PPH Rate with date filtering
    Returns: (numerator, denominator, value)
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Filter by specific date column if provided
    if DELIVERY_DATE_COL in filtered_df.columns:
        # Convert to datetime and filter out rows without this date
        filtered_df[DELIVERY_DATE_COL] = pd.to_datetime(
            filtered_df[DELIVERY_DATE_COL], errors="coerce"
        )
        filtered_df = filtered_df[filtered_df[DELIVERY_DATE_COL].notna()].copy()

        # Apply date range filtering if provided
        if date_range_filters:
            start_date = date_range_filters.get("start_date")
            end_date = date_range_filters.get("end_date")

            if start_date and end_date:
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)

                filtered_df = filtered_df[
                    (filtered_df[DELIVERY_DATE_COL] >= start_dt)
                    & (filtered_df[DELIVERY_DATE_COL] < end_dt)
                ].copy()

    if filtered_df.empty:
        return (0, 0, 0.0)

    # Compute KPI on date-filtered data
    rate, pph_count, total_deliveries = compute_pph_kpi(filtered_df, facility_uids)

    return (pph_count, total_deliveries, rate)


def compute_obstetric_condition_distribution(df, facility_uids=None):
    """
    Compute distribution of all obstetric complications from patient-level data
    """
    if df is None or df.empty:
        return pd.DataFrame()

    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    if PPH_CONDITION_COL not in filtered_df.columns:
        return pd.DataFrame()

    # Get obstetric condition data
    condition_series = filtered_df[PPH_CONDITION_COL].dropna()

    if condition_series.empty:
        return pd.DataFrame()

    # Condition mapping
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

    # Expand multi-code values
    expanded_data = []
    valid_codes = set(condition_mapping.keys())

    for value in condition_series:
        if pd.isna(value):
            continue

        value_str = str(value).strip()

        # Skip empty values
        if not value_str or value_str.lower() in ["nan", "null", ""]:
            continue

        # Split by comma or semicolon
        codes = [
            c.strip()
            for c in value_str.replace(";", ",").split(",")
            if c.strip() and c.strip() in valid_codes
        ]

        for code in codes:
            expanded_data.append(code)

    if not expanded_data:
        return pd.DataFrame()

    # Count occurrences
    condition_counts = pd.Series(expanded_data).value_counts().reset_index()
    condition_counts.columns = ["condition_code", "count"]

    # Map codes to readable names
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
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names=None,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
    facility_uids=None,
):
    """Render trend chart for PPH rate with same styling as other KPIs"""
    from utils.kpi_utils import auto_text_color

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

    # Create overall row
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
            st.metric("📈 Trend from Previous", f"{trend_change:.2f} {trend_symbol}")

    # Download button
    summary_df = df.copy().reset_index(drop=True)

    if "numerator" in summary_df.columns and "denominator" in summary_df.columns:
        summary_df = summary_df[
            [x_axis_col, "numerator", "denominator", value_col]
        ].copy()

        summary_df = summary_df.rename(
            columns={
                "numerator": numerator_name,
                "denominator": denominator_name,
                value_col: title,
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
                title: [overall_value],
            }
        )

        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    else:
        summary_df = summary_df[[x_axis_col, value_col]].copy()
        summary_df = summary_df.rename(columns={value_col: title})
        summary_table = summary_df.copy()

        overall_value = summary_table[title].mean() if not summary_table.empty else 0
        overall_row = pd.DataFrame({x_axis_col: ["Overall"], title: [overall_value]})
        summary_table = pd.concat([summary_table, overall_row], ignore_index=True)

    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Chart Data as CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_chart_data.csv",
        mime="text/csv",
        help="Download the exact x, y, and value components shown in the chart",
    )


def render_pph_facility_comparison_chart(
    df,
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    facility_names,
    facility_uids,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
):
    """Render facility comparison chart with same logic as other KPIs"""
    from utils.kpi_utils import auto_text_color, format_period_month_year

    if text_color is None:
        text_color = auto_text_color(bg_color)

    # STANDARDIZE COLUMN NAMES
    if "orgUnit" not in df.columns:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["orgunit", "facility_uid", "facility_id", "uid", "ou"]:
                df = df.rename(columns={col: "orgUnit"})

    # Check for facility name column
    if "orgUnit_name" in df.columns:
        df = df.rename(columns={"orgUnit_name": "Facility"})
    elif "Facility" not in df.columns:
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

    # Filter out facilities that have no data
    facilities_with_data = []
    for facility_name in comparison_df["Facility"].unique():
        facility_data = comparison_df[comparison_df["Facility"] == facility_name]
        if not (
            facility_data["numerator"].sum() == 0
            and facility_data["denominator"].sum() == 0
        ):
            facilities_with_data.append(facility_name)

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

    # Download functionality
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
    period_col,
    value_col,
    title,
    bg_color,
    text_color,
    region_names,
    region_mapping,
    facilities_by_region,
    numerator_name="PPH Cases",
    denominator_name="Total Deliveries",
):
    """Render region comparison chart with same logic as other KPIs"""
    from utils.kpi_utils import auto_text_color, format_period_month_year

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

    # Filter out regions that have no data
    regions_with_data = []
    for region_name in comparison_df["Region"].unique():
        region_data = comparison_df[comparison_df["Region"] == region_name]
        if not (
            region_data["numerator"].sum() == 0
            and region_data["denominator"].sum() == 0
        ):
            regions_with_data.append(region_name)

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

    # Download functionality
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


def render_obstetric_condition_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of obstetric conditions"""
    from utils.kpi_utils import auto_text_color

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
            height=500,
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
