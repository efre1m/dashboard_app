import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import compute_total_deliveries, auto_text_color


# ---------------- Uterotonic KPI Constants ----------------
UTEROTONIC_UID = "yVRLuRU943e"  # Numerator: Uterotonic given within 1 min
DELIVERY_MODE_UID = "z9wWxK7fw8W"  # Delivery Mode
DELIVERY_TYPE_UID = "lphtwP2ViZU"  # Delivery Type

# Uterotonic options
UTEROTONIC_OPTIONS = {
    "1": "Ergometrine",
    "2": "Oxytocin",
    "3": "Misoprostol",
    "0": "None",  # Excluded from numerator
}

# Valid delivery codes for denominator
VALID_DELIVERY_CODES = {"1", "2"}  # 1 = SVD, 2 = C-Section


# ---------------- Uterotonic KPI Computation Functions ----------------
def compute_uterotonic_numerator(df, facility_uids=None):
    """
    Compute numerator for uterotonic KPI: Count of women who received uterotonic within 1 min

    Formula: Count where Uterotonic given = 1 (Ergometrine), 2 (Oxytocin), or 3 (Misoprostol)
    Exclude: 0 (None)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for uterotonic administration events with valid codes (1, 2, 3)
    uterotonic_cases = df[
        (df["dataElement_uid"] == UTEROTONIC_UID)
        & (df["value"].isin(["1", "2", "3"]))
        & df["value"].notna()
    ]

    return uterotonic_cases["tei_id"].nunique()


def compute_uterotonic_by_type(df, facility_uids=None):
    """
    Compute distribution of uterotonic types

    Returns:
        Dictionary with counts for each uterotonic type
    """
    if df is None or df.empty:
        return {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for uterotonic administration events
    uterotonic_events = df[
        (df["dataElement_uid"] == UTEROTONIC_UID) & df["value"].notna()
    ]

    if uterotonic_events.empty:
        return {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    # Count occurrences of each uterotonic type
    type_counts = uterotonic_events["value"].value_counts()

    # Map codes to names and ensure all types are represented
    result = {
        "Ergometrine": int(type_counts.get("1", 0)),
        "Oxytocin": int(type_counts.get("2", 0)),
        "Misoprostol": int(type_counts.get("3", 0)),
        "total": int(type_counts.sum()),
    }

    return result


def compute_uterotonic_kpi(df, facility_uids=None):
    """
    Compute uterotonic KPI for the given dataframe

    Formula: Uterotonic within 1 min rate (%) =
             (Count of women who received uterotonic within 1 min) ÷ (Total Deliveries) × 100

    Returns:
        Dictionary with uterotonic metrics including drug-specific counts
    """
    if df is None or df.empty:
        return {
            "uterotonic_rate": 0.0,
            "uterotonic_count": 0,
            "total_deliveries": 0,
            "uterotonic_types": {
                "Ergometrine": 0,
                "Oxytocin": 0,
                "Misoprostol": 0,
                "total": 0,
            },
        }

    # Filter by facilities if specified
    if facility_uids:
        # Handle both single facility UID and list of UIDs
        if isinstance(facility_uids, str):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Use the same total deliveries calculation as other KPIs
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Count uterotonic administration cases
    uterotonic_count = compute_uterotonic_numerator(df, facility_uids)

    # Get distribution of uterotonic types
    uterotonic_types = compute_uterotonic_by_type(df, facility_uids)

    # Calculate uterotonic rate
    uterotonic_rate = (
        (uterotonic_count / total_deliveries * 100) if total_deliveries > 0 else 0.0
    )

    return {
        "uterotonic_rate": float(uterotonic_rate),
        "uterotonic_count": int(uterotonic_count),
        "total_deliveries": int(total_deliveries),
        "uterotonic_types": uterotonic_types,
    }


# ---------------- Uterotonic Chart Functions ----------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import compute_total_deliveries, auto_text_color


# ---------------- Uterotonic KPI Constants ----------------
UTEROTONIC_UID = "yVRLuRU943e"  # Numerator: Uterotonic given within 1 min
DELIVERY_MODE_UID = "z9wWxK7fw8W"  # Delivery Mode
DELIVERY_TYPE_UID = "lphtwP2ViZU"  # Delivery Type

# Uterotonic options
UTEROTONIC_OPTIONS = {
    "1": "Ergometrine",
    "2": "Oxytocin",
    "3": "Misoprostol",
    "0": "None",  # Excluded from numerator
}

# Valid delivery codes for denominator
VALID_DELIVERY_CODES = {"1", "2"}  # 1 = SVD, 2 = C-Section


# ---------------- Uterotonic KPI Computation Functions ----------------
def compute_uterotonic_numerator(df, facility_uids=None):
    """
    Compute numerator for uterotonic KPI: Count of women who received uterotonic within 1 min

    Formula: Count where Uterotonic given = 1 (Ergometrine), 2 (Oxytocin), or 3 (Misoprostol)
    Exclude: 0 (None)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for uterotonic administration events with valid codes (1, 2, 3)
    uterotonic_cases = df[
        (df["dataElement_uid"] == UTEROTONIC_UID)
        & (df["value"].isin(["1", "2", "3"]))
        & df["value"].notna()
    ]

    return uterotonic_cases["tei_id"].nunique()


def compute_uterotonic_by_type(df, facility_uids=None):
    """
    Compute distribution of uterotonic types

    Returns:
        Dictionary with counts for each uterotonic type
    """
    if df is None or df.empty:
        return {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for uterotonic administration events
    uterotonic_events = df[
        (df["dataElement_uid"] == UTEROTONIC_UID) & df["value"].notna()
    ]

    if uterotonic_events.empty:
        return {"Ergometrine": 0, "Oxytocin": 0, "Misoprostol": 0, "total": 0}

    # Count occurrences of each uterotonic type
    type_counts = uterotonic_events["value"].value_counts()

    # Map codes to names and ensure all types are represented
    result = {
        "Ergometrine": int(type_counts.get("1", 0)),
        "Oxytocin": int(type_counts.get("2", 0)),
        "Misoprostol": int(type_counts.get("3", 0)),
        "total": int(type_counts.sum()),
    }

    return result


def compute_uterotonic_kpi(df, facility_uids=None):
    """
    Compute uterotonic KPI for the given dataframe

    Formula: Uterotonic within 1 min rate (%) =
             (Count of women who received uterotonic within 1 min) ÷ (Total Deliveries) × 100

    Returns:
        Dictionary with uterotonic metrics including drug-specific counts
    """
    if df is None or df.empty:
        return {
            "uterotonic_rate": 0.0,
            "uterotonic_count": 0,
            "total_deliveries": 0,
            "uterotonic_types": {
                "Ergometrine": 0,
                "Oxytocin": 0,
                "Misoprostol": 0,
                "total": 0,
            },
        }

    # Filter by facilities if specified
    if facility_uids:
        # Handle both single facility UID and list of UIDs
        if isinstance(facility_uids, str):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Use the same total deliveries calculation as other KPIs
    total_deliveries = compute_total_deliveries(df, facility_uids)

    # Count uterotonic administration cases
    uterotonic_count = compute_uterotonic_numerator(df, facility_uids)

    # Get distribution of uterotonic types
    uterotonic_types = compute_uterotonic_by_type(df, facility_uids)

    # Calculate uterotonic rate
    uterotonic_rate = (
        (uterotonic_count / total_deliveries * 100) if total_deliveries > 0 else 0.0
    )

    return {
        "uterotonic_rate": float(uterotonic_rate),
        "uterotonic_count": int(uterotonic_count),
        "total_deliveries": int(total_deliveries),
        "uterotonic_types": uterotonic_types,
    }


# ---------------- Uterotonic Chart Functions ----------------
def render_uterotonic_trend_chart(
    df,
    period_col="period_display",
    value_col="uterotonic_rate",
    title="Uterotonic Administration Rate Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="Uterotonic Cases",
    denominator_name="Total Deliveries",
    facility_uids=None,
):
    """Render a trend chart for uterotonic administration rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Chart options
    chart_options = ["Line Chart", "Stacked Percentage Bar Chart"]
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

    # Create chart based on selected type
    if chart_type == "line chart":
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

        # Update traces for line chart
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
        )

    elif chart_type == "stacked percentage bar chart":
        # Check if we have the drug-specific rate columns
        if all(
            col in df.columns
            for col in ["ergometrine_rate", "oxytocin_rate", "misoprostol_rate"]
        ):
            # Create a new figure
            fig = go.Figure()

            # Add traces for each uterotonic type as percentages
            fig.add_trace(
                go.Bar(
                    name="Ergometrine",
                    x=df[period_col],
                    y=df["ergometrine_rate"],
                    marker_color="#1f77b4",
                    hovertemplate="<b>%{x}</b><br>Ergometrine: %{y:.2f}%<br>Count: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
                    customdata=np.column_stack(
                        (df["ergometrine_count"], df["Total Deliveries"])
                    ),
                )
            )

            fig.add_trace(
                go.Bar(
                    name="Oxytocin",
                    x=df[period_col],
                    y=df["oxytocin_rate"],
                    marker_color="#ff7f0e",
                    hovertemplate="<b>%{x}</b><br>Oxytocin: %{y:.2f}%<br>Count: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
                    customdata=np.column_stack(
                        (df["oxytocin_count"], df["Total Deliveries"])
                    ),
                )
            )

            fig.add_trace(
                go.Bar(
                    name="Misoprostol",
                    x=df[period_col],
                    y=df["misoprostol_rate"],
                    marker_color="#2ca02c",
                    hovertemplate="<b>%{x}</b><br>Misoprostol: %{y:.2f}%<br>Count: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
                    customdata=np.column_stack(
                        (df["misoprostol_count"], df["Total Deliveries"])
                    ),
                )
            )

            # Update layout for stacked bar chart
            fig.update_layout(
                barmode="stack",
                title="Uterotonic Administration Rate (%)",
                height=400,
                yaxis_title="Uterotonic Administration Rate (%)",
                xaxis_title="Period",
            )

            # Set appropriate y-axis range to make small values visible
            max_rate = max(
                df["ergometrine_rate"].max(),
                df["oxytocin_rate"].max(),
                df["misoprostol_rate"].max(),
                df[value_col].max(),
            )

            # Add a buffer to make small values visible
            y_upper_limit = min(max_rate * 1.2, 100)  # Don't exceed 100%
            fig.update_layout(yaxis_range=[0, y_upper_limit])

        else:
            # Fall back to regular line chart if type data is not available
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

            # Update traces for line chart
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=7),
                hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
            )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title=(
            "Uterotonic Administration Rate (%)"
            if chart_type == "line chart"
            else "Uterotonic Administration Rate (%)"
        ),
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
            f'<p style="font-size:1.2rem;font-weight:600;">Latest Value: {last_value:.2f}% <span class="{trend_class}">{trend_symbol}</span></p>',
            unsafe_allow_html=True,
        )

    # Summary table - include drug-specific rate columns if available
    st.subheader(f"📋 {title} Summary Table")

    # First Table: Main KPI Overview
    st.markdown("**Main KPI Overview**")
    main_summary_df = df.copy().reset_index(drop=True)
    main_summary_df = main_summary_df[
        [period_col, numerator_name, denominator_name, value_col]
    ]

    # Calculate overall value for main table
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

    # Second Table: Uterotonic Type Breakdown (only if we have the data)
    if all(
        col in df.columns
        for col in ["ergometrine_rate", "oxytocin_rate", "misoprostol_rate"]
    ):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Uterotonic Type Breakdown**")

        # Create drug breakdown table
        drug_summary_df = df.copy().reset_index(drop=True)
        drug_summary_df = drug_summary_df[
            [
                period_col,
                denominator_name,
                "ergometrine_count",
                "ergometrine_rate",
                "oxytocin_count",
                "oxytocin_rate",
                "misoprostol_count",
                "misoprostol_rate",
            ]
        ]

        # Calculate overall values for drug table
        total_ergometrine_count = drug_summary_df["ergometrine_count"].sum()
        total_oxytocin_count = drug_summary_df["oxytocin_count"].sum()
        total_misoprostol_count = drug_summary_df["misoprostol_count"].sum()
        total_deliveries = drug_summary_df[denominator_name].sum()

        overall_ergometrine_rate = (
            (total_ergometrine_count / total_deliveries * 100)
            if total_deliveries > 0
            else 0
        )
        overall_oxytocin_rate = (
            (total_oxytocin_count / total_deliveries * 100)
            if total_deliveries > 0
            else 0
        )
        overall_misoprostol_rate = (
            (total_misoprostol_count / total_deliveries * 100)
            if total_deliveries > 0
            else 0
        )
        overall_total_rate = (
            overall_ergometrine_rate + overall_oxytocin_rate + overall_misoprostol_rate
        )

        overall_drug_row = pd.DataFrame(
            {
                period_col: [f"Overall {title}"],
                denominator_name: [total_deliveries],
                "ergometrine_count": [total_ergometrine_count],
                "ergometrine_rate": [overall_ergometrine_rate],
                "oxytocin_count": [total_oxytocin_count],
                "oxytocin_rate": [overall_oxytocin_rate],
                "misoprostol_count": [total_misoprostol_count],
                "misoprostol_rate": [overall_misoprostol_rate],
            }
        )

        drug_summary_table = pd.concat(
            [drug_summary_df, overall_drug_row], ignore_index=True
        )
        drug_summary_table.insert(0, "No", range(1, len(drug_summary_table) + 1))

        # Format drug table with compact column names
        drug_summary_table = drug_summary_table.rename(
            columns={
                "ergometrine_count": "Ergo<br>Count",
                "ergometrine_rate": "Ergo<br>Rate",
                "oxytocin_count": "Oxy<br>Count",
                "oxytocin_rate": "Oxy<br>Rate",
                "misoprostol_count": "Miso<br>Count",
                "misoprostol_rate": "Miso<br>Rate",
                denominator_name: "Total<br>Deliveries",
            }
        )

        # Format drug table
        styled_drug_table = (
            drug_summary_table.style.format(
                {
                    "Total<br>Deliveries": "{:,.0f}",
                    "Ergo<br>Count": "{:,.0f}",
                    "Ergo<br>Rate": "{:.1f}%",
                    "Oxy<br>Count": "{:,.0f}",
                    "Oxy<br>Rate": "{:.1f}%",
                    "Miso<br>Count": "{:,.0f}",
                    "Miso<br>Rate": "{:.1f}%",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_drug_table.to_html(), unsafe_allow_html=True)

    # Download button for both tables combined
    combined_table = (
        pd.concat([main_summary_table, drug_summary_table], axis=1)
        if "drug_summary_table" in locals()
        else main_summary_table
    )
    csv = combined_table.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_summary.csv",
        mime="text/csv",
    )


def render_uterotonic_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="uterotonic_rate",
    title="Uterotonic Administration Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="Uterotonic Cases",
    denominator_name="Total Deliveries",
):
    """Render a comparison chart showing each facility's uterotonic performance"""
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

    for facility_uid in facility_uids:
        facility_df = filtered_df[filtered_df["orgUnit"] == facility_uid]
        if not facility_df.empty:
            uterotonic_data = compute_uterotonic_kpi(facility_df, [facility_uid])
            comparison_data.append(
                {
                    "Facility": facility_uid_to_name[facility_uid],
                    "value": uterotonic_data["uterotonic_rate"],
                    "uterotonic_count": uterotonic_data["uterotonic_count"],
                    "total_deliveries": uterotonic_data["total_deliveries"],
                    "ergometrine_count": uterotonic_data["uterotonic_types"][
                        "Ergometrine"
                    ],
                    "oxytocin_count": uterotonic_data["uterotonic_types"]["Oxytocin"],
                    "misoprostol_count": uterotonic_data["uterotonic_types"][
                        "Misoprostol"
                    ],
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
        # For line chart, we need time series data
        time_series_data = []
        for period_display in period_order:
            period_df = filtered_df[filtered_df["period_display"] == period_display]

            for facility_uid in facility_uids:
                facility_period_df = period_df[period_df["orgUnit"] == facility_uid]
                if not facility_period_df.empty:
                    uterotonic_data = compute_uterotonic_kpi(
                        facility_period_df, [facility_uid]
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": uterotonic_data["uterotonic_rate"],
                            "uterotonic_count": uterotonic_data["uterotonic_count"],
                            "total_deliveries": uterotonic_data["total_deliveries"],
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
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>Uterotonic Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (time_series_df["uterotonic_count"], time_series_df["total_deliveries"])
            ),
        )
    else:  # Bar Chart
        # UPDATED: Bar chart shows all facilities with their overall values
        fig = px.bar(
            comparison_df,
            x="Facility",
            y="value",
            title=title,
            height=500,
            color="Facility",
            hover_data=[
                "uterotonic_count",
                "total_deliveries",
                "ergometrine_count",
                "oxytocin_count",
                "misoprostol_count",
            ],
        )

        # UPDATED: Custom hover template for bar chart
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Uterotonic Rate: %{y:.2f}%<br>Uterotonic Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<br>Ergometrine: %{customdata[2]}<br>Oxytocin: %{customdata[3]}<br>Misoprostol: %{customdata[4]}<extra></extra>",
            customdata=np.column_stack(
                (
                    comparison_df["uterotonic_count"],
                    comparison_df["total_deliveries"],
                    comparison_df["ergometrine_count"],
                    comparison_df["oxytocin_count"],
                    comparison_df["misoprostol_count"],
                )
            ),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="Uterotonic Administration Rate (%)",
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
            uterotonic_data = compute_uterotonic_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Uterotonic Cases": uterotonic_data["uterotonic_count"],
                    "Total Deliveries": uterotonic_data["total_deliveries"],
                    "Uterotonic Rate (%)": uterotonic_data["uterotonic_rate"],
                    "Ergometrine": uterotonic_data["uterotonic_types"]["Ergometrine"],
                    "Oxytocin": uterotonic_data["uterotonic_types"]["Oxytocin"],
                    "Misoprostol": uterotonic_data["uterotonic_types"]["Misoprostol"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Uterotonic Cases"].sum()
    total_denominator = facility_table_df["Total Deliveries"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    total_ergometrine = facility_table_df["Ergometrine"].sum()
    total_oxytocin = facility_table_df["Oxytocin"].sum()
    total_misoprostol = facility_table_df["Misoprostol"].sum()

    overall_row = {
        "Facility Name": "Overall",
        "Uterotonic Cases": total_numerator,
        "Total Deliveries": total_denominator,
        "Uterotonic Rate (%)": overall_value,
        "Ergometrine": total_ergometrine,
        "Oxytocin": total_oxytocin,
        "Misoprostol": total_misoprostol,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Uterotonic Cases": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "Uterotonic Rate (%)": "{:.2f}%",
                "Ergometrine": "{:,.0f}",
                "Oxytocin": "{:,.0f}",
                "Misoprostol": "{:,.0f}",
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
        file_name="uterotonic_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_uterotonic_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="uterotonic_rate",
    title="Uterotonic Administration Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="Uterotonic Cases",
    denominator_name="Total Deliveries",
):
    """Render a comparison chart showing each region's uterotonic performance"""
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

    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        region_df = filtered_df[filtered_df["orgUnit"].isin(region_facility_uids)]

        if not region_df.empty:
            uterotonic_data = compute_uterotonic_kpi(region_df, region_facility_uids)
            comparison_data.append(
                {
                    "Region": region_name,
                    "value": uterotonic_data["uterotonic_rate"],
                    "uterotonic_count": uterotonic_data["uterotonic_count"],
                    "total_deliveries": uterotonic_data["total_deliveries"],
                    "ergometrine_count": uterotonic_data["uterotonic_types"][
                        "Ergometrine"
                    ],
                    "oxytocin_count": uterotonic_data["uterotonic_types"]["Oxytocin"],
                    "misoprostol_count": uterotonic_data["uterotonic_types"][
                        "Misoprostol"
                    ],
                }
            )

    if not comparison_data:
        st.info("⚠️ No data available for region comparison.")
        return

    comparison_df = pd.DataFrame(comparison_data)

    # Chart options
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
        # For line chart, we need time series data
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
                    uterotonic_data = compute_uterotonic_kpi(
                        region_period_df, region_facility_uids
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": uterotonic_data["uterotonic_rate"],
                            "uterotonic_count": uterotonic_data["uterotonic_count"],
                            "total_deliveries": uterotonic_data["total_deliveries"],
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
            hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.2f}%<br>Uterotonic Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack(
                (time_series_df["uterotonic_count"], time_series_df["total_deliveries"])
            ),
        )
    else:  # Bar Chart
        # UPDATED: Bar chart shows all regions with their overall values
        fig = px.bar(
            comparison_df,
            x="Region",
            y="value",
            title=title,
            height=500,
            color="Region",
            hover_data=[
                "uterotonic_count",
                "total_deliveries",
                "ergometrine_count",
                "oxytocin_count",
                "misoprostol_count",
            ],
        )

        # UPDATED: Custom hover template for bar chart
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Uterotonic Rate: %{y:.2f}%<br>Uterotonic Cases: %{customdata[0]}<br>Total Deliveries: %{customdata[1]}<br>Ergometrine: %{customdata[2]}<br>Oxytocin: %{customdata[3]}<br>Misoprostol: %{customdata[4]}<extra></extra>",
            customdata=np.column_stack(
                (
                    comparison_df["uterotonic_count"],
                    comparison_df["total_deliveries"],
                    comparison_df["ergometrine_count"],
                    comparison_df["oxytocin_count"],
                    comparison_df["misoprostol_count"],
                )
            ),
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="Uterotonic Administration Rate (%)",
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
            uterotonic_data = compute_uterotonic_kpi(region_df, region_facility_uids)
            region_table_data.append(
                {
                    "Region Name": region_name,
                    "Uterotonic Cases": uterotonic_data["uterotonic_count"],
                    "Total Deliveries": uterotonic_data["total_deliveries"],
                    "Uterotonic Rate (%)": uterotonic_data["uterotonic_rate"],
                    "Ergometrine": uterotonic_data["uterotonic_types"]["Ergometrine"],
                    "Oxytocin": uterotonic_data["uterotonic_types"]["Oxytocin"],
                    "Misoprostol": uterotonic_data["uterotonic_types"]["Misoprostol"],
                }
            )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Uterotonic Cases"].sum()
    total_denominator = region_table_df["Total Deliveries"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    total_ergometrine = region_table_df["Ergometrine"].sum()
    total_oxytocin = region_table_df["Oxytocin"].sum()
    total_misoprostol = region_table_df["Misoprostol"].sum()

    overall_row = {
        "Region Name": "Overall",
        "Uterotonic Cases": total_numerator,
        "Total Deliveries": total_denominator,
        "Uterotonic Rate (%)": overall_value,
        "Ergometrine": total_ergometrine,
        "Oxytocin": total_oxytocin,
        "Misoprostol": total_misoprostol,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Uterotonic Cases": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "Uterotonic Rate (%)": "{:.2f}%",
                "Ergometrine": "{:,.0f}",
                "Oxytocin": "{:,.0f}",
                "Misoprostol": "{:,.0f}",
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
        file_name="uterotonic_rate_region_comparison.csv",
        mime="text/csv",
    )


def render_uterotonic_type_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of uterotonic types"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute uterotonic type distribution
    type_data = compute_uterotonic_by_type(df, facility_uids)

    if type_data["total"] == 0:
        st.info("⚠️ No data available for uterotonic type distribution.")
        return

    # Prepare data for visualization
    pie_data = {
        "Type": ["Oxytocin", "Ergometrine", "Misoprostol"],
        "Count": [
            type_data["Oxytocin"],
            type_data["Ergometrine"],
            type_data["Misoprostol"],
        ],
    }

    pie_df = pd.DataFrame(pie_data)

    # Calculate percentages
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

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
        key=f"uterotonic_chart_type_{str(facility_uids)}",
    )

    # Create chart with reduced size
    if chart_type == "Pie Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,  # Slightly increased height
        )
    else:  # Donut Chart
        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            hole=0.4,
        )

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

    # FIX: COMPLETELY REMOVE ANY TITLE FROM THE LAYOUT
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        height=500,  # SAME HEIGHT AS PPH
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
        margin=dict(l=0, r=150, t=20, b=20),  # SAME MARGINS AS PPH
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
            '<div class="pie-chart-title">Distribution of Uterotonic Types</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 Uterotonic Type Summary")
    summary_df = pie_df.copy()
    summary_df.insert(0, "No", range(1, len(summary_df) + 1))

    styled_table = (
        summary_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"})
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Add download button for CSV
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="uterotonic_type_distribution.csv",
        mime="text/csv",
    )


def render_uterotonic_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="uterotonic_rate",
    title="Uterotonic Administration Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="Uterotonic Cases",
    denominator_name="Total Deliveries",
):
    """SIMPLIFIED VERSION: Render facility comparison without numerator/denominator in hover"""
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
                    uterotonic_data = compute_uterotonic_kpi(
                        facility_period_df, [facility_uid]
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": uterotonic_data["uterotonic_rate"],
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
                uterotonic_data = compute_uterotonic_kpi(facility_df, [facility_uid])
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": uterotonic_data["uterotonic_rate"],
                        "uterotonic_count": uterotonic_data["uterotonic_count"],
                        "total_deliveries": uterotonic_data["total_deliveries"],
                        "ergometrine_count": uterotonic_data["uterotonic_types"][
                            "Ergometrine"
                        ],
                        "oxytocin_count": uterotonic_data["uterotonic_types"][
                            "Oxytocin"
                        ],
                        "misoprostol_count": uterotonic_data["uterotonic_types"][
                            "Misoprostol"
                        ],
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
            hovertemplate="<b>%{x}</b><br>Uterotonic Rate: %{y:.2f}%<extra></extra>"
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="Uterotonic Administration Rate (%)",
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
            uterotonic_data = compute_uterotonic_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "Uterotonic Cases": uterotonic_data["uterotonic_count"],
                    "Total Deliveries": uterotonic_data["total_deliveries"],
                    "Uterotonic Rate (%)": uterotonic_data["uterotonic_rate"],
                    "Ergometrine": uterotonic_data["uterotonic_types"]["Ergometrine"],
                    "Oxytocin": uterotonic_data["uterotonic_types"]["Oxytocin"],
                    "Misoprostol": uterotonic_data["uterotonic_types"]["Misoprostol"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["Uterotonic Cases"].sum()
    total_denominator = facility_table_df["Total Deliveries"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    total_ergometrine = facility_table_df["Ergometrine"].sum()
    total_oxytocin = facility_table_df["Oxytocin"].sum()
    total_misoprostol = facility_table_df["Misoprostol"].sum()

    overall_row = {
        "Facility Name": "Overall",
        "Uterotonic Cases": total_numerator,
        "Total Deliveries": total_denominator,
        "Uterotonic Rate (%)": overall_value,
        "Ergometrine": total_ergometrine,
        "Oxytocin": total_oxytocin,
        "Misoprostol": total_misoprostol,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "Uterotonic Cases": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "Uterotonic Rate (%)": "{:.2f}%",
                "Ergometrine": "{:,.0f}",
                "Oxytocin": "{:,.0f}",
                "Misoprostol": "{:,.0f}",
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
        file_name="uterotonic_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_uterotonic_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="uterotonic_rate",
    title="Uterotonic Administration Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="Uterotonic Cases",
    denominator_name="Total Deliveries",
):
    """SIMPLIFIED VERSION: Render region comparison without numerator/denominator in hover"""
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
                    uterotonic_data = compute_uterotonic_kpi(
                        region_period_df, region_facility_uids
                    )
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": uterotonic_data["uterotonic_rate"],
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
                uterotonic_data = compute_uterotonic_kpi(
                    region_df, region_facility_uids
                )
                bar_data.append(
                    {
                        "Region": region_name,
                        "value": uterotonic_data["uterotonic_rate"],
                        "uterotonic_count": uterotonic_data["uterotonic_count"],
                        "total_deliveries": uterotonic_data["total_deliveries"],
                        "ergometrine_count": uterotonic_data["uterotonic_types"][
                            "Ergometrine"
                        ],
                        "oxytocin_count": uterotonic_data["uterotonic_types"][
                            "Oxytocin"
                        ],
                        "misoprostol_count": uterotonic_data["uterotonic_types"][
                            "Misoprostol"
                        ],
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
            hovertemplate="<b>%{x}</b><br>Uterotonic Rate: %{y:.2f}%<extra></extra>"
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="Uterotonic Administration Rate (%)",
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
            uterotonic_data = compute_uterotonic_kpi(region_df, region_facility_uids)
            region_table_data.append(
                {
                    "Region Name": region_name,
                    "Uterotonic Cases": uterotonic_data["uterotonic_count"],
                    "Total Deliveries": uterotonic_data["total_deliveries"],
                    "Uterotonic Rate (%)": uterotonic_data["uterotonic_rate"],
                    "Ergometrine": uterotonic_data["uterotonic_types"]["Ergometrine"],
                    "Oxytocin": uterotonic_data["uterotonic_types"]["Oxytocin"],
                    "Misoprostol": uterotonic_data["uterotonic_types"]["Misoprostol"],
                }
            )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["Uterotonic Cases"].sum()
    total_denominator = region_table_df["Total Deliveries"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    total_ergometrine = region_table_df["Ergometrine"].sum()
    total_oxytocin = region_table_df["Oxytocin"].sum()
    total_misoprostol = region_table_df["Misoprostol"].sum()

    overall_row = {
        "Region Name": "Overall",
        "Uterotonic Cases": total_numerator,
        "Total Deliveries": total_denominator,
        "Uterotonic Rate (%)": overall_value,
        "Ergometrine": total_ergometrine,
        "Oxytocin": total_oxytocin,
        "Misoprostol": total_misoprostol,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "Uterotonic Cases": "{:,.0f}",
                "Total Deliveries": "{:,.0f}",
                "Uterotonic Rate (%)": "{:.2f}%",
                "Ergometrine": "{:,.0f}",
                "Oxytocin": "{:,.0f}",
                "Misoprostol": "{:,.0f}",
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
        file_name="uterotonic_rate_region_comparison.csv",
        mime="text/csv",
    )


def render_uterotonic_type_pie_chart(
    df, facility_uids=None, bg_color="#FFFFFF", text_color=None
):
    """Render a pie chart showing distribution of uterotonic types"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Compute uterotonic type distribution
    type_data = compute_uterotonic_by_type(df, facility_uids)

    if type_data["total"] == 0:
        st.info("⚠️ No data available for uterotonic type distribution.")
        return

    # Prepare data for visualization
    pie_data = {
        "Type": ["Oxytocin", "Ergometrine", "Misoprostol"],
        "Count": [
            type_data["Oxytocin"],
            type_data["Ergometrine"],
            type_data["Misoprostol"],
        ],
    }

    pie_df = pd.DataFrame(pie_data)

    # Calculate percentages
    total = pie_df["Count"].sum()
    pie_df["Percentage"] = (pie_df["Count"] / total * 100) if total > 0 else 0

    # UPDATED: Move chart type selection to the top and make it more visible
    st.markdown("### 📊 Chart Type Selection")
    chart_type = st.selectbox(
        "Choose how to display the uterotonic type distribution:",
        options=["Pie Chart", "Donut Chart"],
        index=0,  # Default to Pie Chart
        key=f"uterotonic_chart_type_{str(facility_uids)}",
    )

    st.markdown("---")  # Add a separator

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

    # Create chart with consistent height
    if chart_type == "Pie Chart":
        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            color="Type",
            color_discrete_map={
                "Oxytocin": "#ff7f0e",
                "Ergometrine": "#1f77b4",
                "Misoprostol": "#2ca02c",
            },
        )
    else:  # Donut Chart
        fig = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hover_data=["Percentage"],
            labels={"Count": "Count", "Percentage": "Percentage"},
            height=500,
            hole=0.4,
            color="Type",
            color_discrete_map={
                "Oxytocin": "#ff7f0e",
                "Ergometrine": "#1f77b4",
                "Misoprostol": "#2ca02c",
            },
        )

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
    fig.update_layout(title=None)
    fig.layout.pop("title", None)

    # Use container to control layout
    with st.container():
        st.markdown(
            '<div class="pie-chart-title">Distribution of Uterotonic Types</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show summary table
    st.subheader("📋 Uterotonic Type Summary")
    summary_df = pie_df.copy()
    summary_df.insert(0, "No", range(1, len(summary_df) + 1))

    styled_table = (
        summary_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.2f}%"})
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Add download button for CSV
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="uterotonic_type_distribution.csv",
        mime="text/csv",
    )
