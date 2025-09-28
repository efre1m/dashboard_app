import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- ARV KPI Constants ----------------
ARV_RX_NEWBORN_UID = "H7J2SxBpObS"  # Numerator: ARV Rx for Newborn
HIV_RESULT_UID = "tTrH9cOQRnZ"  # HIV Result
BIRTH_OUTCOME_UID = "wZig9cek3Gv"  # Birth Outcome
NUMBER_OF_NEWBORNS_UID = "VzwnSBROvUm"  # Number of Newborns
OTHER_NUMBER_OF_NEWBORNS_UID = "tIa0WvbPGLk"  # Other Number of Newborns
HIV_POSITIVE_VALUE = "1"  # HIV positive result
BIRTH_OUTCOME_ALIVE = "1"  # Alive birth outcome


# ---------------- ARV KPI Computation Functions ----------------
def compute_arv_numerator(df, facility_uids=None):
    """
    Compute numerator for ARV KPI: Count of infants who received ARV prophylaxis

    Formula: Count where ARV Rx for Newborn is not empty
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Filter for ARV administration events with non-empty values
    arv_cases = df[
        (df["dataElement_uid"] == ARV_RX_NEWBORN_UID)
        & df["value"].notna()
        & (df["value"] != "")
    ]

    return arv_cases["tei_id"].nunique()


def compute_arv_denominator(df, facility_uids=None):
    """
    Compute denominator for ARV KPI: Count of live infants born to HIV+ women

    Formula:
        1. Identify mothers with HIV+ result (HIV Result = 1)
        2. For each HIV+ mother, count live infants (Birth Outcome = 1)
        3. Sum Number of Newborns + Other Number of Newborns
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Step 1: Identify HIV+ mothers
    hiv_positive_mothers = df[
        (df["dataElement_uid"] == HIV_RESULT_UID) & (df["value"] == HIV_POSITIVE_VALUE)
    ]["tei_id"].unique()

    if len(hiv_positive_mothers) == 0:
        return 0

    # Step 2: Filter for HIV+ mothers and count live births
    hiv_positive_df = df[df["tei_id"].isin(hiv_positive_mothers)]

    # Count live births from birth outcome data
    live_births = hiv_positive_df[
        (hiv_positive_df["dataElement_uid"] == BIRTH_OUTCOME_UID)
        & (hiv_positive_df["value"] == BIRTH_OUTCOME_ALIVE)
    ]["tei_id"].nunique()

    # Step 3: Sum number of newborns from multiple birth fields
    number_of_newborns = (
        hiv_positive_df[hiv_positive_df["dataElement_uid"] == NUMBER_OF_NEWBORNS_UID][
            "value"
        ]
        .astype(float)
        .sum()
    )

    other_number_of_newborns = (
        hiv_positive_df[
            hiv_positive_df["dataElement_uid"] == OTHER_NUMBER_OF_NEWBORNS_UID
        ]["value"]
        .astype(float)
        .sum()
    )

    total_infants = number_of_newborns + other_number_of_newborns

    # Use the maximum of live_births or total_infants to handle data completeness
    return max(live_births, total_infants)


def compute_arv_kpi(df, facility_uids=None):
    """
    Compute ARV KPI for the given dataframe

    Formula: ARV Prophylaxis Rate (%) =
             (Count of infants who received ARV prophylaxis) ÷ (Live infants born to HIV+ women) × 100

    Returns:
        Dictionary with ARV metrics
    """
    if df is None or df.empty:
        return {
            "arv_rate": 0.0,
            "arv_count": 0,
            "hiv_exposed_infants": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        # Handle both single facility UID and list of UIDs
        if isinstance(facility_uids, str):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count ARV administration cases
    arv_count = compute_arv_numerator(df, facility_uids)

    # Count HIV-exposed infants (denominator)
    hiv_exposed_infants = compute_arv_denominator(df, facility_uids)

    # Calculate ARV rate
    arv_rate = (
        (arv_count / hiv_exposed_infants * 100) if hiv_exposed_infants > 0 else 0.0
    )

    return {
        "arv_rate": float(arv_rate),
        "arv_count": int(arv_count),
        "hiv_exposed_infants": int(hiv_exposed_infants),
    }


def compute_arv_trend_data(df, period_col="period_display", facility_uids=None):
    """
    Compute ARV trend data by period

    Returns:
        DataFrame with columns: period_display, hiv_exposed_infants, arv_count, arv_rate
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Ensure period column exists
    if period_col not in df.columns:
        return pd.DataFrame()

    trend_data = []

    for period in df[period_col].unique():
        period_df = df[df[period_col] == period]
        arv_data = compute_arv_kpi(period_df, facility_uids)

        trend_data.append(
            {
                period_col: period,
                "hiv_exposed_infants": arv_data["hiv_exposed_infants"],
                "arv_count": arv_data["arv_count"],
                "arv_rate": arv_data["arv_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- ARV Chart Functions ----------------
def render_arv_trend_chart(
    df,
    period_col="period_display",
    value_col="arv_rate",
    title="ARV Prophylaxis Rate Trend",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="ARV Cases",
    denominator_name="HIV-Exposed Infants",
    facility_uids=None,
):
    """Render a trend chart for ARV prophylaxis rate"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # Chart options
    chart_options = ["Line Chart", "Area Chart", "Bar Chart"]
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

    elif chart_type == "area chart":
        fig = px.area(
            df,
            x=period_col,
            y=value_col,
            title=title,
            height=400,
            hover_data=hover_data,
        )

        fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
            fill="tozeroy",
        )

    else:  # Bar Chart
        fig = px.bar(
            df,
            x=period_col,
            y=value_col,
            title=title,
            height=400,
            hover_data=hover_data,
        )

        fig.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.2f}}%<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>",
        )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="ARV Prophylaxis Rate (%)",
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

    # Summary table
    st.subheader(f"📋 {title} Summary Table")
    summary_df = df.copy().reset_index(drop=True)
    summary_df = summary_df[[period_col, numerator_name, denominator_name, value_col]]

    # Calculate overall value
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
    summary_table.insert(0, "No", range(1, len(summary_table) + 1))

    # Format table
    styled_table = (
        summary_table.style.format(
            {
                value_col: "{:.1f}%",
                numerator_name: "{:,.0f}",
                denominator_name: "{:,.0f}",
            }
        )
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


def render_arv_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="arv_rate",
    title="ARV Prophylaxis Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="ARV Cases",
    denominator_name="HIV-Exposed Infants",
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
                    arv_data = compute_arv_kpi(facility_period_df, [facility_uid])
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Facility": facility_uid_to_name[facility_uid],
                            "value": arv_data["arv_rate"],
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
                arv_data = compute_arv_kpi(facility_df, [facility_uid])
                bar_data.append(
                    {
                        "Facility": facility_uid_to_name[facility_uid],
                        "value": arv_data["arv_rate"],
                        "arv_count": arv_data["arv_count"],
                        "hiv_exposed_infants": arv_data["hiv_exposed_infants"],
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
            hovertemplate="<b>%{x}</b><br>ARV Rate: %{y:.2f}%<extra></extra>"
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Facility",
        yaxis_title="ARV Prophylaxis Rate (%)",
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
            arv_data = compute_arv_kpi(facility_df, [facility_uid])
            facility_table_data.append(
                {
                    "Facility Name": facility_name,
                    "ARV Cases": arv_data["arv_count"],
                    "HIV-Exposed Infants": arv_data["hiv_exposed_infants"],
                    "ARV Rate (%)": arv_data["arv_rate"],
                }
            )

    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return

    facility_table_df = pd.DataFrame(facility_table_data)

    # Calculate overall
    total_numerator = facility_table_df["ARV Cases"].sum()
    total_denominator = facility_table_df["HIV-Exposed Infants"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Facility Name": "Overall",
        "ARV Cases": total_numerator,
        "HIV-Exposed Infants": total_denominator,
        "ARV Rate (%)": overall_value,
    }

    facility_table_df = pd.concat(
        [facility_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    facility_table_df.insert(0, "No", range(1, len(facility_table_df) + 1))

    # Format table
    styled_table = (
        facility_table_df.style.format(
            {
                "ARV Cases": "{:,.0f}",
                "HIV-Exposed Infants": "{:,.0f}",
                "ARV Rate (%)": "{:.2f}%",
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
        file_name="arv_rate_facility_comparison.csv",
        mime="text/csv",
    )


def render_arv_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="arv_rate",
    title="ARV Prophylaxis Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="ARV Cases",
    denominator_name="HIV-Exposed Infants",
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
                    arv_data = compute_arv_kpi(region_period_df, region_facility_uids)
                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": arv_data["arv_rate"],
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
                arv_data = compute_arv_kpi(region_df, region_facility_uids)
                bar_data.append(
                    {
                        "Region": region_name,
                        "value": arv_data["arv_rate"],
                        "arv_count": arv_data["arv_count"],
                        "hiv_exposed_infants": arv_data["hiv_exposed_infants"],
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
            hovertemplate="<b>%{x}</b><br>ARV Rate: %{y:.2f}%<extra></extra>"
        )

    # Common layout updates
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period" if chart_type == "Line Chart" else "Region",
        yaxis_title="ARV Prophylaxis Rate (%)",
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
            arv_data = compute_arv_kpi(region_df, region_facility_uids)
            region_table_data.append(
                {
                    "Region Name": region_name,
                    "ARV Cases": arv_data["arv_count"],
                    "HIV-Exposed Infants": arv_data["hiv_exposed_infants"],
                    "ARV Rate (%)": arv_data["arv_rate"],
                }
            )

    if not region_table_data:
        st.info("⚠️ No data available for region comparison table.")
        return

    region_table_df = pd.DataFrame(region_table_data)

    # Calculate overall
    total_numerator = region_table_df["ARV Cases"].sum()
    total_denominator = region_table_df["HIV-Exposed Infants"].sum()
    overall_value = (
        (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
    )

    overall_row = {
        "Region Name": "Overall",
        "ARV Cases": total_numerator,
        "HIV-Exposed Infants": total_denominator,
        "ARV Rate (%)": overall_value,
    }

    region_table_df = pd.concat(
        [region_table_df, pd.DataFrame([overall_row])], ignore_index=True
    )
    region_table_df.insert(0, "No", range(1, len(region_table_df) + 1))

    # Format table
    styled_table = (
        region_table_df.style.format(
            {
                "ARV Cases": "{:,.0f}",
                "HIV-Exposed Infants": "{:,.0f}",
                "ARV Rate (%)": "{:.2f}%",
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
        file_name="arv_rate_region_comparison.csv",
        mime="text/csv",
    )
