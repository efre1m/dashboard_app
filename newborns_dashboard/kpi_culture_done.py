# kpi_culture_done.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color


# ---------------- KPI Constants ----------------
BLOOD_CULTURE_UID = "A94ibeuO9GL"  # Blood culture for suspected sepsis
ANTIBIOTICS_ADMINISTERED_UID = "sxtsEDilKZd"  # Were antibiotics administered

# Program Stage UID (Microbiology and labs)
LABS_PROGRAM_STAGE_UID = "aCrttmnx7FI"

# Value codes for blood culture results (from your clarification)
CULTURE_DONE_VALUES = ["1", "2", "3"]  # 1=Negative, 2=Positive, 3=Unknown
ANTIBIOTICS_YES_VALUE = "1"  # Yes code value


# ---------------- KPI Computation Functions ----------------
def compute_culture_done_numerator(df, facility_uids=None):
    """
    Compute numerator for Culture Done KPI:
    Count of babies on antibiotics who had blood culture done

    Formula: Count where:
        - Blood culture for suspected sepsis = "1", "2", or "3"
          (1=Done - Culture Negative, 2=Done - Culture Positive, 3=Done but Unknown Result)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count babies with blood culture done (any result)
    culture_cases = df[
        (df["dataElement_uid"] == BLOOD_CULTURE_UID)
        & df["value"].notna()
        & (df["value"].isin(CULTURE_DONE_VALUES))
    ]["tei_id"].nunique()

    return culture_cases


def compute_antibiotics_denominator(df, facility_uids=None):
    """
    Compute denominator for Culture Done KPI:
    Total count of babies who received antibiotics

    Formula: Count where:
        - Were antibiotics administered = "1" (Yes)
    """
    if df is None or df.empty:
        return 0

    # Filter by facilities if specified
    if facility_uids and not isinstance(facility_uids, list):
        facility_uids = [facility_uids]

    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count babies who received antibiotics
    antibiotics_cases = df[
        (df["dataElement_uid"] == ANTIBIOTICS_ADMINISTERED_UID)
        & df["value"].notna()
        & (df["value"] == ANTIBIOTICS_YES_VALUE)
    ]["tei_id"].nunique()

    return antibiotics_cases


def compute_culture_done_kpi(df, facility_uids=None, tei_df=None):
    """
    Compute Culture Done KPI for the given dataframe

    Formula: % Culture Done for Babies on Antibiotics =
             (Babies on antibiotics with blood culture done) ÷ (Total babies on antibiotics) × 100

    Returns:
        Dictionary with culture metrics
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "culture_rate": 0.0,
            "culture_count": 0,
            "antibiotics_count": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Count babies with culture done (numerator)
    culture_count = compute_culture_done_numerator(df, facility_uids)

    # Count babies on antibiotics (denominator)
    antibiotics_count = compute_antibiotics_denominator(df, facility_uids)

    # Calculate culture rate
    culture_rate = (
        (culture_count / antibiotics_count * 100) if antibiotics_count > 0 else 0.0
    )

    return {
        "culture_rate": float(culture_rate),
        "culture_count": int(culture_count),
        "antibiotics_count": int(antibiotics_count),
    }


def compute_culture_done_trend_data(
    df, period_col="period_display", facility_uids=None, tei_df=None
):
    """
    Compute Culture Done trend data by period

    Returns:
        DataFrame with columns: period_display, antibiotics_count, culture_count, culture_rate
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

    # Sort periods chronologically
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

    for period in periods:
        period_df = df[df[period_col] == period]

        # Compute KPI for this period
        period_kpi = compute_culture_done_kpi(period_df, facility_uids, tei_df)

        trend_data.append(
            {
                period_col: period,
                "antibiotics_count": period_kpi["antibiotics_count"],
                "culture_count": period_kpi["culture_count"],
                "culture_rate": period_kpi["culture_rate"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- CHART FUNCTIONS ----------------
def render_culture_done_trend_chart(
    df,
    period_col="period_display",
    title="Culture Done for Babies on Antibiotics",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
    tei_df=None,
):
    """
    Render trend chart for Culture Done KPI
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Compute trend data
    trend_df = compute_culture_done_trend_data(df, period_col, facility_uids, tei_df)

    if trend_df.empty:
        st.info("⚠️ No trend data available.")
        return

    # Sort periods chronologically
    if "period_sort" in df.columns:
        period_sort_mapping = df[["period_display", "period_sort"]].drop_duplicates()
        trend_df = trend_df.merge(period_sort_mapping, on="period_display", how="left")
        trend_df = trend_df.sort_values("period_sort")
        period_order = trend_df["period_display"].tolist()
    else:
        try:
            trend_df["period_datetime"] = pd.to_datetime(
                trend_df[period_col], errors="coerce"
            )
            trend_df = trend_df.sort_values("period_datetime")
            period_order = trend_df[period_col].tolist()
        except:
            period_order = sorted(trend_df[period_col].unique())

    # Create line chart
    fig = px.line(
        trend_df,
        x=period_col,
        y="culture_rate",
        markers=True,
        line_shape="linear",
        title=title,
        height=400,
        category_orders={period_col: period_order},
    )

    fig.update_traces(
        line=dict(width=3, color="#1f77b4"),
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="<b>%{x}</b><br>Culture Rate: %{y:.1f}%<br>Culture Done: %{customdata[0]}/%{customdata[1]}<extra></extra>",
        customdata=trend_df[["culture_count", "antibiotics_count"]].values,
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Culture Done Rate (%)",
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            range=[0, 105] if trend_df["culture_rate"].max() > 0 else [0, 100],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # Show trend indicator
    if len(trend_df) > 1:
        last_value = trend_df["culture_rate"].iloc[-1]
        prev_value = trend_df["culture_rate"].iloc[-2]
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
            f'<p style="font-size:1.2rem;font-weight:600;">Latest Value: {last_value:.1f}% <span class="{trend_class}">{trend_symbol}</span></p>',
            unsafe_allow_html=True,
        )

    # Show summary table
    st.subheader("📋 Culture Done Trend by Period")

    if not trend_df.empty:
        # Calculate overall totals
        overall_data = compute_culture_done_kpi(df, facility_uids, tei_df)

        # Prepare table data - CORRECT COLUMN ORDER: Culture Done first, then Total on Antibiotics
        table_data = trend_df.copy()

        # Remove period_sort if it exists to avoid issues
        if "period_sort" in table_data.columns:
            table_data = table_data.drop(columns=["period_sort"])

        # Rename columns with CORRECT ORDER
        table_data = table_data.rename(
            columns={
                period_col: "Period",
                "culture_count": "Culture Done",  # FIRST
                "antibiotics_count": "Total babies on Antibiotics",  # SECOND
                "culture_rate": "Culture Rate (%)",
            }
        )

        # ✅ FIX: Create Overall row with CORRECT COLUMN ORDER
        overall_row = {
            "Period": "Overall",
            "Culture Done": overall_data["culture_count"],  # FIRST
            "Total babies on Antibiotics": overall_data["antibiotics_count"],  # SECOND
            "Culture Rate (%)": overall_data["culture_rate"],
        }

        # Add overall row
        table_data = pd.concat(
            [table_data, pd.DataFrame([overall_row])], ignore_index=True
        )

        # ✅ FIX: Ensure CORRECT COLUMN ORDER in the final table
        column_order = [
            "Period",
            "Culture Done",
            "Total babies on Antibiotics",
            "Culture Rate (%)",
        ]
        table_data = table_data[column_order]

        # Add serial number
        table_data.insert(0, "No", range(1, len(table_data) + 1))

        # Format table
        styled_table = (
            table_data.style.format(
                {
                    "Culture Done": "{:,.0f}",
                    "Total babies on Antibiotics": "{:,.0f}",
                    "Culture Rate (%)": "{:.1f}%",
                }
            )
            .set_table_attributes('class="summary-table"')
            .hide(axis="index")
        )

        st.markdown(styled_table.to_html(), unsafe_allow_html=True)

        # Download button
        csv = table_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="culture_done_trend.csv",
            mime="text/csv",
        )
    else:
        st.info("⚠️ No data available for trend table.")


def render_culture_done_facility_comparison_chart(
    df,
    period_col="period_display",
    title="Culture Done for Babies on Antibiotics - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    tei_df=None,
):
    """
    Render facility comparison chart for Culture Done KPI
    """
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

    # Compute time series data for line chart
    time_series_data = []
    all_periods = sorted(df["period_display"].unique())

    for period_display in all_periods:
        period_df = filtered_df[filtered_df["period_display"] == period_display]

        for facility_uid, facility_name in facility_uid_to_name.items():
            facility_period_df = period_df[period_df["orgUnit"] == facility_uid]

            if not facility_period_df.empty:
                facility_kpi = compute_culture_done_kpi(
                    facility_period_df, [facility_uid], tei_df
                )

                time_series_data.append(
                    {
                        "period_display": period_display,
                        "Facility": facility_name,
                        "value": facility_kpi["culture_rate"],
                        "culture_count": facility_kpi["culture_count"],
                        "antibiotics_count": facility_kpi["antibiotics_count"],
                    }
                )

    if not time_series_data:
        st.info("⚠️ No time series data available for facility comparison.")
        return

    time_series_df = pd.DataFrame(time_series_data)

    # Create line chart
    fig = px.line(
        time_series_df,
        x="period_display",
        y="value",
        color="Facility",
        markers=True,
        title=title,
        height=500,
        category_orders={"period_display": all_periods},
    )

    # Update traces
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<br>Culture Done: %{customdata[0]}/%{customdata[1]}<extra></extra>",
        customdata=time_series_df[["culture_count", "antibiotics_count"]].values,
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Culture Done Rate (%)",
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            range=[0, 105] if time_series_df["value"].max() > 0 else [0, 100],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        showlegend=True,
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # Facility comparison table
    st.subheader("📋 Facility Comparison Summary")

    table_data = []
    for facility_uid, facility_name in facility_uid_to_name.items():
        facility_df = filtered_df[filtered_df["orgUnit"] == facility_uid]

        if not facility_df.empty:
            culture_data = compute_culture_done_kpi(facility_df, [facility_uid], tei_df)

            table_data.append(
                {
                    "Facility Name": facility_name,
                    # ✅ CORRECT ORDER: Culture Done first, then Total on Antibiotics
                    "Culture Done": culture_data["culture_count"],
                    "Total babies on Antibiotics": culture_data["antibiotics_count"],
                    "Culture Rate (%)": culture_data["culture_rate"],
                }
            )

    if not table_data:
        st.info("⚠️ No facility summary data available.")
        return

    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values("Culture Rate (%)", ascending=False)

    # Calculate overall totals
    overall_data = compute_culture_done_kpi(filtered_df, facility_uids, tei_df)

    # ✅ FIX: Create Overall row with CORRECT ORDER
    overall_row = {
        "Facility Name": "Overall",
        "Culture Done": overall_data["culture_count"],  # FIRST
        "Total babies on Antibiotics": overall_data["antibiotics_count"],  # SECOND
        "Culture Rate (%)": overall_data["culture_rate"],
    }

    table_df = pd.concat([table_df, pd.DataFrame([overall_row])], ignore_index=True)

    # ✅ FIX: Ensure CORRECT COLUMN ORDER
    column_order = [
        "Facility Name",
        "Culture Done",
        "Total babies on Antibiotics",
        "Culture Rate (%)",
    ]
    table_df = table_df[column_order]

    table_df.insert(0, "No", range(1, len(table_df) + 1))

    # Format table
    styled_table = (
        table_df.style.format(
            {
                "Culture Done": "{:,.0f}",
                "Total babies on Antibiotics": "{:,.0f}",
                "Culture Rate (%)": "{:.1f}%",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="culture_done_facility_comparison.csv",
        mime="text/csv",
    )


def render_culture_done_region_comparison_chart(
    df,
    period_col="period_display",
    title="Culture Done for Babies on Antibiotics - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    tei_df=None,
):
    """
    Render region comparison chart for Culture Done KPI
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if not region_names or not facilities_by_region:
        st.info("⚠️ No regions selected for comparison.")
        return

    # Compute time series data for line chart
    time_series_data = []
    all_periods = sorted(df["period_display"].unique())

    for period_display in all_periods:
        period_df = df[df["period_display"] == period_display]

        for region_name in region_names:
            region_facility_uids = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

            if region_facility_uids:
                region_period_df = period_df[
                    period_df["orgUnit"].isin(region_facility_uids)
                ]

                if not region_period_df.empty:
                    region_kpi = compute_culture_done_kpi(
                        region_period_df, region_facility_uids, tei_df
                    )

                    time_series_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": region_kpi["culture_rate"],
                            "culture_count": region_kpi["culture_count"],
                            "antibiotics_count": region_kpi["antibiotics_count"],
                        }
                    )

    if not time_series_data:
        st.info("⚠️ No time series data available for region comparison.")
        return

    time_series_df = pd.DataFrame(time_series_data)

    # Create line chart
    fig = px.line(
        time_series_df,
        x="period_display",
        y="value",
        color="Region",
        markers=True,
        title=title,
        height=500,
        category_orders={"period_display": all_periods},
    )

    # Update traces
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{data.name}: %{y:.1f}%<br>Culture Done: %{customdata[0]}/%{customdata[1]}<extra></extra>",
        customdata=time_series_df[["culture_count", "antibiotics_count"]].values,
    )

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Culture Done Rate (%)",
        xaxis=dict(
            type="category",
            tickangle=-45,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            range=[0, 105] if time_series_df["value"].max() > 0 else [0, 100],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            zerolinecolor="rgba(128,128,128,0.5)",
        ),
        showlegend=True,
    )

    fig.update_layout(yaxis_tickformat=".1f")
    st.plotly_chart(fig, use_container_width=True)

    # Region comparison table
    st.subheader("📋 Region Comparison Summary")

    table_data = []
    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]

        if region_facility_uids:
            region_df = df[df["orgUnit"].isin(region_facility_uids)]

            if not region_df.empty:
                culture_data = compute_culture_done_kpi(
                    region_df, region_facility_uids, tei_df
                )

                table_data.append(
                    {
                        "Region Name": region_name,
                        # ✅ CORRECT ORDER: Culture Done first, then Total on Antibiotics
                        "Culture Done": culture_data["culture_count"],
                        "Total babies on Antibiotics": culture_data[
                            "antibiotics_count"
                        ],
                        "Culture Rate (%)": culture_data["culture_rate"],
                    }
                )

    if not table_data:
        st.info("⚠️ No region summary data available.")
        return

    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values("Culture Rate (%)", ascending=False)

    # Calculate overall totals
    all_region_facility_uids = []
    for region_name in region_names:
        region_facility_uids = [
            uid for _, uid in facilities_by_region.get(region_name, [])
        ]
        all_region_facility_uids.extend(region_facility_uids)

    overall_df = df[df["orgUnit"].isin(all_region_facility_uids)]
    overall_data = compute_culture_done_kpi(
        overall_df, all_region_facility_uids, tei_df
    )

    # ✅ FIX: Create Overall row with CORRECT ORDER
    overall_row = {
        "Region Name": "Overall",
        "Culture Done": overall_data["culture_count"],  # FIRST
        "Total babies on Antibiotics": overall_data["antibiotics_count"],  # SECOND
        "Culture Rate (%)": overall_data["culture_rate"],
    }

    table_df = pd.concat([table_df, pd.DataFrame([overall_row])], ignore_index=True)

    # ✅ FIX: Ensure CORRECT COLUMN ORDER
    column_order = [
        "Region Name",
        "Culture Done",
        "Total babies on Antibiotics",
        "Culture Rate (%)",
    ]
    table_df = table_df[column_order]

    table_df.insert(0, "No", range(1, len(table_df) + 1))

    # Format table
    styled_table = (
        table_df.style.format(
            {
                "Culture Done": "{:,.0f}",
                "Total babies on Antibiotics": "{:,.0f}",
                "Culture Rate (%)": "{:.1f}%",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="culture_done_region_comparison.csv",
        mime="text/csv",
    )


# ---------------- COMPREHENSIVE SUMMARY ----------------
def render_culture_done_comprehensive_summary(
    df,
    title="Culture Done for Babies on Antibiotics - Summary",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    tei_df=None,
):
    """
    Render a comprehensive summary for Culture Done KPI
    """
    if text_color is None:
        text_color = auto_text_color(bg_color)

    st.subheader(title)

    if df is None or df.empty:
        st.info("⚠️ No data available for summary.")
        return

    # Compute KPI
    culture_data = compute_culture_done_kpi(df, facility_uids, tei_df)

    # Create summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Culture Done Rate",
            value=f"{culture_data['culture_rate']:.1f}%",
            help=f"Percentage of babies on antibiotics who had blood culture done",
        )

    with col2:
        # ✅ CORRECT ORDER: Culture Done Cases first
        st.metric(
            label="Culture Done Cases",
            value=f"{culture_data['culture_count']:,}",
            help=f"Number of babies on antibiotics with blood culture done",
        )

    with col3:
        # ✅ CORRECT ORDER: Total on Antibiotics second
        st.metric(
            label="Total babies on Antibiotics",
            value=f"{culture_data['antibiotics_count']:,}",
            help=f"Total number of babies who received antibiotics",
        )

    # Create detailed summary table
    st.subheader("📊 Detailed Summary")

    # ✅ FIX: Create summary table with CORRECT ORDER
    summary_data = {
        "Metric": [
            "Culture Done Rate",
            "Culture Done Cases",  # FIRST
            "Total babies on Antibiotics",  # SECOND
        ],
        "Value": [
            f"{culture_data['culture_rate']:.1f}%",
            f"{culture_data['culture_count']:,}",  # FIRST
            f"{culture_data['antibiotics_count']:,}",  # SECOND
        ],
        "Description": [
            "Percentage of babies on antibiotics who had blood culture done",
            "Number of babies on antibiotics with blood culture done",
            "Total number of babies who received antibiotics",
        ],
    }

    summary_df = pd.DataFrame(summary_data)

    styled_summary = summary_df.style.set_table_attributes(
        'class="summary-table"'
    ).hide(axis="index")

    st.markdown(styled_summary.to_html(), unsafe_allow_html=True)

    # Add culture result breakdown if available
    st.subheader("📊 Culture Result Breakdown")

    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df_filtered = df[df["orgUnit"].isin(facility_uids)]
    else:
        df_filtered = df

    # Get all culture events
    culture_events = df_filtered[df_filtered["dataElement_uid"] == BLOOD_CULTURE_UID]

    if not culture_events.empty:
        culture_counts = culture_events["value"].value_counts()

        # Map codes to labels
        result_labels = {
            "1": "Done - Culture Negative",
            "2": "Done - Culture Positive",
            "3": "Done but Unknown Result",
        }

        result_data = []
        total_culture_cases = 0

        for code, label in result_labels.items():
            count = culture_counts.get(code, 0)
            total_culture_cases += count
            if count > 0:
                result_data.append(
                    {
                        "Result": label,
                        "Count": count,
                        "Percentage": (
                            (count / len(culture_events) * 100)
                            if len(culture_events) > 0
                            else 0
                        ),
                    }
                )

        if result_data:
            result_df = pd.DataFrame(result_data)
            result_df = result_df.sort_values("Count", ascending=False)

            # Add total row
            total_row = {
                "Result": "Total Culture Done",
                "Count": total_culture_cases,
                "Percentage": 100.0,
            }
            result_df = pd.concat(
                [result_df, pd.DataFrame([total_row])], ignore_index=True
            )

            styled_results = (
                result_df.style.format({"Count": "{:,.0f}", "Percentage": "{:.1f}%"})
                .set_table_attributes('class="summary-table"')
                .hide(axis="index")
            )

            st.markdown(styled_results.to_html(), unsafe_allow_html=True)
        else:
            st.info("No culture results available in the selected data.")
    else:
        st.info("No culture data available for breakdown.")


# ---------------- MAIN EXPORT ----------------
__all__ = [
    # Computation functions
    "compute_culture_done_numerator",
    "compute_antibiotics_denominator",
    "compute_culture_done_kpi",
    "compute_culture_done_trend_data",
    # Chart rendering functions
    "render_culture_done_trend_chart",
    "render_culture_done_facility_comparison_chart",
    "render_culture_done_region_comparison_chart",
    "render_culture_done_comprehensive_summary",
    # Constants
    "BLOOD_CULTURE_UID",
    "ANTIBIOTICS_ADMINISTERED_UID",
    "CULTURE_DONE_VALUES",
    "ANTIBIOTICS_YES_VALUE",
]
