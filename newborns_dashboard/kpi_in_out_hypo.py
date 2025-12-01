import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# Import shared utilities
from utils.kpi_utils import auto_text_color

# ---------------- Import EXACT Inborn KPI Functions ----------------
from newborns_dashboard.kpi_inborn import (
    compute_inborn_numerator,
    compute_inborn_trend_data,
)

# ---------------- KPI Constants ----------------
# Temperature on admission
TEMPERATURE_ON_ADMISSION_UID = "gZi9y12E9i7"  # Temperature on admission (°C)
HYPOTHERMIA_THRESHOLD = 36.5  # in degree Celsius

# Birth location
BIRTH_LOCATION_UID = "aK5txmRYpVX"  # birth location
INBORN_CODE = "1"  # inborn code value
OUTBORN_CODE = "2"  # outborn code value


# ---------------- Computation Functions ----------------
def compute_hypothermia_for_birth_location(
    df, facility_uids=None, birth_location_code=INBORN_CODE
):
    """
    Compute hypothermia cases for specific birth location (inborn or outborn)
    """
    if df is None or df.empty:
        return {"hypothermia_count": 0, "total_count": 0, "hypothermia_rate": 0.0}

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Step 1: Get all newborns with the specified birth location
    birth_location_mask = (df["dataElement_uid"] == BIRTH_LOCATION_UID) & (
        df["value"] == birth_location_code
    )
    birth_location_events = df[birth_location_mask]

    if birth_location_events.empty:
        return {"hypothermia_count": 0, "total_count": 0, "hypothermia_rate": 0.0}

    # Get unique TEIs with this birth location
    birth_location_teis = set(birth_location_events["tei_id"].unique())
    total_count = len(birth_location_teis)

    if total_count == 0:
        return {"hypothermia_count": 0, "total_count": 0, "hypothermia_rate": 0.0}

    # Step 2: Get temperature readings for these TEIs
    temp_events = df[
        (df["dataElement_uid"] == TEMPERATURE_ON_ADMISSION_UID)
        & df["value"].notna()
        & df["tei_id"].isin(birth_location_teis)
    ].copy()

    if temp_events.empty:
        return {
            "hypothermia_count": 0,
            "total_count": total_count,
            "hypothermia_rate": 0.0,
        }

    # Convert temperature values to numeric
    temp_events["temp_value"] = pd.to_numeric(temp_events["value"], errors="coerce")

    # Step 3: Count hypothermia cases among these TEIs
    hypothermia_teis = set(
        temp_events[temp_events["temp_value"] < HYPOTHERMIA_THRESHOLD][
            "tei_id"
        ].unique()
    )
    hypothermia_count = len(hypothermia_teis)

    # Calculate rate
    hypothermia_rate = (
        (hypothermia_count / total_count * 100) if total_count > 0 else 0.0
    )

    return {
        "hypothermia_count": hypothermia_count,
        "total_count": total_count,
        "hypothermia_rate": float(hypothermia_rate),
    }


def compute_inborn_outborn_hypothermia_kpi(df, facility_uids=None, tei_df=None):
    """
    Compute hypothermia rates separately for inborn and outborn babies
    USES EXACT SAME INBORN NUMERATOR as kpi_inborn module
    """
    if df is None or df.empty:
        return {
            "inborn_hypothermia": {
                "hypothermia_count": 0,
                "total_inborn": 0,
                "hypothermia_rate": 0.0,
            },
            "outborn_hypothermia": {
                "hypothermia_count": 0,
                "total_outborn": 0,
                "hypothermia_rate": 0.0,
            },
            "total_admitted": 0,
        }

    # Filter by facilities if specified
    if facility_uids:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]
        df = df[df["orgUnit"].isin(facility_uids)]

    # Get total unique TEIs
    total_admitted = df["tei_id"].nunique()

    # ✅ USE EXACT SAME INBORN NUMERATOR as kpi_inborn module
    inborn_count = compute_inborn_numerator(df, facility_uids)

    # Calculate outborn count
    # Get all birth location events
    birth_location_events = df[df["dataElement_uid"] == BIRTH_LOCATION_UID].copy()

    if not birth_location_events.empty:
        # Count unique outborn TEIs
        outborn_mask = birth_location_events["value"] == OUTBORN_CODE
        outborn_count = birth_location_events[outborn_mask]["tei_id"].nunique()
    else:
        outborn_count = 0

    # Compute hypothermia for inborn
    inborn_hypothermia = compute_hypothermia_for_birth_location(
        df, facility_uids, INBORN_CODE
    )

    # Compute hypothermia for outborn
    outborn_hypothermia = compute_hypothermia_for_birth_location(
        df, facility_uids, OUTBORN_CODE
    )

    return {
        "inborn_hypothermia": {
            "hypothermia_count": inborn_hypothermia["hypothermia_count"],
            "total_inborn": inborn_count,  # ✅ Uses EXACT same numerator as inborn KPI
            "hypothermia_rate": inborn_hypothermia["hypothermia_rate"],
        },
        "outborn_hypothermia": {
            "hypothermia_count": outborn_hypothermia["hypothermia_count"],
            "total_outborn": outborn_count,
            "hypothermia_rate": outborn_hypothermia["hypothermia_rate"],
        },
        "total_admitted": total_admitted,
    }


def compute_inborn_outborn_hypothermia_trend_data(
    df, period_col="period_display", facility_uids=None, tei_df=None
):
    """
    Compute trend data for inborn and outborn hypothermia rates by period
    USES EXACT SAME INBORN TREND DATA as kpi_inborn module
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # ✅ GET EXACT SAME INBORN TREND DATA as kpi_inborn module
    inborn_trend_df = compute_inborn_trend_data(df, period_col, facility_uids, tei_df)

    if inborn_trend_df.empty:
        return pd.DataFrame()

    trend_data = []

    for _, row in inborn_trend_df.iterrows():
        period = row[period_col]
        total_inborn = row[
            "inborn_count"
        ]  # ✅ This is the EXACT inborn count from the trend

        # Filter data for this period
        period_df = df[df[period_col] == period]

        # Compute hypothermia for inborn in this period
        inborn_hypothermia = compute_hypothermia_for_birth_location(
            period_df, facility_uids, INBORN_CODE
        )

        # Compute hypothermia for outborn in this period
        outborn_hypothermia = compute_hypothermia_for_birth_location(
            period_df, facility_uids, OUTBORN_CODE
        )

        trend_data.append(
            {
                period_col: period,
                "inborn_hypothermia_count": inborn_hypothermia["hypothermia_count"],
                "total_inborn": total_inborn,  # ✅ EXACT same as inborn KPI trend
                "inborn_hypothermia_rate": inborn_hypothermia["hypothermia_rate"],
                "outborn_hypothermia_count": outborn_hypothermia["hypothermia_count"],
                "total_outborn": outborn_hypothermia["total_count"],
                "outborn_hypothermia_rate": outborn_hypothermia["hypothermia_rate"],
                "total_admitted": row["total_admitted_newborns"],
            }
        )

    return pd.DataFrame(trend_data)


# ---------------- Chart Functions ----------------
def render_inborn_outborn_hypothermia_trend_chart(
    df,
    period_col="period_display",
    title="Hypothermia at Admission by Birth Location",
    bg_color="#FFFFFF",
    text_color=None,
    facility_uids=None,
    tei_df=None,
):
    """Render dual line chart for inborn and outborn hypothermia rates trend"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Compute trend data using EXACT same inborn trend
    trend_df = compute_inborn_outborn_hypothermia_trend_data(
        df, period_col, facility_uids, tei_df
    )

    if trend_df.empty:
        st.info("⚠️ No hypothermia data available by birth location.")
        return

    trend_df = trend_df.copy()

    # Ensure chronological ordering - same as inborn KPI
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

    # Create dual line chart
    fig = go.Figure()

    # Add inborn line
    fig.add_trace(
        go.Scatter(
            x=trend_df[period_col],
            y=trend_df["inborn_hypothermia_rate"],
            mode="lines+markers",
            name="Inborn Hypothermia Rate",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=7, color="#1f77b4"),
            hovertemplate="<b>Inborn</b><br>Period: %{x}<br>Rate: %{y:.2f}%<br>Cases: %{customdata[0]:.0f} of %{customdata[1]:.0f}<extra></extra>",
            customdata=trend_df[["inborn_hypothermia_count", "total_inborn"]].values,
        )
    )

    # Add outborn line
    fig.add_trace(
        go.Scatter(
            x=trend_df[period_col],
            y=trend_df["outborn_hypothermia_rate"],
            mode="lines+markers",
            name="Outborn Hypothermia Rate",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
            marker=dict(size=7, color="#ff7f0e"),
            hovertemplate="<b>Outborn</b><br>Period: %{x}<br>Rate: %{y:.2f}%<br>Cases: %{customdata[0]:.0f} of %{customdata[1]:.0f}<extra></extra>",
            customdata=trend_df[["outborn_hypothermia_count", "total_outborn"]].values,
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        height=500,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Hypothermia Rate (%)",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=period_order,
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("📋 Hypothermia by Birth Location - Trend Summary")
    summary_df = trend_df.copy().reset_index(drop=True)

    # Calculate overall rates
    total_inborn_hypothermia = summary_df["inborn_hypothermia_count"].sum()
    total_inborn = summary_df["total_inborn"].sum()
    overall_inborn_rate = (
        (total_inborn_hypothermia / total_inborn * 100) if total_inborn > 0 else 0
    )

    total_outborn_hypothermia = summary_df["outborn_hypothermia_count"].sum()
    total_outborn = summary_df["total_outborn"].sum()
    overall_outborn_rate = (
        (total_outborn_hypothermia / total_outborn * 100) if total_outborn > 0 else 0
    )

    # Create display dataframe
    display_df = summary_df[
        [
            period_col,
            "inborn_hypothermia_rate",
            "inborn_hypothermia_count",
            "total_inborn",
            "outborn_hypothermia_rate",
            "outborn_hypothermia_count",
            "total_outborn",
        ]
    ].copy()

    # Rename columns for display
    display_df = display_df.rename(
        columns={
            period_col: "Period",
            "inborn_hypothermia_rate": "Inborn Hypothermia Rate (%)",
            "inborn_hypothermia_count": "Inborn Hypothermia Cases",
            "total_inborn": "Total Inborn Babies",
            "outborn_hypothermia_rate": "Outborn Hypothermia Rate (%)",
            "outborn_hypothermia_count": "Outborn Hypothermia Cases",
            "total_outborn": "Total Outborn Babies",
        }
    )

    # Add overall row
    overall_row = pd.DataFrame(
        {
            "Period": ["Overall"],
            "Inborn Hypothermia Rate (%)": [overall_inborn_rate],
            "Inborn Hypothermia Cases": [total_inborn_hypothermia],
            "Total Inborn Babies": [total_inborn],
            "Outborn Hypothermia Rate (%)": [overall_outborn_rate],
            "Outborn Hypothermia Cases": [total_outborn_hypothermia],
            "Total Outborn Babies": [total_outborn],
        }
    )

    display_df = pd.concat([display_df, overall_row], ignore_index=True)
    display_df.insert(0, "No", range(1, len(display_df) + 1))

    # Format table
    styled_table = (
        display_df.style.format(
            {
                "Inborn Hypothermia Rate (%)": "{:.1f}%",
                "Outborn Hypothermia Rate (%)": "{:.1f}%",
                "Inborn Hypothermia Cases": "{:,.0f}",
                "Outborn Hypothermia Cases": "{:,.0f}",
                "Total Inborn Babies": "{:,.0f}",
                "Total Outborn Babies": "{:,.0f}",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hypothermia_by_birth_location_trend.csv",
        mime="text/csv",
    )


def render_inborn_outborn_hypothermia_comparison_chart(
    df,
    period_col="period_display",
    title="Hypothermia at Admission Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    comparison_type="facility",  # 'facility' or 'region'
    tei_df=None,
):
    """Render comparison chart for inborn/outborn hypothermia rates"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if comparison_type == "facility":
        if (
            not facility_names
            or not facility_uids
            or len(facility_names) != len(facility_uids)
        ):
            st.info("⚠️ No facilities selected for comparison.")
            return

        # Create mapping
        facility_uid_to_name = dict(zip(facility_uids, facility_names))

        # Compute data for each facility
        comparison_data = []

        for facility_uid in facility_uids:
            facility_df = df[df["orgUnit"] == facility_uid]

            if not facility_df.empty:
                # ✅ Use EXACT same computation as inborn KPI
                kpi_data = compute_inborn_outborn_hypothermia_kpi(
                    facility_df, [facility_uid], tei_df
                )

                comparison_data.append(
                    {
                        "Name": facility_uid_to_name[facility_uid],
                        "Type": comparison_type.capitalize(),
                        "Inborn Hypothermia Rate": kpi_data["inborn_hypothermia"][
                            "hypothermia_rate"
                        ],
                        "Inborn Hypothermia Cases": kpi_data["inborn_hypothermia"][
                            "hypothermia_count"
                        ],
                        "Total Inborn": kpi_data["inborn_hypothermia"]["total_inborn"],
                        "Outborn Hypothermia Rate": kpi_data["outborn_hypothermia"][
                            "hypothermia_rate"
                        ],
                        "Outborn Hypothermia Cases": kpi_data["outborn_hypothermia"][
                            "hypothermia_count"
                        ],
                        "Total Outborn": kpi_data["outborn_hypothermia"][
                            "total_outborn"
                        ],
                    }
                )

        if not comparison_data:
            st.info("⚠️ No data available for comparison.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        # Create grouped bar chart
        fig = go.Figure()

        # Add inborn bars
        fig.add_trace(
            go.Bar(
                x=comparison_df["Name"],
                y=comparison_df["Inborn Hypothermia Rate"],
                name="Inborn Hypothermia Rate",
                marker_color="#1f77b4",
                hovertemplate="<b>%{x} - Inborn</b><br>Rate: %{y:.2f}%<br>Cases: %{customdata[0]:.0f} of %{customdata[1]:.0f}<extra></extra>",
                customdata=comparison_df[
                    ["Inborn Hypothermia Cases", "Total Inborn"]
                ].values,
            )
        )

        # Add outborn bars
        fig.add_trace(
            go.Bar(
                x=comparison_df["Name"],
                y=comparison_df["Outborn Hypothermia Rate"],
                name="Outborn Hypothermia Rate",
                marker_color="#ff7f0e",
                hovertemplate="<b>%{x} - Outborn</b><br>Rate: %{y:.2f}%<br>Cases: %{customdata[0]:.0f} of %{customdata[1]:.0f}<extra></extra>",
                customdata=comparison_df[
                    ["Outborn Hypothermia Cases", "Total Outborn"]
                ].values,
            )
        )

        # Calculate overall
        total_inborn_hypothermia = comparison_df["Inborn Hypothermia Cases"].sum()
        total_inborn = comparison_df["Total Inborn"].sum()
        overall_inborn_rate = (
            (total_inborn_hypothermia / total_inborn * 100) if total_inborn > 0 else 0
        )

        total_outborn_hypothermia = comparison_df["Outborn Hypothermia Cases"].sum()
        total_outborn = comparison_df["Total Outborn"].sum()
        overall_outborn_rate = (
            (total_outborn_hypothermia / total_outborn * 100)
            if total_outborn > 0
            else 0
        )

    else:  # region comparison
        # This would be implemented similarly with region-based aggregation
        st.info("⚠️ Region comparison not yet implemented.")
        return

    # Update layout for grouped bar chart
    fig.update_layout(
        title=title,
        barmode="group",
        height=500,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title=comparison_type.capitalize(),
        yaxis_title="Hypothermia Rate (%)",
        xaxis=dict(
            tickangle=-45 if len(comparison_df) > 5 else 0,
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.update_layout(yaxis_tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader(
        f"📋 Hypothermia by Birth Location - {comparison_type.capitalize()} Comparison"
    )

    # Create display dataframe
    display_df = comparison_df[
        [
            "Name",
            "Inborn Hypothermia Rate",
            "Inborn Hypothermia Cases",
            "Total Inborn",
            "Outborn Hypothermia Rate",
            "Outborn Hypothermia Cases",
            "Total Outborn",
        ]
    ].copy()

    # Add overall row
    overall_row = pd.DataFrame(
        {
            "Name": ["Overall"],
            "Inborn Hypothermia Rate": [overall_inborn_rate],
            "Inborn Hypothermia Cases": [total_inborn_hypothermia],
            "Total Inborn": [total_inborn],
            "Outborn Hypothermia Rate": [overall_outborn_rate],
            "Outborn Hypothermia Cases": [total_outborn_hypothermia],
            "Total Outborn": [total_outborn],
        }
    )

    display_df = pd.concat([display_df, overall_row], ignore_index=True)
    display_df.insert(0, "No", range(1, len(display_df) + 1))

    # Format table
    styled_table = (
        display_df.style.format(
            {
                "Inborn Hypothermia Rate": "{:.1f}%",
                "Outborn Hypothermia Rate": "{:.1f}%",
                "Inborn Hypothermia Cases": "{:,.0f}",
                "Outborn Hypothermia Cases": "{:,.0f}",
                "Total Inborn": "{:,.0f}",
                "Total Outborn": "{:,.0f}",
            }
        )
        .set_table_attributes('class="summary-table"')
        .hide(axis="index")
    )

    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"hypothermia_by_birth_location_{comparison_type}_comparison.csv",
        mime="text/csv",
    )


def render_inborn_outborn_hypothermia_summary(df, facility_uids=None, tei_df=None):
    """Render summary metrics for inborn/outborn hypothermia rates"""
    if df is None or df.empty:
        st.info("⚠️ No data available for summary.")
        return

    # Compute KPI using EXACT same computation
    kpi_data = compute_inborn_outborn_hypothermia_kpi(df, facility_uids, tei_df)

    inborn_data = kpi_data["inborn_hypothermia"]
    outborn_data = kpi_data["outborn_hypothermia"]

    # Create metrics columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Admitted Newborns",
            value=f"{kpi_data['total_admitted']:,}",
            help="Total unique newborns in the selected period",
        )

    with col2:
        st.metric(
            label="Inborn Babies",
            value=f"{inborn_data['total_inborn']:,}",
            delta=f"{inborn_data['hypothermia_rate']:.1f}% hypothermia rate",
            delta_color="inverse",
            help=f"Newborns with birth location = 'Inborn' ({INBORN_CODE})",
        )

    with col3:
        st.metric(
            label="Outborn Babies",
            value=f"{outborn_data['total_outborn']:,}",
            delta=f"{outborn_data['hypothermia_rate']:.1f}% hypothermia rate",
            delta_color="inverse",
            help=f"Newborns with birth location = 'Outborn' ({OUTBORN_CODE})",
        )

    # Hypothermia metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Inborn Hypothermia Cases",
            value=f"{inborn_data['hypothermia_count']:,}",
            help="Inborn babies with temperature < 36.5°C on admission",
        )

    with col2:
        st.metric(
            label="Outborn Hypothermia Cases",
            value=f"{outborn_data['hypothermia_count']:,}",
            help="Outborn babies with temperature < 36.5°C on admission",
        )

    with col3:
        # Calculate overall hypothermia rate
        total_hypothermia = (
            inborn_data["hypothermia_count"] + outborn_data["hypothermia_count"]
        )
        total_with_loc = inborn_data["total_inborn"] + outborn_data["total_outborn"]
        overall_rate = (
            (total_hypothermia / total_with_loc * 100) if total_with_loc > 0 else 0
        )

        st.metric(
            label="Overall Hypothermia Rate",
            value=f"{overall_rate:.1f}%",
            help="Hypothermia rate across all babies with birth location recorded",
        )

    # Display the formulas
    st.info(
        f"""
    **Formulas:**
    - **% Hypothermic at admission (inborn)** = (Temperature < 36.5°C for babies with BirthLocation = 'Inborn') ÷ (Total babies with BirthLocation = 'Inborn') × 100
    - **% Hypothermic at admission (outborn)** = (Temperature < 36.5°C for babies with BirthLocation = 'Outborn') ÷ (Total babies with BirthLocation = 'Outborn') × 100
    
    **Note:** These rates use the EXACT same inborn count computation as the 'Inborn Babies (%)' KPI.
    Babies without birth location data are excluded from these calculations.
    """
    )
