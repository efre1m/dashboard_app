import pandas as pd
import streamlit as st
from utils.time_filter import get_date_range, assign_period, get_available_aggregations
import datetime
import logging

from newborns_dashboard.kpi_utils_newborn import (
    compute_newborn_kpis,
    auto_text_color,
    prepare_data_for_newborn_trend_chart,
    extract_period_columns_newborn,
    get_relevant_date_column_for_newborn_kpi,
    get_numerator_denominator_for_newborn_kpi,
    render_newborn_trend_chart,
    render_newborn_facility_comparison_chart,
    render_newborn_region_comparison_chart,
    render_admitted_newborns_trend_chart,
    render_admitted_newborns_facility_comparison_chart,
    render_admitted_newborns_region_comparison_chart,
)

# KPI mapping for newborn comparison charts - UPDATED WITH ADMITTED NEWBORNS
NEWBORN_KPI_MAPPING = {
    "Inborn Rate (%)": {
        "title": "Inborn Babies (%)",
        "numerator_name": "Inborn Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Outborn Rate (%)": {
        "title": "Outborn Babies (%)",
        "numerator_name": "Outborn Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Hypothermia on Admission Rate (%)": {
        "title": "Hypothermia on Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Hypothermia After Admission Rate (%)": {
        "title": "Hypothermia After Admission (%)",
        "numerator_name": "Hypothermia Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    "Neonatal Mortality Rate (%)": {
        "title": "Neonatal Mortality Rate (%)",
        "numerator_name": "Dead Cases",
        "denominator_name": "Total Admitted Newborns",
    },
    # ADD THESE TWO NEW KPIs
    "Inborn Hypothermia Rate (%)": {
        "title": "Hypothermia in Inborn Babies (%)",
        "numerator_name": "Inborn Hypothermia Cases",
        "denominator_name": "Total Inborn Babies",
    },
    "Outborn Hypothermia Rate (%)": {
        "title": "Hypothermia in Outborn Babies (%)",
        "numerator_name": "Outborn Hypothermia Cases",
        "denominator_name": "Total Outborn Babies",
    },
    # ADD ADMITTED NEWBORNS KPI
    "Admitted Newborns": {
        "title": "Total Admitted Newborns",
        "value_name": "Admitted Newborns",
    },
}

# KPI options for newborn dashboard
NEWBORN_KPI_OPTIONS = [
    "Inborn Rate (%)",
    "Outborn Rate (%)",
    "Hypothermia on Admission Rate (%)",
    "Hypothermia After Admission Rate (%)",
    "Neonatal Mortality Rate (%)",
    # ADD THESE TWO
    "Inborn Hypothermia Rate (%)",
    "Outborn Hypothermia Rate (%)",
    # ADD ADMITTED NEWBORNS
    "Admitted Newborns",
]

# KPI Groups for Tab Navigation - UPDATED WITH ADMITTED NEWBORNS
NEWBORN_KPI_GROUPS = {
    "👶 Birth Location": [
        "Inborn Rate (%)",
        "Outborn Rate (%)",
    ],
    "🌡️ Hypothermia": [
        "Hypothermia on Admission Rate (%)",
        "Hypothermia After Admission Rate (%)",
        # ADD THESE TWO INSTEAD OF COMPARISON
        "Inborn Hypothermia Rate (%)",
        "Outborn Hypothermia Rate (%)",
    ],
    "📊 Outcomes": [
        "Neonatal Mortality Rate (%)",
    ],
    # ADD NEW ENROLLMENT TAB
    "📈 Enrollment": [
        "Admitted Newborns",
    ],
}

# KPI Column Requirements - What each newborn KPI actually needs
NEWBORN_KPI_COLUMN_REQUIREMENTS = {
    "Inborn Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_location_admission_information",
        "temperature_on_admission_degc_observations_and_nursing_care_1",
        "event_date_admission_information",
    ],
    "Outborn Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_location_admission_information",
        "temperature_on_admission_degc_observations_and_nursing_care_1",
        "event_date_admission_information",
    ],
    "Hypothermia on Admission Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "temperature_on_admission_degc_observations_and_nursing_care_1",
        "event_date_observations_and_nursing_care_1",
    ],
    "Hypothermia After Admission Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "lowest_recorded_temperature_celsius_observations_and_nursing_care_2",
        "event_date_observations_and_nursing_care_2",
    ],
    "Neonatal Mortality Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "newborn_status_at_discharge_discharge_and_final_diagnosis",
        "event_date_discharge_and_final_diagnosis",
    ],
    # ADD THESE TWO NEW KPIs
    "Inborn Hypothermia Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_location_admission_information",
        "temperature_on_admission_degc_observations_and_nursing_care_1",
        "event_date_admission_information",
    ],
    "Outborn Hypothermia Rate (%)": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
        "birth_location_admission_information",
        "temperature_on_admission_degc_observations_and_nursing_care_1",
        "event_date_admission_information",
    ],
    # ADD ADMITTED NEWBORNS COLUMNS
    "Admitted Newborns": [
        "orgUnit",
        "tei_id",
        "enrollment_date",
    ],
}


def get_newborn_kpi_filtered_dataframe(df, kpi_name):
    """
    Filter DataFrame to only include columns needed for the selected newborn KPI
    """
    if df is None or df.empty:
        return df

    if kpi_name not in NEWBORN_KPI_COLUMN_REQUIREMENTS:
        # If KPI not in mapping, return all columns
        return df.copy()

    # Get required columns for this KPI
    required_columns = NEWBORN_KPI_COLUMN_REQUIREMENTS[kpi_name]

    # Find which columns actually exist in the DataFrame
    available_columns = []
    for col in required_columns:
        if col in df.columns:
            available_columns.append(col)

    # Always include these essential columns if they exist
    essential_cols = ["orgUnit", "tei_id", "enrollment_date"]
    for col in essential_cols:
        if col in df.columns and col not in available_columns:
            available_columns.append(col)

    # If we have no columns after filtering, return original DataFrame
    if not available_columns:
        return df.copy()

    # Create filtered DataFrame
    filtered_df = df[available_columns].copy()

    # Log the optimization
    original_size = len(df.columns)
    filtered_size = len(filtered_df.columns)
    logging.info(f"✅ Newborn KPI Filter: {kpi_name}")
    logging.info(f"   Before: {original_size} columns")
    logging.info(f"   After: {filtered_size} columns")
    logging.info(
        f"   Reduction: {((original_size - filtered_size) / original_size * 100):.1f}%"
    )

    return filtered_df


def get_text_color(bg_color):
    """Get auto text color for background"""
    return auto_text_color(bg_color)


def get_newborn_kpi_config(kpi_selection):
    """Get newborn KPI configuration"""
    return NEWBORN_KPI_MAPPING.get(kpi_selection, {})


def render_newborn_kpi_tab_navigation():
    """Render professional tab navigation for Neonatal KPI selection - UPDATED WITH ADMITTED NEWBORNS"""

    st.markdown(
        """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #1f77b4 !important;
        color: white !important;
        border-color: #1a6790 !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background-color: #1668a1 !important;
        color: white !important;
        border-color: #145a8c !important;
    }
    
    div.stButton > button[kind="secondary"] {
        background-color: #f8f9fa !important;
        color: #495057 !important;
        border-color: #dee2e6 !important;
        font-weight: 500 !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e9ecef !important;
        color: #495057 !important;
        border-color: #ced4da !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # KPI Grouping for Tab Navigation - UPDATED WITH ADMITTED NEWBORNS
    NEWBORN_KPI_GROUPS = {
        "👶 Birth Location": [
            "Inborn Rate (%)",
            "Outborn Rate (%)",
        ],
        "🌡️ Hypothermia": [
            "Hypothermia on Admission Rate (%)",
            "Hypothermia After Admission Rate (%)",
            "Inborn Hypothermia Rate (%)",
            "Outborn Hypothermia Rate (%)",
        ],
        "📊 Outcomes": [
            "Neonatal Mortality Rate (%)",
        ],
        "📈 Enrollment": [
            "Admitted Newborns",
        ],
    }

    # Initialize session state for newborn KPI selection
    if "selected_newborn_kpi" not in st.session_state:
        st.session_state.selected_newborn_kpi = "Inborn Rate (%)"

    # Create main KPI group tabs - UPDATED WITH 4TH TAB
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👶 **Birth Location**",
            "🌡️ **Hypothermia**",
            "📊 **Outcomes**",
            "📈 **Enrollment**",
        ]
    )

    selected_kpi = st.session_state.selected_newborn_kpi

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📊 Inborn Rate",
                key="inborn_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Inborn Rate (%)" else "secondary"),
            ):
                selected_kpi = "Inborn Rate (%)"

        with col2:
            if st.button(
                "📊 Outborn Rate",
                key="outborn_btn",
                use_container_width=True,
                type=("primary" if selected_kpi == "Outborn Rate (%)" else "secondary"),
            ):
                selected_kpi = "Outborn Rate (%)"

    with tab2:
        # NOW WE HAVE 4 BUTTONS IN HYPOTHERMIA TAB
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            if st.button(
                "🌡️ Hypothermia on Admission",
                key="hypo_admission_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia on Admission Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia on Admission Rate (%)"

        with col2:
            if st.button(
                "🌡️ Hypothermia After Admission",
                key="hypo_after_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Hypothermia After Admission Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Hypothermia After Admission Rate (%)"

        with col3:
            if st.button(
                "🌡️ Inborn Hypothermia",
                key="inborn_hypo_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Inborn Hypothermia Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Inborn Hypothermia Rate (%)"

        with col4:
            if st.button(
                "🌡️ Outborn Hypothermia",
                key="outborn_hypo_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Outborn Hypothermia Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Outborn Hypothermia Rate (%)"

    with tab3:
        col1 = st.columns(1)[0]

        with col1:
            if st.button(
                "📊 Neonatal Mortality",
                key="nmr_btn",
                use_container_width=True,
                type=(
                    "primary"
                    if selected_kpi == "Neonatal Mortality Rate (%)"
                    else "secondary"
                ),
            ):
                selected_kpi = "Neonatal Mortality Rate (%)"

    with tab4:
        col1 = st.columns(1)[0]

        with col1:
            if st.button(
                "📈 Admitted Newborns",
                key="admitted_newborns_btn",
                use_container_width=True,
                type=(
                    "primary" if selected_kpi == "Admitted Newborns" else "secondary"
                ),
            ):
                selected_kpi = "Admitted Newborns"

    # Update session state with final selection
    if selected_kpi != st.session_state.selected_newborn_kpi:
        st.session_state.selected_newborn_kpi = selected_kpi
        st.rerun()

    return st.session_state.selected_newborn_kpi


def get_period_columns(df):
    """Get period column names from patient data"""
    period_col = None
    period_display_col = None
    period_sort_col = None

    # Look for period columns in the patient data
    for col in df.columns:
        if "period_display" in col:
            period_display_col = col
        elif "period_sort" in col:
            period_sort_col = col
        elif col == "period":
            period_col = col

    # Fallback to first column if needed
    if not period_col:
        period_col = "period" if "period" in df.columns else df.columns[0]

    return period_col, period_display_col, period_sort_col


def render_newborn_trend_chart_section(
    kpi_selection,
    patient_df,
    facility_uids,
    display_names,
    bg_color,
    text_color,
    comparison_mode="overall",
    facilities_by_region=None,
    region_names=None,
):
    """Render the trend chart for newborn KPIs with KPI-specific program stage dates"""

    # STANDARDIZE COLUMN NAMES
    if "orgUnit" not in patient_df.columns:
        for col in patient_df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                patient_df = patient_df.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in patient_df.columns:
        st.error("❌ 'orgUnit' column not found in data. Cannot filter by UIDs.")
        return

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_newborn_kpi_filtered_dataframe(patient_df, kpi_selection)

    # Get KPI configuration for labels
    kpi_config = get_newborn_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)
    value_name = kpi_config.get("value_name", "Value")

    if kpi_df.empty:
        st.info("⚠️ No data available for trend analysis.")
        return

    # Apply UID filter
    working_df = kpi_df.copy()
    if facility_uids and "orgUnit" in working_df.columns:
        working_df = working_df[working_df["orgUnit"].isin(facility_uids)].copy()

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_selection)

    # Get date range filters from session state
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    # CRITICAL FIX: Use prepare_data_for_newborn_trend_chart to get data filtered by KPI-specific dates AND date range
    prepared_df, date_column_used = prepare_data_for_newborn_trend_chart(
        working_df, kpi_selection, facility_uids, date_range_filters
    )

    if prepared_df.empty:
        st.warning(
            f"⚠️ No data available for {kpi_selection} using its specific program stage date: '{date_column_used}'"
        )
        return

    # Ensure period columns exist with proper formatting
    if "period_display" not in prepared_df.columns:
        if "event_date" in prepared_df.columns:
            prepared_df["period_display"] = (
                prepared_df["event_date"].dt.strftime("%b-%y").str.capitalize()
            )  # Format as "Sep-25"
        else:
            prepared_df["period_display"] = "Period"

    if "period_sort" not in prepared_df.columns:
        if "event_date" in prepared_df.columns:
            try:
                prepared_df["period_sort"] = pd.to_datetime(
                    prepared_df["event_date"]
                ).dt.strftime("%Y%m")
            except:
                prepared_df["period_sort"] = prepared_df.index
        else:
            prepared_df["period_sort"] = prepared_df.index

    # Get unique periods in order
    unique_periods = prepared_df[["period_display", "period_sort"]].drop_duplicates()
    unique_periods = unique_periods.sort_values("period_sort")

    # Create period data - SIMPLIFIED APPROACH
    period_data = []

    for _, row in unique_periods.iterrows():
        period_display = row["period_display"]
        period_sort = row["period_sort"]

        # Get data for this period from prepared_df (already filtered by KPI-specific dates)
        period_df = prepared_df[prepared_df["period_display"] == period_display]

        if not period_df.empty:
            # Get TEI IDs from this period (already filtered by KPI-specific dates)
            period_tei_ids = period_df["tei_id"].dropna().unique()

            if not period_tei_ids.size:
                continue

            # CRITICAL: Get the original data for these TEI IDs, but FILTERED by KPI-specific date
            # Use working_df (which already has facility filter applied)
            period_patient_data = working_df.copy()

            # Filter by TEI IDs AND KPI-specific date
            if date_column_used and date_column_used in period_patient_data.columns:
                period_patient_data[date_column_used] = pd.to_datetime(
                    period_patient_data[date_column_used], errors="coerce"
                )

                # Get TEI IDs that have the KPI-specific date
                tei_ids_with_date = period_patient_data[
                    period_patient_data[date_column_used].notna()
                ]["tei_id"].unique()

                # Intersection: TEI IDs that are in our period AND have KPI-specific date
                valid_tei_ids = set(period_tei_ids) & set(tei_ids_with_date)

                if valid_tei_ids:
                    period_patient_data = period_patient_data[
                        period_patient_data["tei_id"].isin(valid_tei_ids)
                        & period_patient_data[date_column_used].notna()
                    ].copy()
                else:
                    # No valid TEI IDs with KPI-specific date
                    period_data.append(
                        {
                            "period": period_display,
                            "period_display": period_display,
                            "period_sort": period_sort,
                            "value": 0,
                            "numerator": 0,
                            "denominator": 0,
                        }
                    )
                    continue
            else:
                # Fallback: just filter by TEI IDs
                period_patient_data = period_patient_data[
                    period_patient_data["tei_id"].isin(period_tei_ids)
                ].copy()

            # SPECIAL HANDLING FOR ADMITTED NEWBORNS
            if kpi_selection == "Admitted Newborns":
                # For Admitted Newborns, use the new function
                numerator, denominator, _ = get_numerator_denominator_for_newborn_kpi(
                    period_patient_data,
                    kpi_selection,
                    facility_uids,
                    date_range_filters,
                )

                # For Admitted Newborns, the value is the count (numerator)
                # NOT a percentage calculation
                value = float(numerator)  # Just use the count as value
            else:
                # Compute KPI using date-filtered data
                numerator, denominator, _ = get_numerator_denominator_for_newborn_kpi(
                    period_patient_data,
                    kpi_selection,
                    facility_uids,
                    date_range_filters,
                )
                # Calculate value as percentage
                value = (numerator / denominator * 100) if denominator > 0 else 0

            period_data.append(
                {
                    "period": period_display,
                    "period_display": period_display,
                    "period_sort": period_sort,
                    "value": value,
                    "numerator": int(numerator),
                    "denominator": int(denominator),
                }
            )

    if not period_data:
        st.info("⚠️ No period data available for chart.")
        return

    # Create DataFrame
    group = pd.DataFrame(period_data)
    group = group.sort_values("period_sort")

    # Render the chart WITH TABLE
    try:
        # SPECIAL HANDLING FOR ADMITTED NEWBORNS
        if kpi_selection == "Admitted Newborns":
            render_admitted_newborns_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                value_name,
                facility_uids,
            )
        else:
            # Standard trend chart for all other KPIs
            render_newborn_trend_chart(
                group,
                "period_display",
                "value",
                chart_title,
                bg_color,
                text_color,
                display_names,
                numerator_label,
                denominator_label,
                facility_uids,
            )
    except Exception as e:
        st.error(f"Error rendering chart for {kpi_selection}: {str(e)}")


def render_newborn_comparison_chart(
    kpi_selection,
    patient_df=None,
    comparison_mode="facility",
    display_names=None,
    facility_uids=None,
    facilities_by_region=None,
    region_names=None,
    bg_color="#FFFFFF",
    text_color=None,
    is_national=False,
    filtered_patients=None,
):
    # Get date range filters
    date_range_filters = {}
    if "filters" in st.session_state:
        date_range_filters = {
            "start_date": st.session_state.filters.get("start_date"),
            "end_date": st.session_state.filters.get("end_date"),
        }

    """Render comparison charts for both national and regional views WITH TABLES using KPI-specific dates"""

    df_to_use = filtered_patients if filtered_patients is not None else patient_df

    if df_to_use is None or df_to_use.empty:
        st.info("⚠️ No data available for comparison.")
        return

    # OPTIMIZATION: Filter DataFrame for this KPI
    kpi_df = get_newborn_kpi_filtered_dataframe(df_to_use, kpi_selection)
    df_to_use = kpi_df  # Use the filtered DataFrame

    kpi_config = get_newborn_kpi_config(kpi_selection)
    numerator_label = kpi_config.get("numerator_name", "Numerator")
    denominator_label = kpi_config.get("denominator_name", "Denominator")
    chart_title = kpi_config.get("title", kpi_selection)
    value_name = kpi_config.get("value_name", "Value")

    # STANDARDIZE COLUMN NAMES - Ensure we have orgUnit
    if "orgUnit" not in df_to_use.columns:
        # Try to find the facility ID column
        for col in df_to_use.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                df_to_use = df_to_use.rename(columns={col: "orgUnit"})
                break

    if "orgUnit" not in df_to_use.columns:
        st.error("❌ Facility identifier column not found. Cannot perform comparison.")
        return

    # Get the SPECIFIC date column for this KPI
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_selection)

    # CRITICAL FIX: Prepare data using prepare_data_for_newborn_trend_chart for EACH facility/region
    if comparison_mode == "facility":
        comparison_data = []

        # For each facility, prepare its own trend data
        for facility_uid, facility_name in zip(facility_uids, display_names):
            # Filter data for this specific facility
            facility_df = df_to_use[df_to_use["orgUnit"] == facility_uid].copy()

            if facility_df.empty:
                # Add zero entries for all periods
                comparison_data.append(
                    {
                        "period_display": "All Periods",
                        "orgUnit": facility_uid,
                        "orgUnit_name": facility_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Prepare facility-specific data with KPI-specific dates AND date range filters
            prepared_df, _ = prepare_data_for_newborn_trend_chart(
                facility_df, kpi_selection, [facility_uid], date_range_filters
            )

            if prepared_df.empty:
                # Add zero entry
                comparison_data.append(
                    {
                        "period_display": "All Periods",
                        "orgUnit": facility_uid,
                        "orgUnit_name": facility_name,
                        "value": 0,
                        "numerator": 0,
                        "denominator": 0,
                    }
                )
                continue

            # Group by period for this facility
            for period_display, period_group in prepared_df.groupby("period_display"):
                if not period_group.empty:
                    # Get TEI IDs from this period (already filtered by KPI-specific dates)
                    period_tei_ids = period_group["tei_id"].dropna().unique()

                    # Get data for these TEI IDs from the ORIGINAL facility data
                    # but ONLY if they have the KPI-specific date
                    if date_column and date_column in facility_df.columns:
                        period_data = facility_df.copy()
                        period_data[date_column] = pd.to_datetime(
                            period_data[date_column], errors="coerce"
                        )
                        # Filter by TEI IDs AND KPI-specific date
                        period_data = period_data[
                            period_data["tei_id"].isin(period_tei_ids)
                            & period_data[date_column].notna()
                        ].copy()
                    else:
                        period_data = facility_df[
                            facility_df["tei_id"].isin(period_tei_ids)
                        ].copy()

                    # SPECIAL HANDLING FOR ADMITTED NEWBORNS
                    if kpi_selection == "Admitted Newborns":
                        # For Admitted Newborns, we need to handle it differently
                        numerator, denominator, _ = (
                            get_numerator_denominator_for_newborn_kpi(
                                period_data,
                                kpi_selection,
                                [facility_uid],
                                date_range_filters,
                            )
                        )

                        # For Admitted Newborns, value is the count (numerator)
                        value = float(numerator)
                    else:
                        # For all other KPIs, use the standard function
                        numerator, denominator, _ = (
                            get_numerator_denominator_for_newborn_kpi(
                                period_data,
                                kpi_selection,
                                [facility_uid],
                                date_range_filters,
                            )
                        )

                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                    comparison_data.append(
                        {
                            "period_display": period_display,
                            "orgUnit": facility_uid,
                            "orgUnit_name": facility_name,
                            "value": value,
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        if not comparison_data:
            st.info("⚠️ No comparison data available.")
            return

        comparison_df = pd.DataFrame(comparison_data)

        # Ensure proper column names
        if "orgUnit_name" in comparison_df.columns:
            comparison_df = comparison_df.rename(columns={"orgUnit_name": "Facility"})

        # Call the appropriate chart function
        if kpi_selection == "Admitted Newborns":
            render_admitted_newborns_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                value_name=value_name,
            )
        else:
            # Standard facility comparison chart for all other KPIs
            render_newborn_facility_comparison_chart(
                df=comparison_df,
                period_col="period_display",
                value_col="value",
                title=f"{chart_title} - Facility Comparison",
                bg_color=bg_color,
                text_color=text_color,
                facility_names=display_names,
                facility_uids=facility_uids,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )

    elif comparison_mode == "region" and is_national:
        # Region comparison - similar fix needed
        region_data = []

        # Get facility UIDs for each region
        region_facility_mapping = {}
        for region_name in region_names:
            region_facility_mapping[region_name] = [
                uid for _, uid in facilities_by_region.get(region_name, [])
            ]

        for region_name in region_names:
            region_facility_uids = region_facility_mapping.get(region_name, [])
            if not region_facility_uids:
                continue

            # Get data for this region
            region_df = df_to_use[
                df_to_use["orgUnit"].isin(region_facility_uids)
            ].copy()

            if region_df.empty:
                continue

            # Prepare region-specific data with KPI-specific dates AND date range filters
            prepared_df, _ = prepare_data_for_newborn_trend_chart(
                region_df, kpi_selection, region_facility_uids, date_range_filters
            )

            if prepared_df.empty:
                continue

            # Group by period for this region
            for period_display, period_group in prepared_df.groupby("period_display"):
                if not period_group.empty:
                    # Get TEI IDs from this period (already filtered by KPI-specific dates)
                    period_tei_ids = period_group["tei_id"].dropna().unique()

                    # Get data for these TEI IDs
                    if date_column and date_column in region_df.columns:
                        period_data = region_df.copy()
                        period_data[date_column] = pd.to_datetime(
                            period_data[date_column], errors="coerce"
                        )
                        period_data = period_data[
                            period_data["tei_id"].isin(period_tei_ids)
                            & period_data[date_column].notna()
                        ].copy()
                    else:
                        period_data = region_df[
                            region_df["tei_id"].isin(period_tei_ids)
                        ].copy()

                    # SPECIAL HANDLING FOR ADMITTED NEWBORNS
                    if kpi_selection == "Admitted Newborns":
                        # For Admitted Newborns, we need to handle it differently
                        numerator, denominator, _ = (
                            get_numerator_denominator_for_newborn_kpi(
                                period_data,
                                kpi_selection,
                                region_facility_uids,
                                date_range_filters,
                            )
                        )

                        # For Admitted Newborns, value is the count (numerator)
                        value = float(numerator)
                    else:
                        # For all other KPIs, use the standard function
                        numerator, denominator, _ = (
                            get_numerator_denominator_for_newborn_kpi(
                                period_data,
                                kpi_selection,
                                region_facility_uids,
                                date_range_filters,
                            )
                        )

                        value = (
                            (numerator / denominator * 100) if denominator > 0 else 0
                        )

                    region_data.append(
                        {
                            "period_display": period_display,
                            "Region": region_name,
                            "value": value,
                            "numerator": int(numerator),
                            "denominator": int(denominator),
                        }
                    )

        if not region_data:
            st.info("⚠️ No comparison data available for regions.")
            return

        region_df = pd.DataFrame(region_data)

        # Call the appropriate region comparison function
        if kpi_selection == "Admitted Newborns":
            render_admitted_newborns_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=chart_title,
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                value_name=value_name,
            )
        else:
            # Standard region comparison chart for all other KPIs
            render_newborn_region_comparison_chart(
                df=region_df,
                period_col="period_display",
                value_col="value",
                title=f"{chart_title} - Region Comparison",
                bg_color=bg_color,
                text_color=text_color,
                region_names=region_names,
                region_mapping=facilities_by_region,
                facilities_by_region=facilities_by_region,
                numerator_name=numerator_label,
                denominator_name=denominator_label,
            )
    else:
        if comparison_mode == "region":
            st.info(
                "⚠️ Region comparison is only available in national view when region data is provided."
            )
        else:
            st.info("⚠️ Invalid comparison mode selected.")


def render_newborn_additional_analytics(
    kpi_selection, patient_df, facility_uids, bg_color, text_color
):
    """Render additional analytics charts for newborn KPIs"""
    # For now, no additional analytics
    # Could add things like birth weight distribution charts, etc.
    pass


def normalize_newborn_patient_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single datetime column 'event_date' exists for newborn patient data"""
    if df.empty:
        return df

    df = df.copy()

    # Get current KPI to use the right date column
    current_kpi = st.session_state.get("selected_newborn_kpi", "Inborn Rate (%)")

    # Get the SPECIFIC date column for this KPI
    kpi_date_column = get_relevant_date_column_for_newborn_kpi(current_kpi)

    # Try KPI-specific date column first
    if kpi_date_column and kpi_date_column in df.columns:
        df["event_date"] = pd.to_datetime(df[kpi_date_column], errors="coerce")
        logging.info(
            f"✅ normalize_newborn_patient_dates: Using KPI-specific '{kpi_date_column}'"
        )
    elif "combined_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["combined_date"], errors="coerce")
        logging.info("✅ normalize_newborn_patient_dates: Using combined_date")
    else:
        # Look for program stage event dates
        program_stage_date_columns = [
            col
            for col in df.columns
            if col.startswith("event_date_") or col == "event_date"
        ]

        for col in program_stage_date_columns:
            try:
                df["event_date"] = pd.to_datetime(df[col], errors="coerce")
                if not df["event_date"].isna().all():
                    logging.info(
                        f"✅ normalize_newborn_patient_dates: Using {col} for event_date"
                    )
                    break
            except:
                continue

    # If no program stage date found, try enrollment_date
    if (
        "event_date" not in df.columns or df["event_date"].isna().all()
    ) and "enrollment_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["enrollment_date"], errors="coerce")
        logging.info(
            "✅ normalize_newborn_patient_dates: Using enrollment_date for event_date"
        )

    # If still no date found, use current date
    if "event_date" not in df.columns or df["event_date"].isna().all():
        df["event_date"] = pd.Timestamp.now().normalize()
        logging.warning(
            "⚠️ normalize_newborn_patient_dates: No valid date column found, using current date"
        )

    return df


def render_newborn_patient_filter_controls(
    patient_df, container=None, context="newborn"
):
    """Simple filter controls for newborn patient data"""
    if container is None:
        container = st

    filters = {}

    # Generate unique key suffix
    key_suffix = f"_{context}"

    # Time Period options
    time_options = [
        "All Time",
        "Custom Range",
        "Today",
        "This Week",
        "Last Week",
        "This Month",
        "Last Month",
        "This Year",
        "Last Year",
    ]

    # Get current selection
    current_selection = st.session_state.get(f"quick_range{key_suffix}", "All Time")

    # Ensure the value is valid
    if current_selection not in time_options:
        current_selection = "All Time"

    filters["quick_range"] = container.selectbox(
        "📅 Time Period",
        time_options,
        index=time_options.index(current_selection),
        key=f"quick_range{key_suffix}",
    )

    # Get REAL VALID dates from patient dataframe
    min_date, max_date = _get_newborn_patient_date_range(patient_df)

    # Ensure min_date is not earlier than 2000
    if min_date < datetime.date(2000, 1, 1):
        min_date = datetime.date(2000, 1, 1)

    # Ensure max_date is not later than 2035
    if max_date > datetime.date(2035, 12, 31):
        max_date = datetime.date(2035, 12, 31)

    # Set dates based on selection
    if filters["quick_range"] == "All Time":
        filters["start_date"] = min_date
        filters["end_date"] = max_date

    elif filters["quick_range"] == "Custom Range":
        # For Custom Range, use CURRENT YEAR as default
        today = datetime.date.today()
        default_start = datetime.date(today.year, 1, 1)
        default_end = datetime.date(today.year, 12, 31)

        # Ensure defaults are within min/max bounds
        if default_start < min_date:
            default_start = min_date
        if default_end > max_date:
            default_end = max_date

        col1, col2 = container.columns(2)
        with col1:
            filters["start_date"] = col1.date_input(
                "Start Date",
                value=default_start,
                min_value=datetime.date(2000, 1, 1),
                max_value=max_date,
                key=f"start_date{key_suffix}",
            )
        with col2:
            filters["end_date"] = col2.date_input(
                "End Date",
                value=default_end,
                min_value=min_date,
                max_value=datetime.date(2035, 12, 31),
                key=f"end_date{key_suffix}",
            )
    else:
        # Use the FULL patient dataframe directly
        start_date, end_date = get_date_range(patient_df, filters["quick_range"])
        filters["start_date"] = start_date
        filters["end_date"] = end_date
        logging.info(
            f"📅 Filter dates set: {start_date} to {end_date} for '{filters['quick_range']}'"
        )

    # Aggregation Level
    available_aggregations = get_available_aggregations(
        filters["start_date"], filters["end_date"]
    )

    default_index = (
        available_aggregations.index("Monthly")
        if "Monthly" in available_aggregations
        else 0
    )

    # Get current value from session state if exists
    current_period_label = st.session_state.get("period_label", "Monthly")

    if current_period_label in available_aggregations:
        default_index = available_aggregations.index(current_period_label)

    filters["period_label"] = container.selectbox(
        "⏰ Aggregation Level",
        available_aggregations,
        index=default_index,
        key=f"period_label{key_suffix}",
    )

    # Save to session state
    st.session_state.period_label = filters["period_label"]

    # Also save to filters dictionary in session state
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters["period_label"] = filters["period_label"]
    st.session_state.filters["start_date"] = filters["start_date"]
    st.session_state.filters["end_date"] = filters["end_date"]

    # Background Color
    filters["bg_color"] = container.color_picker(
        "🎨 Chart Background", "#FFFFFF", key=f"bg_color{key_suffix}"
    )
    filters["text_color"] = auto_text_color(filters["bg_color"])

    # Add placeholder for kpi_selection
    filters["kpi_selection"] = st.session_state.get(
        "selected_newborn_kpi", "Inborn Rate (%)"
    )

    return filters


def _get_newborn_patient_date_range(patient_df):
    """Get REAL min/max dates from newborn patient dataframe - HARDCODED ALL TIME"""
    import datetime
    import logging

    # ALWAYS return the full 2000-2035 range
    all_time_start = datetime.date(2000, 1, 1)
    all_time_end = datetime.date(2035, 12, 31)

    logging.info(
        f"📅 _get_newborn_patient_date_range: Using ALL TIME range: {all_time_start} to {all_time_end}"
    )

    return all_time_start, all_time_end


def apply_newborn_patient_filters(patient_df, filters, facility_uids=None):
    """Apply filters to newborn patient dataframe - FIXED DATE FILTERING"""
    if patient_df.empty:
        return patient_df

    df = patient_df.copy()

    # Get current KPI selection from session state
    current_kpi = st.session_state.get("selected_newborn_kpi", "Inborn Rate (%)")

    # Get the SPECIFIC date column for this KPI
    kpi_date_column = get_relevant_date_column_for_newborn_kpi(current_kpi)

    logging.info(f"🔍 apply_newborn_patient_filters for KPI: {current_kpi}")
    logging.info(f"   - Using date column: {kpi_date_column}")
    logging.info(f"   - Quick range: {filters.get('quick_range', 'N/A')}")

    # STANDARDIZE COLUMN NAMES - Ensure we have orgUnit
    if "orgUnit" not in df.columns:
        # Try to find the facility ID column
        for col in df.columns:
            if col.lower() in ["orgunit", "facility_uid", "facility_id", "ou", "uid"]:
                df = df.rename(columns={col: "orgUnit"})
                break

    # CRITICAL FIX: Apply facility filter FIRST
    if facility_uids and "orgUnit" in df.columns:
        if not isinstance(facility_uids, list):
            facility_uids = [facility_uids]

        # Check if this is "All Facilities" or actual facility UIDs
        is_all_facilities = (
            len(facility_uids) == 0
            or facility_uids == ["All Facilities"]
            or (len(facility_uids) == 1 and facility_uids[0] == "All Facilities")
        )

        if not is_all_facilities:
            facility_mask = df["orgUnit"].isin(facility_uids)
            df = df[facility_mask].copy()
            logging.info(f"   - After facility filter: {len(df)} patients")
        else:
            logging.info("   - Skipping facility filter (All Facilities selected)")
    else:
        logging.info("   - No facility filter applied")

    # STEP 1: Use KPI-specific date column OR enrollment_date as fallback
    date_column_to_use = None

    # Try KPI-specific date column first
    if kpi_date_column and kpi_date_column in df.columns:
        date_column_to_use = kpi_date_column
        logging.info(f"   ✅ Using KPI-specific date column: {date_column_to_use}")
    elif "enrollment_date" in df.columns:
        date_column_to_use = "enrollment_date"
        logging.info(f"   ⚠️ KPI-specific date not found, using enrollment_date")
    elif "event_date" in df.columns:
        date_column_to_use = "event_date"
        logging.info(f"   ⚠️ Using generic event_date")
    else:
        # Find any date column
        date_cols = [col for col in df.columns if "date" in col.lower()]
        if date_cols:
            date_column_to_use = date_cols[0]
            logging.info(f"   ⚠️ Using found date column: {date_column_to_use}")
        else:
            logging.warning("   ❌ No date column found for filtering")
            date_column_to_use = None

    # STEP 2: Apply date filtering if we have a date column
    should_filter_by_date = (
        date_column_to_use is not None
        and filters.get("quick_range") != "All Time"  # Not "All Time"
        and filters.get("start_date")  # Has start date
        and filters.get("end_date")  # Has end date
    )

    if should_filter_by_date:
        try:
            # Get the start and end dates from filters
            start_date = pd.Timestamp(filters["start_date"])
            end_date = pd.Timestamp(filters["end_date"]) + pd.Timedelta(
                days=1
            )  # Include end date

            logging.info(
                f"🔍 DATE FILTER: Filtering '{date_column_to_use}' from {start_date.date()} to {end_date.date()}"
            )

            # Convert date column to datetime
            df[date_column_to_use] = pd.to_datetime(
                df[date_column_to_use], errors="coerce"
            )

            # Count valid dates before filtering
            valid_dates_before = df[date_column_to_use].notna().sum()
            logging.info(
                f"📅 Valid dates before filtering: {valid_dates_before}/{len(df)}"
            )

            if valid_dates_before > 0:
                # Apply date filter
                date_mask = (df[date_column_to_use] >= start_date) & (
                    df[date_column_to_use] < end_date
                )

                df_filtered = df[date_mask].copy()

                # Log filtering results
                patients_before = len(df)
                patients_after = len(df_filtered)
                patients_lost = patients_before - patients_after

                logging.info(
                    f"✅ DATE FILTER COMPLETE: {patients_after} patients remain (lost {patients_lost})"
                )

                # Sample filtered dates
                if patients_after > 0:
                    sample_dates = (
                        df_filtered[date_column_to_use]
                        .head(3)
                        .dt.strftime("%Y-%m-%d")
                        .tolist()
                    )
                    logging.info(f"📅 Sample filtered dates: {sample_dates}")

                df = df_filtered
            else:
                logging.warning(
                    "⚠️ No valid dates found for filtering. Keeping all patients."
                )
                df = df

        except Exception as error:
            logging.error(f"❌ ERROR in date filtering: {error}")
            # If there's an error, keep all patients (don't filter)
            df = df
    else:
        logging.info("⏰ Skipping date filtering (All Time selected or no dates)")
        df = df

    # STEP 3: Ensure we have event_date column for period assignment
    if "event_date" not in df.columns and date_column_to_use:
        # Use the filtered date column for period assignment
        df["event_date"] = pd.to_datetime(df[date_column_to_use], errors="coerce")
    elif "event_date" not in df.columns:
        # Try to find any date column
        df = normalize_newborn_patient_dates(df)

    # STEP 4: Assign periods for trend analysis
    if "period_label" not in filters:
        filters["period_label"] = st.session_state.get("period_label", "Monthly")

    try:
        valid_date_rows = df["event_date"].notna().sum()
        if valid_date_rows > 0:
            df = assign_period(df, "event_date", filters["period_label"])
            logging.info(
                f"   📊 Assigned {filters['period_label']} periods: {valid_date_rows} valid dates"
            )
        else:
            # Create empty period columns
            df["period_display"] = ""
            df["period_sort"] = 0
            logging.warning("   ⚠️ No valid dates for period assignment")
    except Exception as e:
        # Create empty period columns as fallback
        df["period_display"] = ""
        df["period_sort"] = 0
        logging.error(f"   ❌ Error assigning periods: {e}")

    # Save to session state
    st.session_state.period_label = filters["period_label"]
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    st.session_state.filters.update(filters)

    # OPTIMIZATION: Filter for current KPI
    if current_kpi in NEWBORN_KPI_COLUMN_REQUIREMENTS:
        df = get_newborn_kpi_filtered_dataframe(df, current_kpi)
        logging.info(f"   🔍 KPI filter applied for '{current_kpi}'")

    logging.info(f"🔄 apply_newborn_patient_filters: Final result - {len(df)} patients")
    return df


def get_newborn_period_data_for_kpi(kpi_selection, patient_df, facility_uids):
    """Get period-based data for a specific newborn KPI from patient data"""
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_selection)

    # Extract period columns using program stage specific date - CHANGED: using extract_period_columns_newborn
    df_with_periods = extract_period_columns_newborn(patient_df, date_column)  # CHANGED

    if df_with_periods.empty:
        return pd.DataFrame()

    # Filter by facilities if specified
    if facility_uids:
        df_with_periods = df_with_periods[
            df_with_periods["orgUnit"].isin(facility_uids)
        ]

    # Sort by period
    if "period_sort" in df_with_periods.columns:
        df_with_periods = df_with_periods.sort_values("period_sort")

    return df_with_periods


def compute_newborn_kpi_for_period(kpi_selection, period_df, facility_uids):
    """Compute newborn KPI for a specific period using patient data - ULTRA SIMPLE VERSION"""
    if period_df.empty:
        return _get_newborn_default_kpi_data(kpi_selection)

    # For ALL KPIs, use get_numerator_denominator_for_newborn_kpi
    numerator, denominator, value = get_numerator_denominator_for_newborn_kpi(
        period_df, kpi_selection, facility_uids
    )

    # Return SIMPLE, consistent structure
    return {
        "value": float(value),
        "numerator": int(numerator),
        "denominator": int(denominator),
        "kpi_name": kpi_selection,
    }


def _get_newborn_default_kpi_data(kpi_selection):
    """Get default empty data for newborn KPIs"""
    return {"value": 0.0, "numerator": 0, "denominator": 0}


def get_date_column_from_newborn_df(df, kpi_selection):
    """Get the appropriate date column from newborn patient-level data based on KPI - USE SPECIFIC PROGRAM STAGE"""
    # Use the same mapping as in kpi_utils_newborn
    date_column = get_relevant_date_column_for_newborn_kpi(kpi_selection)

    # Check if this specific column exists
    if date_column in df.columns:
        return date_column

    # If not, warn and don't use wrong dates
    st.warning(f"⚠️ Required date column '{date_column}' not found for {kpi_selection}")
    return None
