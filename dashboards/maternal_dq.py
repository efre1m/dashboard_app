# dashboards/maternal_dq.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.patient_mapping import get_patient_name_from_tei

# Critical elements for maternal data quality checks
MATERNAL_CRITICAL_ELEMENTS = {
    "xm2Z3lvBJY8": {
        "name": "Systolic Blood Pressure",
        "impossible_range": (50, 250),
        "unit": "mmHg",
    },
    "IxvvFNeB42i": {
        "name": "Diastolic Blood Pressure",
        "impossible_range": (30, 150),
        "unit": "mmHg",
    },
    "UzMzF0nuz4h": {
        "name": "Fetal Heart Rate",
        "impossible_range": (60, 240),
        "unit": "bpm",
    },
    "aXFLfbJzYH5": {
        "name": "Hemoglobin Level",
        "impossible_range": (3, 25),
        "unit": "g/dL",
    },
    "QUlJEvzGcQK": {
        "name": "Birth Weight",
        "impossible_range": (500, 8000),
        "unit": "grams",
    },
    "khlYIOsLWhU": {
        "name": "Random Blood Sugar",
        "impossible_range": (20, 800),
        "unit": "mg/dL",
    },
    "cbgZGJITGA3": {"name": "Temperature", "impossible_range": (28, 43), "unit": "°C"},
    "JNQ9EQ246aW": {
        "name": "Glasgow Coma Scale",
        "impossible_range": (2, 15),
        "unit": "score",
    },
}


def safe_numeric_conversion(value):
    """Safely convert value to numeric"""
    if pd.isna(value) or value is None or value == "":
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    try:
        cleaned_value = str(value).strip()
        return float(cleaned_value)
    except (ValueError, TypeError):
        return np.nan


def get_region_from_facility(facility_name):
    """Get region name from facility name using the mapping"""
    if (
        hasattr(st, "session_state")
        and "facility_to_region_map" in st.session_state
        and facility_name in st.session_state.facility_to_region_map
    ):
        return st.session_state.facility_to_region_map[facility_name]
    return "Unknown Region"


def check_maternal_outliers(events_df):
    """Check for outliers in maternal critical data elements"""
    if events_df.empty:
        return pd.DataFrame()

    outliers = []

    for data_element_uid, element_info in MATERNAL_CRITICAL_ELEMENTS.items():
        element_name = element_info["name"]
        min_val, max_val = element_info["impossible_range"]
        unit = element_info["unit"]

        # Filter rows for this data element
        element_rows = events_df[
            events_df["dataElement_uid"] == data_element_uid
        ].copy()

        if not element_rows.empty:
            # Convert to numeric
            element_rows["numeric_value"] = element_rows["value"].apply(
                safe_numeric_conversion
            )
            numeric_rows = element_rows[element_rows["numeric_value"].notna()]

            if not numeric_rows.empty:
                # Find outliers
                outlier_mask = (numeric_rows["numeric_value"] < min_val) | (
                    numeric_rows["numeric_value"] > max_val
                )
                outlier_rows = numeric_rows[outlier_mask]

                for _, row in outlier_rows.iterrows():
                    outlier_value = row["numeric_value"]

                    if outlier_value < min_val:
                        issue_type = f"Too Low (< {min_val} {unit})"
                    else:
                        issue_type = f"Too High (> {max_val} {unit})"

                    # Get proper patient name using TEI API
                    tei_id = row.get("tei_id")
                    patient_name = get_patient_name_from_tei(tei_id, "maternal")

                    facility_name = row.get("orgUnit_name", "Unknown Facility")
                    region = get_region_from_facility(facility_name)
                    program_stage = row.get(
                        "program_stage", row.get("programStageName", "Unknown Stage")
                    )
                    event_date = row.get(
                        "event_date", row.get("eventDate", "Unknown Date")
                    )

                    outliers.append(
                        {
                            "Patient Name": patient_name,
                            "Region": region,
                            "Facility": facility_name,
                            "Program Stage": program_stage,
                            "Data Element": element_name,
                            "Value": outlier_value,
                            "Unit": unit,
                            "Issue Type": issue_type,
                            "Expected Range": f"{min_val} - {max_val} {unit}",
                            "Event Date": event_date,
                            "TEI ID": tei_id,
                        }
                    )

    return pd.DataFrame(outliers)


def render_maternal_data_quality():
    """Render Maternal Data Quality Analysis using data from session state"""

    events_df = st.session_state.maternal_events_df

    # Check for outliers
    with st.spinner("🔍 Analyzing maternal data quality..."):
        outliers_df = check_maternal_outliers(events_df)

    if outliers_df.empty:
        st.success("✅ No data quality issues found in critical maternal data elements")
        return

    st.error(f"🚨 Found {len(outliers_df)} data quality issues")

    # Professional Filter Section with same style as LBW tables
    st.markdown("### 🔍 Filter Issues")

    # Create filter container
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            # Region filter with dropdown
            all_regions = ["All Regions"] + sorted(outliers_df["Region"].unique())
            selected_region = st.selectbox(
                "Select Region",
                options=all_regions,
                index=0,
                help="Filter by region",
                key="maternal_region_filter",
            )

        with col2:
            # Facility filter with dropdown
            all_facilities = ["All Facilities"] + sorted(
                outliers_df["Facility"].unique()
            )
            selected_facility = st.selectbox(
                "Select Facility",
                options=all_facilities,
                index=0,
                help="Filter by facility",
                key="maternal_facility_filter",
            )

        with col3:
            # Data Element filter with dropdown
            all_elements = ["All Data Elements"] + sorted(
                outliers_df["Data Element"].unique()
            )
            selected_element = st.selectbox(
                "Select Data Element",
                options=all_elements,
                index=0,
                help="Filter by data element",
                key="maternal_element_filter",
            )

    # Apply filters
    filtered_df = outliers_df.copy()

    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    if selected_facility != "All Facilities":
        filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]

    if selected_element != "All Data Elements":
        filtered_df = filtered_df[filtered_df["Data Element"] == selected_element]

    if filtered_df.empty:
        st.info("No data quality issues match the selected filters.")
        return

    # Display the main table with professional styling (same as LBW tables)
    st.markdown("### 📋 Data Quality Issues")

    # Prepare table data with numbering
    display_df = filtered_df.copy().reset_index(drop=True)
    display_df.insert(0, "No", range(1, len(display_df) + 1))

    # Select columns to display
    display_columns = [
        "No",
        "Patient Name",
        "Region",
        "Facility",
        "Program Stage",
        "Data Element",
        "Value",
        "Unit",
        "Issue Type",
        "Expected Range",
        "Event Date",
    ]

    # Apply custom CSS for professional table styling (same as summary tables)
    st.markdown(
        """
    <style>
    .summary-table-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }

    .summary-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Arial', sans-serif;
        font-size: 14px;
    }

    .summary-table thead tr {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
    }

    .summary-table th {
        color: white;
        padding: 14px 16px;
        text-align: left;
        font-weight: 600;
        font-size: 14px;
        border: none;
    }

    .summary-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #f0f0f0;
        font-size: 14px;
        background-color: white;
    }

    .summary-table tbody tr:last-child td {
        border-bottom: none;
    }

    .summary-table tbody tr:hover td {
        background-color: #f8f9fa;
    }

    /* Number column styling */
    .summary-table td:first-child {
        font-weight: 600;
        color: #666;
        text-align: center;
    }

    .summary-table th:first-child {
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Format the dataframe for display
    styled_df = display_df[display_columns].copy()
    styled_df["Value"] = styled_df["Value"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )

    # Create styled table
    styled_table = (
        styled_df.style.set_table_attributes('class="summary-table"')
        .set_properties(**{"text-align": "left"})
        .hide(axis="index")
    )

    # Display the table in container
    st.markdown('<div class="summary-table-container">', unsafe_allow_html=True)
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download Data Quality Report",
        data=csv,
        file_name="maternal_data_quality_issues.csv",
        mime="text/csv",
        use_container_width=True,
    )
