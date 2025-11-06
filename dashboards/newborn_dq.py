# dashboards/newborn_dq.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.patient_mapping import get_patient_name_from_tei

# Critical elements for newborn data quality checks
NEWBORN_CRITICAL_ELEMENTS = {
    "CzIgD0rsk52": {
        "name": "Birth Weight",
        "impossible_range": (400, 8000),
        "unit": "grams",
    },
    "yxWUMt3sCil": {
        "name": "Weight on Admission",
        "impossible_range": (400, 6000),
        "unit": "grams",
    },
    "yBCwmQP0A6a": {
        "name": "Discharge Weight",
        "impossible_range": (400, 6000),
        "unit": "grams",
    },
    "gZi9y12E9i7": {
        "name": "Temperature on Admission",
        "impossible_range": (28, 42),
        "unit": "°C",
    },
    "nIKIu6f5vbW": {
        "name": "Lowest Temperature",
        "impossible_range": (28, 42),
        "unit": "°C",
    },
    "sCDFBFReCco": {
        "name": "Highest Temperature",
        "impossible_range": (28, 43),
        "unit": "°C",
    },
    "xnU9AiiCD9v": {
        "name": "Oxygen Saturation on Admission",
        "impossible_range": (40, 100),
        "unit": "%",
    },
    "j4W59YyYG04": {
        "name": "Lowest Oxygen Saturation",
        "impossible_range": (30, 100),
        "unit": "%",
    },
    "ydFKE4DmfrT": {
        "name": "Highest Oxygen Saturation",
        "impossible_range": (50, 100),
        "unit": "%",
    },
    "F4NIANzeaTe": {
        "name": "Blood Sugar (mmol/L)",
        "impossible_range": (0.5, 50),
        "unit": "mmol/L",
    },
    "zYFS8Cc5vqp": {
        "name": "Blood Sugar (mg/dL)",
        "impossible_range": (10, 900),
        "unit": "mg/dL",
    },
    "d3PkVNa0ozG": {
        "name": "Lowest Blood Sugar (mmol/L)",
        "impossible_range": (0.1, 30),
        "unit": "mmol/L",
    },
    "GHfKwL7n4v5": {
        "name": "Lowest Blood Sugar (mg/dL)",
        "impossible_range": (2, 540),
        "unit": "mg/dL",
    },
    "p4NR33eio81": {
        "name": "Highest Bilirubin",
        "impossible_range": (0, 40),
        "unit": "mg/dL",
    },
    "r5JV9avYdmB": {
        "name": "Gestational Age Months",
        "impossible_range": (5, 10),
        "unit": "months",
    },
    "c3QaY9N6Ll7": {
        "name": "Gestational Age Weeks",
        "impossible_range": (20, 44),
        "unit": "weeks",
    },
    "tsM5JUrghn3": {
        "name": "Gestational Age Days",
        "impossible_range": (0, 30),
        "unit": "days",
    },
    "TdD5Sk7leqZ": {
        "name": "Maternal Age",
        "impossible_range": (12, 55),
        "unit": "years",
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


def check_newborn_outliers(events_df):
    """Check for outliers in newborn critical data elements"""
    if events_df.empty:
        return pd.DataFrame()

    outliers = []

    for data_element_uid, element_info in NEWBORN_CRITICAL_ELEMENTS.items():
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
                    patient_name = get_patient_name_from_tei(tei_id, "newborn")

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


def render_newborn_data_quality():
    """Render Newborn Data Quality Analysis using data from session state"""

    events_df = st.session_state.newborn_events_df

    # Check for outliers
    with st.spinner("🔍 Analyzing newborn data quality..."):
        outliers_df = check_newborn_outliers(events_df)

    if outliers_df.empty:
        st.success("✅ No data quality issues found in critical newborn data elements")
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
                key="newborn_region_filter",
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
                key="newborn_facility_filter",
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
                key="newborn_element_filter",
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
        background: linear-gradient(135deg, #3498db, #2980b9);
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
        file_name="newborn_data_quality_issues.csv",
        mime="text/csv",
        use_container_width=True,
    )
