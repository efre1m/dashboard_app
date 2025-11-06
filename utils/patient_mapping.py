# utils/patient_mapping_tei.py
import streamlit as st
import pandas as pd


def get_patient_name_from_tei(tei_id, program_type):
    """
    Fetch patient name from cached TEI DataFrame using TEI ID and attribute UID
    program_type: 'maternal' or 'newborn'
    Returns: (first_name, father_name/last_name)
    """
    if not tei_id or tei_id == "N/A" or pd.isna(tei_id):
        return "Unknown Patient", ""

    try:
        # Get the appropriate TEI DataFrame from session state
        if program_type == "maternal":
            tei_df = st.session_state.get("maternal_tei_df", pd.DataFrame())
            first_name_attribute_id = "X9OEn2gPhss"  # Mothers First Name attribute ID
            father_name_attribute_id = "NZramclXjFR"  # Mothers Father Name attribute ID
        elif program_type == "newborn":
            tei_df = st.session_state.get("newborn_tei_df", pd.DataFrame())
            first_name_attribute_id = "tiMSRGbNJJz"  # First Name of Baby attribute ID
            father_name_attribute_id = "RRFJFC0UyrX"  # Last Name of Baby attribute ID
        else:
            return "Unknown Patient", ""

        if tei_df.empty:
            return "Unknown Patient", ""

        # Find TEI ID column (handle different column names)
        tei_id_column = "tei_id"
        if tei_id_column not in tei_df.columns:
            # Look for alternative TEI ID columns
            possible_columns = [
                col
                for col in tei_df.columns
                if "tei" in col.lower() or "id" in col.lower()
            ]
            if possible_columns:
                tei_id_column = possible_columns[0]
            else:
                return "Unknown Patient", ""

        # Filter for this specific TEI
        tei_records = tei_df[tei_df[tei_id_column] == tei_id]

        if tei_records.empty:
            return "Unknown Patient", ""

        # Get first name
        first_name_rows = tei_records[
            tei_records["attribute"] == first_name_attribute_id
        ]
        first_name = "Unknown Patient"
        if not first_name_rows.empty:
            first_name_value = first_name_rows["value"].iloc[0]
            if pd.notna(first_name_value) and str(first_name_value).strip() != "":
                first_name = str(first_name_value).strip()

        # Get father name/last name
        father_name_rows = tei_records[
            tei_records["attribute"] == father_name_attribute_id
        ]
        father_name = ""
        if not father_name_rows.empty:
            father_name_value = father_name_rows["value"].iloc[0]
            if pd.notna(father_name_value) and str(father_name_value).strip() != "":
                father_name = str(father_name_value).strip()

        return first_name, father_name

    except Exception as e:
        print(f"Error fetching patient name for TEI {tei_id}: {e}")
        return "Unknown Patient", ""
