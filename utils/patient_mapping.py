# utils/patient_mapping.py
import requests
import streamlit as st
from utils.config import settings


def get_patient_name_from_tei(tei_id, program_type):
    """
    Fetch patient name from Tracked Entity Instance API
    program_type: 'maternal' or 'newborn'
    """
    if not tei_id or tei_id == "N/A":
        return "Unknown Patient"

    # Cache the results to avoid repeated API calls
    cache_key = f"patient_name_{tei_id}"
    if hasattr(st, "session_state") and cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        url = f"{settings.DHIS2_BASE_URL}/api/trackedEntityInstances/{tei_id}.json"
        params = {"fields": "attributes[attribute[id,name,displayName],value]"}
        auth = (settings.DHIS2_USERNAME, settings.DHIS2_PASSWORD)

        response = requests.get(
            url, params=params, auth=auth, timeout=settings.DHIS2_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        # Debug: print the actual API response to see what we're getting
        print(f"DEBUG - TEI {tei_id} response: {data}")

        # Extract patient name based on program type
        patient_name = extract_patient_name(data, program_type)

        # Cache the result
        if hasattr(st, "session_state"):
            st.session_state[cache_key] = patient_name

        return patient_name

    except Exception as e:
        print(f"Error fetching patient name for TEI {tei_id}: {e}")
        return "Unknown Patient"


def extract_patient_name(tei_data, program_type):
    """Extract patient name from TEI attributes"""
    attributes = tei_data.get("attributes", [])

    print(f"DEBUG - All attributes for {program_type}: {attributes}")

    if program_type == "maternal":
        # Try multiple possible attribute IDs for maternal name
        name_attributes = ["X9OEn2gPhss", "firstName", "m_mth_fst_nm"]
        for attr in attributes:
            attr_id = attr.get("attribute", {}).get("id", "")
            attr_value = attr.get("value", "")
            print(f"DEBUG - Maternal attribute: {attr_id} = {attr_value}")
            if attr_id in name_attributes and attr_value:
                return attr_value

    elif program_type == "newborn":
        # Try multiple possible attribute IDs for newborn name
        name_attributes = ["tiMSRGbNJJz", "firstName", "in_bynam_fst"]
        for attr in attributes:
            attr_id = attr.get("attribute", {}).get("id", "")
            attr_value = attr.get("value", "")
            print(f"DEBUG - Newborn attribute: {attr_id} = {attr_value}")
            if attr_id in name_attributes and attr_value:
                return attr_value

    # If no name found, return first non-empty attribute value as fallback
    for attr in attributes:
        if attr.get("value"):
            return attr.get("value")

    return "Unknown Patient"
