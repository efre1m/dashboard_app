# utils/code_utils.py

COUNTRY_ID = "1"


def get_region_code(display_region_id: str) -> str:
    """Return region code as country_id + display_region_id"""
    return COUNTRY_ID + str(display_region_id)


def get_facility_code(facility_id: str, display_region_id: str) -> str:
    """Return facility code as country_id + display_region_id + facility_id"""
    return COUNTRY_ID + str(display_region_id) + str(facility_id)
