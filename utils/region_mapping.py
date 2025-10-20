# utils/region_mapping.py
"""
Mapping of database region_id values to ODK form region codes.
The numeric codes (1-6) are values found in the 'region' or 'reg-region' columns of ODK forms.
"""

# Mapping of DATABASE region_id to ODK numeric codes used in form columns
REGION_TO_ODK_CODE_MAPPING = {
    # Database region_id: [ODK numeric codes that should map to this region]
    1: ["3"],  # Tigray (Database ID 1) -> ODK Code "3"
    3: ["4", "6"],  # South Ethiopia (Database ID 3) -> ODK Codes "4" AND "6"
    4: ["4", "6"],  # Sidama (Database ID 4) -> ODK Codes "4" AND "6"
    5: ["1"],  # Oromia (Database ID 5) -> ODK Code "1"
    6: ["4", "6"],  # Central Ethiopia (Database ID 6) -> ODK Codes "4" AND "6"
    7: ["4", "6"],  # South West Ethiopia (Database ID 7) -> ODK Codes "4" AND "6"
    8: ["5"],  # Afar (Database ID 8) -> ODK Code "5"
    9: ["2"],  # Amhara (Database ID 9) -> ODK Code "2"
}

# Database region_id to region name mapping
DATABASE_REGION_NAMES = {
    1: "Tigray",
    3: "South Ethiopia",
    4: "Sidama",
    5: "Oromia",
    6: "Central Ethiopia",
    7: "South West Ethiopia",
    8: "Afar",
    9: "Amhara",
}


def get_odk_region_codes(database_region_id: int) -> list[str]:
    """Get ODK numeric region codes for a given database region_id."""
    return REGION_TO_ODK_CODE_MAPPING.get(database_region_id, [])


def get_region_name_from_database_id(database_region_id: int) -> str:
    """Get region name from database region_id."""
    return DATABASE_REGION_NAMES.get(database_region_id, "Unknown Region")


def get_supported_regions() -> dict:
    """Get all regions that have ODK code mappings."""
    return DATABASE_REGION_NAMES


def get_odk_code_mapping_display() -> dict:
    """Get ODK code to database regions mapping for display."""
    mapping = {}
    for odk_code in ["1", "2", "3", "4", "5", "6"]:
        database_ids = []
        for db_region_id, codes in REGION_TO_ODK_CODE_MAPPING.items():
            if odk_code in [str(code).strip() for code in codes]:
                database_ids.append(db_region_id)
        region_names = [
            DATABASE_REGION_NAMES.get(rid, f"ID:{rid}") for rid in database_ids
        ]
        mapping[odk_code] = region_names
    return mapping
