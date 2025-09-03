# utils/queries.py
from typing import List, Tuple, Optional
from utils.db import get_db_connection
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_orgunit_uids_for_user(user: dict) -> List[Tuple[str, str]]:
    """
    Fetch DHIS2 OrgUnit UIDs and facility/region names accessible for a user.
    Works for facility, regional, and national roles.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    role: str = user.get("role", "")
    ous: List[Tuple[str, str]] = []

    if role == "facility" and user.get("facility_id"):
        cur.execute("SELECT dhis2_uid, facility_name FROM facilities WHERE facility_id=%s",
                    (user["facility_id"],))
        ous = cur.fetchall()
    elif role == "regional" and user.get("region_id"):
        # Fetch regional UID from regions table
        cur.execute("SELECT dhis2_regional_uid, region_name FROM regions WHERE region_id=%s",
                    (user["region_id"],))
        row = cur.fetchone()
        if row:
            ous = [(row[0], row[1])]  # single entry: (DHIS2 UID, region name)
    elif role == "national" and user.get("country_id"):
        cur.execute("""SELECT f.dhis2_uid, f.facility_name
                       FROM facilities f
                       JOIN regions r ON r.region_id = f.region_id
                       WHERE r.country_id=%s ORDER BY f.facility_name""",
                    (user["country_id"],))
        ous = cur.fetchall()

    cur.close()
    conn.close()

    logging.info("OrgUnits fetched for user '%s': %s", user.get("username"), ous)
    return [(ou, name) for ou, name in ous if ou]


def get_program_uid(program_name: str = "Maternal Inpatient Data") -> Optional[str]:
    """
    Fetch DHIS2 program UID from the database.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT program_uid FROM programs WHERE program_name=%s", (program_name,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    program_uid: Optional[str] = row[0] if row else None
    logging.info("Program UID for '%s': %s", program_name, program_uid)
    return program_uid


def get_facility_name_by_dhis_uid(dhis_uid: str) -> Optional[str]:
    """
    Fetch facility_name from dhis2_uid.
    """
    if not dhis_uid:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT facility_name FROM facilities WHERE dhis2_uid=%s", (dhis_uid,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    return row[0] if row else None


def get_region_name_by_dhis_uid(dhis_uid: str) -> Optional[str]:
    """
    Fetch region_name from dhis2_regional_uid.
    """
    if not dhis_uid:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT region_name FROM regions WHERE dhis2_regional_uid=%s", (dhis_uid,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    return row[0] if row else None
