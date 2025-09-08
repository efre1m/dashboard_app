# utils/queries.py
from typing import List, Tuple, Optional
from utils.db import get_db_connection
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_orgunit_uids_for_user(user: dict) -> List[Tuple[str, str]]:
    """
    Fetch DHIS2 OrgUnit UIDs and names accessible for a user.
    Returns list of tuples: (dhis2_uid, display_name)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    role: str = user.get("role", "")
    ous: List[Tuple[str, str]] = []

    try:
        if role == "facility" and user.get("facility_id"):
            cur.execute("SELECT dhis2_uid, facility_name FROM facilities WHERE facility_id=%s",
                        (user["facility_id"],))
            ous = cur.fetchall()
            
        elif role == "regional" and user.get("region_id"):
            # For regional users, return the regional UID (will use DESCENDANTS mode)
            cur.execute("SELECT dhis2_regional_uid, region_name FROM regions WHERE region_id=%s",
                        (user["region_id"],))
            row = cur.fetchone()
            if row:
                ous = [(row[0], row[1])]
                
        elif role == "national" and user.get("country_id"):
            # For national users, return the country UID (will use DESCENDANTS mode)
            cur.execute("SELECT dhis2_uid, country_name FROM countries WHERE country_id=%s",
                        (user["country_id"],))
            row = cur.fetchone()
            if row:
                ous = [(row[0], row[1])]
                
    except Exception as e:
        logging.error(f"Error fetching orgUnits for user: {e}")
    finally:
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
    program_uid: Optional[str] = None
    
    try:
        cur.execute("SELECT program_uid FROM programs WHERE program_name=%s", (program_name,))
        row = cur.fetchone()
        program_uid = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching program UID: {e}")
    finally:
        cur.close()
        conn.close()

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
    facility_name = None
    
    try:
        cur.execute("SELECT facility_name FROM facilities WHERE dhis2_uid=%s", (dhis_uid,))
        row = cur.fetchone()
        facility_name = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching facility name: {e}")
    finally:
        cur.close()
        conn.close()

    return facility_name


def get_region_name_by_dhis_uid(dhis_uid: str) -> Optional[str]:
    """
    Fetch region_name from dhis2_regional_uid.
    """
    if not dhis_uid:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    region_name = None
    
    try:
        cur.execute("SELECT region_name FROM regions WHERE dhis2_regional_uid=%s", (dhis_uid,))
        row = cur.fetchone()
        region_name = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching region name: {e}")
    finally:
        cur.close()
        conn.close()

    return region_name


def get_country_name_by_dhis_uid(dhis_uid: str) -> Optional[str]:
    """
    Fetch country_name from dhis2_uid (countries table).
    """
    if not dhis_uid:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    country_name = None
    
    try:
        cur.execute("SELECT country_name FROM countries WHERE dhis2_uid=%s", (dhis_uid,))
        row = cur.fetchone()
        country_name = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching country name: {e}")
    finally:
        cur.close()
        conn.close()

    return country_name