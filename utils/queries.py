# utils/queries.py
from typing import List, Tuple, Optional, Dict
from utils.db import get_db_connection
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_all_programs() -> List[Dict]:
    """
    Fetch all programs from the database dynamically.
    Returns list of programs with program_id, program_name, and program_uid.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    programs: List[Dict] = []

    try:
        cur.execute(
            "SELECT program_id, program_name, program_uid FROM programs ORDER BY program_name"
        )
        rows = cur.fetchall()

        for row in rows:
            programs.append(
                {"program_id": row[0], "program_name": row[1], "program_uid": row[2]}
            )

        logging.info(f"Fetched {len(programs)} programs from database")

    except Exception as e:
        logging.error(f"Error fetching programs: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    return programs


def get_default_program() -> Optional[Dict]:
    """
    Get the default program (prefer 'Maternal Inpatient Data' if present).
    Falls back to the first program if the preferred one is not found.
    """
    programs = get_all_programs()
    for program in programs:
        if program.get("program_name") == "Maternal Inpatient Data":
            return program
    return programs[0] if programs else None


def get_program_uid(program_name: Optional[str] = None) -> Optional[str]:
    """
    Fetch DHIS2 program UID from the database.
    If program_name is provided, returns the UID for that program.
    If not provided, returns the default program UID (if any).
    """
    if program_name:
        conn = get_db_connection()
        cur = conn.cursor()
        program_uid: Optional[str] = None
        try:
            cur.execute(
                "SELECT program_uid FROM programs WHERE program_name=%s",
                (program_name,),
            )
            row = cur.fetchone()
            program_uid = row[0] if row else None
        except Exception as e:
            logging.error(f"Error fetching program UID for '{program_name}': {e}")
        finally:
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        logging.info("Program UID for '%s': %s", program_name, program_uid)
        return program_uid
    else:
        # fallback to default program
        default_program = get_default_program()
        program_uid = default_program["program_uid"] if default_program else None
        logging.info("Default program UID: %s", program_uid)
        return program_uid


def get_program_by_uid(program_uid: str) -> Optional[Dict]:
    """
    Get program details by program UID.
    Returns a dict with keys program_id, program_name, program_uid or None if not found.
    """
    if not program_uid:
        return None

    conn = get_db_connection()
    cur = conn.cursor()
    program: Optional[Dict] = None
    try:
        cur.execute(
            "SELECT program_id, program_name, program_uid FROM programs WHERE program_uid=%s",
            (program_uid,),
        )
        row = cur.fetchone()
        if row:
            program = {
                "program_id": row[0],
                "program_name": row[1],
                "program_uid": row[2],
            }
    except Exception as e:
        logging.error(f"Error fetching program by UID '{program_uid}': {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    return program


# ---- Organizational / Facility functions ----


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
            cur.execute(
                "SELECT dhis2_uid, facility_name FROM facilities WHERE facility_id=%s",
                (user["facility_id"],),
            )
            ous = cur.fetchall()

        elif role == "regional" and user.get("region_id"):
            # For regional users, return the regional UID (will use DESCENDANTS mode)
            cur.execute(
                "SELECT dhis2_regional_uid, region_name FROM regions WHERE region_id=%s",
                (user["region_id"],),
            )
            row = cur.fetchone()
            if row:
                ous = [(row[0], row[1])]

        elif role == "national" and user.get("country_id"):
            # For national users, return the country UID (will use DESCENDANTS mode)
            cur.execute(
                "SELECT dhis2_uid, country_name FROM countries WHERE country_id=%s",
                (user["country_id"],),
            )
            row = cur.fetchone()
            if row:
                ous = [(row[0], row[1])]

    except Exception as e:
        logging.error(f"Error fetching orgUnits for user: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    logging.info("OrgUnits fetched for user '%s': %s", user.get("username"), ous)
    return [(ou, name) for ou, name in ous if ou]


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
        cur.execute(
            "SELECT facility_name FROM facilities WHERE dhis2_uid=%s", (dhis_uid,)
        )
        row = cur.fetchone()
        facility_name = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching facility name: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

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
        cur.execute(
            "SELECT region_name FROM regions WHERE dhis2_regional_uid=%s", (dhis_uid,)
        )
        row = cur.fetchone()
        region_name = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching region name: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

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
        cur.execute(
            "SELECT country_name FROM countries WHERE dhis2_uid=%s", (dhis_uid,)
        )
        row = cur.fetchone()
        country_name = row[0] if row else None
    except Exception as e:
        logging.error(f"Error fetching country name: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    return country_name


def get_facilities_for_user(user: dict) -> List[Tuple[str, str]]:
    """
    Get facilities for the current user from database.
    Returns list of tuples: (facility_name, dhis2_uid)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    facilities: List[Tuple[str, str]] = []

    try:
        role = user.get("role", "")

        if role == "national":
            # National users can see all facilities
            cur.execute("SELECT facility_name, dhis2_uid FROM facilities")
            facilities = cur.fetchall()
        elif role == "regional" and user.get("region_id"):
            # Get all facilities in the user's region
            cur.execute(
                "SELECT facility_name, dhis2_uid FROM facilities WHERE region_id = %s",
                (user["region_id"],),
            )
            facilities = cur.fetchall()
        elif role == "facility" and user.get("facility_id"):
            # Get the specific facility for facility users
            cur.execute(
                "SELECT facility_name, dhis2_uid FROM facilities WHERE facility_id = %s",
                (user["facility_id"],),
            )
            facilities = cur.fetchall()
    except Exception as e:
        logging.error(f"Error fetching facilities from database: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    return facilities


def get_facility_mapping_for_user(user: dict) -> Dict[str, str]:
    """
    Get facility ID to UID mapping for the current user.
    Returns dict: {facility_id: dhis2_uid}
    """
    conn = get_db_connection()
    cur = conn.cursor()
    facility_mapping = {}

    try:
        role = user.get("role", "")

        if role == "national":
            cur.execute("SELECT facility_id, dhis2_uid FROM facilities")
            rows = cur.fetchall()
            facility_mapping = {
                str(row[0]): row[1] for row in rows
            }  # Convert ID to string

        elif role == "regional" and user.get("region_id"):
            cur.execute(
                "SELECT facility_id, dhis2_uid FROM facilities WHERE region_id = %s",
                (user["region_id"],),
            )
            rows = cur.fetchall()
            facility_mapping = {
                str(row[0]): row[1] for row in rows
            }  # Convert ID to string

        elif role == "facility" and user.get("facility_id"):
            cur.execute(
                "SELECT facility_id, dhis2_uid FROM facilities WHERE facility_id = %s",
                (user["facility_id"],),
            )
            row = cur.fetchone()
            if row:
                facility_mapping = {str(row[0]): row[1]}  # Convert ID to string

    except Exception as e:
        logging.error(f"Error fetching facility mapping: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    # DEBUG: Log what we got
    logging.info(
        f"🔍 get_facility_mapping_for_user: Found {len(facility_mapping)} facilities"
    )
    if facility_mapping:
        sample = list(facility_mapping.items())[:3]
        logging.info(f"🔍 Sample facility mapping: {sample}")

    return facility_mapping


def get_facilities_grouped_by_region(
    user: dict,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Get facilities grouped by DISPLAY region ID (mapped from actual region_id).
    Returns dict: {display_region_id: [(facility_id, facility_name, dhis2_uid), ...]}
    """
    conn = get_db_connection()
    cur = conn.cursor()
    facilities_by_region: Dict[str, List[Tuple[str, str, str]]] = {}

    try:
        role = user.get("role", "")

        if role == "national":
            # Get ALL facilities with MAPPED display IDs
            cur.execute(
                """
                SELECT 
                    -- Map region_id to display ID
                    CASE r.region_name
                        WHEN 'Oromia' THEN 1
                        WHEN 'Amhara' THEN 2
                        WHEN 'Tigray' THEN 3
                        WHEN 'South Ethiopia' THEN 4
                        WHEN 'Afar' THEN 5
                        WHEN 'Sidama' THEN 6
                        WHEN 'Central Ethiopia' THEN 7
                        WHEN 'South West Ethiopia' THEN 8
                        ELSE r.region_id
                    END as display_region_id,
                    f.facility_id, 
                    f.facility_name, 
                    f.dhis2_uid 
                FROM facilities f
                JOIN regions r ON f.region_id = r.region_id
                ORDER BY 
                    CASE r.region_name
                        WHEN 'Oromia' THEN 1
                        WHEN 'Amhara' THEN 2
                        WHEN 'Tigray' THEN 3
                        WHEN 'South Ethiopia' THEN 4
                        WHEN 'Afar' THEN 5
                        WHEN 'Sidama' THEN 6
                        WHEN 'Central Ethiopia' THEN 7
                        WHEN 'South West Ethiopia' THEN 8
                        ELSE 99
                    END ASC,
                    f.facility_id ASC
                """
            )
            facilities = cur.fetchall()

        elif role == "regional" and user.get("region_id"):
            # For regional users, map their region_id too
            cur.execute(
                """
                SELECT 
                    CASE r.region_name
                        WHEN 'Oromia' THEN 1
                        WHEN 'Amhara' THEN 2
                        WHEN 'Tigray' THEN 3
                        WHEN 'South Ethiopia' THEN 4
                        WHEN 'Afar' THEN 5
                        WHEN 'Sidama' THEN 6
                        WHEN 'Central Ethiopia' THEN 7
                        WHEN 'South West Ethiopia' THEN 8
                        ELSE r.region_id
                    END as display_region_id,
                    f.facility_id, 
                    f.facility_name, 
                    f.dhis2_uid 
                FROM facilities f
                JOIN regions r ON f.region_id = r.region_id
                WHERE r.region_id = %s
                ORDER BY f.facility_id ASC
                """,
                (user["region_id"],),
            )
            facilities = cur.fetchall()

        elif role == "facility" and user.get("facility_id"):
            # For facility users
            cur.execute(
                """
                SELECT 
                    CASE r.region_name
                        WHEN 'Oromia' THEN 1
                        WHEN 'Amhara' THEN 2
                        WHEN 'Tigray' THEN 3
                        WHEN 'South Ethiopia' THEN 4
                        WHEN 'Afar' THEN 5
                        WHEN 'Sidama' THEN 6
                        WHEN 'Central Ethiopia' THEN 7
                        WHEN 'South West Ethiopia' THEN 8
                        ELSE r.region_id
                    END as display_region_id,
                    f.facility_id, 
                    f.facility_name, 
                    f.dhis2_uid 
                FROM facilities f
                JOIN regions r ON f.region_id = r.region_id
                WHERE f.facility_id = %s
                """,
                (user["facility_id"],),
            )
            facilities = cur.fetchall()
        else:
            facilities = []

        # Group by DISPLAY region_id
        for display_region_id, facility_id, facility_name, dhis2_uid in facilities:
            region_key = str(display_region_id)
            if region_key not in facilities_by_region:
                facilities_by_region[region_key] = []
            facilities_by_region[region_key].append(
                (str(facility_id), facility_name, dhis2_uid)
            )

    except Exception as e:
        logging.error(f"Error fetching facilities grouped by region: {e}")
    finally:
        try:
            cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    return facilities_by_region


def get_all_facilities_flat(user: dict) -> List[Tuple[str, str]]:
    """
    Get all facilities as a flat list for the current user.
    Returns list of tuples: (facility_name, dhis2_uid)
    """
    facilities_by_region = get_facilities_grouped_by_region(user)
    all_facilities: List[Tuple[str, str]] = []

    for region_facilities in facilities_by_region.values():
        all_facilities.extend(region_facilities)

    return all_facilities
