import requests
import logging
import time
from typing import List, Dict, Any, Set, Optional
from utils.config import settings
from requests.adapters import HTTPAdapter, Retry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_session: requests.Session | None = None

# Define REQUIRED data elements SEPARATED by program stage
MATERNAL_HEALTH_ELEMENTS = {
    # Core Maternal Health Indicators
    "Q1p7CxWGUoi",  # FP Counseling and Method Provided pp (IPPCAR)
    "lphtwP2ViZU",  # Mode of Delivery maternal (Denominator for multiple KPIs)
    "wZig9cek3Gv",  # Birth Outcome (Stillbirth, Live birth)
    "VzwnSBROvUm",  # Number of Newborns (Multiple births)
    "tIa0WvbPGLk",  # Other Number of Newborns (Multiple births)
    "z7Eb2yFLOBI",  # Date/stay pp (Early PNC)
    "TjQOcW6tm8k",  # Condition of Discharge (Maternal Death)
    "CJiTafFo0TS",  # Obstetric condition at delivery (PPH)
    "yVRLuRU943e",  # Uterotonics given (Uterotonic KPI)
    # HIV/ARV Indicators (Maternal context)
    "tTrH9cOQRnZ",  # HIV Result
    "H7J2SxBpObS",  # ARV Rx for Newborn (By type) pp
    # Birth Weight (Maternal context - recorded at birth)
    "QUlJEvzGcQK",  # Birth Weight (grams)
    # Assisted Delivery Indicator - NEW
    "K8BCYRU1TUP",  # Instrumental delivery (Assisted Delivery KPI)
}

NEWBORN_HEALTH_ELEMENTS = {
    # Newborn Health Indicators (Admission/Newborn stage)
    "QK7Fi6OwtDC",  # KMC Administered
    "yxWUMt3sCil",  # Weight on admission
    "gZi9y12E9i7",  # Temperature on admission (°C)
    "UOmhJkyAK6h",  # Date of Admission
    "wlHEf9FdmJM",  # CPAP Administered
    # Reason for Admission (Newborn)
    "T30GbTiVgFR",  # First Reason for Admission
    "OpHw2X58x5i",  # Second Reason for Admission
    "gJH6PkYI6IV",  # Third Reason for Admission
    "aK5txmRYpVX",  # birth location
    "vmOAGuFcaz4",  # Newborn Status at Discharge
    "yBCwmQP0A6a",  # Discharge Weight (grams):
    "nIKIu6f5vbW",  # lowest recorded temperature (Celsius)
    "sxtsEDilKZd",  # Were antibiotics administered?
    "wn0tHaHcceW",  # Sub-Categories of Infection:
}

# Combine for complete coverage
REQUIRED_DATA_ELEMENTS = MATERNAL_HEALTH_ELEMENTS | NEWBORN_HEALTH_ELEMENTS

# Map data element UIDs to human-readable names
DATA_ELEMENT_NAMES = {
    # Maternal Health
    "Q1p7CxWGUoi": "FP Counseling and Method Provided pp",
    "lphtwP2ViZU": "Mode of Delivery maternal",
    "wZig9cek3Gv": "Birth Outcome",
    "VzwnSBROvUm": "Number of Newborns",
    "tIa0WvbPGLk": "Other Number of Newborns",
    "z7Eb2yFLOBI": "Date/stay pp",
    "TjQOcW6tm8k": "Condition of Discharge",
    "CJiTafFo0TS": "Obstetric condition at delivery",
    "yVRLuRU943e": "Uterotonics given",
    # HIV/ARV
    "tTrH9cOQRnZ": "HIV Result",
    "H7J2SxBpObS": "ARV Rx for Newborn (By type) pp",
    # Birth Weight
    "QUlJEvzGcQK": "Birth Weight (grams)",
    # Assisted Delivery - NEW
    "K8BCYRU1TUP": "Instrumental delivery",
    # Newborn Health
    "QK7Fi6OwtDC": "KMC Administered",
    "yxWUMt3sCil": "Weight on admission",
    "gZi9y12E9i7": "Temperature on admission (°C)",
    "UOmhJkyAK6h": "Date of Admission",
    "wlHEf9FdmJM": "CPAP Administered",
    # Reason for Admission
    "T30GbTiVgFR": "First Reason for Admission",
    "OpHw2X58x5i": "Second Reason for Admission",
    "gJH6PkYI6IV": "Third Reason for Admission",
    "aK5txmRYpVX": "birth location",
    "vmOAGuFcaz4": "Newborn Status at Discharge",
    "yBCwmQP0A6a": "Discharge Weight (grams)",
    "nIKIu6f5vbW": "lowest recorded temperature (Celsius)",
    "sxtsEDilKZd": "Were antibiotics administered?",
    "wn0tHaHcceW": "Sub-Categories of Infection",
}


def _get_session() -> requests.Session:
    """Create/reuse a requests session with retries configured."""
    global _session
    if _session is None:
        s = requests.Session()
        s.auth = (settings.DHIS2_USERNAME, settings.DHIS2_PASSWORD)
        s.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            }
        )
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        _session = s
        logging.info("DHIS2 session initialized.")
    return _session


def fetch_top_regions() -> Dict[str, str]:
    """Fetch all top-level regions (level=2)."""
    url = f"{settings.DHIS2_BASE_URL}/api/organisationUnits.json?level=2&fields=id,displayName&paging=false"
    try:
        resp = _get_session().get(url, timeout=60)
        resp.raise_for_status()
        regions = {
            ou["id"]: ou["displayName"]
            for ou in resp.json().get("organisationUnits", [])
        }
        logging.info(f"Top-level regions fetched: {len(regions)}")
        return regions
    except Exception as e:
        logging.error(f"Failed to fetch top regions: {e}")
        return {}


def fetch_orgunit_names() -> Dict[str, str]:
    """Fetch all orgUnit UIDs and names from DHIS2."""
    url = f"{settings.DHIS2_BASE_URL}/api/organisationUnits.json?fields=id,displayName&paging=false"
    try:
        resp = _get_session().get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ou_dict = {
            ou["id"]: ou["displayName"] for ou in data.get("organisationUnits", [])
        }
        logging.info(f"Fetched {len(ou_dict)} orgUnits from DHIS2.")
        return ou_dict
    except Exception as e:
        logging.error(f"Failed to fetch orgUnit names: {e}")
        return {}


def fetch_patient_data(
    program_uid: str,
    ou_uid: str,
    user_role: str,
    required_elements: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch paginated TEIs for a program + orgUnit with proper hierarchical support.
    Fetches ALL data but will filter data values later.
    """
    url = f"{settings.DHIS2_BASE_URL}/api/trackedEntityInstances.json"
    all_patients: List[Dict[str, Any]] = []
    page = 1

    # Build fields parameter - fetch ALL data elements
    base_fields = (
        "trackedEntityInstance,"
        "orgUnit,"
        "attributes[attribute,value],"
        "enrollments[enrollment,program,status,enrollmentDate,orgUnit,"
        "events[event,programStage,status,eventDate,orgUnit,"
        "dataValues[dataElement,value]]]"
    )

    logging.info(f"Fetching patient data for OU: {ou_uid}, role: {user_role}")
    if required_elements:
        logging.info(f"Will filter for {len(required_elements)} required data elements")

    while True:
        params = {
            "program": program_uid,
            "ou": ou_uid,
            "fields": base_fields,
            "pageSize": 5000,
            "page": page,
            "paging": "true",
        }

        # Apply hierarchy rules
        if user_role in ["regional", "national"]:
            params["ouMode"] = "DESCENDANTS"
            logging.info(f"Using DESCENDANTS mode for {user_role} with OU {ou_uid}")

        try:
            resp = _get_session().get(url, params=params, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            teis = data.get("trackedEntityInstances", [])

            # Filter data values if required elements specified
            if required_elements:
                filtered_teis = _filter_data_values(teis, required_elements)
                all_patients.extend(filtered_teis)
            else:
                all_patients.extend(teis)

            pager = data.get("pager", {})
            if page >= pager.get("pageCount", 0) or not teis:
                break
            page += 1
            time.sleep(0.3)
        except requests.RequestException as e:
            logging.error(f"Patient data fetch failed for OU {ou_uid}: {e}")
            break

    logging.info(f"Fetched {len(all_patients)} TEIs for OU: {ou_uid}")
    return all_patients


def _filter_data_values(
    teis: List[Dict[str, Any]], required_elements: Set[str]
) -> List[Dict[str, Any]]:
    """
    Filter data values to only include required elements while preserving TEI structure.
    This ensures TEI-data element associations remain correct.
    """
    filtered_teis = []

    for tei in teis:
        filtered_tei = tei.copy()
        filtered_enrollments = []

        for enrollment in tei.get("enrollments", []):
            filtered_enrollment = enrollment.copy()
            filtered_events = []

            for event in enrollment.get("events", []):
                # Keep only required data elements
                required_data_values = [
                    dv
                    for dv in event.get("dataValues", [])
                    if dv.get("dataElement") in required_elements
                ]

                # Only include event if it has required data OR if we want to preserve all events
                # For now, we'll preserve events even if they have no required data
                # to maintain TEI-event associations
                filtered_event = event.copy()
                filtered_event["dataValues"] = required_data_values
                filtered_events.append(filtered_event)

            filtered_enrollment["events"] = filtered_events
            filtered_enrollments.append(filtered_enrollment)

        filtered_tei["enrollments"] = filtered_enrollments
        filtered_teis.append(filtered_tei)

    return filtered_teis


def fetch_national_data(
    program_uid: str, required_elements: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch TEIs region by region for full national coverage.
    """
    regions = fetch_top_regions()
    all_patients = []

    for ou_uid, name in regions.items():
        logging.info(f"Fetching TEIs for region: {name}")
        region_patients = fetch_patient_data(
            program_uid, ou_uid, "national", required_elements
        )
        all_patients.extend(region_patients)
        logging.info(f"Total TEIs collected so far: {len(all_patients)}")
        time.sleep(1)

    return all_patients


def fetch_dhis2_data_for_ous(
    program_uid: str,
    ou_uids: List[str],
    user_role: str,
    required_elements: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Fetch all TEIs for all OUs in program.
    If required_elements is specified, filter data values to only those elements.
    """
    all_patients: List[Dict[str, Any]] = []

    if required_elements:
        logging.info(
            f"🚀 OPTIMIZED FETCH: Will filter for {len(required_elements)} required data elements"
        )
        logging.info(f"📋 Maternal Health elements: {len(MATERNAL_HEALTH_ELEMENTS)}")
        logging.info(f"📋 Newborn Health elements: {len(NEWBORN_HEALTH_ELEMENTS)}")
    else:
        logging.info("📋 FULL FETCH: Fetching all data elements")

    logging.info(f"Fetching DHIS2 data for role: {user_role}")

    # For national level, use the regional iteration approach
    if user_role == "national":
        logging.info("Using national data fetching strategy (region by region)")
        all_patients = fetch_national_data(program_uid, required_elements)
    else:
        # For facility and regional levels
        logging.info(f"Fetching DHIS2 data for {len(ou_uids)} OUs, role: {user_role}")
        for ou_uid in ou_uids:
            patients = fetch_patient_data(
                program_uid, ou_uid, user_role, required_elements
            )
            all_patients.extend(patients)
            time.sleep(1)

    # Fetch all orgUnit names
    orgunit_names = fetch_orgunit_names()

    # Fetch only required data elements if specified
    if required_elements:
        de_ids = ",".join(required_elements)
        try:
            de_resp = _get_session().get(
                f"{settings.DHIS2_BASE_URL}/api/dataElements.json?fields=id,displayName&filter=id:in:[{de_ids}]&paging=false"
            )
            de_resp.raise_for_status()
            de_dict = {
                de["id"]: de["displayName"]
                for de in de_resp.json().get("dataElements", [])
            }
            logging.info(f"Fetched names for {len(de_dict)} required data elements")
        except Exception as e:
            logging.warning(f"Failed to fetch required dataElement names: {e}")
            # Fall back to our predefined names
            de_dict = {
                uid: DATA_ELEMENT_NAMES.get(uid, uid) for uid in required_elements
            }
    else:
        try:
            de_resp = _get_session().get(
                f"{settings.DHIS2_BASE_URL}/api/dataElements.json?fields=id,displayName&paging=false"
            )
            de_resp.raise_for_status()
            de_dict = {
                de["id"]: de["displayName"]
                for de in de_resp.json().get("dataElements", [])
            }
        except Exception as e:
            logging.warning(f"Failed to fetch dataElement names: {e}")
            de_dict = {}

    # Fetch programStages
    try:
        ps_resp = _get_session().get(
            f"{settings.DHIS2_BASE_URL}/api/programStages.json?program={program_uid}&fields=id,displayName&paging=false"
        )
        ps_resp.raise_for_status()
        ps_dict = {
            ps["id"]: ps["displayName"]
            for ps in ps_resp.json().get("programStages", [])
        }
    except Exception as e:
        logging.warning(f"Failed to fetch program stage names: {e}")
        ps_dict = {}

    # Count statistics
    total_teis = len(all_patients)

    # Count TEIs with required data
    teis_with_required_data = 0
    for tei in all_patients:
        has_required_data = any(
            any(
                dv.get("dataElement") in (required_elements or set())
                for dv in event.get("dataValues", [])
            )
            for enrollment in tei.get("enrollments", [])
            for event in enrollment.get("events", [])
        )
        if has_required_data:
            teis_with_required_data += 1

    logging.info(f"📊 DATA SUMMARY:")
    logging.info(f"   Total TEIs: {total_teis}")
    if required_elements:
        logging.info(f"   TEIs with required data: {teis_with_required_data}")
        logging.info(
            f"   Required data coverage: {teis_with_required_data}/{total_teis} = {(teis_with_required_data/total_teis*100):.1f}%"
        )

    result = {
        "patients": all_patients,
        "dataElements": de_dict,
        "programStages": ps_dict,
        "orgUnitNames": orgunit_names,
        "total_ous": (
            len(ou_uids) if user_role != "national" else len(fetch_top_regions())
        ),
        "total_teis": total_teis,
        "optimization_stats": {
            "required_data_elements_count": (
                len(required_elements) if required_elements else 0
            ),
            "teis_with_required_data": teis_with_required_data,
            "maternal_elements_count": len(MATERNAL_HEALTH_ELEMENTS),
            "newborn_elements_count": len(NEWBORN_HEALTH_ELEMENTS),
        },
    }

    logging.info(f"✅ Completed fetching data. Total TEIs: {total_teis}")
    return result
