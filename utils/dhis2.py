import requests
import logging
import time
from typing import List, Dict, Any
from utils.config import settings
from requests.adapters import HTTPAdapter, Retry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_session: requests.Session | None = None


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
    program_uid: str, ou_uid: str, user_role: str
) -> List[Dict[str, Any]]:
    """
    Fetch paginated TEIs for a program + orgUnit with proper hierarchical support.
    - Facility: fetch only from that facility
    - Regional: fetch all descendants of the region
    - National: fetch all descendants of the region (for regional iteration)
    """
    url = f"{settings.DHIS2_BASE_URL}/api/trackedEntityInstances.json"
    all_patients: List[Dict[str, Any]] = []
    page = 1
    logging.info(f"Fetching patient data for OU: {ou_uid}, role: {user_role}")

    while True:
        params = {
            "program": program_uid,
            "ou": ou_uid,
            "fields": (
                "trackedEntityInstance,"
                "orgUnit,"
                "attributes[attribute,value],"
                "enrollments[enrollment,program,status,enrollmentDate,orgUnit,"
                "events[event,programStage,status,eventDate,orgUnit,"
                "dataValues[dataElement,value]]]"
            ),
            "pageSize": 5000,  # large enough to capture all data per page
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
            all_patients.extend(teis)

            pager = data.get("pager", {})
            if page >= pager.get("pageCount", 0) or not teis:
                break
            page += 1
            time.sleep(0.3)  # avoid hammering DHIS2
        except requests.RequestException as e:
            logging.error(f"Patient data fetch failed for OU {ou_uid}: {e}")
            break

    logging.info(f"Fetched {len(all_patients)} TEIs for OU: {ou_uid}")
    return all_patients


def fetch_national_data(program_uid: str) -> List[Dict[str, Any]]:
    """
    Fetch TEIs region by region for full national coverage.
    This matches the approach in your working regional script.
    """
    regions = fetch_top_regions()
    all_patients = []

    for ou_uid, name in regions.items():
        logging.info(f"Fetching TEIs for region: {name}")
        region_patients = fetch_patient_data(program_uid, ou_uid, "national")
        all_patients.extend(region_patients)
        logging.info(f"Total TEIs collected so far: {len(all_patients)}")
        time.sleep(1)  # Be gentle with the API

    return all_patients


def fetch_dhis2_data_for_ous(
    program_uid: str, ou_uids: List[str], user_role: str
) -> Dict[str, Any]:
    """
    Fetch all TEIs for all OUs in program, enrich events with dataElement + programStage names,
    and include orgUnit names for each UID.

    For national level, use the regional iteration approach instead of direct OU fetching.
    """
    all_patients: List[Dict[str, Any]] = []
    logging.info(f"Fetching DHIS2 data for role: {user_role}")

    # For national level, use the regional iteration approach
    if user_role == "national":
        logging.info("Using national data fetching strategy (region by region)")
        all_patients = fetch_national_data(program_uid)
    else:
        # For facility and regional levels, use the original approach
        logging.info(f"Fetching DHIS2 data for {len(ou_uids)} OUs, role: {user_role}")
        for ou_uid in ou_uids:
            patients = fetch_patient_data(program_uid, ou_uid, user_role)
            all_patients.extend(patients)
            time.sleep(1)

    # Fetch all orgUnit names
    orgunit_names = fetch_orgunit_names()

    # Fetch dataElements
    try:
        de_resp = _get_session().get(
            f"{settings.DHIS2_BASE_URL}/api/dataElements.json?fields=id,displayName&paging=false"
        )
        de_resp.raise_for_status()
        de_dict = {
            de["id"]: de["displayName"] for de in de_resp.json().get("dataElements", [])
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

    # Inject display names into events
    for tei in all_patients:
        for enrollment in tei.get("enrollments", []):
            for event in enrollment.get("events", []):
                event["programStageName"] = ps_dict.get(
                    event.get("programStage"), event.get("programStage")
                )
                for dv in event.get("dataValues", []):
                    dv["dataElementName"] = de_dict.get(
                        dv.get("dataElement"), dv.get("dataElement")
                    )

    result = {
        "patients": all_patients,
        "dataElements": de_dict,
        "programStages": ps_dict,
        "orgUnitNames": orgunit_names,
        "total_ous": (
            len(ou_uids) if user_role != "national" else len(fetch_top_regions())
        ),
        "total_teis": len(all_patients),
    }

    logging.info(f"Completed fetching data. Total TEIs: {len(all_patients)}")
    return result
