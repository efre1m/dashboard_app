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
        s.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))
        _session = s
        logging.info("DHIS2 session initialized.")
    return _session


def fetch_orgunit_names() -> Dict[str, str]:
    """Fetch all orgUnit UIDs and names from DHIS2."""
    url = f"{settings.DHIS2_BASE_URL}/api/organisationUnits.json?fields=id,displayName&paging=false"
    try:
        resp = _get_session().get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ou_dict = {ou["id"]: ou["displayName"] for ou in data.get("organisationUnits", [])}
        logging.info(f"Fetched {len(ou_dict)} orgUnits from DHIS2.")
        return ou_dict
    except Exception as e:
        logging.error(f"Failed to fetch orgUnit names: {e}")
        return {}


def fetch_patient_data(program_uid: str, ou_uid: str, user_role: str) -> List[Dict[str, Any]]:
    """
    Fetch paginated TEIs for a program + orgUnit with proper hierarchical support.
    For regional and national users, use DESCENDANTS mode to get all child facilities.
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
            "pageSize": 100,
            "page": page,
            "paging": "true"
        }
        
        # Use DESCENDANTS mode for regional and national users
        if user_role in ["regional", "national"]:
            params["ouMode"] = "DESCENDANTS"
            logging.info(f"Using DESCENDANTS mode for {user_role} user with OU: {ou_uid}")

        try:
            resp = _get_session().get(url, params=params, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            teis = data.get("trackedEntityInstances", [])
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


def fetch_dhis2_data_for_ous(program_uid: str, ou_uids: List[str], user_role: str) -> Dict[str, Any]:
    """
    Fetch all TEIs for all OUs in program, enrich events with dataElement + programStage names,
    and include orgUnit names for each UID.
    """
    all_patients: List[Dict[str, Any]] = []
    logging.info(f"Fetching hierarchical DHIS2 data for {len(ou_uids)} OUs, role: {user_role}")

    # Fetch all orgUnit names once
    orgunit_names = fetch_orgunit_names()

    for ou_uid in ou_uids:
        patients = fetch_patient_data(program_uid, ou_uid, user_role)
        all_patients.extend(patients)
        time.sleep(1)

    # Fetch dataElements
    try:
        de_resp = _get_session().get(
            f"{settings.DHIS2_BASE_URL}/api/dataElements.json?fields=id,displayName&paging=false"
        )
        de_resp.raise_for_status()
        de_dict = {de["id"]: de["displayName"] for de in de_resp.json().get("dataElements", [])}
    except Exception as e:
        logging.warning(f"Failed to fetch dataElement names: {e}")
        de_dict = {}

    # Fetch programStages
    try:
        ps_resp = _get_session().get(
            f"{settings.DHIS2_BASE_URL}/api/programStages.json?program={program_uid}&fields=id,displayName&paging=false"
        )
        ps_resp.raise_for_status()
        ps_dict = {ps["id"]: ps["displayName"] for ps in ps_resp.json().get("programStages", [])}
    except Exception as e:
        logging.warning(f"Failed to fetch program stage names: {e}")
        ps_dict = {}

    # Inject display names into events
    for tei in all_patients:
        for enrollment in tei.get("enrollments", []):
            for event in enrollment.get("events", []):
                event["programStageName"] = ps_dict.get(event.get("programStage"), event.get("programStage"))
                for dv in event.get("dataValues", []):
                    dv["dataElementName"] = de_dict.get(dv.get("dataElement"), dv.get("dataElement"))

    result = {
        "patients": all_patients,
        "dataElements": de_dict,
        "programStages": ps_dict,
        "orgUnitNames": orgunit_names,
        "total_ous": len(ou_uids),
        "total_teis": len(all_patients)
    }

    logging.info(f"Completed fetching hierarchical data. Total TEIs: {len(all_patients)}")
    return result