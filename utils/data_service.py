# utils/data_service.py
from typing import Optional, Dict, List
import pandas as pd
import logging
from utils.queries import get_orgunit_uids_for_user, get_program_by_uid
from utils.dhis2 import fetch_dhis2_data_for_ous
from utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_program_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch DHIS2 program data for a given user and return structured DataFrames.

    Args:
        user: User dictionary with role and access info
        program_uid: Specific program UID to fetch data for
        facility_uids: Optional list of facility UIDs to filter by
        period_label: Period aggregation level
    """
    # If no program_uid provided, we can't fetch data
    if not program_uid:
        logging.warning("No program UID provided.")
        return {}

    # Get program info for display purposes
    program_info = get_program_by_uid(program_uid)
    if not program_info:
        logging.warning(f"Program with UID {program_uid} not found in database.")
        return {}

    # Get OU access based on role
    if facility_uids:
        ou_uids = facility_uids
        all_ou_pairs = get_orgunit_uids_for_user(user)
        ou_names = {uid: name for uid, name in all_ou_pairs if uid in facility_uids}
    else:
        ou_pairs = get_orgunit_uids_for_user(user)
        ou_uids = [ou for ou, _ in ou_pairs]
        ou_names = {uid: name for uid, name in ou_pairs}

    if not ou_uids:
        logging.warning("No accessible orgUnits found.")
        return {}

    user_role = user.get("role", "")
    dhis2_data = fetch_dhis2_data_for_ous(program_uid, ou_uids, user_role)
    patients = dhis2_data.get("patients", [])
    de_dict = dhis2_data.get("dataElements", {})
    ps_dict = dhis2_data.get("programStages", {})
    dhis2_ou_names = dhis2_data.get("orgUnitNames", {})

    # Merge local OU names with DHIS2 OU names
    final_ou_names = {**dhis2_ou_names, **ou_names}

    if not patients:
        logging.warning(
            f"No patient data found for program {program_info['program_name']}."
        )
        return {"program_info": program_info}

    # Helper: map UID -> name
    def map_org_name(uid: str) -> str:
        return final_ou_names.get(uid, "Unknown OrgUnit")

    # --- TEI DataFrame ---
    tei_df = pd.json_normalize(
        patients,
        record_path=["attributes"],
        meta=["trackedEntityInstance", "orgUnit"],
        meta_prefix="tei_",
        errors="ignore",
    ).rename(
        columns={"tei_trackedEntityInstance": "tei_id", "tei_orgUnit": "tei_orgUnit"}
    )
    if not tei_df.empty:
        tei_df["orgUnit_name"] = tei_df["tei_orgUnit"].apply(map_org_name)

    # --- Enrollment DataFrame ---
    enr_df = pd.json_normalize(
        patients,
        record_path=["enrollments"],
        meta=["trackedEntityInstance", "orgUnit"],
        meta_prefix="tei_",
        errors="ignore",
    ).rename(
        columns={"tei_trackedEntityInstance": "tei_id", "tei_orgUnit": "tei_orgUnit"}
    )
    if not enr_df.empty:
        enr_df["orgUnit_name"] = enr_df["tei_orgUnit"].apply(map_org_name)

    # --- Events DataFrame ---
    events_list = []
    for tei in patients:
        tei_id = tei.get("trackedEntityInstance")
        tei_org_unit = tei.get("orgUnit")
        for enrollment in tei.get("enrollments", []):
            for event in enrollment.get("events", []):
                event_orgUnit = (
                    event.get("orgUnit") or enrollment.get("orgUnit") or tei_org_unit
                )
                for dv in event.get("dataValues", []):
                    event_data = {
                        "tei_id": tei_id,
                        "event": event.get("event"),
                        "programStage_uid": event.get("programStage"),
                        "programStageName": event.get(
                            "programStageName",
                            ps_dict.get(
                                event.get("programStage"), event.get("programStage")
                            ),
                        ),
                        "orgUnit": event_orgUnit,
                        "orgUnit_name": map_org_name(event_orgUnit),
                        "eventDate": event.get("eventDate"),
                        "dataElement_uid": dv.get("dataElement"),
                        "dataElementName": dv.get(
                            "dataElementName",
                            de_dict.get(dv.get("dataElement"), dv.get("dataElement")),
                        ),
                        "value": dv.get("value"),
                    }
                    events_list.append(event_data)

    evt_df = pd.DataFrame(events_list)

    # --- Handle period labeling ---
    if not evt_df.empty and "eventDate" in evt_df.columns:
        evt_df["event_date"] = pd.to_datetime(evt_df["eventDate"], errors="coerce")
        if period_label == "Daily":
            evt_df["period"] = evt_df["event_date"].dt.date
        elif period_label == "Monthly":
            evt_df["period"] = evt_df["event_date"].dt.to_period("M").astype(str)
        elif period_label == "Quarterly":
            evt_df["period"] = evt_df["event_date"].dt.to_period("Q").astype(str)
        else:
            evt_df["period"] = evt_df["event_date"].dt.to_period("Y").astype(str)

    # --- Convert enrollment dates ---
    if not enr_df.empty and "enrollmentDate" in enr_df.columns:
        enr_df["enrollmentDate"] = pd.to_datetime(
            enr_df["enrollmentDate"], errors="coerce"
        )

    return {
        "program_info": program_info,
        "raw_json": patients,
        "tei": tei_df,
        "enrollments": enr_df,
        "events": evt_df,
    }
