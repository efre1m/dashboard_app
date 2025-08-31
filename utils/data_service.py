from typing import Optional, Dict
import pandas as pd
import logging
from utils.queries import get_orgunit_uids_for_user, get_program_uid
from utils.dhis2 import fetch_dhis2_data_for_ous
from utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_program_data_for_user(
    user: dict,
    period_label: str = "Monthly",
    program_name: str = "Maternal Inpatient Data"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch DHIS2 program data for a given user and return structured DataFrames.
    Includes TEI, enrollment, and flattened events with both UIDs and display names.
    """
    program_uid: Optional[str] = get_program_uid(program_name)
    if not program_uid:
        logging.warning("No program UID found.")
        return {}

    ou_pairs = get_orgunit_uids_for_user(user)
    ou_uids = [ou for ou, _ in ou_pairs]
    if not ou_uids:
        logging.warning("No accessible orgUnits found.")
        return {}

    dhis2_data = fetch_dhis2_data_for_ous(program_uid, ou_uids)
    patients = dhis2_data.get("patients", [])
    de_dict = dhis2_data.get("dataElements", {})
    ps_dict = dhis2_data.get("programStages", {})

    # Tracked Entity Instances (TEI) DataFrame
    tei_df = pd.json_normalize(
        patients,
        record_path=["attributes"],
        meta=["trackedEntityInstance", "orgUnit"],
        meta_prefix="tei_",
        errors="ignore"
    ).rename(columns={
        "tei_trackedEntityInstance": "tei_id",
        "tei_orgUnit": "tei_orgUnit"
    })

    # Enrollment DataFrame
    enr_df = pd.json_normalize(
        patients,
        record_path=["enrollments"],
        meta=["trackedEntityInstance", "orgUnit"],
        meta_prefix="tei_",
        errors="ignore"
    ).rename(columns={
        "tei_trackedEntityInstance": "tei_id",
        "tei_orgUnit": "tei_orgUnit"
    })

    # Events DataFrame (flatten)
    events_list = []
    for tei in patients:
        tei_id = tei.get("trackedEntityInstance")
        for enrollment in tei.get("enrollments", []):
            for event in enrollment.get("events", []):
                event_orgUnit = event.get("orgUnit") or enrollment.get("orgUnit") or tei.get("orgUnit")
                event_date = event.get("eventDate")
                programStage = event.get("programStage")
                event_id = event.get("event")
                for dv in event.get("dataValues", []):
                    events_list.append({
                        "tei_id": tei_id,
                        "event": event_id,
                        "programStage_uid": programStage,
                        "programStageName": event.get("programStageName", ps_dict.get(programStage, programStage)),
                        "orgUnit": event_orgUnit,
                        "eventDate": event_date,
                        "dataElement_uid": dv.get("dataElement"),
                        "dataElementName": dv.get("dataElementName", de_dict.get(dv.get("dataElement"), dv.get("dataElement"))),
                        "value": dv.get("value")
                    })

    evt_df = pd.DataFrame(events_list)

    if not evt_df.empty:
        evt_df["event_date"] = pd.to_datetime(evt_df.get("eventDate", pd.NaT), errors="coerce")

        # Period column based on label
        if period_label == "Daily":
            evt_df["period"] = evt_df["event_date"].dt.date
        elif period_label == "Monthly":
            evt_df["period"] = evt_df["event_date"].dt.to_period("M").astype(str)
        elif period_label == "Quarterly":
            evt_df["period"] = evt_df["event_date"].dt.to_period("Q").astype(str)
        else:
            evt_df["period"] = evt_df["event_date"].dt.to_period("Y").astype(str)

    # Enrollment Dates
    if not enr_df.empty and "enrollmentDate" in enr_df.columns:
        enr_df["enrollmentDate"] = pd.to_datetime(enr_df["enrollmentDate"], errors="coerce")

    return {
        "raw_json": patients,
        "tei": tei_df,
        "enrollments": enr_df,
        "events": evt_df,
    }
