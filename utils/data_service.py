# utils/data_service.py
from typing import Optional, Dict, List, Union
import pandas as pd
import logging
from utils.queries import get_orgunit_uids_for_user, get_program_by_uid
from utils.dhis2 import fetch_dhis2_data_for_ous
from utils.odk_api import fetch_all_forms_as_dataframes, fetch_form_csv, list_forms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_program_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Fetch DHIS2 program data for a given user and return structured DataFrames.
    """
    if not program_uid:
        logging.warning("No program UID provided.")
        return {}

    program_info = get_program_by_uid(program_uid)
    if not program_info:
        logging.warning(f"Program with UID {program_uid} not found in database.")
        return {}

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

    final_ou_names = {**dhis2_ou_names, **ou_names}

    if not patients:
        logging.warning(
            f"No patient data found for program {program_info['program_name']}."
        )
        return {"program_info": program_info}

    def map_org_name(uid: str) -> str:
        return final_ou_names.get(uid, "Unknown OrgUnit")

    # TEI DataFrame
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

    # Enrollment DataFrame
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

    # Events DataFrame - FIXED: Include ALL TEIs even if they have no events
    events_list = []

    # Track all TEI IDs to ensure we include everyone
    all_tei_ids = {tei.get("trackedEntityInstance") for tei in patients}

    # First, process all actual events
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
                        "has_actual_event": True,  # Mark as real event
                    }
                    events_list.append(event_data)

    # Second, create placeholder events for TEIs without any events
    teis_with_events = {event["tei_id"] for event in events_list}
    teis_without_events = all_tei_ids - teis_with_events

    logging.info(
        f"📊 TEI Statistics: Total={len(all_tei_ids)}, With Events={len(teis_with_events)}, Without Events={len(teis_without_events)}"
    )

    for tei_id in teis_without_events:
        # Find the TEI to get orgUnit info
        tei_data = next(
            (tei for tei in patients if tei.get("trackedEntityInstance") == tei_id),
            None,
        )
        if tei_data:
            # Use enrollment date as event date if available
            event_date = None
            if tei_data.get("enrollments"):
                event_date = tei_data["enrollments"][0].get("enrollmentDate")

            placeholder_event = {
                "tei_id": tei_id,
                "event": f"placeholder_{tei_id}",
                "programStage_uid": "NO_EVENTS",
                "programStageName": "No Events Recorded",
                "orgUnit": tei_data.get("orgUnit"),
                "orgUnit_name": map_org_name(tei_data.get("orgUnit")),
                "eventDate": event_date,
                "dataElement_uid": "NO_EVENTS",
                "dataElementName": "No Data Elements",
                "value": "NO_EVENTS",
                "has_actual_event": False,  # Mark as placeholder
            }
            events_list.append(placeholder_event)

    evt_df = pd.DataFrame(events_list)

    # Handle period labeling
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

    # Convert enrollment dates
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


def fetch_odk_data_for_user(
    user: dict, form_id: str = None
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Fetch ODK form data filtered by user's access level.
    Preserves all columns exactly as they are from ODK.
    """
    odk_data = {}

    try:
        if form_id:
            df = fetch_form_csv(form_id)
            if not df.empty and user:
                from utils.odk_api import filter_by_region

                user_role = user.get("role", "")
                if user_role == "regional" and user.get("region_id"):
                    df = filter_by_region(df, user.get("region_id"))
                    logging.info(
                        f"Applied regional filtering for form {form_id}, region_id: {user.get('region_id')}"
                    )
                elif user_role == "national":
                    logging.info(
                        f"✅ National user - NO filtering for form {form_id} - returning ALL data"
                    )
                # No else - national users get all data without filtering

            if not df.empty:
                odk_data[form_id] = df
                logging.info(
                    f"Retrieved ODK form '{form_id}' with {len(df)} records and {len(df.columns)} columns"
                )
            else:
                logging.warning(f"No data found for ODK form '{form_id}'")
        else:
            form_dfs = fetch_all_forms_as_dataframes(user)
            odk_data.update(form_dfs)

            user_role = user.get("role", "unknown") if user else "none"
            logging.info(
                f"✅ Retrieved {len(form_dfs)} ODK forms for user role: {user_role}"
            )

    except Exception as e:
        logging.error(f"Error fetching ODK data: {e}")

    return {"odk_forms": odk_data}


def fetch_combined_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
    odk_form_id: str = None,
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Fetch both DHIS2 program data and ODK form data for a user.
    """
    result = {}

    if program_uid:
        dhis2_data = fetch_program_data_for_user(
            user, program_uid, facility_uids, period_label
        )
        result.update(dhis2_data)

    odk_data = fetch_odk_data_for_user(user, odk_form_id)
    result.update(odk_data)

    return result


def list_available_odk_forms() -> List[dict]:
    """List all available ODK forms in the project."""
    try:
        forms = list_forms()
        return forms
    except Exception as e:
        logging.error(f"Error listing ODK forms: {e}")
        return []
