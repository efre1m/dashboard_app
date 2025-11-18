# utils/data_service.py
from typing import Optional, Dict, List, Union, Set
import pandas as pd
import logging
from utils.queries import get_orgunit_uids_for_user, get_program_by_uid
from utils.dhis2 import (
    fetch_dhis2_data_for_ous,
    REQUIRED_DATA_ELEMENTS,
    MATERNAL_HEALTH_ELEMENTS,
    NEWBORN_HEALTH_ELEMENTS,
    DATA_ELEMENT_NAMES,
)
from utils.odk_api import fetch_all_forms_as_dataframes, fetch_form_csv, list_forms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_program_data_for_user(
    user: dict,
    program_uid: str = None,
    facility_uids: List[str] = None,
    period_label: str = "Monthly",
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Fetch DHIS2 program data optimized for ALL required KPI data elements.
    Processes only the data elements needed for ALL KPIs including PPH, Uterotonic,
    Maternal Health, and Newborn Health indicators.
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

    # Use optimized fetch with ALL required data elements
    logging.info(
        f"🚀 OPTIMIZED FETCH: Using {len(REQUIRED_DATA_ELEMENTS)} required data elements"
    )
    logging.info(f"📋 Maternal Health elements: {len(MATERNAL_HEALTH_ELEMENTS)}")
    logging.info(f"📋 Newborn Health elements: {len(NEWBORN_HEALTH_ELEMENTS)}")

    dhis2_data = fetch_dhis2_data_for_ous(
        program_uid, ou_uids, user_role, required_elements=REQUIRED_DATA_ELEMENTS
    )

    patients = dhis2_data.get("patients", [])
    de_dict = dhis2_data.get("dataElements", {})
    ps_dict = dhis2_data.get("programStages", {})
    dhis2_ou_names = dhis2_data.get("orgUnitNames", {})
    optimization_stats = dhis2_data.get("optimization_stats", {})

    final_ou_names = {**dhis2_ou_names, **ou_names}

    if not patients:
        logging.warning(
            f"No patient data found for program {program_info['program_name']}."
        )
        return {"program_info": program_info}

    def map_org_name(uid: str) -> str:
        return final_ou_names.get(uid, "Unknown OrgUnit")

    # TEI DataFrame - ALL TEIs INCLUDED
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

    # Enrollment DataFrame - ALL enrollments included
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

    # Events DataFrame - Process ONLY required data elements
    events_list = []
    required_events_count = 0

    # Track all TEI IDs
    all_tei_ids = {tei.get("trackedEntityInstance") for tei in patients}

    # Process only events with required data elements
    for tei in patients:
        tei_id = tei.get("trackedEntityInstance")
        tei_org_unit = tei.get("orgUnit")

        for enrollment in tei.get("enrollments", []):
            for event in enrollment.get("events", []):
                event_orgUnit = (
                    event.get("orgUnit") or enrollment.get("orgUnit") or tei_org_unit
                )
                event_date = event.get("eventDate")

                # Process only required data elements
                for dv in event.get("dataValues", []):
                    data_element_uid = dv.get("dataElement")
                    data_value = dv.get("value")

                    # Only process if it's a required element
                    if data_element_uid in REQUIRED_DATA_ELEMENTS:
                        data_element_name = DATA_ELEMENT_NAMES.get(
                            data_element_uid,
                            de_dict.get(data_element_uid, data_element_uid),
                        )

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
                            "eventDate": event_date,
                            "dataElement_uid": data_element_uid,
                            "dataElementName": data_element_name,
                            "value": data_value,
                            "has_actual_event": True,
                        }
                        events_list.append(event_data)
                        required_events_count += 1

    # Create minimal placeholders only for TEIs with no events at all
    teis_with_events = {event["tei_id"] for event in events_list}
    teis_without_events = all_tei_ids - teis_with_events

    logging.info(
        f"📊 EVENT STATISTICS: Total TEIs={len(all_tei_ids)}, With Events={len(teis_with_events)}, Without Events={len(teis_without_events)}"
    )
    logging.info(f"📈 Required data events collected: {required_events_count}")

    # Create one placeholder per TEI without events (not per data element)
    for tei_id in teis_without_events:
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
                "dataElement_uid": "NO_DATA",
                "dataElementName": "No Data Available",
                "value": "",
                "has_actual_event": False,
            }
            events_list.append(placeholder_event)

    evt_df = pd.DataFrame(events_list)

    # Handle period labeling
    if not evt_df.empty and "eventDate" in evt_df.columns:
        evt_df["event_date"] = pd.to_datetime(evt_df["eventDate"], errors="coerce")
        has_date_mask = evt_df["event_date"].notna()

        if period_label == "Daily":
            evt_df.loc[has_date_mask, "period"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.date.astype(str)
            evt_df.loc[has_date_mask, "period_display"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y-%m-%d")
            evt_df.loc[has_date_mask, "period_sort"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y%m%d")
        elif period_label == "Monthly":
            evt_df.loc[has_date_mask, "period"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("M").astype(str)
            )
            evt_df.loc[has_date_mask, "period_display"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%b %Y")
            evt_df.loc[has_date_mask, "period_sort"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y%m")
        elif period_label == "Quarterly":
            evt_df.loc[has_date_mask, "period"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("Q").astype(str)
            )
            evt_df.loc[has_date_mask, "period_display"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("Q").astype(str)
            )
            evt_df.loc[has_date_mask, "period_sort"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.year.astype(str)
                + "Q"
                + evt_df.loc[has_date_mask, "event_date"].dt.quarter.astype(str)
            )
        else:
            evt_df.loc[has_date_mask, "period"] = (
                evt_df.loc[has_date_mask, "event_date"].dt.to_period("Y").astype(str)
            )
            evt_df.loc[has_date_mask, "period_display"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.strftime("%Y")
            evt_df.loc[has_date_mask, "period_sort"] = evt_df.loc[
                has_date_mask, "event_date"
            ].dt.year.astype(str)

    # Convert enrollment dates
    if not enr_df.empty and "enrollmentDate" in enr_df.columns:
        enr_df["enrollmentDate"] = pd.to_datetime(
            enr_df["enrollmentDate"], errors="coerce"
        )

    # Data verification - Check key data elements by category
    if not evt_df.empty:
        # MATERNAL HEALTH DATA VERIFICATION
        maternal_data = evt_df[evt_df["dataElement_uid"].isin(MATERNAL_HEALTH_ELEMENTS)]
        maternal_real_data = maternal_data[maternal_data["has_actual_event"] == True]

        # Key Maternal elements
        pph_data = evt_df[evt_df["dataElement_uid"] == "CJiTafFo0TS"]
        pph_real_data = pph_data[pph_data["has_actual_event"] == True]

        delivery_data = evt_df[evt_df["dataElement_uid"] == "lphtwP2ViZU"]
        delivery_real_data = delivery_data[delivery_data["has_actual_event"] == True]

        uterotonic_data = evt_df[evt_df["dataElement_uid"] == "yVRLuRU943e"]
        uterotonic_real_data = uterotonic_data[
            uterotonic_data["has_actual_event"] == True
        ]

        # NEWBORN HEALTH DATA VERIFICATION
        newborn_data = evt_df[evt_df["dataElement_uid"].isin(NEWBORN_HEALTH_ELEMENTS)]
        newborn_real_data = newborn_data[newborn_data["has_actual_event"] == True]

        logging.info(f"✅ MATERNAL HEALTH DATA:")
        logging.info(f"   📊 Total Maternal records: {len(maternal_real_data)}")
        logging.info(f"   🩸 PPH records: {len(pph_real_data)} real")
        logging.info(f"   👶 Delivery records: {len(delivery_real_data)} real")
        logging.info(f"   💊 Uterotonic records: {len(uterotonic_real_data)} real")

        logging.info(f"✅ NEWBORN HEALTH DATA:")
        logging.info(f"   📊 Total Newborn records: {len(newborn_real_data)}")

        # Log sample values for key indicators
        if not pph_real_data.empty:
            pph_values = pph_real_data["value"].value_counts()
            logging.info(f"   🔍 PPH value distribution: {dict(pph_values.head())}")

        if not uterotonic_real_data.empty:
            uterotonic_values = uterotonic_real_data["value"].value_counts()
            logging.info(
                f"   💊 Uterotonic value distribution: {dict(uterotonic_values.head())}"
            )

    return {
        "program_info": program_info,
        "raw_json": patients,
        "tei": tei_df,
        "enrollments": enr_df,
        "events": evt_df,
        "optimization_stats": optimization_stats,
    }


# Rest of the functions remain the same...
def fetch_odk_data_for_user(
    user: dict, form_id: str = None
) -> Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Fetch ODK form data filtered by user's access level."""
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
