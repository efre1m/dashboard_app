#!/usr/bin/env python3
"""
Fetch DHIS2 events for the IMNID assessment program and export to CSV with
region and facility mapping.

Usage (from repo root):
  python imnid_assessment_fetch.py --org-unit WWIYKuA6f3s

Defaults:
  program: zXzz681r0Z2
  ouMode: DESCENDANTS
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

DATA_ELEMENT_NAME_MAP: Dict[str, str] = {
    "xkPiGZho4Hg": "NDQA Total  maternal admission on week",
    "URABbZdEPUQ": "NDQA Total Neontal admission in a week",
    "QQLrQ9kTgH6": "NDQA Birth outcome - Paper Recorded",
    "F0RA6yMAPsD": "NDQA Birth weight - Paper Recorded",
    "Y2c5eki6TmA": "NDQA CPAP - Paper Recorded",
    "JSzUmvp3JTG": "NDQA KMC (Kangaroo Mother Care) - Paper Recorded",
    "ZpTVZ6iOcLk": "NDQA Maternal discharge summary - Paper Recorded",
    "eZTUmV6aznh": "NDQA Mode of delivery - Paper Recorded",
    "W2g3O7Zwz7S": "NDQA Neonatal status at discharge - Paper Recorded",
    "PkVdeqWXp6R": "NDQA Temperature at neonatal admission - Paper Recorded",
    "K2iQurm35Hu": "NDQA Birth outcome - DHIS2 Recorded",
    "vXphURrsTGI": "NDQA Birth weight - DHIS2 Recorded",
    "vLTVpZjfo8y": "NDQA CPAP - DHIS2 Recorded",
    "gLgkv8ge1Gh": "NDQA KMC (Kangaroo Mother Care) - DHIS2 Recorded",
    "r1IIkKoVvR1": "NDQA Maternal discharge summary - DHIS2 Recorded",
    "b2vE4gjyh3O": "NDQA Mode of delivery - DHIS2 Recorded",
    "Wa1zc97vx8i": "NDQA Neonatal status at discharge - DHIS2 Recorded",
    "zYsl1phevoZ": "NDQA Temperature at neonatal admission - DHIS2 Recorded",
    "kplC9sV0TsM": "NDQA Birth outcome - Consistent Across Systems",
    "XmkfLWR05je": "NDQA Birth weight - Consistent Across Systems",
    "aOKSHCruCnJ": "NDQA CPAP - Consistent Across Systems",
    "NYS8MjwGLhJ": "NDQA KMC (Kangaroo Mother Care) - Consistent Across Systems",
    "PRisGq8Tnd5": "NDQA Maternal discharge summary - Consistent Across Systems",
    "BwkEcS3aMaB": "NDQA Mode of delivery - Consistent Across Systems",
    "WpAk8yj4zri": "NDQA Neonatal status at discharge - Consistent Across Systems",
    "VuNGpgJ2pcv": "NDQA Temperature at neonatal admission - Consistent Across Systems",
    "AXWhkNG9Tpg": "NDQA CPAP - Reason not recorded in Paper",
    "aHOM6LlC38i": "NDQA KMC (Kangaroo Mother Care) - Reason not recorded in Paper",
    "Ng69O2avVzd": "NDQA Neonatal status at discharge - Reason not recorded in Paper",
    "mmHRZUVLPGS": "NDQA Birth weight - Reason not recorded in DHIS2",
    "ArNxkuwg5d4": "NDQA CPAP - Reason not recorded in DHIS2",
    "C7e3z5pQauk": "NDQA KMC (Kangaroo Mother Care) - Reason not recorded in DHIS2",
    "ZupAhOapGvg": "NDQA Maternal discharge summary - Reason not recorded in DHIS2",
    "FjcIPo45fyW": "NDQA Neonatal status at discharge - Reason not recorded in DHIS2",
    "H3tjl6MjOBg": "NDQA Temperature at neonatal admission - Reason not recorded in DHIS2",
    "LLFe4HEHAkh": "NDQA Comments",
    "bxro4K5ZyLj": "NDQA Paper data entry date of the Mother/newborn record being checked",
    "DHaFz2MmMfK": "NDQA MRN of the Mother/newborn record being checked",
    "j0miklPFg7K": "NDQA Enrollment date of the Mother/newborn record being checked",
    "Hbhiw6jDVSX": "NDQA Which Data are you checking",
}

PROGRAM_STAGE_NAME_MAP: Dict[str, str] = {
    "eY6dYXsn0Kh": "IMNID Data Quality Assessment",
}


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def load_settings() -> Tuple[str, str, str, int]:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(repo_root, ".env"))

    base_url = (os.getenv("DHIS2_BASE_URL") or "").rstrip("/")
    username = os.getenv("DHIS2_USERNAME") or ""
    password = os.getenv("DHIS2_PASSWORD") or ""
    timeout = int(os.getenv("DHIS2_TIMEOUT", "60"))

    if not base_url or not username or not password:
        raise RuntimeError(
            "Missing DHIS2 credentials. Ensure DHIS2_BASE_URL, DHIS2_USERNAME, "
            "and DHIS2_PASSWORD are set in .env."
        )

    return base_url, username, password, timeout


def fetch_events(
    session: requests.Session,
    base_url: str,
    program_uid: str,
    org_unit_uid: str,
    ou_mode: str,
    page_size: int,
    timeout: int,
) -> List[dict]:
    url = f"{base_url}/api/29/events.json"

    fields = (
        "event,program,programStage,orgUnit,orgUnitName,eventDate,created,"
        "lastUpdated,status,deleted,attributeOptionCombo,"
        "dataValues[dataElement,value]"
    )

    all_events: List[dict] = []
    page = 1

    while True:
        params = {
            "program": program_uid,
            "orgUnit": org_unit_uid,
            "ouMode": ou_mode,
            "pageSize": page_size,
            "page": page,
            "fields": fields,
        }

        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        events = data.get("events", [])
        all_events.extend(events)

        pager = data.get("pager", {})
        is_last = pager.get("isLastPage", False)
        page_count = pager.get("pageCount")
        current_page = pager.get("page", page)

        if is_last:
            break
        if page_count is not None and current_page >= page_count:
            break
        if not pager and not events:
            break

        page += 1
        time.sleep(0.2)

    return all_events


def fetch_orgunit_region_mapping(
    session: requests.Session, base_url: str, org_unit_ids: List[str], timeout: int
) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping:
      orgUnitId -> {"facility_name": ..., "region_name": ..., "region_uid": ...}
    """
    if not org_unit_ids:
        return {}

    url = f"{base_url}/api/organisationUnits.json"
    mapping: Dict[str, Dict[str, str]] = {}

    for batch in chunked(sorted(set(org_unit_ids)), 50):
        ids = ",".join(batch)
        params = {
            "fields": "id,displayName,level,ancestors[id,displayName,level]",
            "filter": f"id:in:[{ids}]",
            "paging": False,
        }
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        for ou in data.get("organisationUnits", []):
            region_name = ""
            region_uid = ""
            for anc in ou.get("ancestors", []) or []:
                if anc.get("level") == 2:
                    region_name = anc.get("displayName", "")
                    region_uid = anc.get("id", "")
                    break

            mapping[ou["id"]] = {
                "facility_name": ou.get("displayName", ""),
                "region_name": region_name,
                "region_uid": region_uid,
            }

    return mapping


def events_to_dataframe(
    events: List[dict], orgunit_map: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    rows: List[dict] = []
    unknown_data_elements: set[str] = set()

    for ev in events:
        ou_id = ev.get("orgUnit", "")
        ou_name = ev.get("orgUnitName") or orgunit_map.get(ou_id, {}).get("facility_name", "")
        region_name = orgunit_map.get(ou_id, {}).get("region_name", "")
        stage_name = PROGRAM_STAGE_NAME_MAP.get(ev.get("programStage", ""), "")

        row = {
            "region_name": region_name,
            "facility_name": ou_name,
            "program_stage": stage_name,
            "eventDate": ev.get("eventDate", ""),
            "status": ev.get("status", ""),
            "created": ev.get("created", ""),
            "lastUpdated": ev.get("lastUpdated", ""),
            "deleted": ev.get("deleted", ""),
        }

        for dv in ev.get("dataValues", []) or []:
            de = dv.get("dataElement")
            if not de:
                continue
            de_name = DATA_ELEMENT_NAME_MAP.get(de)
            if not de_name:
                unknown_data_elements.add(de)
                continue
            row[de_name] = dv.get("value", "")

        rows.append(row)

    df = pd.DataFrame(rows)
    if unknown_data_elements:
        print(
            "Warning: Skipped data elements without names: "
            + ", ".join(sorted(unknown_data_elements))
        )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch IMNID assessment events from DHIS2 to CSV.")
    parser.add_argument("--program", default="zXzz681r0Z2", help="Program UID")
    parser.add_argument("--org-unit", required=True, help="Root orgUnit UID")
    parser.add_argument("--ou-mode", default="DESCENDANTS", help="OrgUnit mode")
    parser.add_argument("--page-size", type=int, default=1000, help="DHIS2 page size")
    parser.add_argument("--out", default="", help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_url, username, password, timeout = load_settings()
    session = requests.Session()
    session.auth = (username, password)

    events = fetch_events(
        session,
        base_url,
        args.program,
        args.org_unit,
        args.ou_mode,
        args.page_size,
        timeout,
    )

    org_unit_ids = [e.get("orgUnit", "") for e in events if e.get("orgUnit")]
    orgunit_map = fetch_orgunit_region_mapping(session, base_url, org_unit_ids, timeout)

    df = events_to_dataframe(events, orgunit_map)

    out_path = args.out or f"imnid_assessment_{args.program}_{args.org_unit}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
