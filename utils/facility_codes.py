from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _to_code_str(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    # Preserve user-provided string formatting (e.g., "01") exactly.
    if isinstance(value, str):
        text = value.strip()
        return text or None
    as_num = pd.to_numeric(value, errors="coerce")
    if pd.notna(as_num):
        return str(int(as_num))
    text = str(value).strip()
    return text or None


@lru_cache(maxsize=1)
def _get_maps() -> tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      - facility_name_normalized -> facility_code
      - dhis2_uid_normalized -> facility_code
      - facility_name_normalized -> region_code
      - region_name_normalized -> region_code
    """
    mapping_path = Path(__file__).resolve().parent / "IMNID_facility_code.xlsx"
    if not mapping_path.exists():
        return {}, {}, {}, {}

    try:
        # Read as text to preserve any leading zeros present in the file.
        ref_df = pd.read_excel(mapping_path, dtype=str)
    except Exception:
        return {}, {}, {}, {}

    if "new_facility_code" in ref_df.columns:
        code_col = "new_facility_code"
    elif "facility_code" in ref_df.columns:
        code_col = "facility_code"
    else:
        return {}, {}, {}, {}

    name_to_code: Dict[str, str] = {}
    uid_to_code: Dict[str, str] = {}
    name_to_region: Dict[str, str] = {}
    region_name_to_code: Dict[str, str] = {}

    for _, row in ref_df.iterrows():
        code = _to_code_str(row.get(code_col))
        # new_facility_code is expected to keep leading zeros and be 7-digit formatted.
        if code_col == "new_facility_code" and code and code.isdigit():
            code = code.zfill(7)
        if not code:
            continue

        name_key = _norm(row.get("facility_name"))
        if name_key:
            name_to_code[name_key] = code

        uid_key = _norm(row.get("dhis2_uid"))
        if uid_key:
            uid_to_code[uid_key] = code

        region_code = _to_code_str(row.get("region_code"))
        # region_code must keep leading zero form like "01", "08".
        if region_code and region_code.isdigit():
            region_code = region_code.zfill(2)
        if region_code and name_key:
            name_to_region[name_key] = region_code
        region_name_key = _norm(row.get("region_name"))
        if region_code and region_name_key:
            region_name_to_code[region_name_key] = region_code

    return name_to_code, uid_to_code, name_to_region, region_name_to_code


def get_facility_code(
    facility_name: object = None, dhis2_uid: object = None, fallback: object = None
) -> str:
    """Resolve facility code by UID first, then by name; return fallback if not found."""
    name_to_code, uid_to_code, _, _ = _get_maps()

    uid_key = _norm(dhis2_uid)
    if uid_key and uid_key in uid_to_code:
        return uid_to_code[uid_key]

    name_key = _norm(facility_name)
    if name_key and name_key in name_to_code:
        return name_to_code[name_key]

    if fallback is not None:
        return str(fallback)
    if facility_name is not None:
        return str(facility_name)
    return ""


def get_region_code_from_facility_code(code: object) -> str:
    code_str = _to_code_str(code)
    return code_str[0] if code_str else ""


def get_region_code(
    facility_name: object = None,
    region_name: object = None,
    facility_code: object = None,
    fallback: object = None,
) -> str:
    """Resolve region code using mapping file (preferred), then fallback."""
    _, _, name_to_region, region_name_to_code = _get_maps()

    name_key = _norm(facility_name)
    if name_key and name_key in name_to_region:
        return name_to_region[name_key]

    region_key = _norm(region_name)
    if region_key and region_key in region_name_to_code:
        return region_name_to_code[region_key]

    code_from_facility = get_region_code_from_facility_code(facility_code)
    if code_from_facility:
        return code_from_facility

    if fallback is not None:
        return str(fallback)
    return ""


def apply_facility_codes_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace common facility name columns with facility code where mapping exists.
    Keeps original values when no mapping is found.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    uid_series = None
    if "orgUnit" in out.columns:
        uid_series = out["orgUnit"]
    elif "dhis2_uid" in out.columns:
        uid_series = out["dhis2_uid"]

    def _map_column(col_name: str):
        if col_name not in out.columns:
            return
        src = out[col_name]
        if uid_series is not None:
            out[col_name] = [
                get_facility_code(name, uid, fallback=name)
                for name, uid in zip(src, uid_series)
            ]
        else:
            out[col_name] = [get_facility_code(name, None, fallback=name) for name in src]

    for col in ("orgUnit_name", "facility_name", "Facility"):
        _map_column(col)

    facility_ref_col = None
    for c in ("orgUnit_name", "facility_name", "Facility"):
        if c in out.columns:
            facility_ref_col = c
            break

    if facility_ref_col:
        facility_codes = [get_facility_code(v, None, fallback=v) for v in out[facility_ref_col]]
        if "region_name" in out.columns:
            out["region_name"] = [
                get_region_code(facility_name=f_name, region_name=r_name, facility_code=f_code, fallback=r_name)
                for f_name, r_name, f_code in zip(out[facility_ref_col], out["region_name"], facility_codes)
            ]
        if "Region" in out.columns:
            out["Region"] = [
                get_region_code(facility_name=f_name, region_name=r_name, facility_code=f_code, fallback=r_name)
                for f_name, r_name, f_code in zip(out[facility_ref_col], out["Region"], facility_codes)
            ]

    return out
