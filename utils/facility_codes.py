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
    as_num = pd.to_numeric(value, errors="coerce")
    if pd.notna(as_num):
        return str(int(as_num))
    text = str(value).strip()
    return text or None


@lru_cache(maxsize=1)
def _get_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
      - facility_name_normalized -> facility_code
      - dhis2_uid_normalized -> facility_code
    """
    mapping_path = Path(__file__).resolve().parent / "IMNID_facility_code.xlsx"
    if not mapping_path.exists():
        return {}, {}

    try:
        ref_df = pd.read_excel(mapping_path)
    except Exception:
        return {}, {}

    if "facility_code" not in ref_df.columns:
        return {}, {}

    name_to_code: Dict[str, str] = {}
    uid_to_code: Dict[str, str] = {}

    for _, row in ref_df.iterrows():
        code = _to_code_str(row.get("facility_code"))
        if not code:
            continue

        name_key = _norm(row.get("facility_name"))
        if name_key:
            name_to_code[name_key] = code

        uid_key = _norm(row.get("dhis2_uid"))
        if uid_key:
            uid_to_code[uid_key] = code

    return name_to_code, uid_to_code


def get_facility_code(
    facility_name: object = None, dhis2_uid: object = None, fallback: object = None
) -> str:
    """Resolve facility code by UID first, then by name; return fallback if not found."""
    name_to_code, uid_to_code = _get_maps()

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

    return out

