import hashlib
import difflib
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

from utils.ethiopian_periods import (
    add_ethiopian_period_metadata,
    filter_periods_by_overlap,
    map_gregorian_dates_to_ethiopian_yearmonths,
)
from newborns_dashboard.kpi_utils_newborn import (
    auto_text_color,
    compute_admitted_newborns_count,
    render_newborn_facility_comparison_chart,
    render_newborn_region_comparison_chart,
    render_newborn_trend_chart,
)
from utils.queries import get_facility_mapping_for_user

# ---------------- Caching Setup ----------------
if "newborn_coverage_rate_cache" not in st.session_state:
    st.session_state.newborn_coverage_rate_cache = {}


def get_newborn_coverage_rate_cache_key(df, facility_uids=None, computation_type=""):
    """Generate a unique cache key for Newborn Coverage Rate computations."""
    key_data = {
        "computation_type": computation_type,
        "facility_uids": tuple(sorted(facility_uids)) if facility_uids else None,
        "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest(),
        "data_shape": df.shape,
    }
    return str(key_data)


def clear_newborn_coverage_rate_cache():
    """Clear the Newborn Coverage Rate cache."""
    st.session_state.newborn_coverage_rate_cache = {}


# ---------------- Denominator Loading ----------------
DENOMINATOR_CACHE_VERSION = 4
DENOMINATOR_FILENAME_CANDIDATES = (
    "aggregated_admission_newborn.xlsx",
    "aggregated admission newborn.xlsx",
)


def _normalize_name(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text


def _normalize_region_key(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _normalize_facility_key(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())

    # Standardize common suffix patterns (abbrev <-> full name)
    suffix_replacements = [
        (r"\bprimary hospital\b", "ph"),
        (r"\bgeneral hospital\b", "gh"),
        (r"\bcomprehensive specialized hospital\b", "csh"),
        (r"\bcomprehensive specialised hospital\b", "csh"),
        (r"\breferral hospital\b", "rh"),
        (r"\bspecialized hospital\b", "sh"),
        (r"\bspecialised hospital\b", "sh"),
        (r"\bhealth center\b", "hc"),
        (r"\bhealth centre\b", "hc"),
    ]
    for pattern, replacement in suffix_replacements:
        text = re.sub(pattern, replacement, text)

    return " ".join(text.split())


_FACILITY_TYPE_SUFFIX_TOKENS = {
    "ph",
    "gh",
    "csh",
    "rh",
    "sh",
    "hc",
    "hospital",
}


def _strip_facility_type_suffix(key: str) -> str:
    parts = (key or "").split()
    while parts and parts[-1] in _FACILITY_TYPE_SUFFIX_TOKENS:
        parts = parts[:-1]
    return " ".join(parts)


def _get_denominator_file_path():
    utils_dir = Path(__file__).resolve().parents[1] / "utils"
    for filename in DENOMINATOR_FILENAME_CANDIDATES:
        candidate = utils_dir / filename
        if candidate.exists():
            return candidate
    return None


def _extract_yearmonth_columns(df):
    yearmonth_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.isdigit() and len(col_str) == 6:
            yearmonth_cols.append(col)
    return yearmonth_cols


def _wide_to_long_denominator(df_wide):
    if df_wide is None or df_wide.empty:
        return pd.DataFrame(
            columns=[
                "region_name",
                "region_name_norm",
                "org_unit_name",
                "org_unit_name_norm",
                "yearmonth",
                "year",
                "denominator",
            ]
        )

    wide = df_wide.copy()

    # Column name normalization (handle minor naming differences)
    col_map = {c: str(c).strip() for c in wide.columns}
    wide = wide.rename(columns=col_map)

    if "region_name" not in wide.columns or "org_unit_name" not in wide.columns:
        # Try some common alternatives
        alt_region = next(
            (c for c in wide.columns if _normalize_name(c) in {"region", "region name"}),
            None,
        )
        alt_org = next(
            (
                c
                for c in wide.columns
                if _normalize_name(c)
                in {
                    "org_unit_name",
                    "org unit name",
                    "orgunit_name",
                    "facility_name",
                    "organisation unit name",
                    "organization unit name",
                }
            ),
            None,
        )
        if alt_region and "region_name" not in wide.columns:
            wide = wide.rename(columns={alt_region: "region_name"})
        if alt_org and "org_unit_name" not in wide.columns:
            wide = wide.rename(columns={alt_org: "org_unit_name"})

    if "region_name" not in wide.columns or "org_unit_name" not in wide.columns:
        logging.error(
            "Newborn Coverage Rate: Denominator file missing required columns 'region_name' and/or 'org_unit_name'."
        )
        return pd.DataFrame()

    ym_cols = _extract_yearmonth_columns(wide)
    if not ym_cols:
        logging.error(
            "Newborn Coverage Rate: No YearMonth columns found in denominator file."
        )
        return pd.DataFrame()

    long_df = wide.melt(
        id_vars=["region_name", "org_unit_name"],
        value_vars=ym_cols,
        var_name="yearmonth",
        value_name="denominator",
    )

    long_df["region_name_norm"] = long_df["region_name"].apply(_normalize_region_key)
    long_df["org_unit_name_norm"] = long_df["org_unit_name"].apply(_normalize_facility_key)
    long_df["yearmonth"] = pd.to_numeric(long_df["yearmonth"], errors="coerce")
    long_df["yearmonth"] = long_df["yearmonth"].astype("Int64")
    long_df["year"] = (long_df["yearmonth"] // 100).astype("Int64")
    long_df["denominator"] = pd.to_numeric(long_df["denominator"], errors="coerce").fillna(
        0
    )

    return add_ethiopian_period_metadata(long_df)


def load_newborn_coverage_denominator():
    """
    Load aggregated newborn admissions (denominator) from utils Excel file.
    Returns a LONG dataframe with columns:
      - region_name
      - region_name_norm
      - org_unit_name
      - org_unit_name_norm
      - yearmonth (Int64, e.g., 202503)
      - year (Int64)
      - denominator (numeric)
    """
    cache = st.session_state.newborn_coverage_rate_cache
    path = _get_denominator_file_path()
    if path is None:
        logging.error(
            "Newborn Coverage Rate: Denominator file not found in utils folder."
        )
        return pd.DataFrame()

    try:
        fingerprint = f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}:v{DENOMINATOR_CACHE_VERSION}"
    except Exception:
        fingerprint = f"{path}:v{DENOMINATOR_CACHE_VERSION}"

    cached_fp = cache.get("denominator_fingerprint")
    cached_df = cache.get("denominator_long_df")
    if cached_fp == fingerprint and isinstance(cached_df, pd.DataFrame):
        return cached_df

    try:
        wide = pd.read_excel(path)
    except Exception as e:
        logging.error(f"Newborn Coverage Rate: Failed to read denominator file: {e}")
        return pd.DataFrame()

    long_df = _wide_to_long_denominator(wide)
    cache["denominator_fingerprint"] = fingerprint
    cache["denominator_long_df"] = long_df
    return long_df


def _get_uid_to_facility_name_map():
    """
    Resolve UID -> facility name mapping using session_state if available,
    otherwise fallback to a database-backed lookup (cached in session_state).
    """
    cache = st.session_state.newborn_coverage_rate_cache

    # Prefer session_state mappings already loaded by dashboards.
    for key in ("facility_mapping", "facility_mapping_facility"):
        mapping = st.session_state.get(key)
        if isinstance(mapping, dict) and mapping:
            return {uid: name for name, uid in mapping.items()}

    user = st.session_state.get("user", {}) or {}
    cache_key = (
        f"uid_to_name::{user.get('role')}::{user.get('region_id')}::{user.get('facility_id')}"
    )
    if cache_key in cache:
        return cache[cache_key]

    try:
        name_to_uid = get_facility_mapping_for_user(user)
    except Exception as e:
        logging.warning(
            f"Newborn Coverage Rate: Failed to load facility mapping for user: {e}"
        )
        name_to_uid = {}

    uid_to_name = {uid: name for name, uid in name_to_uid.items()}
    cache[cache_key] = uid_to_name
    return uid_to_name


def _resolve_facility_names(facility_uids, df=None):
    """
    Resolve facility names from UIDs (preferred). If UIDs are not provided,
    fallback to UIDs present in df['orgUnit'].
    """
    uids = list(facility_uids or [])
    if not uids and df is not None and "orgUnit" in df.columns:
        uids = (
            pd.Series(df["orgUnit"])
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

    if not uids:
        return []

    uid_to_name = _get_uid_to_facility_name_map()
    names = []
    for uid in uids:
        name = uid_to_name.get(str(uid))
        if name:
            names.append(name)
    return names


def _sum_denominator_for_facilities(den_df, facility_names, yearmonths=None, years=None):
    if den_df is None or den_df.empty or not facility_names:
        return 0

    norms = {_normalize_facility_key(n) for n in facility_names if n}
    if not norms:
        return 0

    den_keys = set(den_df["org_unit_name_norm"].dropna().astype(str).unique().tolist())

    matched_keys = set()
    missing_keys = []
    for key in norms:
        if key in den_keys:
            matched_keys.add(key)
        else:
            missing_keys.append(key)

    if missing_keys:
        # Try base-key matching (strip facility type suffixes such as "ph"/"gh"/"hospital").
        den_base_map = {}
        for dk in den_keys:
            base = _strip_facility_type_suffix(dk)
            if base:
                den_base_map.setdefault(base, []).append(dk)

        for key in list(missing_keys):
            base = _strip_facility_type_suffix(key)
            candidates = den_base_map.get(base, [])
            if len(candidates) == 1:
                matched_keys.add(candidates[0])
                missing_keys.remove(key)

    if missing_keys:
        # Final fallback: fuzzy match (high cutoff to avoid incorrect facility pairing).
        den_key_list = sorted(den_keys)
        for key in missing_keys:
            best = difflib.get_close_matches(key, den_key_list, n=1, cutoff=0.92)
            if best:
                matched_keys.add(best[0])

    if not matched_keys:
        return 0

    working = den_df[den_df["org_unit_name_norm"].isin(matched_keys)].copy()
    if working.empty:
        return 0

    if yearmonths:
        ym = pd.to_numeric(list(yearmonths), errors="coerce")
        ym = [int(x) for x in ym if not pd.isna(x)]
        working = working[working["yearmonth"].isin(ym)].copy()
    elif years:
        ys = pd.to_numeric(list(years), errors="coerce")
        ys = [int(x) for x in ys if not pd.isna(x)]
        working = working[working["year"].isin(ys)].copy()

    denom = pd.to_numeric(working["denominator"], errors="coerce").fillna(0).sum()
    if pd.isna(denom):
        return 0
    return int(denom)


def _infer_full_regions_for_selection(facility_uids, facilities_by_region):
    if not facility_uids or not facilities_by_region:
        return None

    selected = {str(uid) for uid in facility_uids if uid}
    if not selected:
        return None

    region_uid_sets = {}
    for region_name, facilities in facilities_by_region.items():
        region_uid_sets[region_name] = {str(uid) for _, uid in facilities if uid}

    selected_regions = []
    covered = set()
    for region_name, region_uids in region_uid_sets.items():
        if region_uids and region_uids.issubset(selected):
            selected_regions.append(region_name)
            covered |= region_uids

    if selected_regions and covered == selected:
        return selected_regions

    return None


def _sum_denominator_for_regions(den_df, region_names, yearmonths=None, years=None):
    if den_df is None or den_df.empty or not region_names:
        return 0

    region_keys = {_normalize_region_key(r) for r in region_names if r}
    if not region_keys:
        return 0

    if "region_name_norm" in den_df.columns:
        working = den_df[den_df["region_name_norm"].isin(region_keys)].copy()
        if working.empty:
            available = sorted(
                set(den_df["region_name_norm"].dropna().astype(str).unique().tolist())
            )
            matched = set()
            for key in region_keys:
                best = difflib.get_close_matches(key, available, n=1, cutoff=0.85)
                if best:
                    matched.add(best[0])
            if matched:
                working = den_df[den_df["region_name_norm"].isin(matched)].copy()
    else:
        working = den_df.copy()
        working["_region_norm"] = working.get("region_name", "").apply(_normalize_region_key)
        working = working[working["_region_norm"].isin(region_keys)].copy()

    if working.empty:
        return 0

    if yearmonths:
        ym = pd.to_numeric(list(yearmonths), errors="coerce")
        ym = [int(x) for x in ym if not pd.isna(x)]
        working = working[working["yearmonth"].isin(ym)].copy()
    elif years:
        ys = pd.to_numeric(list(years), errors="coerce")
        ys = [int(x) for x in ys if not pd.isna(x)]
        working = working[working["year"].isin(ys)].copy()

    denom = pd.to_numeric(working["denominator"], errors="coerce").fillna(0).sum()
    if pd.isna(denom):
        return 0
    return int(denom)


def get_numerator_denominator_for_newborn_coverage_rate(
    df, facility_uids=None, date_range_filters=None
):
    """
    Newborn Coverage Rate =
      total admitted newborns (numerator) / total aggregated newborn admissions (denominator)

    - Numerator is computed from patient-level data using kpi_utils_newborn.
    - Denominator is loaded from utils/aggregated_admission_newborn.xlsx (wide YearMonth format).
    - Returns: (numerator, denominator, value) where value is percentage (0-100).
    """
    if df is None or df.empty:
        return (0, 0, 0.0)

    working_df = df.copy()

    # Optional date range filter (defensive; most callers already filter before grouping)
    if date_range_filters and "enrollment_date" in working_df.columns:
        start_date = date_range_filters.get("start_date")
        end_date = date_range_filters.get("end_date")
        if start_date and end_date:
            working_df["enrollment_date"] = pd.to_datetime(
                working_df["enrollment_date"], errors="coerce"
            )
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            working_df = working_df[
                (working_df["enrollment_date"] >= start_dt)
                & (working_df["enrollment_date"] < end_dt)
            ].copy()

    if working_df.empty:
        return (0, 0, 0.0)

    den_long = load_newborn_coverage_denominator()
    if den_long is None or den_long.empty:
        return (0, 0, 0.0)

    # Derive YearMonth / Year from patient-level dates
    dates = None
    if "enrollment_date" in working_df.columns:
        dates = pd.to_datetime(working_df["enrollment_date"], errors="coerce")
    elif "event_date" in working_df.columns:
        dates = pd.to_datetime(working_df["event_date"], errors="coerce")

    if dates is None or dates.dropna().empty:
        return (0, 0, 0.0)

    available_yearmonths = den_long["yearmonth"].dropna().astype(int).unique().tolist()
    period_df = filter_periods_by_overlap(
        den_long[["yearmonth", "gc_start", "gc_end"]].drop_duplicates(),
        start_date=date_range_filters.get("start_date") if date_range_filters else None,
        end_date=date_range_filters.get("end_date") if date_range_filters else None,
    )
    if not period_df.empty:
        available_yearmonths = period_df["yearmonth"].dropna().astype(int).tolist()

    working_df = working_df.copy()
    working_df["_ethiopian_yearmonth"] = map_gregorian_dates_to_ethiopian_yearmonths(
        dates, available_yearmonths
    )
    working_df = working_df[working_df["_ethiopian_yearmonth"].notna()].copy()
    if working_df.empty:
        return (0, 0, 0.0)

    numerator = int(compute_admitted_newborns_count(working_df, facility_uids))
    yearmonths = (
        working_df["_ethiopian_yearmonth"].dropna().astype("Int64").astype(int).unique().tolist()
    )
    years = sorted({int(ym) // 100 for ym in yearmonths})

    user = st.session_state.get("user", {}) or {}
    role = str(user.get("role") or "").lower()

    # Prefer region-based denominator when the selection is clearly a full-region union
    # (fixes All Facilities and By Region modes, and enforces role scope for regional users).
    denominator = 0
    if role in {"national", "regional"}:
        try:
            from utils.queries import get_facilities_grouped_by_region

            facilities_by_region = get_facilities_grouped_by_region(user)
        except Exception:
            facilities_by_region = {}

        inferred_regions = None
        if facilities_by_region:
            inferred_regions = _infer_full_regions_for_selection(facility_uids, facilities_by_region)

            # If facility_uids is empty/None, interpret as "all accessible" for national/regional.
            if inferred_regions is None and not facility_uids:
                inferred_regions = list(facilities_by_region.keys())

        if inferred_regions:
            denominator = _sum_denominator_for_regions(
                den_long, inferred_regions, yearmonths=yearmonths, years=years
            )

    if denominator == 0:
        facility_names = _resolve_facility_names(facility_uids, df=working_df)
        denominator = _sum_denominator_for_facilities(
            den_long, facility_names, yearmonths=yearmonths, years=years
        )

    value = (numerator / denominator * 100) if denominator > 0 else 0.0
    if np.isnan(value) or np.isinf(value):
        value = 0.0

    return (numerator, int(denominator), float(value))


# ---------------- Rendering Helpers (Dashboard Integration) ----------------
def render_newborn_coverage_rate_trend_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Newborn Coverage Rate",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    numerator_name="Admitted Newborns",
    denominator_name="Aggregated Admissions",
    facility_uids=None,
    **kwargs,
):
    """Render trend chart for Newborn Coverage Rate using standard newborn chart renderer."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    return render_newborn_trend_chart(
        df,
        period_col,
        value_col,
        title,
        bg_color,
        text_color,
        facility_names=facility_names,
        numerator_name=numerator_name,
        denominator_name=denominator_name,
        facility_uids=facility_uids,
        **kwargs,
    )


def render_newborn_coverage_rate_facility_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Newborn Coverage Rate - Facility Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    facility_names=None,
    facility_uids=None,
    numerator_name="Admitted Newborns",
    denominator_name="Aggregated Admissions",
    **kwargs,
):
    """Render facility comparison chart for Newborn Coverage Rate."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    return render_newborn_facility_comparison_chart(
        df,
        period_col,
        value_col,
        title,
        bg_color,
        text_color,
        facility_names=facility_names or [],
        facility_uids=facility_uids or [],
        numerator_name=numerator_name,
        denominator_name=denominator_name,
        **kwargs,
    )


def render_newborn_coverage_rate_region_comparison_chart(
    df,
    period_col="period_display",
    value_col="value",
    title="Newborn Coverage Rate - Region Comparison",
    bg_color="#FFFFFF",
    text_color=None,
    region_names=None,
    region_mapping=None,
    facilities_by_region=None,
    numerator_name="Admitted Newborns",
    denominator_name="Aggregated Admissions",
    **kwargs,
):
    """Render region comparison chart for Newborn Coverage Rate."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    return render_newborn_region_comparison_chart(
        df,
        period_col=period_col,
        value_col=value_col,
        title=title,
        bg_color=bg_color,
        text_color=text_color,
        region_names=region_names,
        region_mapping=region_mapping,
        facilities_by_region=facilities_by_region,
        numerator_name=numerator_name,
        denominator_name=denominator_name,
        **kwargs,
    )
