# utils/odk_api.py
import os
import io
import logging
import requests
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
from utils.region_mapping import get_odk_region_codes, get_region_name_from_database_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Load environment variables
ODK_BASE_URL = os.getenv("ODK_BASE_URL", "https://dhicenter.com").rstrip("/")
ODK_API_ROOT = os.getenv("ODK_API_ROOT", "/v1")
ODK_PROJECT_ID = os.getenv("ODK_PROJECT_ID", "14")
ODK_USERNAME = os.getenv("ODK_USERNAME", "")
ODK_PASSWORD = os.getenv("ODK_PASSWORD", "")
ODK_TIMEOUT = int(os.getenv("ODK_TIMEOUT", "60"))

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Create or reuse a requests session with retries configured."""
    global _session
    if _session is None:
        s = requests.Session()
        s.auth = (ODK_USERNAME, ODK_PASSWORD)
        s.headers.update(
            {"Accept": "application/json", "User-Agent": "ODK-DataFetcher/1.0"}
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
        logging.info("ODK Central session initialized.")
    return _session


def list_forms() -> list[dict]:
    """Return all forms in the given ODK project."""
    url = f"{ODK_BASE_URL}{ODK_API_ROOT}/projects/{ODK_PROJECT_ID}/forms"
    try:
        resp = _get_session().get(url, timeout=ODK_TIMEOUT)
        resp.raise_for_status()
        forms = resp.json()
        logging.info(f"Retrieved {len(forms)} forms from project {ODK_PROJECT_ID}.")
        return forms
    except Exception as e:
        logging.error(f"Failed to list ODK forms: {e}")
        return []


def fetch_form_csv(form_id: str) -> pd.DataFrame:
    """
    Fetch a form's submissions as a live DataFrame (no file saved).
    Preserves all columns exactly as they are from ODK.
    """
    url = f"{ODK_BASE_URL}{ODK_API_ROOT}/projects/{ODK_PROJECT_ID}/forms/{form_id}/submissions.csv"
    try:
        resp = _get_session().get(url, timeout=ODK_TIMEOUT)
        resp.raise_for_status()

        # Read CSV and preserve ALL columns exactly as they are
        df = pd.read_csv(io.BytesIO(resp.content))

        logging.info(
            f"Retrieved {len(df)} records from form '{form_id}' with {len(df.columns)} columns."
        )
        return df
    except Exception as e:
        logging.error(f"Failed to fetch CSV for form {form_id}: {e}")
        return pd.DataFrame()


def filter_by_region(df: pd.DataFrame, database_region_id: int) -> pd.DataFrame:
    """
    Filter the DataFrame based on a regional user's database region_id.
    Uses numeric region codes (1-6) found in ODK form columns.
    Preserves all columns exactly as they are.
    """
    if df.empty:
        return df

    # Get the ODK numeric codes for this database region_id
    target_codes = get_odk_region_codes(database_region_id)
    if not target_codes:
        logging.warning(
            f"No ODK region mapping found for database region_id={database_region_id}"
        )
        return df

    # Look for region columns that contain numeric codes
    region_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(
            region_keyword in col_lower
            for region_keyword in ["region", "reg-region", "reg_region"]
        ):
            region_col = col
            break

    if not region_col:
        logging.warning(
            f"No region column found in form for database_region_id={database_region_id}. Available columns: {list(df.columns)}"
        )
        return df

    # Convert region column to string and clean
    df[region_col] = df[region_col].astype(str).str.strip()

    # Filter by numeric region codes
    allowed_codes = [str(code).strip() for code in target_codes]

    filtered = df[df[region_col].isin(allowed_codes)]

    region_name = get_region_name_from_database_id(database_region_id)
    logging.info(
        f"Filtered {len(filtered)}/{len(df)} records for database_region_id={database_region_id} ({region_name}) using ODK codes: {', '.join(allowed_codes)}"
    )

    return filtered


def filter_by_facility(df: pd.DataFrame, facility_names: list[str]) -> pd.DataFrame:
    """
    Filter the DataFrame based on facility names.
    Preserves all columns exactly as they are.
    """
    if df.empty:
        return df

    if not facility_names:
        logging.warning("No facility names provided for filtering")
        return df

    # Try common facility column names
    facility_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(
            facility_keyword in col_lower
            for facility_keyword in ["facility", "health_facility", "healthfacility"]
        ):
            facility_col = col
            break

    if not facility_col:
        logging.warning(
            f"No facility column found in form data. Available columns: {list(df.columns)}"
        )
        return df

    # Filter by facility names (case-insensitive)
    df[facility_col] = df[facility_col].astype(str).str.strip().str.lower()
    allowed = [f.strip().lower() for f in facility_names]

    filtered = df[df[facility_col].isin(allowed)]
    logging.info(
        f"Filtered {len(filtered)}/{len(df)} records for facilities: {', '.join(facility_names)}"
    )
    return filtered


def fetch_all_forms_as_dataframes(
    user: dict = None, facility_names: list[str] = None
) -> dict[str, pd.DataFrame]:
    """
    Fetch all form submissions as DataFrames with user-based filtering.
    Preserves all columns exactly as they are from ODK.

    Args:
        user: User dictionary with role and access info
        facility_names: List of facility names for filtering

    Returns:
        Dictionary of form_id to DataFrame mappings
    """
    forms = list_forms()
    form_dfs: dict[str, pd.DataFrame] = {}

    for form in forms:
        form_id = form.get("xmlFormId")
        if not form_id:
            continue

        df = fetch_form_csv(form_id)

        # Apply user-based filtering
        if user and not df.empty:
            user_role = user.get("role", "")

            if user_role == "regional" and user.get("region_id"):
                # Filter by region for regional users using database region_id
                database_region_id = user.get("region_id")
                df = filter_by_region(df, database_region_id)
                logging.info(
                    f"Applied regional filtering for database_region_id: {database_region_id}"
                )

            elif user_role == "facility" and facility_names:
                # Filter by specific facilities for facility users
                df = filter_by_facility(df, facility_names)
                logging.info(
                    f"Applied facility filtering for facilities: {facility_names}"
                )

            # National users get all data (no filtering)
            elif user_role == "national":
                logging.info("National user - no ODK data filtering applied")

        if not df.empty:
            form_dfs[form_id] = df

    logging.info(
        f"Fetched {len(form_dfs)} DataFrames from ODK Central for user role: {user.get('role', 'unknown') if user else 'none'}"
    )
    return form_dfs
