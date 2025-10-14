# utils/odk_api.py
import os
import io
import logging
import requests
import pandas as pd
from requests.adapters import HTTPAdapter, Retry

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
    Fetch a form’s submissions as a live DataFrame (no file saved).
    Example:
        df = fetch_form_csv("final_bmet_form")
    """
    url = f"{ODK_BASE_URL}{ODK_API_ROOT}/projects/{ODK_PROJECT_ID}/forms/{form_id}/submissions.csv"
    try:
        resp = _get_session().get(url, timeout=ODK_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        logging.info(f"Retrieved {len(df)} records from form '{form_id}'.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch CSV for form {form_id}: {e}")
        return pd.DataFrame()


def fetch_all_forms_as_dataframes() -> dict[str, pd.DataFrame]:
    """Fetch all form submissions as DataFrames."""
    forms = list_forms()
    form_dfs: dict[str, pd.DataFrame] = {}

    for form in forms:
        form_id = form.get("xmlFormId")
        if not form_id:
            continue
        df = fetch_form_csv(form_id)
        if not df.empty:
            form_dfs[form_id] = df

    logging.info(f"Fetched {len(form_dfs)} DataFrames from ODK Central.")
    return form_dfs
