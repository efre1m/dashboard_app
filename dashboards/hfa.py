"""HFA dashboard tab (Streamlit)

Reads only from the HFA directory (`utils/HFA`) to display facility-level
Health Facility Assessment (HFA) Excel analyses. The tab exposes Region and
Facility (DHIS2 name) filters and renders the selected facility's Excel sheet
as a clean, scrollable table. Access is limited to national and regional users;
facility-level users see an access notice instead of data.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
import re

# ---------------------------------------------------------------------------
# Constants and paths (restrict all file I/O to utils/HFA)
# ---------------------------------------------------------------------------

HFA_DIR = Path("D:/dashboard_app/utils/HFA").resolve()
MAPPING_PATH = HFA_DIR / "dhis2_to_hfafile_facility_mapping.xlsx"
ALLOWED_ROLES = {"national", "regional"}

# Build a case-insensitive lookup for region folders (only inside utils/HFA)
REGION_FOLDERS: Dict[str, Path] = {
    p.name.lower(): p for p in HFA_DIR.iterdir() if p.is_dir()
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_mapping() -> pd.DataFrame:
    """Load facility mapping once; trims whitespace and enforces required cols."""
    df = pd.read_excel(MAPPING_PATH)
    df.columns = [c.strip() for c in df.columns]
    required = ["Region", "dhis2 name", "facility name in HFA file"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Mapping file missing columns: {', '.join(missing)}")
    return df


def region_values() -> List[str]:
    df = load_mapping()
    regions = sorted(df["Region"].dropna().unique().tolist())
    return regions


def facilities_for_region(region: Optional[str]) -> List[str]:
    if not region:
        return []
    df = load_mapping()
    subset = df[df["Region"].str.lower() == region.lower()]
    names = subset["dhis2 name"].dropna().unique().tolist()
    names.sort()
    return names


def _resolve_region_folder(region: str) -> Path:
    """Return the folder for a region (case-insensitive), defaulting to HFA root."""
    return REGION_FOLDERS.get(region.lower(), HFA_DIR)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy Excel sheet for display: fill blanks and drop empty rows/cols."""
    df = df.copy()

    # Rename unnamed columns to meaningful defaults (Value, Status, extras)
    unnamed_idx = 0
    new_cols = []
    for c in df.columns:
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed"):
            unnamed_idx += 1
            if unnamed_idx == 1:
                name = "Value"
            elif unnamed_idx == 2:
                name = "Status"
            else:
                name = f"Column {unnamed_idx}"
        new_cols.append(name)
    df.columns = new_cols

    # Ensure first column is labeled Indicator
    if df.columns.size > 0:
        df.columns = ["Indicator"] + list(df.columns[1:])

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Remove top metadata rows (first two rows) if present
    if len(df) > 2:
        df = df.iloc[2:]

    df = df.fillna("")
    # Drop rows and cols that are entirely empty strings
    df = df.loc[:, ~(df.eq("").all())]
    df = df.loc[~(df.eq("").all(axis=1))]
    df = df.reset_index(drop=True)
    return df


def load_facility_dataframe(region: str, dhis2_name: str) -> pd.DataFrame:
    """Load Excel for selected facility as a DataFrame."""
    df_map = load_mapping()
    match = df_map[
        (df_map["Region"].str.lower() == region.lower())
        & (df_map["dhis2 name"].str.lower() == dhis2_name.lower())
    ]
    if match.empty:
        raise FileNotFoundError("Facility not found in mapping file.")

    hfa_stub = match.iloc[0]["facility name in HFA file"]
    region_dir = _resolve_region_folder(region)
    excel_path = (region_dir / f"{hfa_stub}.xlsx").resolve()

    # Safety: ensure the resolved path stays inside HFA_DIR
    if HFA_DIR not in excel_path.parents:
        raise PermissionError("Resolved HFA path is outside the allowed directory.")
    if not excel_path.exists():
        raise FileNotFoundError(f"HFA file not found: {excel_path.name}")

    df_excel = pd.read_excel(excel_path, sheet_name=0, dtype=str)
    df_excel = _clean_dataframe(df_excel)
    return df_excel


def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert dataframe to Excel bytes for download."""
    buf = BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def _safe_filename(base: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return clean or "HFA_export"


# ---------------------------------------------------------------------------
# Streamlit renderer
# ---------------------------------------------------------------------------

def render_hfa_tab(user: Dict, *, key_prefix: str = "hfa") -> None:
    """Render HFA tab inside Streamlit for national/regional users."""
    role = (user or {}).get("role", "").lower()
    if role not in ALLOWED_ROLES:
        st.info("The HFA dashboard is available only to national or regional users.")
        return

    user_region = (
        user.get("region")
        or user.get("Region")
        or user.get("user_region")
        or user.get("region_name")
    )

    all_regions = region_values()
    if role == "regional":
        regions = [r for r in all_regions if r.lower() == (user_region or "").lower()]
    else:
        regions = all_regions

    if not regions:
        st.warning("No regions available for this user.")
        return

    st.markdown(
        """
        <style>
        .hfa-panel { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; }
        .hfa-filters { display: flex; gap: 12px; flex-wrap: wrap; }
        .hfa-filter { min-width: 240px; flex: 1; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 40%, #0ea5e9 100%);
            color: #f8fafc;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 0.2px;
            box-shadow: 0 6px 16px rgba(14, 165, 233, 0.25);
            margin-bottom: 12px;
        ">
            Health Facility Assessment (HFA)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-column layout: table left (wide), filters right (narrow)
    main_col, filter_col = st.columns([3, 1], gap="large")

    region_index = 0
    if role != "regional" and user_region:
        for idx, reg in enumerate(regions):
            if reg.lower() == user_region.lower():
                region_index = idx
                break

    with filter_col:
        st.markdown("**Filters**")
        region = st.selectbox(
            "Region",
            regions,
            index=region_index if regions else None,
            key=f"{key_prefix}_region",
        )

        facility_options = facilities_for_region(region)
        facility = st.selectbox(
            "Facility (DHIS2 name)",
            facility_options,
            index=0 if len(facility_options) == 1 else None,
            key=f"{key_prefix}_facility",
        )

    with main_col:
        if not region or not facility:
            st.info("Select a region and facility to view the HFA analysis.")
            return

        try:
            df = load_facility_dataframe(region, facility)
        except Exception as err:  # broad to surface message to user
            st.error(f"Unable to load HFA data: {err}")
            return

        if df.empty:
            st.warning("No data found in the selected HFA file.")
            return

        st.markdown(
            f"""
            <div style="
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 800;
                color: #0f172a;
                margin: 0 0 8px 0;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
            ">
                {facility} — HFA Profile ({region})
            </div>
            """,
            unsafe_allow_html=True,
        )

        download_bytes = _df_to_excel_bytes(df)
        download_name = _safe_filename(f"{facility}_{region}_HFA.xlsx")
        with filter_col:
            st.download_button(
                "Download as Excel",
                data=download_bytes,
                file_name=download_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # Build a styled view that mimics the Excel formatting
        def _style_sections(row):
            first = str(row.iloc[0]).strip()
            is_section = (
                first.lower().startswith(tuple(f"{i}." for i in range(1, 30)))
                or "profile" in first.lower()
                or "capacity" in first.lower()
                or "overview" in first.lower()
            )
            if is_section:
                return [
                    "background-color: #dbeafe; font-weight: 700; color: #0f172a; border: 1px solid #94a3b8; text-align: left;"
                    for _ in row
                ]
            return ["" for _ in row]

        styler = (
            df.style
            .set_table_styles(
                [
                    {"selector": "table", "props": "border-collapse:collapse; border:2px solid #94a3b8; width:100%; border-radius:12px; overflow:hidden;"},
                    {"selector": "th", "props": "background-color:#0f172a; color:#f8fafc; font-weight:800; text-align:center; padding:10px; border:1px solid #475569; letter-spacing:0.1px; font-size:13px;"},
                    {"selector": "td", "props": "padding:8px; border:1px solid #cbd5e1; font-size:12px; color:#0f172a;"},
                    {"selector": "tbody tr:nth-child(odd)", "props": "background-color:#f8fafc;"},
                    {"selector": "tbody tr:nth-child(even)", "props": "background-color:#ffffff;"},
                ]
            )
            .hide(axis="index")
            .set_properties(**{"text-align": "left"})
            .apply(_style_sections, axis=1)
        )
        if "Value" in df.columns:
            styler = styler.set_properties(subset=["Value"], **{"text-align": "center"})
        if "Status" in df.columns:
            styler = styler.set_properties(subset=["Status"], **{"text-align": "center"})

        table_css = """
        <style>
        .hfa-table-container {
            border-radius: 14px;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.14);
            border: 1px solid #cbd5e1;
            overflow: auto;
            background: #ffffff;
            margin-top: 4px;
            max-height: 70vh;
        }
        .hfa-table-container table { width: 100%; }
        </style>
        """

        html_table = styler.set_table_attributes('class="hfa-table"').to_html()
        components.html(table_css + f'<div class="hfa-table-container">{html_table}</div>', height=640, scrolling=True)
