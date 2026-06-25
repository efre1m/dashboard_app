import os
import pandas as pd

IMNID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imnid")
NEWBORN_DIR = os.path.join(IMNID_DIR, "newborn")

# Source → target pairs
PAIRS = [
    ("national_olenchity.csv", "national_newborn.csv"),
    ("regioinal_oromia_olenchity.csv", "regional_Oromia_newborn.csv"),
]

# Column rename mapping: old NID source column → new NCF target column
COLUMN_MAPPING = {
    "was_phototherapy_administered?_interventions": "phototherapy_administered?_medication_sheet",
    "was_a_transfusion_given?_interventions": "transfusion_given?_medication_sheet",
    "was_bilirubin_tested?_observations_and_nursing_care_2": "bilirubin_tested?_nurse_followup_sheet",
    "were_antibiotics_administered?_interventions": "are_antibiotics_administered?_medication_sheet",
    "csf_culture_for_suspected_meningitis_microbiology_and_labs": "csf_culture_for_suspected_meningitis_investigation_sheet",
    "blood_culture_for_suspected_sepsis_microbiology_and_labs": "blood_culture_for_suspected_sepsis_investigation_sheet",
}

# Value mapping for each old source column (unrecognized → N/A)
VALUE_MAPPING = {
    "was_phototherapy_administered?_interventions": {"1": "1", "0": "0"},
    "was_a_transfusion_given?_interventions": {"1": "1", "2": "2", "0": "0"},
    "was_bilirubin_tested?_observations_and_nursing_care_2": {"1": "1", "0": "0"},
    "were_antibiotics_administered?_interventions": {"1": "1", "0": "0"},
    "csf_culture_for_suspected_meningitis_microbiology_and_labs": {"0": "0", "1": "1", "2": "2", "3": "3"},
    "blood_culture_for_suspected_sepsis_microbiology_and_labs": {"0": "0", "1": "1", "2": "2", "3": "3"},
}


def _remap_source_columns(src: pd.DataFrame) -> pd.DataFrame:
    """Rename old NID-style columns to new NCF names, applying value mapping."""
    df = src.copy()
    for old_col, new_col in COLUMN_MAPPING.items():
        if old_col not in df.columns:
            continue
        val_map = VALUE_MAPPING.get(old_col, {})
        df[new_col] = df[old_col].apply(
            lambda v: val_map.get(str(v).strip(), "N/A")
        )
        df = df.drop(columns=[old_col])
    return df


def merge_olenchity_into_newborn(source_name: str, target_name: str) -> int:
    source_path = os.path.join(IMNID_DIR, source_name)
    target_path = os.path.join(NEWBORN_DIR, target_name)

    if not os.path.exists(source_path):
        print(f"  SKIP  {source_name} — not found")
        return 0
    if not os.path.exists(target_path):
        print(f"  SKIP  {target_name} — not found")
        return 0

    src = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    tgt = pd.read_csv(target_path, dtype=str, keep_default_na=False)

    src.columns = src.columns.str.strip()
    tgt.columns = tgt.columns.str.strip()

    src["tei_id"] = src["tei_id"].astype(str).str.strip()
    tgt["tei_id"] = tgt["tei_id"].astype(str).str.strip()

    # Remap old NID columns → new NCF columns with value conversion
    src = _remap_source_columns(src)

    # Keep only columns that exist in target
    keep_cols = [c for c in src.columns if c in tgt.columns]
    src = src[keep_cols]

    # Add target columns missing from source, filled with N/A
    for col in tgt.columns:
        if col not in src.columns:
            src[col] = "N/A"

    # Reorder to match target
    src = src[tgt.columns]

    # Remove any rows with empty tei_id
    src = src[src["tei_id"] != ""]
    if src.empty:
        print(f"  DONE  {source_name} → {target_name} — 0 rows (empty source)")
        return 0

    existing_teis = set(tgt["tei_id"])
    incoming_teis = set(src["tei_id"])
    overlap = incoming_teis & existing_teis
    new_teis = incoming_teis - existing_teis

    if overlap:
        # Update existing rows
        tgt = tgt[~tgt["tei_id"].isin(overlap)].copy()
        new_rows = src[src["tei_id"].isin(overlap)].copy()
        tgt = pd.concat([tgt, new_rows], ignore_index=True)

    # Append truly new rows
    new_rows = src[src["tei_id"].isin(new_teis)].copy()
    if not new_rows.empty:
        tgt = pd.concat([tgt, new_rows], ignore_index=True)

    tgt.to_csv(target_path, index=False, encoding="utf-8")
    total = len(overlap) + len(new_teis)
    print(
        f"  DONE  {source_name} -> {target_name}: "
        f"updated {len(overlap)}, added {len(new_teis)}"
    )
    return total


if __name__ == "__main__":
    print("=" * 60)
    print("Merging Olenchity data into newborn CSVs")
    print("=" * 60)
    total = 0
    for source_name, target_name in PAIRS:
        total += merge_olenchity_into_newborn(source_name, target_name)
    print("=" * 60)
    print(f"Total rows processed: {total}")
    print("=" * 60)
