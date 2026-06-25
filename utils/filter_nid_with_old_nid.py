"""
Filter each CSV in nid/ to keep only rows whose tei_id also appears
in the corresponding CSV in nid_old/ (same filename).

Manual equivalent in Excel:
  1. Copy tei_id column from the old file
  2. Paste into the new file's tei_id column
  3. Duplicates get colored (tei_ids that exist in BOTH old and new)
  4. Remove uncolored rows (tei_ids only in new, not in old)
"""

import os
import pandas as pd

IMNID_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imnid")
NID_OLD_DIR = os.path.join(IMNID_DIR, "nid_old")
NID_DIR = os.path.join(IMNID_DIR, "nid")


def filter_single_file(fname: str) -> None:
    old_path = os.path.join(NID_OLD_DIR, fname)
    new_path = os.path.join(NID_DIR, fname)

    if not os.path.exists(old_path):
        print(f"  SKIP  {fname} — no old version found")
        return
    if not os.path.exists(new_path):
        print(f"  SKIP  {fname} — no new version found")
        return

    old_ids = pd.read_csv(old_path, dtype=str, usecols=["tei_id"])
    old_set = set(old_ids["tei_id"].dropna().str.strip())
    print(f"  Old file: {len(old_ids):,} rows, {len(old_set):,} unique TEI IDs")

    df_new = pd.read_csv(new_path, dtype=str)
    before = len(df_new)

    mask = df_new["tei_id"].str.strip().isin(old_set)
    df_filtered = df_new[mask].copy()
    after = len(df_filtered)

    if after < before:
        df_filtered.to_csv(new_path, index=False, encoding="utf-8")

    pct = 100.0 * after / before if before > 0 else 0
    print(f"  {fname}: {before:,} → {after:,} rows ({pct:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("Filter each nid/ CSV to TEI IDs that also exist in nid_old/")
    print("=" * 60)
    print()
    for fname in sorted(os.listdir(NID_DIR)):
        if not fname.endswith(".csv"):
            continue
        filter_single_file(fname)
    print()
    print("Done.")
