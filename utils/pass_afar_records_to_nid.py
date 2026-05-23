import os
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMNID_DIR = os.path.join(CURRENT_DIR, "imnid")
AFAR_DIR = os.path.join(IMNID_DIR, "afar_nid")
NID_DIR = os.path.join(IMNID_DIR, "nid")

SOURCE_FILE = os.path.join(AFAR_DIR, "regional_afara_newborn_nid.csv")

TARGET_FILES = [
    os.path.join(NID_DIR, "regional_Afar_newborn_nid.csv"),
    os.path.join(NID_DIR, "national_newborn_nid.csv"),
]


def pick_source_file() -> str:
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(
            f"No Afar source file found: {SOURCE_FILE}"
        )
    return SOURCE_FILE


def append_missing_teis(source_path: str, target_path: str) -> int:
    source_df = pd.read_csv(source_path, dtype=str, keep_default_na=False)

    if "tei_id" not in source_df.columns:
        raise ValueError(f"Source missing tei_id: {source_path}")

    source_df.columns = source_df.columns.str.strip()
    source_df["tei_id"] = source_df["tei_id"].astype(str).str.strip()
    source_df = source_df[source_df["tei_id"] != ""].copy()
    source_df = source_df.drop_duplicates(subset=["tei_id"], keep="first")

    if os.path.exists(target_path):
        target_df = pd.read_csv(target_path, dtype=str, keep_default_na=False)
        target_df.columns = target_df.columns.str.strip()

        if "tei_id" not in target_df.columns:
            raise ValueError(f"Target missing tei_id: {target_path}")

        target_df["tei_id"] = target_df["tei_id"].astype(str).str.strip()
        existing_teis = set(target_df["tei_id"])

        to_add = source_df[~source_df["tei_id"].isin(existing_teis)].copy()
        if to_add.empty:
            print(f"No new TEIs for {os.path.basename(target_path)}")
            return 0

        # Align columns to target structure
        for col in target_df.columns:
            if col not in to_add.columns:
                to_add[col] = "N/A"
        for col in to_add.columns:
            if col not in target_df.columns:
                target_df[col] = "N/A"

        to_add = to_add[target_df.columns]
        combined_df = pd.concat([target_df, to_add], ignore_index=True)
        combined_df.to_csv(target_path, index=False, encoding="utf-8")

        print(
            f"Updated {os.path.basename(target_path)}: +{len(to_add)} TEIs (total {len(combined_df)})"
        )
        return len(to_add)

    # If target file does not exist, create it from source
    source_df.to_csv(target_path, index=False, encoding="utf-8")
    print(
        f"Created {os.path.basename(target_path)} with {len(source_df)} TEIs"
    )
    return len(source_df)


def main() -> None:
    os.makedirs(NID_DIR, exist_ok=True)

    source_path = pick_source_file()
    print(f"Using source: {source_path}")

    total_added = 0
    for target_path in TARGET_FILES:
        added = append_missing_teis(source_path, target_path)
        total_added += added

    print("=" * 80)
    print(f"Done. Total additions across target files: {total_added}")
    print("=" * 80)


if __name__ == "__main__":
    main()
