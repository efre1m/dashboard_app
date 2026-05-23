import os
import pandas as pd


OLD_FACILITY_NAME = "Suhul2 general hospital"
NEW_FACILITY_NAME = "Suhul general hospital"
OLD_ORGUNIT_ID = "HZdhQgsycUM"
NEW_ORGUNIT_ID = "lpFA6qFp9kG"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMNID_DIR = os.path.join(CURRENT_DIR, "imnid")

TARGET_FILES = [
    os.path.join(IMNID_DIR, "maternal", "national_maternal.csv"),
    os.path.join(IMNID_DIR, "maternal", "regional_Tigray_maternal.csv"),
    os.path.join(IMNID_DIR, "newborn", "national_newborn.csv"),
    os.path.join(IMNID_DIR, "newborn", "regional_Tigray_newborn.csv"),
]


def replace_value_in_columns(df: pd.DataFrame, old_value: str, new_value: str, columns: list[str]) -> int:
    updated = 0
    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].astype(str)
        mask = series.str.strip() == old_value
        count = int(mask.sum())
        if count > 0:
            df.loc[mask, col] = new_value
            updated += count
    return updated


def process_file(path: str) -> tuple[int, int]:
    if not os.path.exists(path):
        print(f"[SKIP] File not found: {path}")
        return 0, 0

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    name_updates = replace_value_in_columns(
        df,
        OLD_FACILITY_NAME,
        NEW_FACILITY_NAME,
        ["orgUnit_name"],
    )
    id_updates = replace_value_in_columns(
        df,
        OLD_ORGUNIT_ID,
        NEW_ORGUNIT_ID,
        ["orgUnit"],
    )

    if name_updates > 0 or id_updates > 0:
        df.to_csv(path, index=False, encoding="utf-8")
        print(
            f"[UPDATED] {os.path.basename(path)} | name updates: {name_updates}, id updates: {id_updates}"
        )
    else:
        print(f"[NO CHANGE] {os.path.basename(path)}")

    return name_updates, id_updates


def main() -> None:
    print("=" * 80)
    print("Renaming Suhul facility and orgUnit ID in maternal/newborn target files")
    print("=" * 80)

    total_name_updates = 0
    total_id_updates = 0

    for target in TARGET_FILES:
        name_count, id_count = process_file(target)
        total_name_updates += name_count
        total_id_updates += id_count

    print("=" * 80)
    print(f"Done. Total name updates: {total_name_updates}")
    print(f"Done. Total ID updates: {total_id_updates}")
    print("=" * 80)


if __name__ == "__main__":
    main()
