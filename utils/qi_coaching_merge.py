import os

import pandas as pd


REQUIRED_QI_COLUMNS = [
    "SubmissionDate",
    "Qi",
    "reg-region",
    "reg-round",
    "reg-hospital",
    "general_info-assessment_date",
    "general_info-qi_coaches",
    "part8_project-q801",
    "internal_coach-coach_name",
    "meta-instanceID",
    "KEY",
]


def _read_csv_preserve_values(path: str) -> pd.DataFrame:
    """Read CSV as text to prevent accidental type conversion during merge."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalized_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _validate_required_columns(df: pd.DataFrame, required_columns, label: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in {label}: {missing_columns}")


def _validate_no_duplicate_columns(df: pd.DataFrame, label: str) -> None:
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(f"Duplicate columns found in {label}: {duplicate_columns}")


def _validate_exact_copy(
    source_df: pd.DataFrame, aligned_df: pd.DataFrame, columns_to_check
) -> None:
    mismatched_columns = []
    for column_name in columns_to_check:
        if column_name not in source_df.columns or column_name not in aligned_df.columns:
            continue
        if not _normalized_series(source_df[column_name]).equals(
            _normalized_series(aligned_df[column_name])
        ):
            mismatched_columns.append(column_name)

    if mismatched_columns:
        raise ValueError(
            "Column alignment mismatch detected for qi coaching columns: "
            f"{mismatched_columns}"
        )


def merge_qi_coaching_forms() -> str:
    """
    Append qi_coaching_full rows to Final_qi_coaching_full using strict
    same-name column alignment only. This avoids cross-column mixing when the
    two form versions do not share an identical schema.
    """
    base_dir = os.path.dirname(__file__)
    qi_file = os.path.join(base_dir, "mentorship", "qi_coaching_full.csv")
    final_qi_file = os.path.join(base_dir, "mentorship", "Final_qi_coaching_full.csv")
    merged_file = os.path.join(base_dir, "mentorship", "merged_qi_coaching.csv")

    missing_files = [p for p in (qi_file, final_qi_file) if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing_files)
        )

    qi_df = _read_csv_preserve_values(qi_file)
    final_qi_df = _read_csv_preserve_values(final_qi_file)

    _validate_no_duplicate_columns(qi_df, "qi_coaching_full.csv")
    _validate_no_duplicate_columns(final_qi_df, "Final_qi_coaching_full.csv")
    _validate_required_columns(qi_df, REQUIRED_QI_COLUMNS, "qi_coaching_full.csv")
    _validate_required_columns(
        final_qi_df, REQUIRED_QI_COLUMNS, "Final_qi_coaching_full.csv"
    )

    common_cols = [col for col in qi_df.columns if col in final_qi_df.columns]
    only_in_final = [col for col in final_qi_df.columns if col not in qi_df.columns]
    only_in_qi = [col for col in qi_df.columns if col not in final_qi_df.columns]

    # Reindex against the final schema so appended values can only land in the
    # identically named target column.
    qi_aligned = qi_df.reindex(columns=final_qi_df.columns)

    # Guard against accidental reordering or cross-column movement.
    _validate_exact_copy(qi_df, qi_aligned, common_cols)

    merged_df = pd.concat([final_qi_df, qi_aligned], ignore_index=True)

    print(f"Qi coaching rows: {len(qi_df)}")
    print(f"Final qi coaching rows: {len(final_qi_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns ({len(common_cols)}): {common_cols}")
    print(f"Columns only in Final_qi_coaching_full.csv: {only_in_final}")
    print(f"Columns only in qi_coaching_full.csv: {only_in_qi}")

    if only_in_qi or only_in_final:
        print(
            "Unshared columns are left unmapped by design to avoid mixing values "
            "across different question fields."
        )

    try:
        merged_df.to_csv(merged_file, index=False)
        print(f"Output path: {merged_file}")
    except PermissionError:
        print(f"Could not write output file (permission denied): {merged_file}")
        print("Close the file if it is open, then run again.")

    return merged_file


if __name__ == "__main__":
    merge_qi_coaching_forms()
