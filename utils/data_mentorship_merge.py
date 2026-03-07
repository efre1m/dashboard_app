import os

import pandas as pd


CRITICAL_ANALYSIS_COLUMNS = [
    "region",
    "hospital",
    "round",
    "q21",
    "q22",
    "q23",
    "q24",
    "q31",
    "q32",
    "q33",
    "q34",
    "q35",
    "q36",
    "q37",
    "q38",
    "q39",
    "q391",
    "q41",
    "q42",
    "q43",
    "q44",
    "q51",
    "q52",
    "q53",
    "q54",
    "q55",
    "q56",
    "q57",
]


def _read_csv_preserve_values(path: str) -> pd.DataFrame:
    """Read CSV as text to avoid unintended type coercion during merge."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalized_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _validate_required_columns(df: pd.DataFrame, required_columns, label: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in {label}: {missing_columns}")


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
            "Column alignment mismatch detected for critical mentorship columns: "
            f"{mismatched_columns}"
        )


def merge_data_mentorship_forms() -> str:
    """
    Append Data_Mentorship_form rows to Final_Data_Mentorship_form with strict
    column-name alignment only. No heuristic remapping is allowed here because
    the mentorship indicator columns must stay bound to their exact source fields.
    """
    base_dir = os.path.dirname(__file__)
    data_file = os.path.join(base_dir, "mentorship", "Data_Mentorship_form.csv")
    final_data_file = os.path.join(
        base_dir, "mentorship", "Final_Data_Mentorship_form.csv"
    )
    merged_file = os.path.join(base_dir, "mentorship", "merged_data.csv")

    missing_files = [p for p in (data_file, final_data_file) if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing_files)
        )

    data_df = _read_csv_preserve_values(data_file)
    final_data_df = _read_csv_preserve_values(final_data_file)

    _validate_required_columns(
        data_df, CRITICAL_ANALYSIS_COLUMNS, "Data_Mentorship_form.csv"
    )
    _validate_required_columns(
        final_data_df, CRITICAL_ANALYSIS_COLUMNS, "Final_Data_Mentorship_form.csv"
    )

    common_cols = [col for col in data_df.columns if col in final_data_df.columns]
    only_in_final = [col for col in final_data_df.columns if col not in data_df.columns]
    only_in_data = [col for col in data_df.columns if col not in final_data_df.columns]

    # Reindex against the final schema so every appended value lands in the
    # same-named final column and nowhere else.
    data_aligned = data_df.reindex(columns=final_data_df.columns)

    # Validate the critical columns used by indicator analysis before writing.
    _validate_exact_copy(data_df, data_aligned, CRITICAL_ANALYSIS_COLUMNS)

    # Validate all direct same-name columns as an extra guard against drift.
    _validate_exact_copy(data_df, data_aligned, common_cols)

    merged_df = pd.concat([final_data_df, data_aligned], ignore_index=True)

    print(f"Data mentorship rows: {len(data_df)}")
    print(f"Final data mentorship rows: {len(final_data_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns ({len(common_cols)}): {common_cols}")
    print(f"Columns only in Final_Data_Mentorship_form.csv: {only_in_final}")
    print(f"Columns only in Data_Mentorship_form.csv: {only_in_data}")

    try:
        merged_df.to_csv(merged_file, index=False)
        print(f"Output path: {merged_file}")
    except PermissionError:
        print(f"Could not write output file (permission denied): {merged_file}")
        print("Close the file if it is open, then run again.")

    return merged_file


if __name__ == "__main__":
    merge_data_mentorship_forms()
