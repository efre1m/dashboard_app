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
    # Check that all required columns exist, with flexible matching for different column naming conventions.
    # For q-prefixed columns (q21, q22, etc.), accept both plain form ("q21") and prefixed forms
    # ("system_access-q21", "tracker_data-q31") for broader compatibility.
    missing_columns = []
    for col in required_columns:
        column_found = False
        
        # Check for exact column name match first
        if col in df.columns:
            column_found = True
        # For q-prefixed columns, accept suffix-based matching
        elif col.startswith("q") and col[1:].isdigit():
            for df_col in df.columns:
                # Match by q-suffix (e.g., "system_access-q21" matches "q21")
                if df_col.endswith(f"-{col}"):
                    column_found = True
                    break
        
        if not column_found:
            missing_columns.append(col)
    
    if missing_columns:
        raise KeyError(f"Missing required columns in {label}: {missing_columns}")
    return


def _validate_exact_copy(
    source_df: pd.DataFrame, aligned_df: pd.DataFrame, columns_to_check
) -> None:
    mismatched_columns = []
    for column_name in columns_to_check:
        # Find the actual column names to compare
        source_col = None
        aligned_col = None
        
        # For all cases, match by suffix to handle both plain and prefixed versions
        # This ensures columns like "q21" in source align with "system_access-q21" in aligned
        suffix = column_name
        
        # Find matching columns in both dataframes by suffix
        for col in source_df.columns:
            if col.endswith(f"-{suffix}"):
                source_col = col
                break
        
        for col in aligned_df.columns:
            if col.endswith(f"-{suffix}"):
                aligned_col = col
                break
        
        # If we found matching columns in both, compare their values
        if source_col and aligned_col:
            if not _normalized_series(source_df[source_col]).equals(
                _normalized_series(aligned_df[aligned_col])
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

    # Validate the critical columns used by indicator analysis before writing.
    _validate_required_columns(
        data_df, CRITICAL_ANALYSIS_COLUMNS, "Data_Mentorship_form.csv"
    )
    _validate_required_columns(
        final_data_df, CRITICAL_ANALYSIS_COLUMNS, "Final_Data_Mentorship_form.csv"
    )

    # Reindex against the final schema so every appended value lands in the
    # same-named final column and nowhere else.
    data_aligned = data_df.reindex(columns=final_data_df.columns)

    # Validate the critical columns used by indicator analysis before writing.
    _validate_exact_copy(data_df, data_aligned, CRITICAL_ANALYSIS_COLUMNS)

    merged_df = pd.concat([final_data_df, data_aligned], ignore_index=True)

    print(f"Data mentorship rows: {len(data_df)}")
    print(f"Final data mentorship rows: {len(final_data_df)}")
    print(f"Merged rows: {len(merged_df)}")

    try:
        merged_df.to_csv(merged_file, index=False)
        print(f"Output path: {merged_file}")
    except PermissionError:
        print(f"Could not write output file (permission denied): {merged_file}")
        print("Close the file if it is open, then run again.")

    return merged_file


if __name__ == "__main__":
    merge_data_mentorship_forms()
