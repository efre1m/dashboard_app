import os
import pandas as pd
from datetime import datetime

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

COLUMN_RENAME_MAP = {
    "access": "system_access-summary",
    "q21": "system_access-q21",
    "q22": "system_access-q22",
    "q23": "system_access-q23",
    "q24": "system_access-q24",
    "tracker": "tracker_data-summary",
    "q31": "tracker_data-q31",
    "q32": "tracker_data-q32",
    "q33": "tracker_data-q33",
    "q34": "tracker_data-q34",
    "q35": "tracker_data-q35",
    "q36": "tracker_data-q36",
    "q37": "tracker_data-q37",
    "q38": "tracker_data-q38",
    "q39": "tracker_data-q39",
    "q391": "tracker_data-q391",
    "features": "tracker_features-summary",
    "q41": "tracker_features-q41",
    "q42": "tracker_features-q42",
    "q43": "tracker_features-q43",
    "q44": "tracker_features-q44",
    "analysis": "analysis_vis-summary",
    "q51": "analysis_vis-q51",
    "q52": "analysis_vis-q52",
    "q53": "analysis_vis-q53",
    "q54": "analysis_vis-q54",
    "q55": "analysis_vis-q55",
    "q56": "analysis_vis-q56",
    "q57": "analysis_vis-q57",
    "dq": "dqu-summary",
    "q61": "dqu-q61",
    "q62": "dqu-q62",
    "q63": "dqu-q63",
    "q64": "dqu-q64",
    "q65": "dqu-q65",
    "q66": "dqu-q66",
    "q67": "dqu-q67",
    "q68": "dqu-q68",
    "q69": "dqu-q69",
}


def _read_csv_preserve_values(path: str) -> pd.DataFrame:
    """Read CSV as text to avoid unintended type coercion during merge."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _normalized_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _validate_required_columns(df: pd.DataFrame, required_columns, label: str) -> None:
    missing_columns = []
    for col in required_columns:
        column_found = False
        if col in df.columns:
            column_found = True
        elif col.startswith("q") and col[1:].isdigit():
            for df_col in df.columns:
                if df_col.endswith(f"-{col}"):
                    column_found = True
                    break
        if not column_found:
            missing_columns.append(col)
    if missing_columns:
        raise KeyError(f"Missing required columns in {label}: {missing_columns}")


def _validate_exact_copy(
    source_df: pd.DataFrame, aligned_df: pd.DataFrame, columns_to_check
) -> None:
    mismatched_columns = []
    for column_name in columns_to_check:
        source_col = None
        aligned_col = None
        suffix = column_name
        for col in source_df.columns:
            if col.endswith(f"-{suffix}"):
                source_col = col
                break
        for col in aligned_df.columns:
            if col.endswith(f"-{suffix}"):
                aligned_col = col
                break
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
    Merge merged_data_v2.csv with Final_Data_Mentorship_form.csv.
    Filters Final_Data_Mentorship_form.csv to keep only records with
    SubmissionDate >= 2026-06-20, maps flat column names in merged_data_v2
    to prefixed names matching the Final schema, then concatenates and saves
    as merged_data.csv.
    """
    base_dir = os.path.dirname(__file__)
    data_file = os.path.join(base_dir, "mentorship", "merged_data_v2.csv")
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

    # Filter Final to keep only records with SubmissionDate >= 2026-06-20
    if "SubmissionDate" in final_data_df.columns:
        before = len(final_data_df)
        final_data_df["SubmissionDate"] = final_data_df["SubmissionDate"].str.strip()
        final_data_df = final_data_df[
            final_data_df["SubmissionDate"].apply(
                lambda x: _parse_date(x) >= datetime(2026, 6, 20)
                if _parse_date(x)
                else False
            )
        ]
        after = len(final_data_df)
        print(f"Final_Data_Mentorship_form.csv: {before} rows -> {after} rows after date filter")
    else:
        print("Warning: SubmissionDate column not found in Final_Data_Mentorship_form.csv")

    # Validate critical columns
    _validate_required_columns(
        data_df, CRITICAL_ANALYSIS_COLUMNS, "merged_data_v2.csv"
    )
    _validate_required_columns(
        final_data_df, CRITICAL_ANALYSIS_COLUMNS, "Final_Data_Mentorship_form.csv"
    )

    # Map flat column names in data_df to prefixed names matching Final schema
    data_renamed = data_df.rename(columns=COLUMN_RENAME_MAP)

    # Drop columns in data_renamed that don't exist in Final schema (they'd be lost anyway)
    cols_to_drop = [c for c in data_renamed.columns if c not in final_data_df.columns]
    if cols_to_drop:
        print(f"Dropping columns not in Final schema: {cols_to_drop}")
    data_aligned = data_renamed.drop(columns=cols_to_drop, errors="ignore")

    # Reindex against Final schema so missing columns get empty values
    data_aligned = data_aligned.reindex(columns=final_data_df.columns)

    # Validate critical column values transferred correctly
    _validate_exact_copy(data_df, data_aligned, CRITICAL_ANALYSIS_COLUMNS)

    merged_df = pd.concat([final_data_df, data_aligned], ignore_index=True)

    print(f"merged_data_v2 rows: {len(data_df)}")
    print(f"Final data mentorship rows (post-filter): {len(final_data_df)}")
    print(f"Merged rows: {len(merged_df)}")

    try:
        merged_df.to_csv(merged_file, index=False)
        print(f"Output path: {merged_file}")
    except PermissionError:
        print(f"Could not write output file (permission denied): {merged_file}")
        print("Close the file if it is open, then run again.")

    return merged_file


def _parse_date(date_str):
    """Try to parse a date string into a datetime object."""
    if not date_str or date_str == "NA":
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _to_numeric_indicator_value(series: pd.Series) -> pd.Series:
    """
    Convert responses to numeric values for indicator averaging.
    Yes/no values become 1/0, while numeric responses keep their original value.
    """
    cleaned = series.astype(str).str.strip()
    lowered = cleaned.str.lower()
    result = pd.Series(index=series.index, dtype="float64")

    blank_mask = lowered.isin({"", "nan", "none", "null", "na", "n/a"})
    result.loc[lowered.isin({"yes", "true", "y"})] = 1.0
    result.loc[lowered.isin({"no", "false", "n"})] = 0.0

    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric_mask = numeric.notna() & ~blank_mask
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype(float)

    return result


if __name__ == "__main__":
    merge_data_mentorship_forms()
