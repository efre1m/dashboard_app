import pandas as pd
import os

def merge_bmet_forms():
    """Append bmet_form rows to final_bmet_form using column-name alignment."""
    base_dir = os.path.dirname(__file__)
    bmet_file = os.path.join(base_dir, "mentorship", "bmet_form (2).csv")
    final_bmet_file = os.path.join(base_dir, "mentorship", "final_bmet_form (2).csv")
    merged_file = os.path.join(base_dir, "mentorship", "merged_bmet.csv")

    missing_files = [p for p in (bmet_file, final_bmet_file) if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing_files)
        )

    bmet_df = pd.read_csv(bmet_file)
    final_bmet_df = pd.read_csv(final_bmet_file)

    common_cols = [col for col in bmet_df.columns if col in final_bmet_df.columns]
    only_in_final_bmet = [col for col in final_bmet_df.columns if col not in bmet_df.columns]
    only_in_bmet = [col for col in bmet_df.columns if col not in final_bmet_df.columns]
    bmet_aligned = bmet_df.reindex(columns=final_bmet_df.columns)

    explicit_mappings = {
        "facility_assessment-Score": "facility_assessment-score_fac",
    }
    applied_mappings = []
    for source_col, target_col in explicit_mappings.items():
        if source_col in bmet_df.columns and target_col in bmet_aligned.columns:
            bmet_aligned[target_col] = bmet_df[source_col]
            applied_mappings.append(f"{source_col} -> {target_col}")

    merged_df = pd.concat([final_bmet_df, bmet_aligned], ignore_index=True)

    print(f"Merged columns: {common_cols}")
    print(f"Applied explicit mappings: {applied_mappings}")
    print(f"Columns only in final_bmet_form (2).csv: {only_in_final_bmet}")
    print(f"Columns only in bmet_form (2).csv: {only_in_bmet}")
    try:
        merged_df.to_csv(merged_file, index=False)
        print(f"Output path: {merged_file}")
    except PermissionError:
        print(f"Could not write output file (permission denied): {merged_file}")
        print("Close the file if it is open, then run again.")

    return merged_file


if __name__ == "__main__":
    merge_bmet_forms()
