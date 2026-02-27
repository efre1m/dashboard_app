import os
import pandas as pd


def merge_qoc_forms():
    """Append QoC_form rows to final_QoC_form using column-name alignment."""
    base_dir = os.path.dirname(__file__)
    qoc_file = os.path.join(base_dir, "mentorship", "QoC_form.csv")
    final_qoc_file = os.path.join(base_dir, "mentorship", "final_QoC_form.csv")
    merged_file = os.path.join(base_dir, "mentorship", "merged_qoc.csv")

    missing_files = [p for p in (qoc_file, final_qoc_file) if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            "Missing required CSV file(s): " + ", ".join(missing_files)
        )

    qoc_df = pd.read_csv(qoc_file)
    final_qoc_df = pd.read_csv(final_qoc_file)

    common_cols = [col for col in qoc_df.columns if col in final_qoc_df.columns]
    only_in_final_qoc = [col for col in final_qoc_df.columns if col not in qoc_df.columns]
    only_in_qoc = [col for col in qoc_df.columns if col not in final_qoc_df.columns]

    # Align source rows to final schema so column-order and missing fields are safe.
    qoc_aligned = qoc_df.reindex(columns=final_qoc_df.columns)
    merged_df = pd.concat([final_qoc_df, qoc_aligned], ignore_index=True)

    print(f"QoC rows: {len(qoc_df)}")
    print(f"Final QoC rows: {len(final_qoc_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Merged columns count: {len(common_cols)}")
    print(f"Columns only in final_QoC_form.csv: {only_in_final_qoc}")
    print(f"Columns only in QoC_form.csv: {only_in_qoc}")

    try:
        merged_df.to_csv(merged_file, index=False)
        print(f"Output path: {merged_file}")
    except PermissionError:
        print(f"Could not write output file (permission denied): {merged_file}")
        print("Close the file if it is open, then run again.")

    return merged_file


if __name__ == "__main__":
    merge_qoc_forms()

