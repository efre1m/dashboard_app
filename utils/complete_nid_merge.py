"""
SIMPLE NID TO NEWBORN APPEND - ONLY PASS 5→1, OTHERS BECOME N/A
"""

import pandas as pd
import os
import glob

print("=" * 80)
print("NID TO NEWBORN APPEND - ONLY 5→1, OTHERS N/A")
print("=" * 80)

# Import assign_source_column to fix source after merge
from add_source_column import assign_source_column

# ==================== 1. PATHS ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
nid_folder = os.path.join(current_dir, "imnid", "nid")
newborn_folder = os.path.join(current_dir, "imnid", "newborn")

print(f"📂 NID folder: {nid_folder}")
print(f"📂 Newborn folder: {newborn_folder}")

if not os.path.exists(nid_folder):
    print("❌ NID folder not found!")
    exit()

if not os.path.exists(newborn_folder):
    print("❌ Newborn folder not found!")
    exit()

print("✅ Both folders found!")

# ==================== 2. EXACT MAPPING ====================
MAPPING = {
    # Patient identifiers
    "tei_id": "tei_id",
    "orgUnit": "orgUnit",
    "orgUnit_name": "orgUnit_name",
    "enrollment_status": "enrollment_status",
    "enrollment_date": "enrollment_date",
    "incident_date": "incident_date",
    "program_type": "program_type",
    "region_uid": "region_uid",
    "region_name": "region_name",
    # Event dates
    "event_date_admission_information": "event_date_nicu_admission_careform",
    "event_date_maternal_birth_and_infant_details": "event_date_nicu_admission_careform",
    "event_date_interventions": "event_date_nurse_followup_sheet",
    "event_date_interventions_cpap": "event_date_neonatal_referral_form",
    "event_date_discharge_and_final_diagnosis": "event_date_discharge_care_form",
    "event_date_observations_and_nursing_care_2": "event_date_observations_and_nursing_care_2",
    "event_observations_and_nursing_care_2": "event_observations_and_nursing_care_2",
    # Admission data
    "date_of_admission_admission_information": "date_of_admission_n_nicu_admission_careform",
    "birth_location_admission_information": "place_of_delivery_nicu_admission_careform",
    "temperature_on_admission_degc_observations_and_nursing_care_1": "temp_at_admission_nicu_admission_careform",
    "birth_weight_grams_maternal_birth_and_infant_details": "birth_weight_n_nicu_admission_careform",
    "weight_on_admission_admission_information": "weight_at_admission_n_nicu_admission_careform",
    "time_of_birth_admission_information": "time_of_birth_admission_information",
    "time_of_admission_admission_information": "time_of_admission_admission_information",
    "were_antibiotics_administered?_interventions": "maternal_medication_during_pregnancy_and_labor_nicu_admission_careform",
    # Interventions
    "kmc_administered_interventions": "kmc_done_nurse_followup_sheet",
    "cpap_administered_interventions": "baby_placed_on_cpap_neonatal_referral_form",
    "cpap_1_start_date_interventions": "cpap_1_start_date_interventions",
    "cpap_1_start_time_interventions": "cpap_1_start_time_interventions",
    # Discharge data
    "newborn_status_at_discharge_discharge_and_final_diagnosis": "newborn_status_at_discharge_n_discharge_care_form",
    "discharge_weight_grams_discharge_and_final_diagnosis": "weight_on_discharge_discharge_care_form",
    "sub_categories_of_infection_discharge_and_final_diagnosis": "sub_categories_of_infection_n_discharge_care_form",
    # Observations & monitoring
    "lowest_recorded_oxygen_saturation_pct_observations_and_nursing_care_2": "lowest_recorded_oxygen_saturation_pct_observations_and_nursing_care_2",
}

print(f"\n📋 Basic mapping: {len(MAPPING)} columns")

# ==================== 3. FILE MAPPING ====================
FILE_MAPPING = {
    "national_newborn_nid.csv": "national_newborn.csv",
    "regional_Afar_newborn_nid.csv": "regional_Afar_newborn.csv",
    "regional_Amhara_newborn_nid.csv": "regional_Amhara_newborn.csv",
    "regional_Central_Ethiopia_newborn_nid.csv": "regional_Central_Ethiopia_newborn.csv",
    "regional_Oromia_newborn_nid.csv": "regional_Oromia_newborn.csv",
    "regional_Sidama_newborn_nid.csv": "regional_Sidama_newborn.csv",
    "regional_South_Ethiopia_newborn_nid.csv": "regional_South_Ethiopia_newborn.csv",
    "regional_South_West_Ethiopia_newborn_nid.csv": "regional_South_West_Ethiopia_newborn.csv",
    "regional_Tigray_newborn_nid.csv": "regional_Tigray_newborn.csv",
}


# ==================== 4. SIMPLE RULE FOR REASON COLUMNS ====================
def get_reason_value_simple(nid_row):
    """SIMPLE RULE: If ANY reason column has "5" → "1", otherwise → "N/A" """

    # Check all 3 reason columns
    reason_columns = [
        "first_reason_for_admission_admission_information",
        "second_reason_for_admission_admission_information",
        "third_reason_for_admission_admission_information",
    ]

    # Check if ANY column has value "5"
    for col in reason_columns:
        if col in nid_row:
            value = str(nid_row[col]).strip()
            if value == "5":
                return "1"  # 5 becomes 1

    # If no column has "5", return "N/A"
    return "N/A"


# ==================== 5. SIMPLE APPEND FUNCTION ====================
def append_nid_to_newborn(nid_path, newborn_path):
    """Upsert patients from NID into newborn file (update existing TEIs + add new TEIs)."""

    # Check if files exist
    if not os.path.exists(nid_path):
        print(f"  ❌ NID file not found: {os.path.basename(nid_path)}")
        return 0, 0

    if not os.path.exists(newborn_path):
        print(f"  ❌ Newborn file not found: {os.path.basename(newborn_path)}")
        return 0, 0

    try:
        # Read both files
        nid_df = pd.read_csv(nid_path, dtype=str, keep_default_na=False)
        newborn_df = pd.read_csv(newborn_path, dtype=str, keep_default_na=False)

        print(f"  📊 NID patients: {len(nid_df)}")
        print(f"  📊 Existing newborn: {len(newborn_df)}")

        # Clean column names
        nid_df.columns = nid_df.columns.str.strip()
        newborn_df.columns = newborn_df.columns.str.strip()

        # Check for tei_id
        if "tei_id" not in nid_df.columns or "tei_id" not in newborn_df.columns:
            print(f"  ⚠️ Missing tei_id column")
            return 0, 0

        # Prepare TEI sets
        nid_df["tei_id"] = nid_df["tei_id"].astype(str).str.strip()
        newborn_df["tei_id"] = newborn_df["tei_id"].astype(str).str.strip()

        existing_teis = set(newborn_df["tei_id"])
        nid_unique_teis = set(nid_df["tei_id"])
        missing_teis = nid_unique_teis - existing_teis
        overlap_teis = nid_unique_teis.intersection(existing_teis)

        print(f"  🔎 NID unique TEIs: {len(nid_unique_teis)}")
        print(f"  🔎 Existing newborn TEIs: {len(existing_teis)}")
        print(f"  🔎 Missing TEIs to append: {len(missing_teis)}")
        print(f"  🔎 Existing TEIs to update: {len(overlap_teis)}")

        # Use one NID row per TEI for upsert
        nid_rows_for_upsert = nid_df.drop_duplicates(subset=["tei_id"], keep="first").copy()

        if len(nid_rows_for_upsert) == 0:
            print(f"  ⚠️ No NID patients available for upsert")
            return 0, 0

        print(f"  🔄 Processing {len(nid_rows_for_upsert)} NID TEIs for upsert")
        sample_teis = nid_rows_for_upsert["tei_id"].head(5).tolist()
        print(f"  🧪 Sample TEIs from NID: {sample_teis}")

        # Track how many patients have "5" in reason columns
        count_5_to_1 = 0

        # Create new rows with mapped values
        new_rows = []

        for _, nid_row in nid_rows_for_upsert.iterrows():
            # Start with empty dictionary
            row_data = {}

            # 1. Get reason value: ONLY "1" if ANY column has "5", otherwise "N/A"
            reason_value = get_reason_value_simple(nid_row)
            if reason_value == "1":
                count_5_to_1 += 1

            row_data["sub_categories_of_prematurity_n_discharge_care_form"] = (
                reason_value
            )

            # 2. Now handle all other mapped columns
            for nid_col, newborn_col in MAPPING.items():
                if nid_col in nid_row:
                    value = str(nid_row[nid_col]).strip()

                    # Apply 5→1 transformation for prematurity columns
                    if "prematurity" in newborn_col.lower():
                        if value == "5":
                            value = "1"

                    row_data[newborn_col] = value

            new_rows.append(row_data)

        # Show reason column statistics
        print(f"  📊 Reason column stats:")
        print(f"     Patients with '5' → '1': {count_5_to_1:,}")
        print(f"     Patients with 'N/A': {len(nid_rows_for_upsert) - count_5_to_1:,}")

        # Create DataFrame for new rows
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            # Make sure both frames share all columns (newborn existing rows get N/A for new columns)
            for col in newborn_df.columns:
                if col not in new_df.columns:
                    new_df[col] = "N/A"  # Fill missing columns with N/A
            for col in new_df.columns:
                if col not in newborn_df.columns:
                    newborn_df[col] = "N/A"

            # Reorder consistently
            new_df = new_df[newborn_df.columns]

            # Fill empty event dates with enrollment_date
            if "enrollment_date" in new_df.columns:
                event_date_cols = [
                    "event_date_nicu_admission_careform",
                    "event_date_nurse_followup_sheet",
                    "event_date_neonatal_referral_form",
                    "event_date_discharge_care_form",
                    "event_date_observations_and_nursing_care_2",
                ]

                for event_col in event_date_cols:
                    if event_col in new_df.columns:
                        mask = new_df[event_col].astype(str).str.strip() == ""
                        new_df.loc[mask, event_col] = new_df.loc[
                            mask, "enrollment_date"
                        ]

            # Fill remaining empty cells with "N/A"
            for col in new_df.columns:
                mask = new_df[col].astype(str).str.strip() == ""
                new_df.loc[mask, col] = "N/A"

            # Upsert:
            # - keep existing newborn rows not present in NID source
            # - replace overlapping rows with NID-mapped rows
            source_teis = set(new_df["tei_id"].astype(str).str.strip())
            preserved_newborn = newborn_df[
                ~newborn_df["tei_id"].astype(str).str.strip().isin(source_teis)
            ].copy()
            combined_df = pd.concat([preserved_newborn, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["tei_id"], keep="last")

            # Fix source column before saving
            combined_df = assign_source_column(combined_df)

            # Save
            combined_df.to_csv(newborn_path, index=False)
            print(f"  💾 Updated: {os.path.basename(newborn_path)}")
            print(
                f"  📈 New total: {len(combined_df)} patients "
                f"(updated {len(overlap_teis):,}, added {len(missing_teis):,})"
            )

            return len(missing_teis), count_5_to_1

        return 0, 0

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 0, 0


# ==================== 6. PROCESS ALL FILES ====================
print("\n" + "=" * 80)
print("PROCESSING ALL FILES...")
print("=" * 80)

total_new_patients = 0
total_5_to_1 = 0

for nid_filename, newborn_filename in FILE_MAPPING.items():
    nid_path = os.path.join(nid_folder, nid_filename)
    newborn_path = os.path.join(newborn_folder, newborn_filename)

    print(f"\n📄 {nid_filename} → {newborn_filename}")

    added, count_5 = append_nid_to_newborn(nid_path, newborn_path)
    total_new_patients += added
    total_5_to_1 += count_5

# ==================== 7. FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("✅ PROCESSING COMPLETE")
print("=" * 80)

print(f"\n📊 FINAL SUMMARY:")
print(f"  Total new patients added: {total_new_patients:,}")
print(f"  Total '5' → '1' transformations: {total_5_to_1:,}")
print(f"  Expected '5→1' count: 1,645")

# Verify the counts
if total_5_to_1 == 1645:
    print(f"  ✅ PERFECT! Got exactly 1,645 '5→1' transformations")
elif total_5_to_1 < 1645:
    print(f"  ⚠️ UNDERCOUNT: Missing {1645 - total_5_to_1} transformations")
else:
    print(f"  ⚠️ OVERCOUNT: Got {total_5_to_1 - 1645} extra transformations")

print(f"\n📁 Checking newborn files for '1' and 'N/A' values:")
for newborn_filename in FILE_MAPPING.values():
    newborn_path = os.path.join(newborn_folder, newborn_filename)

    if os.path.exists(newborn_path):
        try:
            df = pd.read_csv(newborn_path, dtype=str, keep_default_na=False)

            if "sub_categories_of_prematurity_n_discharge_care_form" in df.columns:
                value_counts = df[
                    "sub_categories_of_prematurity_n_discharge_care_form"
                ].value_counts()
                count_1 = value_counts.get("1", 0)
                count_na = value_counts.get("N/A", 0)

                print(f"\n  📁 {newborn_filename}:")
                print(f"     Total patients: {len(df):,}")
                print(f"     '1' values: {count_1:,}")
                print(f"     'N/A' values: {count_na:,}")

                # Show other values (should be none or very few)
                other_values = {
                    k: v
                    for k, v in value_counts.items()
                    if k not in ["1", "N/A", "", "nan"]
                }
                if other_values:
                    print(f"     Other values (should be 0):")
                    for val, count in other_values.items():
                        print(f"       '{val}': {count:,}")

        except Exception as e:
            print(f"  ⚠️ {newborn_filename}: Error - {str(e)}")

print("\n" + "=" * 80)
print("🎉 DONE! Simple rule applied:")
print("  - If ANY reason column has '5' → put '1'")
print("  - Otherwise → put 'N/A'")
print("  - No other values passed from reason columns")
print("=" * 80)
