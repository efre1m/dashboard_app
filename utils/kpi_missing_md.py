import pandas as pd
import streamlit as st
from utils.kpi_utils import compute_total_deliveries, compute_csection_count
from utils.kpi_svd import compute_svd_count
from utils.kpi_assisted import compute_assisted_count


def compute_missing_md_count(df, facility_uids=None):
    if df is None or df.empty:
        return {
            "missing_md_count": 0,
            "total_deliveries": 0,
            "missing_percentage": 0.0,
        }

    total_deliveries = compute_total_deliveries(df, facility_uids)
    csection_count = compute_csection_count(df, facility_uids)
    svd_count = compute_svd_count(df, facility_uids)
    instrumental_count = compute_assisted_count(df, facility_uids)

    missing_md_count = max(
        0, total_deliveries - (csection_count + svd_count + instrumental_count)
    )

    missing_percentage = (
        (missing_md_count / total_deliveries * 100) if total_deliveries > 0 else 0.0
    )

    return {
        "missing_md_count": missing_md_count,
        "total_deliveries": total_deliveries,
        "missing_percentage": missing_percentage,
        "csection_count": csection_count,
        "svd_count": svd_count,
        "instrumental_count": instrumental_count,
    }


def render_missing_md_simple_table(
    df,
    facility_uids,
    display_names=None,
    comparison_mode="overall",
    facilities_by_region=None,
    region_names=None,
):
    """
    Render missing mode of delivery table with proper region support.

    Parameters:
    - df: DataFrame with event data
    - facility_uids: List of facility IDs
    - display_names: List of facility names (for facility comparison)
    - comparison_mode: "overall", "facility", or "region"
    - facilities_by_region: Dict of region_name -> [(facility_name, facility_uid)]
    - region_names: List of region names (for region comparison)
    """
    st.subheader("📋 Missing Mode of Delivery Analysis")

    table_data = []

    # REGION comparison mode
    if comparison_mode == "region" and region_names and facilities_by_region:
        for region_name in region_names:
            # Get facility UIDs for this region
            region_facility_uids = []
            region_facility_names = []

            if region_name in facilities_by_region:
                for facility_name, facility_uid in facilities_by_region[region_name]:
                    region_facility_uids.append(facility_uid)
                    region_facility_names.append(facility_name)

            if not region_facility_uids:
                continue

            # Filter data for facilities in this region
            region_df = df[df["orgUnit"].isin(region_facility_uids)]
            if region_df.empty:
                continue

            # Calculate missing MD for this region
            md_data = compute_missing_md_count(region_df, region_facility_uids)

            table_data.append(
                {
                    "Region": region_name,
                    "Total": md_data["total_deliveries"],
                    "C-Section": md_data["csection_count"],
                    "SVD": md_data["svd_count"],
                    "Instrumental": md_data["instrumental_count"],
                    "Missing": md_data["missing_md_count"],
                    "Missing %": f"{md_data['missing_percentage']:.1f}%",
                }
            )

    # FACILITY comparison mode
    elif (
        comparison_mode == "facility"
        and display_names
        and len(display_names) == len(facility_uids)
    ):
        for facility_name, facility_uid in zip(display_names, facility_uids):
            facility_df = df[df["orgUnit"] == facility_uid]
            if facility_df.empty:
                continue

            md_data = compute_missing_md_count(facility_df, [facility_uid])

            table_data.append(
                {
                    "Facility": facility_name,
                    "Total": md_data["total_deliveries"],
                    "C-Section": md_data["csection_count"],
                    "SVD": md_data["svd_count"],
                    "Instrumental": md_data["instrumental_count"],
                    "Missing": md_data["missing_md_count"],
                    "Missing %": f"{md_data['missing_percentage']:.1f}%",
                }
            )

    # OVERALL mode (default)
    else:
        md_data = compute_missing_md_count(df, facility_uids)

        table_data.append(
            {
                "Level": "Overall",
                "Total": md_data["total_deliveries"],
                "C-Section": md_data["csection_count"],
                "SVD": md_data["svd_count"],
                "Instrumental": md_data["instrumental_count"],
                "Missing": md_data["missing_md_count"],
                "Missing %": f"{md_data['missing_percentage']:.1f}%",
            }
        )

    if not table_data:
        st.info("No data available.")
        return None

    result_df = pd.DataFrame(table_data)

    # Determine column name based on comparison mode
    if comparison_mode == "region":
        first_column_name = "Region"
    elif comparison_mode == "facility":
        first_column_name = "Facility"
    else:
        first_column_name = "Level"

    # Rename the first column for consistency
    if first_column_name in result_df.columns:
        result_df = result_df.rename(columns={first_column_name: "Name"})
    elif "Facility" in result_df.columns:
        result_df = result_df.rename(columns={"Facility": "Name"})
    elif "Region" in result_df.columns:
        result_df = result_df.rename(columns={"Region": "Name"})
    elif "Level" in result_df.columns:
        result_df = result_df.rename(columns={"Level": "Name"})

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn(width="large"),
            "Total": st.column_config.NumberColumn(format="%d"),
            "C-Section": st.column_config.NumberColumn(format="%d"),
            "SVD": st.column_config.NumberColumn(format="%d"),
            "Instrumental": st.column_config.NumberColumn(format="%d"),
            "Missing": st.column_config.NumberColumn(format="%d"),
            "Missing %": st.column_config.TextColumn(),
        },
    )

    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="missing_mode_of_delivery.csv",
        mime="text/csv",
    )

    return result_df
