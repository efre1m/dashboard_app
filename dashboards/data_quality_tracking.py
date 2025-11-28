# dashboards/data_quality_tracking.py
import streamlit as st
import pandas as pd
import logging
import time
from utils.queries import get_facilities_grouped_by_region

# Import the data quality functions
try:
    from dashboards.maternal_dq import (
        check_maternal_outliers,
        check_missing_data_elements as check_maternal_missing_elements,
        clear_dq_cache,
        is_national_user,
    )
    from dashboards.newborn_dq import (
        check_newborn_outliers,
        check_missing_data_elements as check_newborn_missing_elements,
    )
except ImportError as e:
    logging.error(f"Failed to import DQ modules: {e}")

    # Create fallback functions
    def check_maternal_outliers(events_df, user):
        return pd.DataFrame()

    def check_maternal_missing_elements(events_df, user):
        return pd.DataFrame()

    def check_newborn_outliers(events_df, user):
        return pd.DataFrame()

    def check_newborn_missing_elements(events_df, user):
        return pd.DataFrame()

    def clear_dq_cache(user=None):
        pass

    def is_national_user(user):
        # Fallback implementation - check if user has national role or no region
        user_role = user.get("role", "").lower()
        user_region = user.get("region_name")
        return "national" in user_role or user_region is None or user_region == ""


def check_data_availability():
    """Check if data is available and log details"""
    maternal_events_available = (
        "maternal_events_df" in st.session_state
        and st.session_state.maternal_events_df is not None
        and not st.session_state.maternal_events_df.empty
    )

    maternal_tei_available = (
        "maternal_tei_df" in st.session_state
        and st.session_state.maternal_tei_df is not None
        and not st.session_state.maternal_tei_df.empty
    )

    newborn_events_available = (
        "newborn_events_df" in st.session_state
        and st.session_state.newborn_events_df is not None
        and not st.session_state.newborn_events_df.empty
    )

    newborn_tei_available = (
        "newborn_tei_df" in st.session_state
        and st.session_state.newborn_tei_df is not None
        and not st.session_state.newborn_tei_df.empty
    )

    return {
        "maternal": maternal_events_available and maternal_tei_available,
        "newborn": newborn_events_available and newborn_tei_available,
    }


def check_user_change():
    """Check if user has changed and clear DQ cache if needed"""
    current_user = st.session_state.get("user", {})
    current_user_key = f"{current_user.get('username', 'unknown')}_{current_user.get('role', 'unknown')}"

    # Check if this is a new user session
    if "last_dq_user" not in st.session_state:
        st.session_state.last_dq_user = current_user_key
        logging.info(f"🆕 First DQ session for user: {current_user_key}")

        # Clear all analysis data for new user
        clear_all_analysis_data()
        return True  # Return True for new user

    # Check if user changed
    if st.session_state.last_dq_user != current_user_key:
        logging.info(
            f"👤 User changed from {st.session_state.last_dq_user} to {current_user_key}"
        )
        clear_dq_cache(st.session_state.get("previous_user"))
        st.session_state.last_dq_user = current_user_key

        # Clear all analysis data when user changes
        clear_all_analysis_data()
        return True

    return False


def clear_all_analysis_data():
    """Clear all analysis data from session state"""
    analysis_keys = [
        # Maternal analysis
        "maternal_outliers_df",
        "maternal_missing_df",
        "maternal_outliers_progress",
        "maternal_missing_progress",
        "maternal_outliers_stop",
        "maternal_missing_stop",
        "maternal_outliers_results",
        "maternal_missing_results",
        "maternal_analysis_triggered",
        # Newborn analysis
        "newborn_outliers_df",
        "newborn_missing_df",
        "newborn_outliers_progress",
        "newborn_missing_progress",
        "newborn_outliers_stop",
        "newborn_missing_stop",
        "newborn_outliers_results",
        "newborn_missing_results",
        "newborn_analysis_triggered",
        # Cache keys for current user
        "dq_maternal_outliers",
        "dq_maternal_missing_elements",
        "dq_newborn_outliers",
        "dq_newborn_missing_elements",
    ]

    for key in analysis_keys:
        if key in st.session_state:
            del st.session_state[key]

    logging.info("🧹 Cleared all analysis data for new user session")


def filter_data_by_user_access(events_df, tei_df, user):
    """Filter data based on user access level (regional or national)"""
    if not events_df.empty and not tei_df.empty:
        # Get user region - handle None values safely
        user_region = user.get("region_name")
        is_national = is_national_user(user)

        logging.info(
            f"🔍 Filtering data - User Region: '{user_region}', Is National: {is_national}"
        )

        # For national users, return all data without filtering
        if is_national:
            logging.info(
                f"🌍 National user - showing all data: {len(events_df)} events, {len(tei_df)} TEIs"
            )
            return events_df.copy(), tei_df.copy()

        # For regional users, only filter if we have a valid region
        if user_region:
            user_region_upper = user_region.upper()
            try:
                facilities_by_region = get_facilities_grouped_by_region(user)
                facility_to_region_map = {}
                for region_name, facilities in facilities_by_region.items():
                    for facility_name, dhis2_uid in facilities:
                        facility_to_region_map[facility_name] = region_name.upper()

                # Filter events data based on user's region facilities
                if "Facility" in events_df.columns:
                    events_df["Region"] = events_df["Facility"].map(
                        facility_to_region_map
                    )
                    filtered_events = events_df[
                        events_df["Region"] == user_region_upper
                    ].copy()

                    # Get the TEI IDs from filtered events
                    filtered_tei_ids = filtered_events["TEI ID"].unique()

                    # Filter TEI data based on the same TEI IDs
                    filtered_tei = tei_df[
                        tei_df["TEI ID"].isin(filtered_tei_ids)
                    ].copy()

                    logging.info(
                        f"📍 Regional user data filtered: {len(filtered_events)} events, {len(filtered_tei)} TEIs from {user_region_upper}"
                    )
                    return filtered_events, filtered_tei

            except Exception as e:
                logging.error(f"❌ Error filtering data by region: {e}")

        # If regional user but no region specified, or filtering failed, return all data with warning
        logging.warning(
            f"⚠️ Regional user but no region specified - showing all data: {len(events_df)} events, {len(tei_df)} TEIs"
        )
        return events_df.copy(), tei_df.copy()

    return events_df, tei_df


def analyze_with_progress(analysis_function, events_df, user, analysis_name):
    """Optimized analysis with progress tracking and stop functionality"""
    # Clear previous state for this analysis
    progress_key = f"{analysis_name}_progress"
    stop_key = f"{analysis_name}_stop"
    results_key = f"{analysis_name}_results"

    st.session_state[progress_key] = 0
    st.session_state[stop_key] = False
    st.session_state[results_key] = pd.DataFrame()

    # Create containers
    progress_container = st.empty()
    stop_container = st.empty()
    results_container = st.empty()

    # Store results
    current_results = pd.DataFrame()
    partial_results_shown = False

    # Progress simulation with real analysis
    progress_steps = [0, 30, 60, 90, 100]

    for progress in progress_steps:
        if st.session_state[stop_key]:
            break

        # Update progress
        st.session_state[progress_key] = progress

        # Show progress
        with progress_container:
            st.progress(progress / 100, text=f"🔄 Analyzing... {progress}%")

        # Stop button
        with stop_container:
            if st.button(
                "⏹️ Stop Analysis",
                key=f"stop_{analysis_name}_{progress}",
                use_container_width=True,
            ):
                st.session_state[stop_key] = True
                break

        # Run analysis at key points and show partial results
        if progress in [30, 60, 90, 100]:
            current_results = analysis_function(events_df, user)
            st.session_state[results_key] = current_results

            # Show partial results at each milestone
            if not current_results.empty and not partial_results_shown:
                with results_container:
                    results_container.empty()
                    st.info(
                        f"📊 Progress {progress}% - Found {len(current_results)} issues so far"
                    )
                    display_cols = [
                        col for col in current_results.columns if col not in ["TEI ID"]
                    ]
                    display_df = current_results[display_cols].head(10)
                    st.dataframe(display_df, use_container_width=True)
                    if len(current_results) > 10:
                        st.caption(
                            f"Showing first 10 of {len(current_results)} results found so far"
                        )
                    st.write("---")

                # Only show partial results once to avoid duplication
                if progress >= 60:  # Show substantial results at 60% or more
                    partial_results_shown = True

        # Small delay for smooth progress
        if progress < 100:
            time.sleep(0.5)

    # Final state - ALWAYS show results when stopped or completed
    results_container.empty()  # Clear previous results

    if st.session_state[stop_key]:
        with progress_container:
            st.warning(f"⏹️ Analysis stopped at {st.session_state[progress_key]}%")
        with stop_container:
            st.info("Showing all patients processed so far...")
    else:
        with progress_container:
            st.progress(1.0, text="✅ Analysis complete!")
        with stop_container:
            st.success("✅ Analysis completed successfully!")

    # Display final results - ALL processed patients
    with results_container:
        if not current_results.empty:
            st.info(f"📊 Found {len(current_results)} issues in total")
            display_cols = [
                col for col in current_results.columns if col not in ["TEI ID"]
            ]
            display_df = current_results[display_cols]
            st.dataframe(display_df, use_container_width=True)
            st.caption(f"Showing all {len(current_results)} results")
        else:
            st.info("🔍 No issues found in the analyzed data")

    return current_results


def clear_analysis_state(analysis_type):
    """Clear all analysis state for a specific type"""
    keys_to_clear = [
        f"{analysis_type}_progress",
        f"{analysis_type}_stop",
        f"{analysis_type}_results",
        f"{analysis_type}_df",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def safe_sorted_unique(values):
    """Safely sort unique values, handling mixed data types"""
    if values is None or len(values) == 0:
        return []

    # Convert to list and remove None values
    unique_values = [v for v in values.unique() if v is not None]

    # Separate strings and non-strings
    strings = [str(v) for v in unique_values if isinstance(v, (str,))]
    non_strings = [v for v in unique_values if not isinstance(v, (str,))]

    # Sort each group separately
    sorted_strings = sorted(strings)
    sorted_non_strings = sorted(non_strings)

    # Combine results
    return sorted_strings + sorted_non_strings


def render_data_quality_tracking(user):
    """Main function to render Data Quality Tracking dashboard"""

    st.markdown(
        '<div class="main-header">Data Quality Tracking</div>', unsafe_allow_html=True
    )

    # Check user changes - this will clear data for new users
    check_user_change()

    # Check data availability
    data_availability = check_data_availability()
    maternal_data_available = data_availability["maternal"]
    newborn_data_available = data_availability["newborn"]

    if not maternal_data_available and not newborn_data_available:
        st.warning(
            """
        ⚠️ **No data available for Data Quality Analysis**
        
        To use Data Quality Tracking:
        1. First visit either the **Maternal Dashboard** or **Newborn Dashboard** tab
        2. Wait for the data to load completely (you'll see KPI cards and charts)
        3. Then return to this tab to analyze data quality
        
        The data quality dashboard needs the processed data from the main dashboards.
        """
        )
        return

    # Create facility to region mapping
    try:
        facilities_by_region = get_facilities_grouped_by_region(user)
        facility_to_region_map = {}
        for region_name, facilities in facilities_by_region.items():
            for facility_name, dhis2_uid in facilities:
                facility_to_region_map[facility_name] = region_name
        st.session_state.facility_to_region_map = facility_to_region_map
        logging.info("✅ Created facility to region mapping")
    except Exception as e:
        logging.error(f"❌ Failed to create facility mapping: {e}")
        st.session_state.facility_to_region_map = {}

    # Create tabs
    tab1, tab2 = st.tabs(["Maternal Data Quality", "Newborn Data Quality"])

    with tab1:
        render_maternal_data_quality_manual(user, maternal_data_available)

    with tab2:
        render_newborn_data_quality_manual(user, newborn_data_available)

    # Cache management
    if st.sidebar.button("🧹 Clear All Analysis Data", use_container_width=True):
        clear_all_analysis_data()
        st.sidebar.success("All analysis data cleared!")
        st.rerun()


def render_maternal_data_quality_manual(user, data_available):
    """Render Maternal Data Quality with optimized analysis"""
    if not data_available:
        st.warning(
            """
            ⚠️ **Maternal data not available**
            
            To analyze maternal data quality:
            1. Visit the **Maternal Dashboard** tab first
            2. Wait for data to load completely (KPI cards will appear)
            3. Return to this tab
            """
        )
        return

    st.markdown("### 🤰 Maternal Data Quality Analysis")

    # Initialize session state for analysis triggers
    if "maternal_analysis_triggered" not in st.session_state:
        st.session_state.maternal_analysis_triggered = {
            "outliers": False,
            "missing": False,
        }

    # Filter data based on user access
    maternal_events_df = st.session_state.maternal_events_df
    maternal_tei_df = st.session_state.maternal_tei_df

    filtered_events, filtered_tei = filter_data_by_user_access(
        maternal_events_df, maternal_tei_df, user
    )

    # Analysis controls
    col1, col2 = st.columns(2)

    with col1:
        analyze_outliers = st.button(
            "🔍 Analyze Outliers",
            use_container_width=True,
            key="maternal_analyze_outliers",
        )

    with col2:
        analyze_missing = st.button(
            "📊 Analyze Missing Data",
            use_container_width=True,
            key="maternal_analyze_missing",
        )

    # Set analysis triggers
    if analyze_outliers:
        st.session_state.maternal_analysis_triggered["outliers"] = True
        st.session_state.maternal_analysis_triggered["missing"] = False
        st.rerun()

    if analyze_missing:
        st.session_state.maternal_analysis_triggered["missing"] = True
        st.session_state.maternal_analysis_triggered["outliers"] = False
        st.rerun()

    # Cache status
    current_user_key = (
        f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    )
    cache_key_outliers = f"dq_maternal_outliers_{current_user_key}"
    cache_key_missing = f"dq_maternal_missing_elements_{current_user_key}"

    has_cached_outliers = cache_key_outliers in st.session_state
    has_cached_missing = cache_key_missing in st.session_state

    # OUTLIERS ANALYSIS
    st.markdown("---")
    st.markdown("#### 📈 Outliers Analysis")

    # Run outliers analysis if triggered
    if st.session_state.maternal_analysis_triggered["outliers"]:
        # Clear previous state
        clear_analysis_state("maternal_outliers")

        # Run analysis with filtered data
        outliers_df = analyze_with_progress(
            check_maternal_outliers, filtered_events, user, "maternal_outliers"
        )
        st.session_state.maternal_outliers_df = outliers_df

        # Cache if completed
        if not st.session_state.get("maternal_outliers_stop", True):
            st.session_state[cache_key_outliers] = outliers_df.copy()
            st.session_state[f"{cache_key_outliers}_timestamp"] = time.time()

        # Reset trigger after analysis
        st.session_state.maternal_analysis_triggered["outliers"] = False
        st.rerun()

    # Get current outliers data - show empty until user clicks analyze
    outliers_df = pd.DataFrame()

    if (
        has_cached_outliers
        and not st.session_state.maternal_analysis_triggered["outliers"]
    ):
        outliers_df = st.session_state[cache_key_outliers]
        st.session_state.maternal_outliers_df = outliers_df
        st.success(f"📊 Cached analysis: {len(outliers_df)} outliers")
    elif (
        "maternal_outliers_df" in st.session_state
        and not st.session_state.maternal_outliers_df.empty
        and not st.session_state.maternal_analysis_triggered["outliers"]
    ):
        outliers_df = st.session_state.maternal_outliers_df
        if st.session_state.get("maternal_outliers_stop", False):
            st.warning(f"⏹️ Stopped analysis: {len(outliers_df)} outliers found")
        else:
            st.success(f"📊 Analysis complete: {len(outliers_df)} outliers found")
    else:
        st.info("👆 Click 'Analyze Outliers' to start analysis")

    # Display outliers results (ONLY when not actively analyzing)
    if (
        not outliers_df.empty
        and not st.session_state.maternal_analysis_triggered["outliers"]
    ):
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            if "Region" in outliers_df.columns:
                all_regions = ["All Regions"] + safe_sorted_unique(
                    outliers_df["Region"]
                )
                selected_region = st.selectbox(
                    "Region",
                    options=all_regions,
                    index=0,
                    key="maternal_outlier_region",
                )
            else:
                st.info("No Region data available")
                selected_region = "All Regions"

        with col2:
            if "Facility" in outliers_df.columns:
                all_facilities = ["All Facilities"] + safe_sorted_unique(
                    outliers_df["Facility"]
                )
                selected_facility = st.selectbox(
                    "Facility",
                    options=all_facilities,
                    index=0,
                    key="maternal_outlier_facility",
                )
            else:
                st.info("No Facility data available")
                selected_facility = "All Facilities"

        with col3:
            if "Data Element" in outliers_df.columns:
                all_elements = ["All Data Elements"] + safe_sorted_unique(
                    outliers_df["Data Element"]
                )
                selected_element = st.selectbox(
                    "Data Element",
                    options=all_elements,
                    index=0,
                    key="maternal_outlier_element",
                )
            else:
                st.info("No Data Element data available")
                selected_element = "All Data Elements"

        # Apply filters
        filtered_df = outliers_df.copy()
        if selected_region != "All Regions" and "Region" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]
        if selected_facility != "All Facilities" and "Facility" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]
        if (
            selected_element != "All Data Elements"
            and "Data Element" in filtered_df.columns
        ):
            filtered_df = filtered_df[filtered_df["Data Element"] == selected_element]

        if filtered_df.empty:
            st.info("No outliers match the selected filters")
        else:
            display_columns = [
                "First Name",
                "Father Name",
                "Region",
                "Facility",
                "Data Element",
                "Value",
                "Unit",
                "Issue Type",
            ]
            # Only include columns that actually exist in the dataframe
            available_columns = [
                col for col in display_columns if col in filtered_df.columns
            ]
            st.dataframe(filtered_df[available_columns], use_container_width=True)

    # MISSING DATA ANALYSIS
    st.markdown("---")
    st.markdown("#### 📋 Missing Data Analysis")

    # Run missing data analysis if triggered
    if st.session_state.maternal_analysis_triggered["missing"]:
        # Clear previous state
        clear_analysis_state("maternal_missing")

        # Run analysis with filtered data
        missing_df = analyze_with_progress(
            check_maternal_missing_elements, filtered_events, user, "maternal_missing"
        )
        st.session_state.maternal_missing_df = missing_df

        # Cache if completed
        if not st.session_state.get("maternal_missing_stop", True):
            st.session_state[cache_key_missing] = missing_df.copy()
            st.session_state[f"{cache_key_missing}_timestamp"] = time.time()

        # Reset trigger after analysis
        st.session_state.maternal_analysis_triggered["missing"] = False
        st.rerun()

    # Get current missing data - show empty until user clicks analyze
    missing_df = pd.DataFrame()

    if (
        has_cached_missing
        and not st.session_state.maternal_analysis_triggered["missing"]
    ):
        missing_df = st.session_state[cache_key_missing]
        st.session_state.maternal_missing_df = missing_df
        st.success(f"📊 Cached analysis: {len(missing_df)} missing elements")
    elif (
        "maternal_missing_df" in st.session_state
        and not st.session_state.maternal_missing_df.empty
        and not st.session_state.maternal_analysis_triggered["missing"]
    ):
        missing_df = st.session_state.maternal_missing_df
        if st.session_state.get("maternal_missing_stop", False):
            st.warning(f"⏹️ Stopped analysis: {len(missing_df)} missing elements found")
        else:
            st.success(
                f"📊 Analysis complete: {len(missing_df)} missing elements found"
            )
    else:
        st.info("👆 Click 'Analyze Missing Data' to start analysis")

    # Display missing data results (ONLY when not actively analyzing)
    if (
        not missing_df.empty
        and not st.session_state.maternal_analysis_triggered["missing"]
    ):
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            if "Region" in missing_df.columns:
                all_regions = ["All Regions"] + safe_sorted_unique(missing_df["Region"])
                selected_region = st.selectbox(
                    "Region",
                    options=all_regions,
                    index=0,
                    key="maternal_missing_region",
                )
            else:
                st.info("No Region data available")
                selected_region = "All Regions"

        with col2:
            if "Facility" in missing_df.columns:
                all_facilities = ["All Facilities"] + safe_sorted_unique(
                    missing_df["Facility"]
                )
                selected_facility = st.selectbox(
                    "Facility",
                    options=all_facilities,
                    index=0,
                    key="maternal_missing_facility",
                )
            else:
                st.info("No Facility data available")
                selected_facility = "All Facilities"

        # Apply filters
        filtered_df = missing_df.copy()
        if selected_region != "All Regions" and "Region" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]
        if selected_facility != "All Facilities" and "Facility" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]

        if filtered_df.empty:
            st.info("No missing elements match the selected filters")
        else:
            display_columns = [
                "First Name",
                "Father Name",
                "Region",
                "Facility",
                "Data Element Missing",
            ]
            # Only include columns that actually exist in the dataframe
            available_columns = [
                col for col in display_columns if col in filtered_df.columns
            ]
            st.dataframe(filtered_df[available_columns], use_container_width=True)


def render_newborn_data_quality_manual(user, data_available):
    """Render Newborn Data Quality with optimized analysis"""
    if not data_available:
        st.warning(
            """
            ⚠️ **Newborn data not available**
            
            To analyze newborn data quality:
            1. Visit the **Newborn Dashboard** tab first
            2. Wait for data to load completely (KPI cards will appear)
            3. Return to this tab
            """
        )
        return

    st.markdown("### 👶 Newborn Data Quality Analysis")

    # Initialize session state for analysis triggers
    if "newborn_analysis_triggered" not in st.session_state:
        st.session_state.newborn_analysis_triggered = {
            "outliers": False,
            "missing": False,
        }

    # Filter data based on user access
    newborn_events_df = st.session_state.newborn_events_df
    newborn_tei_df = st.session_state.newborn_tei_df

    filtered_events, filtered_tei = filter_data_by_user_access(
        newborn_events_df, newborn_tei_df, user
    )

    # Analysis controls
    col1, col2 = st.columns(2)

    with col1:
        analyze_outliers = st.button(
            "🔍 Analyze Outliers",
            use_container_width=True,
            key="newborn_analyze_outliers",
        )

    with col2:
        analyze_missing = st.button(
            "📊 Analyze Missing Data",
            use_container_width=True,
            key="newborn_analyze_missing",
        )

    # Set analysis triggers
    if analyze_outliers:
        st.session_state.newborn_analysis_triggered["outliers"] = True
        st.session_state.newborn_analysis_triggered["missing"] = False
        st.rerun()

    if analyze_missing:
        st.session_state.newborn_analysis_triggered["missing"] = True
        st.session_state.newborn_analysis_triggered["outliers"] = False
        st.rerun()

    # Cache status
    current_user_key = (
        f"{user.get('username', 'unknown')}_{user.get('role', 'unknown')}"
    )
    cache_key_outliers = f"dq_newborn_outliers_{current_user_key}"
    cache_key_missing = f"dq_newborn_missing_elements_{current_user_key}"

    has_cached_outliers = cache_key_outliers in st.session_state
    has_cached_missing = cache_key_missing in st.session_state

    # OUTLIERS ANALYSIS
    st.markdown("---")
    st.markdown("#### 📈 Outliers Analysis")

    # Run outliers analysis if triggered
    if st.session_state.newborn_analysis_triggered["outliers"]:
        # Clear previous state
        clear_analysis_state("newborn_outliers")

        # Run analysis with filtered data
        outliers_df = analyze_with_progress(
            check_newborn_outliers, filtered_events, user, "newborn_outliers"
        )
        st.session_state.newborn_outliers_df = outliers_df

        # Cache if completed
        if not st.session_state.get("newborn_outliers_stop", True):
            st.session_state[cache_key_outliers] = outliers_df.copy()
            st.session_state[f"{cache_key_outliers}_timestamp"] = time.time()

        # Reset trigger after analysis
        st.session_state.newborn_analysis_triggered["outliers"] = False
        st.rerun()

    # Get current outliers data - show empty until user clicks analyze
    outliers_df = pd.DataFrame()

    if (
        has_cached_outliers
        and not st.session_state.newborn_analysis_triggered["outliers"]
    ):
        outliers_df = st.session_state[cache_key_outliers]
        st.session_state.newborn_outliers_df = outliers_df
        st.success(f"📊 Cached analysis: {len(outliers_df)} outliers")
    elif (
        "newborn_outliers_df" in st.session_state
        and not st.session_state.newborn_outliers_df.empty
        and not st.session_state.newborn_analysis_triggered["outliers"]
    ):
        outliers_df = st.session_state.newborn_outliers_df
        if st.session_state.get("newborn_outliers_stop", False):
            st.warning(f"⏹️ Stopped analysis: {len(outliers_df)} outliers found")
        else:
            st.success(f"📊 Analysis complete: {len(outliers_df)} outliers found")
    else:
        st.info("👆 Click 'Analyze Outliers' to start analysis")

    # Display outliers results (ONLY when not actively analyzing)
    if (
        not outliers_df.empty
        and not st.session_state.newborn_analysis_triggered["outliers"]
    ):
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            if "Region" in outliers_df.columns:
                all_regions = ["All Regions"] + safe_sorted_unique(
                    outliers_df["Region"]
                )
                selected_region = st.selectbox(
                    "Region", options=all_regions, index=0, key="newborn_outlier_region"
                )
            else:
                st.info("No Region data available")
                selected_region = "All Regions"

        with col2:
            if "Facility" in outliers_df.columns:
                all_facilities = ["All Facilities"] + safe_sorted_unique(
                    outliers_df["Facility"]
                )
                selected_facility = st.selectbox(
                    "Facility",
                    options=all_facilities,
                    index=0,
                    key="newborn_outlier_facility",
                )
            else:
                st.info("No Facility data available")
                selected_facility = "All Facilities"

        with col3:
            if "Data Element" in outliers_df.columns:
                all_elements = ["All Data Elements"] + safe_sorted_unique(
                    outliers_df["Data Element"]
                )
                selected_element = st.selectbox(
                    "Data Element",
                    options=all_elements,
                    index=0,
                    key="newborn_outlier_element",
                )
            else:
                st.info("No Data Element data available")
                selected_element = "All Data Elements"

        # Apply filters
        filtered_df = outliers_df.copy()
        if selected_region != "All Regions" and "Region" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]
        if selected_facility != "All Facilities" and "Facility" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]
        if (
            selected_element != "All Data Elements"
            and "Data Element" in filtered_df.columns
        ):
            filtered_df = filtered_df[filtered_df["Data Element"] == selected_element]

        if filtered_df.empty:
            st.info("No outliers match the selected filters")
        else:
            display_columns = [
                "First Name",
                "Last Name",
                "Region",
                "Facility",
                "Data Element",
                "Value",
                "Unit",
                "Issue Type",
            ]
            # Only include columns that actually exist in the dataframe
            available_columns = [
                col for col in display_columns if col in filtered_df.columns
            ]
            st.dataframe(filtered_df[available_columns], use_container_width=True)

    # MISSING DATA ANALYSIS
    st.markdown("---")
    st.markdown("#### 📋 Missing Data Analysis")

    # Run missing data analysis if triggered
    if st.session_state.newborn_analysis_triggered["missing"]:
        # Clear previous state
        clear_analysis_state("newborn_missing")

        # Run analysis with filtered data
        missing_df = analyze_with_progress(
            check_newborn_missing_elements, filtered_events, user, "newborn_missing"
        )
        st.session_state.newborn_missing_df = missing_df

        # Cache if completed
        if not st.session_state.get("newborn_missing_stop", True):
            st.session_state[cache_key_missing] = missing_df.copy()
            st.session_state[f"{cache_key_missing}_timestamp"] = time.time()

        # Reset trigger after analysis
        st.session_state.newborn_analysis_triggered["missing"] = False
        st.rerun()

    # Get current missing data - show empty until user clicks analyze
    missing_df = pd.DataFrame()

    if (
        has_cached_missing
        and not st.session_state.newborn_analysis_triggered["missing"]
    ):
        missing_df = st.session_state[cache_key_missing]
        st.session_state.newborn_missing_df = missing_df
        st.success(f"📊 Cached analysis: {len(missing_df)} missing elements")
    elif (
        "newborn_missing_df" in st.session_state
        and not st.session_state.newborn_missing_df.empty
        and not st.session_state.newborn_analysis_triggered["missing"]
    ):
        missing_df = st.session_state.newborn_missing_df
        if st.session_state.get("newborn_missing_stop", False):
            st.warning(f"⏹️ Stopped analysis: {len(missing_df)} missing elements found")
        else:
            st.success(
                f"📊 Analysis complete: {len(missing_df)} missing elements found"
            )
    else:
        st.info("👆 Click 'Analyze Missing Data' to start analysis")

    # Display missing data results (ONLY when not actively analyzing)
    if (
        not missing_df.empty
        and not st.session_state.newborn_analysis_triggered["missing"]
    ):
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            if "Region" in missing_df.columns:
                all_regions = ["All Regions"] + safe_sorted_unique(missing_df["Region"])
                selected_region = st.selectbox(
                    "Region", options=all_regions, index=0, key="newborn_missing_region"
                )
            else:
                st.info("No Region data available")
                selected_region = "All Regions"

        with col2:
            if "Facility" in missing_df.columns:
                all_facilities = ["All Facilities"] + safe_sorted_unique(
                    missing_df["Facility"]
                )
                selected_facility = st.selectbox(
                    "Facility",
                    options=all_facilities,
                    index=0,
                    key="newborn_missing_facility",
                )
            else:
                st.info("No Facility data available")
                selected_facility = "All Facilities"

        # Apply filters
        filtered_df = missing_df.copy()
        if selected_region != "All Regions" and "Region" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]
        if selected_facility != "All Facilities" and "Facility" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Facility"] == selected_facility]

        if filtered_df.empty:
            st.info("No missing elements match the selected filters")
        else:
            display_columns = [
                "First Name",
                "Last Name",
                "Region",
                "Facility",
                "Data Element Missing",
            ]
            # Only include columns that actually exist in the dataframe
            available_columns = [
                col for col in display_columns if col in filtered_df.columns
            ]
            st.dataframe(filtered_df[available_columns], use_container_width=True)
