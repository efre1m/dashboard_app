import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO
from datetime import datetime
from utils.kpi_utils import compute_kpis
from utils.auth import get_user_display_info  # Import the auth utility


# ---------------- Utility: Store previous KPI values ----------------
def load_previous_kpis(user_id: str):
    filepath = f"previous_kpis_{user_id}.json"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def save_current_kpis(user_id: str, kpis: dict):
    filepath = f"previous_kpis_{user_id}.json"
    with open(filepath, "w") as f:
        json.dump(kpis, f)


# ---------------- Session State Management ----------------
def initialize_kpi_session_state(user_id: str):
    """Initialize session state for KPI tracking"""
    if f"kpi_initialized_{user_id}" not in st.session_state:
        # Load previous KPIs from file on first load
        st.session_state[f"previous_kpis_{user_id}"] = load_previous_kpis(user_id)
        st.session_state[f"current_kpis_{user_id}"] = {}
        st.session_state[f"kpi_initialized_{user_id}"] = True
        st.session_state[f"kpi_last_saved_{user_id}"] = datetime.now().isoformat()


def should_save_kpis(user_id: str, current_kpis: dict) -> bool:
    """Determine if we should save KPIs (only on actual data changes, not every re-render)"""
    if f"current_kpis_{user_id}" not in st.session_state:
        return True

    # Check if KPIs have actually changed
    previous_current = st.session_state.get(f"current_kpis_{user_id}", {})

    # Compare key KPI values to see if there's a meaningful change
    kpi_keys = [
        "ippcar",
        "stillbirth_rate",
        "pnc_coverage",
        "maternal_death_rate",
        "csection_rate",
    ]

    for key in kpi_keys:
        prev_value = previous_current.get(key, 0)
        curr_value = current_kpis.get(key, 0)

        # If any KPI has changed by more than 0.1, consider it a real change
        if abs(prev_value - curr_value) > 0.1:
            return True

    # Also save if it's been more than 5 minutes since last save (in case of small changes)
    last_saved = st.session_state.get(f"kpi_last_saved_{user_id}")
    if last_saved:
        last_saved_dt = datetime.fromisoformat(last_saved)
        time_diff = (datetime.now() - last_saved_dt).total_seconds()
        if time_diff > 300:  # 5 minutes
            return True

    return False


# ---------------- Trend Comparison with Context ----------------
def compare_trend(current_value, prev_value, higher_is_better=True):
    """
    Compare KPI trends with contextual coloring.
    - higher_is_better=True  -> increase is good (green), decrease is bad (red)
    - higher_is_better=False -> increase is bad (red), decrease is good (green)
    """
    if prev_value is None:
        return "–", "trend-neutral"

    if current_value > prev_value:
        return ("▲", "trend-good") if higher_is_better else ("▲", "trend-bad")
    elif current_value < prev_value:
        return ("▼", "trend-bad") if higher_is_better else ("▼", "trend-good")
    else:
        return "–", "trend-neutral"


def get_trend_text(current_value, prev_value, higher_is_better=True):
    """
    Get trend as text for Excel export
    """
    if prev_value is None:
        return "No Previous Data"

    if current_value > prev_value:
        return "Increasing" if higher_is_better else "Decreasing"
    elif current_value < prev_value:
        return "Decreasing" if higher_is_better else "Increasing"
    else:
        return "No Change"


# ---------------- Excel Export Function ----------------
def generate_kpi_excel_report(
    current_kpis, previous_kpis, session_date=None, location_name="Location"
):
    """
    Generate Excel file with KPI data showing current vs previous values
    """
    if session_date is None:
        session_date = datetime.now().strftime("%Y-%m-%d")

    # KPI configuration
    kpi_config = [
        {
            "name": "IPPCAR",
            "current_key": "ippcar",
            "prev_key": "ippcar",
            "higher_is_better": True,
            "unit": "%",
        },
        {
            "name": "Stillbirth Rate",
            "current_key": "stillbirth_rate",
            "prev_key": "stillbirth_rate",
            "higher_is_better": False,
            "unit": "%",
        },
        {
            "name": "PNC Coverage",
            "current_key": "pnc_coverage",
            "prev_key": "pnc_coverage",
            "higher_is_better": True,
            "unit": "%",
        },
        {
            "name": "Maternal Death Rate",
            "current_key": "maternal_death_rate",
            "prev_key": "maternal_death_rate",
            "higher_is_better": False,
            "unit": "%",
        },
        {
            "name": "C-Section Rate",
            "current_key": "csection_rate",
            "prev_key": "csection_rate",
            "higher_is_better": False,
            "unit": "%",
        },
    ]

    # Create DataFrame with current vs previous comparison
    data = []
    for kpi in kpi_config:
        current_value = current_kpis.get(kpi["current_key"], 0)
        prev_value = previous_kpis.get(kpi["prev_key"])
        trend = get_trend_text(current_value, prev_value, kpi["higher_is_better"])

        # Format values with units for display
        current_display = (
            f"{current_value:.2f}{kpi['unit']}" if current_value is not None else "N/A"
        )
        prev_display = (
            f"{prev_value:.2f}{kpi['unit']}"
            if prev_value is not None
            else "No Previous Data"
        )

        data.append(
            {
                "Date": session_date,
                "KPI Name": kpi["name"],
                "Current KPI Value": current_display,
                "Previous KPI Value": prev_display,
                "Trend": trend,
                "Raw Current Value": current_value,
                "Raw Previous Value": prev_value,
            }
        )

    df = pd.DataFrame(data)

    # Create Excel file with better formatting
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Write main data (without raw values)
        display_df = df.drop(columns=["Raw Current Value", "Raw Previous Value"])
        display_df.to_excel(writer, sheet_name="KPI Comparison Report", index=False)

        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets["KPI Comparison Report"]

        # Merge Date cells for each KPI group (A2:A6, A7:A11, etc. - assuming 5 KPIs per date)
        from openpyxl.styles import Alignment

        for i in range(0, len(df), 5):
            start_row = i + 2  # +2 because Excel is 1-indexed and we have header
            end_row = start_row + 4  # 5 rows total (0-4)
            if end_row <= len(df) + 1:  # +1 for header
                worksheet.merge_cells(f"A{start_row}:A{end_row}")
                cell = worksheet[f"A{start_row}"]
                cell.alignment = Alignment(horizontal="center", vertical="center")

        # Add header styling
        from openpyxl.styles import Font

        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(
                max_length + 2, 50
            )  # Cap at 50 to avoid too wide columns
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Add a summary sheet with just the comparison
        summary_data = []
        for kpi in kpi_config:
            current_value = current_kpis.get(kpi["current_key"], 0)
            prev_value = previous_kpis.get(kpi["prev_key"])

            summary_data.append(
                {
                    "KPI Name": kpi["name"],
                    "Current Value": current_value,
                    "Previous Value": prev_value if prev_value is not None else "N/A",
                    "Change": (
                        current_value - prev_value if prev_value is not None else "N/A"
                    ),
                    "Trend": get_trend_text(
                        current_value, prev_value, kpi["higher_is_better"]
                    ),
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Style summary sheet
        summary_sheet = writer.sheets["Summary"]
        for cell in summary_sheet[1]:
            cell.font = Font(bold=True)

        for column in summary_sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            summary_sheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    return output


# ---------------- Get Location Name for File ----------------
def get_location_name_for_file(user: dict) -> str:
    """
    Extract location name for file naming based on user role
    """
    role = user.get("role", "")

    if role == "facility":
        return user.get("facility_name", "Facility").replace(" ", "_")
    elif role == "regional":
        return user.get("region_name", "Region").replace(" ", "_")
    elif role == "national":
        return user.get("country_name", "National").replace(" ", "_")
    elif role == "admin":
        return "National"  # Admin sees national data by default
    else:
        return "Location"


# ---------------- KPI Card Rendering ----------------
def render_kpi_cards(
    filtered_events_df,  # CHANGED: Now accepts filtered data
    location_name="Location",
    user_id="default_user",
    user=None,
):
    """
    Render KPI cards component with trend indicators that respect ALL user filters.
    """
    if filtered_events_df.empty or "event_date" not in filtered_events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # Initialize session state for this user
    initialize_kpi_session_state(user_id)

    # Compute KPIs for the FILTERED data (no additional filtering needed)
    kpis = compute_kpis(filtered_events_df)  # REMOVED facility_uids parameter

    if not isinstance(kpis, dict):
        st.error("Error computing KPI. Please check data.")
        return

    # Get previous KPIs from session state (persists across re-renders)
    previous_kpis = st.session_state.get(f"previous_kpis_{user_id}", {})

    # Only save KPIs if there's a meaningful data change or it's been a while
    if should_save_kpis(user_id, kpis):
        save_current_kpis(user_id, kpis)
        st.session_state[f"previous_kpis_{user_id}"] = kpis
        st.session_state[f"current_kpis_{user_id}"] = kpis
        st.session_state[f"kpi_last_saved_{user_id}"] = datetime.now().isoformat()

    # Update current KPIs in session state
    st.session_state[f"current_kpis_{user_id}"] = kpis

    # FIX: Store trend calculations in session state to persist across re-renders
    trend_session_key = f"trend_calculations_{user_id}"

    # Only recalculate trends if they don't exist or data has actually changed
    if trend_session_key not in st.session_state or should_save_kpis(user_id, kpis):
        # Compare current vs previous session (context-aware)
        ippcar_trend, ippcar_trend_class = compare_trend(
            kpis.get("ippcar", 0), previous_kpis.get("ippcar"), higher_is_better=True
        )
        pnc_trend, pnc_trend_class = compare_trend(
            kpis.get("pnc_coverage", 0),
            previous_kpis.get("pnc_coverage"),
            higher_is_better=True,
        )
        stillbirth_trend, stillbirth_trend_class = compare_trend(
            kpis.get("stillbirth_rate", 0),
            previous_kpis.get("stillbirth_rate"),
            higher_is_better=False,
        )
        maternal_death_trend, maternal_death_trend_class = compare_trend(
            kpis.get("maternal_death_rate", 0),
            previous_kpis.get("maternal_death_rate"),
            higher_is_better=False,
        )
        csection_trend, csection_trend_class = compare_trend(
            kpis.get("csection_rate", 0),
            previous_kpis.get("csection_rate"),
            higher_is_better=False,
        )

        # Store trends in session state
        st.session_state[trend_session_key] = {
            "ippcar": (ippcar_trend, ippcar_trend_class),
            "pnc": (pnc_trend, pnc_trend_class),
            "stillbirth": (stillbirth_trend, stillbirth_trend_class),
            "maternal_death": (maternal_death_trend, maternal_death_trend_class),
            "csection": (csection_trend, csection_trend_class),
        }
    else:
        # Use stored trends from session state
        stored_trends = st.session_state[trend_session_key]
        ippcar_trend, ippcar_trend_class = stored_trends["ippcar"]
        pnc_trend, pnc_trend_class = stored_trends["pnc"]
        stillbirth_trend, stillbirth_trend_class = stored_trends["stillbirth"]
        maternal_death_trend, maternal_death_trend_class = stored_trends[
            "maternal_death"
        ]
        csection_trend, csection_trend_class = stored_trends["csection"]

    # Get location name for file naming
    if user is None:
        # Try to get user from session state if not provided
        user = st.session_state.get("user", {})

    location_file_name = get_location_name_for_file(user)

    # Generate Excel report with comparison data
    excel_file = generate_kpi_excel_report(
        kpis, previous_kpis, location_name=location_name
    )

    # Add CSS for styling - MODIFIED TO DISPLAY CARDS IN ONE ROW
    st.markdown(
        """
    <style>
    .kpi-value, .kpi-name, .metric-label { color: #000000 !important; }
    .section-header { color: #000000 !important; font-size: 1.2em; font-weight: bold; margin-bottom: 1rem; }
    .trend-good { color: green !important; font-weight: bold; }
    .trend-bad { color: red !important; font-weight: bold; }
    .trend-neutral { color: gray !important; }
    .kpi-grid { 
        display: grid; 
        grid-template-columns: repeat(5, 1fr); 
        gap: 1rem; 
        margin: 1rem 0; 
        overflow-x: auto;
    }
    .kpi-card { 
        background: white; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 4px solid #4CAF50; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        min-width: 220px;
    }
    .kpi-value { font-size: 1.5em; font-weight: bold; margin-bottom: 0.5rem; }
    .kpi-name { font-weight: 600; margin-bottom: 0.5rem; color: #333 !important; font-size: 0.9em; }
    .kpi-metrics { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.8em; }
    .metric-label { padding: 0.1rem 0.5rem; border-radius: 3px; display: inline-block; width: fit-content; }
    .metric-fp { background: #e8f5e8; }
    .metric-stillbirth { background: #ffe8e8; }
    .metric-maternal-death { background: #ffcccc; }
    .metric-csection { background: #fff0cc; }
    .metric-total { background: #f0f0f0; }
    
    /* Responsive design for smaller screens */
    @media (max-width: 1400px) {
        .kpi-grid { 
            grid-template-columns: repeat(5, minmax(200px, 1fr)); 
        }
        .kpi-card { min-width: 200px; }
    }
    
    @media (max-width: 1200px) {
        .kpi-grid { 
            grid-template-columns: repeat(3, 1fr); 
        }
    }
    
    @media (max-width: 768px) {
        .kpi-grid { 
            grid-template-columns: repeat(2, 1fr); 
        }
    }
    
    @media (max-width: 480px) {
        .kpi-grid { 
            grid-template-columns: 1fr; 
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<div class="section-header">📊 Key Performance Indicators</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.download_button(
            label="⬇️ Download KPI Report",
            data=excel_file,
            file_name=f"kpi_comparison_report_{location_file_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download Excel report with current vs previous KPI values comparison",
            use_container_width=True,
        )

    # KPI Cards HTML - Show current values with persistent trend indicators
    # FIXED: Stillbirth and Maternal Death rates are over total deliveries, not total births/live births
    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("ippcar",0):.2f}% <span class="{ippcar_trend_class}">{ippcar_trend}</span></div>
            <div class="kpi-name">IPPCAR (Immediate Postpartum Contraceptive Acceptance Rate)</div>
            <div class="kpi-metrics">
                <span class="metric-label metric-fp">Accepted FP: {kpis.get("fp_acceptance",0)}</span>
                <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("stillbirth_rate",0):.2f}% <span class="{stillbirth_trend_class}">{stillbirth_trend}</span></div>
            <div class="kpi-name">Stillbirth Rate</div>
            <div class="kpi-metrics">
                <span class="metric-label metric-stillbirth">Stillbirths: {kpis.get("stillbirths",0)}</span>
                <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("pnc_coverage",0):.2f}% <span class="{pnc_trend_class}">{pnc_trend}</span></div>
            <div class="kpi-name">Early PNC Coverage (within 48 hrs)</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-fp">PNC ≤48 hrs: {kpis.get("early_pnc",0)}</span>
               <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("maternal_death_rate",0):.2f}% <span class="{maternal_death_trend_class}">{maternal_death_trend}</span></div>
            <div class="kpi-name">Maternal Death Rate</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-maternal-death">Maternal Deaths: {kpis.get("maternal_deaths",0)}</span>
               <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("csection_rate",0):.2f}% <span class="{csection_trend_class}">{csection_trend}</span></div>
            <div class="kpi-name">C-Section Rate</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-csection">C-Sections: {kpis.get("csection_deliveries",0)}</span>
               <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
    </div>
    """

    st.markdown(kpi_html, unsafe_allow_html=True)

    # Add information about trend persistence
    st.caption(
        "💡 Trend indicators compare current values with the last saved baseline. The baseline resets automatically after ~3 minutes, refresh, or new login."
    )


# ---------------- Helper function for dashboard compatibility ----------------
def get_kpi_comparison_data(
    df,
    period_col="period_display",
    value_col="value",
    kpi_name="KPI",
    facility_uids=None,
):
    """
    Helper function to prepare KPI comparison data for dashboard charts
    Compatible with the updated kpi_utils.py functions
    """
    if df.empty:
        return pd.DataFrame()

    # Apply facility filtering if needed
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Group by period
    grouped = filtered_df.groupby(period_col)

    result_data = []
    for period, group_df in grouped:
        # Compute KPIs for this period group
        kpis = compute_kpis(group_df)

        # Extract relevant KPI based on kpi_name
        if "IPPCAR" in kpi_name or "Contraceptive" in kpi_name:
            value = kpis.get("ippcar", 0)
            numerator = kpis.get("fp_acceptance", 0)
            denominator = kpis.get("total_deliveries", 1)
        elif "Stillbirth" in kpi_name:
            value = kpis.get("stillbirth_rate", 0)
            numerator = kpis.get("stillbirths", 0)
            denominator = kpis.get("total_deliveries", 1)
        elif "PNC" in kpi_name or "Postnatal" in kpi_name:
            value = kpis.get("pnc_coverage", 0)
            numerator = kpis.get("early_pnc", 0)
            denominator = kpis.get("total_deliveries", 1)
        elif "Maternal Death" in kpi_name:
            value = kpis.get("maternal_death_rate", 0)
            numerator = kpis.get("maternal_deaths", 0)
            denominator = kpis.get("total_deliveries", 1)
        elif "C-Section" in kpi_name:
            value = kpis.get("csection_rate", 0)
            numerator = kpis.get("csection_deliveries", 0)
            denominator = kpis.get("total_deliveries", 1)
        else:
            value = 0
            numerator = 0
            denominator = 1

        result_data.append(
            {
                period_col: period,
                value_col: value,
                "numerator": numerator,
                "denominator": denominator,
            }
        )

    return pd.DataFrame(result_data)


# ---------------- Additional helper for time series data ----------------
def prepare_kpi_timeseries_data(
    df, kpi_name="KPI", facility_uids=None, period_col="period_display"
):
    """
    Prepare time series data for KPI trend charts
    Works with the updated kpi_utils.py
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure we have period columns
    if period_col not in df.columns and "event_date" in df.columns:
        # Create period columns if not present
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df[period_col] = df["event_date"].dt.strftime("%b-%y")
        df["period_sort"] = df["event_date"].dt.strftime("%Y%m")

    if period_col not in df.columns:
        return pd.DataFrame()

    # Apply facility filtering if needed
    filtered_df = df.copy()
    if facility_uids and "orgUnit" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["orgUnit"].isin(facility_uids)].copy()

    # Sort by period if sort column exists
    if "period_sort" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("period_sort")

    # Get grouped data
    return get_kpi_comparison_data(
        filtered_df, period_col, "value", kpi_name, facility_uids
    )
