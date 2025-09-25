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
            "unit": "per 1000",
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
            "unit": "per 100,000",
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
            f"{current_value:.2f}{kpi['unit'].replace('per ', '/')}"
            if current_value is not None
            else "N/A"
        )
        prev_display = (
            f"{prev_value:.2f}{kpi['unit'].replace('per ', '/')}"
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
    events_df,
    facility_uids=None,
    location_name="Location",
    user_id="default_user",
    user=None,
):
    """
    Render KPI cards component with trend indicators only (no previous values shown in cards).
    Previous values are only included in the Excel export for comparison.
    """
    if events_df.empty or "event_date" not in events_df.columns:
        st.markdown(
            '<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>',
            unsafe_allow_html=True,
        )
        return

    # Compute KPIs for the selected facilities
    kpis = compute_kpis(events_df, facility_uids)
    if not isinstance(kpis, dict):
        st.error("Error computing KPI. Please check data.")
        return

    # Load last KPI values for this user
    previous_kpis = load_previous_kpis(user_id)

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

    # Get location name for file naming
    if user is None:
        # Try to get user from session state if not provided
        user = st.session_state.get("user", {})

    location_file_name = get_location_name_for_file(user)

    # Generate Excel report with comparison data
    excel_file = generate_kpi_excel_report(
        kpis, previous_kpis, location_name=location_name
    )

    # Save new KPI values for the next login/session
    save_current_kpis(user_id, kpis)

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
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header with download button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<div class="section-header">📊 Key Performance Indicators</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.download_button(
            label="⬇️ Download KPI Comparison Report",
            data=excel_file,
            file_name=f"kpi_comparison_report_{location_file_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download Excel report with current vs previous KPI values comparison",
        )

    # KPI Cards HTML - Only show current values with trend indicators
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
            <div class="kpi-value">{kpis.get("stillbirth_rate",0):.2f} <span class="{stillbirth_trend_class}">{stillbirth_trend}</span></div>
            <div class="kpi-name">Stillbirth Rate (per 1000 births)</div>
            <div class="kpi-metrics">
                <span class="metric-label metric-stillbirth">Stillbirths: {kpis.get("stillbirths",0)}</span>
                <span class="metric-label metric-total">Total Births: {kpis.get("total_births",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("pnc_coverage",0):.2f}% <span class="{pnc_trend_class}">{pnc_trend}</span></div>
            <div class="kpi-name">Early PNC Coverage (within 48 hrs)</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-fp">PNC ≤48 hrs: {kpis.get("early_pnc",0)}</span>
               <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries_pnc",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("maternal_death_rate",0):.2f} <span class="{maternal_death_trend_class}">{maternal_death_trend}</span></div>
            <div class="kpi-name">Maternal Death Rate (per 100,000 births)</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-maternal-death">Maternal Deaths: {kpis.get("maternal_deaths",0)}</span>
               <span class="metric-label metric-total">Live Births: {kpis.get("live_births",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("csection_rate",0):.2f}% <span class="{csection_trend_class}">{csection_trend}</span></div>
            <div class="kpi-name">C-Section Rate</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-csection">C-Sections: {kpis.get("csection_deliveries",0)}</span>
               <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries_cs",0)}</span>
            </div>
        </div>
    </div>
    """

    st.markdown(kpi_html, unsafe_allow_html=True)

    # Add a small note about the Excel report
    st.caption(
        "💡 Download the Excel report to see detailed comparison with previous values"
    )
