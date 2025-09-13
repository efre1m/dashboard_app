import streamlit as st
import pandas as pd
import json
import os
from utils.kpi_utils import compute_kpis

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

# ---------------- KPI Card Rendering ----------------
def render_kpi_cards(events_df, facility_uids=None, location_name="Location", user_id="default_user"):
    """
    Render KPI cards component with persistent previous-value comparison.
    - Loads previous KPIs from JSON
    - Compares with current KPIs (context-aware coloring)
    - Displays ▲ ▼ – with colors
    - Saves current KPIs back to JSON
    """
    if events_df.empty or "event_date" not in events_df.columns:
        st.markdown('<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>', unsafe_allow_html=True)
        return

    # Compute KPIs for the selected facilities
    kpis = compute_kpis(events_df, facility_uids)
    if not isinstance(kpis, dict):
        st.error("Error computing KPI. Please check data.")
        return

    # Load last KPI values for this user
    previous_kpis = load_previous_kpis(user_id)

    # Compare current vs previous session (context-aware)
    ippcar_trend, ippcar_trend_class = compare_trend(kpis.get("ippcar", 0), previous_kpis.get("ippcar"), higher_is_better=True)
    pnc_trend, pnc_trend_class = compare_trend(kpis.get("pnc_coverage", 0), previous_kpis.get("pnc_coverage"), higher_is_better=True)
    stillbirth_trend, stillbirth_trend_class = compare_trend(kpis.get("stillbirth_rate", 0), previous_kpis.get("stillbirth_rate"), higher_is_better=False)
    maternal_death_trend, maternal_death_trend_class = compare_trend(kpis.get("maternal_death_rate", 0), previous_kpis.get("maternal_death_rate"), higher_is_better=False)
    csection_trend, csection_trend_class = compare_trend(kpis.get("csection_rate", 0), previous_kpis.get("csection_rate"), higher_is_better=False)

    # Save new KPI values for the next login/session
    save_current_kpis(user_id, kpis)

    # Add CSS for black text + trend colors
    st.markdown("""
    <style>
    .kpi-value, .kpi-name, .metric-label { color: #000000 !important; }
    .section-header { color: #000000 !important; }
    .trend-good { color: green !important; font-weight: bold; }
    .trend-bad { color: red !important; font-weight: bold; }
    .trend-neutral { color: gray !important; }
    </style>
    """, unsafe_allow_html=True)

    # KPI Cards HTML
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

    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    st.markdown(kpi_html, unsafe_allow_html=True)
