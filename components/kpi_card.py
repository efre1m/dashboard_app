import streamlit as st
import pandas as pd
from utils.kpi_utils import compute_kpis

def calculate_trend(df: pd.DataFrame, kpi_type: str, facility_uids=None):
    """
    Compute ▲ ▼ – by comparing the last two period values.
    Assumes df already has a 'period' column.
    """
    if df.empty or "period" not in df.columns:
        return "–", "trend-neutral"

    # Filter by facilities if specified
    if facility_uids and facility_uids != ["All Facilities"]:
        df = df[df["orgUnit"].isin(facility_uids)]

    # Build a compact period-level dataframe with a single 'value' column
    if kpi_type == "ippcar":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": (
                    x[(x["dataElement_uid"] == "Q1p7CxWGUoi") &
                      (x["value"].isin([
                          "sn2MGial4TT", "aB5By4ATx8M", "TAxj9iLvWQ0",
                          "FyCtuLALNpY", "ejFYFZlmlwT"
                      ]))]["tei_id"].nunique()
                    / max(1, x[(x["dataElement_uid"] == "lphtwP2ViZU") & (x["value"].notna())]["tei_id"].nunique())
                ) * 100
            })
        ).reset_index(drop=True)

    elif kpi_type == "stillbirth":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["stillbirth_rate"]
            })
        ).reset_index(drop=True)

    elif kpi_type == "pnc":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["pnc_coverage"]
            })
        ).reset_index(drop=True)

    elif kpi_type == "maternal_death":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["maternal_death_rate"]
            })
        ).reset_index(drop=True)

    elif kpi_type == "csection":
        group = df.groupby("period", as_index=False).apply(
            lambda x: pd.Series({
                "value": compute_kpis(x)["csection_rate"]
            })
        ).reset_index(drop=True)

    else:
        return "–", "trend-neutral"

    if len(group) > 1:
        last_value = group["value"].iloc[-1]
        prev_value = group["value"].iloc[-2]
        if pd.notna(last_value) and pd.notna(prev_value):
            if last_value > prev_value:
                return "▲", "trend-up"
            elif last_value < prev_value:
                return "▼", "trend-down"

    return "–", "trend-neutral"

def render_kpi_cards(events_df, facility_uids=None, location_name="Location"):
    """
    Render KPI cards component with proper CSS styling
    
    Args:
        events_df: DataFrame with events data
        facility_uids: List of facility UIDs to filter by (None for all)
        location_name: Name of the facility/region for display
    """
    if events_df.empty or "event_date" not in events_df.columns:
        st.markdown('<div class="no-data-warning">⚠️ No data available. KPIs and charts are hidden.</div>', unsafe_allow_html=True)
        return

    # Compute KPIs for the selected facilities
    kpis = compute_kpis(events_df, facility_uids)
    if not isinstance(kpis, dict):
        st.error("Error computing KPI. Please check data.")
        return

    # Calculate trends for the cards
    ippcar_trend, ippcar_trend_class = calculate_trend(events_df, "ippcar", facility_uids)
    stillbirth_trend, stillbirth_trend_class = calculate_trend(events_df, "stillbirth", facility_uids)
    pnc_trend, pnc_trend_class = calculate_trend(events_df, "pnc", facility_uids)
    maternal_death_trend, maternal_death_trend_class = calculate_trend(events_df, "maternal_death", facility_uids)
    csection_trend, csection_trend_class = calculate_trend(events_df, "csection", facility_uids)

    # Add CSS for black text styling
    st.markdown("""
    <style>
    .kpi-value, .kpi-name, .metric-label {
        color: #000000 !important;
    }
    .section-header {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # KPI Cards HTML with proper CSS classes
    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("ippcar",0):.1f}% <span class="{ippcar_trend_class}">{ippcar_trend}</span></div>
            <div class="kpi-name">IPPCAR (Immediate Postpartum Contraceptive Acceptance Rate)</div>
            <div class="kpi-metrics">
                <span class="metric-label metric-fp">Accepted FP: {kpis.get("fp_acceptance",0)}</span>
                <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("stillbirth_rate",0):.1f} <span class="{stillbirth_trend_class}">{stillbirth_trend}</span></div>
            <div class="kpi-name">Stillbirth Rate (per 1000 births)</div>
            <div class="kpi-metrics">
                <span class="metric-label metric-stillbirth">Stillbirths: {kpis.get("stillbirths",0)}</span>
                <span class="metric-label metric-total">Total Births: {kpis.get("total_births",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("pnc_coverage",0):.1f}% <span class="{pnc_trend_class}">{pnc_trend}</span></div>
            <div class="kpi-name">Early PNC Coverage (within 48 hrs)</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-fp">PNC ≤48 hrs: {kpis.get("early_pnc",0)}</span>
               <span class="metric-label metric-total">Total Deliveries: {kpis.get("total_deliveries_pnc",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("maternal_death_rate",0):.1f} <span class="{maternal_death_trend_class}">{maternal_death_trend}</span></div>
            <div class="kpi-name">Maternal Death Rate (per 100,000 births)</div>
            <div class="kpi-metrics">
               <span class="metric-label metric-maternal-death">Maternal Deaths: {kpis.get("maternal_deaths",0)}</span>
               <span class="metric-label metric-total">Live Births: {kpis.get("live_births",0)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{kpis.get("csection_rate",0):.1f}% <span class="{csection_trend_class}">{csection_trend}</span></div>
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