# kpi_utils_infection.py - Antibiotics for Clinical Sepsis indicator

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import logging

from utils.kpi_utils import auto_text_color
from utils.time_filter import assign_period
from newborns_dashboard.kpi_utils_newborn_simplified import sort_periods_chronologically

logger = logging.getLogger(__name__)

# Column name constants (merged newborn+NID dataset)
INFECTION_SUB_CATEGORY_COL = "sub_categories_of_infection_n_discharge_care_form"
REASON1_COL = "first_reason_for_admission_admission_information"
REASON2_COL = "second_reason_for_admission_admission_information"
REASON3_COL = "third_reason_for_admission_admission_information"
ANTIBIOTIC_COL = "are_antibiotics_administered?_medication_sheet"

LINE_COLOR = "#1f77b4"


def _filter_by_facility(df, facility_uids):
    if facility_uids and "orgUnit" in df.columns:
        return df[df["orgUnit"].isin(facility_uids)].copy()
    return df.copy()


def _prepare_period_df(df, date_range_filters):
    working_df = df.copy()
    date_col = "enrollment_date"
    if date_col not in working_df.columns:
        return None
    working_df["event_date"] = pd.to_datetime(working_df[date_col], errors="coerce")
    if date_range_filters:
        sd = date_range_filters.get("start_date")
        ed = date_range_filters.get("end_date")
        if sd and ed:
            sd_dt = pd.Timestamp(sd)
            ed_dt = pd.Timestamp(ed) + pd.Timedelta(days=1)
            working_df = working_df[(working_df["event_date"] >= sd_dt) & (working_df["event_date"] < ed_dt)]
    working_df = working_df[working_df["event_date"].notna()].copy()
    if working_df.empty:
        return None
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "event_date", period_label)
    except Exception:
        return None
    return working_df


def _is_clinical_sepsis(row):
    for col in [REASON1_COL, REASON2_COL, REASON3_COL]:
        val = pd.to_numeric(row.get(col, pd.NA), errors="coerce")
        if val == 19:
            return True
    inf_val = pd.to_numeric(row.get(INFECTION_SUB_CATEGORY_COL, pd.NA), errors="coerce")
    if inf_val in (1, 2):
        return True
    return False


def compute_clinical_sepsis_data(df, facility_uids=None):
    working_df = _filter_by_facility(df, facility_uids)
    working_df[INFECTION_SUB_CATEGORY_COL] = pd.to_numeric(
        working_df.get(INFECTION_SUB_CATEGORY_COL, pd.Series(dtype=str)), errors="coerce"
    )
    working_df[ANTIBIOTIC_COL] = pd.to_numeric(
        working_df.get(ANTIBIOTIC_COL, pd.Series(dtype=str)), errors="coerce"
    )
    for col in [REASON1_COL, REASON2_COL, REASON3_COL]:
        if col not in working_df.columns:
            working_df[col] = pd.NA
        else:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    is_sepsis = working_df.apply(_is_clinical_sepsis, axis=1)
    denominator = int(is_sepsis.sum())
    numerator = int((is_sepsis & (working_df[ANTIBIOTIC_COL] == 1)).sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def _widget_key(base_key, key_suffix=None):
    suffix = str(key_suffix).strip() if key_suffix is not None else ""
    return f"{base_key}_{suffix}" if suffix else base_key


def render_infection_coverage_trend_chart(
    df, period_col="period_display", title="Antibiotics for Clinical Sepsis",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs
):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    working_df = _prepare_period_df(df, date_range_filters)
    if working_df is None or working_df.empty:
        st.subheader(title); st.warning("No data available."); return

    if period_col not in working_df.columns:
        st.subheader(title); st.warning("Period column not found."); return

    periods = sort_periods_chronologically(working_df[period_col].unique())
    trend_data = []
    for period in periods:
        pdf = working_df[working_df[period_col] == period]
        n, d, r = compute_clinical_sepsis_data(pdf, facility_uids)
        trend_data.append({"period": period, "numerator": n, "denominator": d, "rate": r})
    trend_df = pd.DataFrame(trend_data)
    if trend_df.empty:
        st.warning("No data."); return

    st.subheader(f"{title} — Monthly")
    fig = go.Figure()
    fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.6,
                  annotation_text="Target: 100%", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=trend_df["period"], y=trend_df["rate"], mode="lines+markers",
        name="Coverage %", line=dict(color=LINE_COLOR, width=3), marker=dict(size=8),
        hovertemplate="Month: %{x}<br>Coverage: %{y:.1f}%<br>Numerator: %{customdata[0]}<br>Denominator: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack((trend_df["numerator"], trend_df["denominator"])),
    ))
    fig.update_layout(
        title=title, height=400,
        yaxis=dict(range=[0, 105], dtick=25, title="Coverage (%)"),
        xaxis=dict(title=""), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color, margin=dict(l=50, r=30, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    overall_n = int(trend_df["numerator"].sum())
    overall_d = int(trend_df["denominator"].sum())
    overall_rate = (overall_n / overall_d * 100) if overall_d > 0 else 0

    display_rows = []
    for _, row in trend_df.iterrows():
        display_rows.append({
            "Period": row["period"],
            "Numerator": int(row["numerator"]),
            "Denominator": int(row["denominator"]),
            "Rate (%)": f"{row['rate']:.1f}%",
        })
    display_rows.append({
        "Period": "Overall",
        "Numerator": overall_n,
        "Denominator": overall_d,
        "Rate (%)": f"{overall_rate:.1f}%",
    })
    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "antibiotics_clinical_sepsis.csv", "text/csv",
                       key=_widget_key("infection_cov_dl", kwargs.get("key_suffix")))

    with st.expander("ℹ️ How this indicator is computed"):
        st.markdown(
            """
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
                <th style="padding:8px; text-align:left;">Target</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Antibiotics for Clinical Sepsis</b></td>
                <td style="padding:8px;">Clinical sepsis babies who received antibiotics</td>
                <td style="padding:8px;">Babies with clinical sepsis — <b>Probable Sepsis</b>, <b>Culture-Positive Sepsis</b>, <b>OR</b> any admission reason is <b>Suspected Infection - Sepsis/Meningitis (Neonatal working diagnosis)</b></td>
                <td style="padding:8px;">100%</td>
            </tr>
            </table>
            <p style="margin-top:8px;"><b>Clinical Sepsis Definition:</b> Infection discharge diagnosis = <b>Probable Sepsis</b> or <b>Culture-Positive Sepsis</b>, <b>OR</b> any admission reason = <b>Suspected Infection - Sepsis/Meningitis (Neonatal working diagnosis)</b>.</p>
            </div>
            """, unsafe_allow_html=True,
        )


def render_infection_facility_comparison(
    df, comparison_mode="facility", display_names=None, facility_uids=None,
    facilities_by_region=None, region_names=None, period_col="period_display",
    title="Antibiotics for Clinical Sepsis Comparison", bg_color="#FFFFFF", text_color=None, **kwargs
):
    if text_color is None:
        text_color = auto_text_color(bg_color)
    if df is None or df.empty:
        st.subheader(title); st.warning("No data available."); return

    if comparison_mode == "facility":
        if not facility_uids or not display_names:
            st.warning("No facilities selected."); return
        entities = dict(zip(facility_uids, display_names))
    else:
        if not region_names:
            st.warning("No regions selected."); return
        entities = {r: r for r in region_names}

    comparison_data = []
    for eid, ename in entities.items():
        if comparison_mode == "facility":
            edf = df[df["orgUnit"] == eid]
        else:
            if not facilities_by_region or eid not in facilities_by_region:
                continue
            facs = [f[1] if isinstance(f, (list, tuple)) and len(f) > 1 else f for f in facilities_by_region[eid]]
            edf = df[df["orgUnit"].isin(facs)]
        if edf.empty:
            continue
        n, d, r = compute_clinical_sepsis_data(edf)
        comparison_data.append({
            "Entity": ename,
            "Coverage": f"{r:.1f}%" if d > 0 else "-",
            "Numerator/Denominator": f"{n}/{d}" if d > 0 else "-",
        })
    if not comparison_data:
        st.warning("No comparison data."); return
    comp_df = pd.DataFrame(comparison_data)
    st.subheader(title)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    csv = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "infection_comparison.csv", "text/csv")


__all__ = [
    "compute_clinical_sepsis_data",
    "render_infection_coverage_trend_chart",
    "render_infection_facility_comparison",
]
