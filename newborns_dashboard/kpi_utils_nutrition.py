# kpi_utils_nutrition.py - Nutrition Dashboard Implementation

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import logging

from utils.kpi_utils import auto_text_color
from utils.time_filter import assign_period
from newborns_dashboard.kpi_utils_newborn_simplified import sort_periods_chronologically

logger = logging.getLogger(__name__)

# --- Column name constants (merged newborn+NID dataset) ---
BREAST_MILK_START_PREFIX = "if_yes_date_of_initiation_of_breast_milk_feeding_kmc_ward_follow_up_sheet"
DATE_OF_DELIVERY_PREFIX = "date_of_delivery_nicu_admission_careform"
FEEDING_DISCHARGE_PREFIX = "type_of_feeding_on_discharge_discharge_care_form"
BLOOD_SUGAR_UNIT_PREFIX = "what_units_is_blood_sugar_measured_in_nicu_admission_careform"
BLOOD_SUGAR_MMOL_PREFIX = "blood_sugar_concentration_on_admission_mmol_l_nicu_admission_careform"
BLOOD_SUGAR_MGDL_PREFIX = "blood_sugar_concentration_on_admission_mg_dl_nicu_admission_careform"

LINE_COLOR = "#1f77b4"

FEEDING_LABELS = {
    "1": "Exclusive breastmilk",
    "2": "Formula only",
    "3": "Fortified breastmilk",
    "4": "Predominant breastmilk",
    "5": "Combination of breastmilk and formula",
    "6": "Unknown / Absconder",
}

FEEDING_COLORS = {
    "Exclusive breastmilk": "#2ca02c",
    "Formula only": "#ff7f0e",
    "Fortified breastmilk": "#1f77b4",
    "Predominant breastmilk": "#9467bd",
    "Combination of breastmilk and formula": "#8c564b",
    "Unknown / Absconder": "#7f7f7f",
}


def _any_version_value(df, col_prefix):
    """Return the first non-empty value across versioned columns matching col_prefix."""
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    if not cols:
        return pd.Series("", index=df.index)
    result = pd.Series("", index=df.index)
    for col in cols:
        mask = result == ""
        result = result.where(~mask | (df[col] == ""), df[col])
    return result


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
            working_df = working_df[
                (working_df["event_date"] >= sd_dt) & (working_df["event_date"] < ed_dt)
            ]
    working_df = working_df[working_df["event_date"].notna()].copy()
    if working_df.empty:
        return None
    period_label = st.session_state.get("period_label", "Monthly")
    try:
        working_df = assign_period(working_df, "event_date", period_label)
    except Exception:
        return None
    return working_df


def _widget_key(base_key, key_suffix=None):
    suffix = str(key_suffix).strip() if key_suffix is not None else ""
    return f"{base_key}_{suffix}" if suffix else base_key


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------

def compute_breastmilk_on_dob_data(df, facility_uids=None):
    """Percentage of babies who got breastmilk on their day of birth.

    Numerator: babies where breast milk start date == date of delivery
    Denominator: total admitted newborns
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()

    bm_start = _any_version_value(working_df, BREAST_MILK_START_PREFIX)
    doDelivery = _any_version_value(working_df, DATE_OF_DELIVERY_PREFIX)

    bm_start_dt = pd.to_datetime(bm_start, errors="coerce")
    doDelivery_dt = pd.to_datetime(doDelivery, errors="coerce")

    has_bm = bm_start_dt.notna()
    has_do = doDelivery_dt.notna()
    both = has_bm & has_do

    numerator = int((both & (bm_start_dt == doDelivery_dt)).sum())
    denominator = len(working_df)
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_exclusive_breastmilk_discharge_data(df, facility_uids=None):
    """Percentage of babies exclusively on breastmilk at discharge.

    Numerator: feeding_on_discharge == 1 (Exclusive breastmilk)
    Denominator: babies with a recorded feeding_on_discharge value (non-empty, not N/A)
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()

    feeding = _any_version_value(working_df, FEEDING_DISCHARGE_PREFIX)
    has_value = feeding.ne("") & feeding.ne("N/A") & feeding.ne("nan") & feeding.notna()

    numerator = int((has_value & (feeding == "1")).sum())
    denominator = int(has_value.sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_not_hypoglycemic_rate_data(df, facility_uids=None):
    """Not Hypoglycemic Rate Among Babies with Documented Blood Sugar.

    Checks blood sugar unit first, then evaluates the value using the correct threshold.
    Unit 1 (mmol/L): threshold > 2.5
    Unit 2 (mg/dL): threshold > 45
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()

    bs_unit = _any_version_value(working_df, BLOOD_SUGAR_UNIT_PREFIX)
    bs_mmol = pd.to_numeric(_any_version_value(working_df, BLOOD_SUGAR_MMOL_PREFIX), errors="coerce")
    bs_mgdl = pd.to_numeric(_any_version_value(working_df, BLOOD_SUGAR_MGDL_PREFIX), errors="coerce")

    is_mmol = bs_unit == "1"
    is_mgdl = bs_unit == "2"

    has_mmol = is_mmol & bs_mmol.notna()
    has_mgdl = is_mgdl & bs_mgdl.notna()

    denominator = int(has_mmol.sum() + has_mgdl.sum())
    not_hypo = int((has_mmol & (bs_mmol > 2.5)).sum() + (has_mgdl & (bs_mgdl > 45)).sum())
    rate = (not_hypo / denominator * 100) if denominator > 0 else 0.0
    return not_hypo, denominator, rate


def compute_feeding_distribution_data(df, facility_uids=None):
    """Percentage distribution of feeding methods at discharge.

    Returns dict: {label: count} for each feeding category.
    Only includes babies with a recorded feeding_on_discharge value.
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()

    feeding = _any_version_value(working_df, FEEDING_DISCHARGE_PREFIX)
    has_value = feeding.ne("") & feeding.ne("N/A") & feeding.ne("nan") & feeding.notna()

    filtered = feeding[has_value]
    counts = filtered.value_counts().to_dict()

    result = {}
    for code, label in FEEDING_LABELS.items():
        result[label] = int(counts.get(code, 0))
    return result


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_single_nutrition_chart(
    working_df, period_col, compute_fn, title, target, csv_name, info_html,
    bg_color, text_color, facility_uids, key_suffix,
):
    """Helper: render one indicator's trend chart + table."""
    periods = sort_periods_chronologically(working_df[period_col].unique())
    trend_data = []

    for period in periods:
        pdf = working_df[working_df[period_col] == period]
        n, d, r = compute_fn(pdf, facility_uids)
        trend_data.append({"period": period, "numerator": n, "denominator": d, "rate": r})

    trend_df = pd.DataFrame(trend_data)
    if trend_df.empty:
        st.caption(f"{title} — No data")
        return

    trend_df = trend_df[trend_df["denominator"] > 0].copy()
    if trend_df.empty:
        st.info(f"No data to display for **{title}**.")
        return

    overall_n = int(trend_df["numerator"].sum())
    overall_d = int(trend_df["denominator"].sum())
    overall_rate = (overall_n / overall_d * 100) if overall_d > 0 else 0

    fig = go.Figure()
    if target is not None:
        fig.add_hline(
            y=target, line_dash="dash", line_color="green", opacity=0.6,
            annotation_text=f"Target: {target}%", annotation_position="top left",
        )
    fig.add_trace(go.Scatter(
        x=trend_df["period"], y=trend_df["rate"], mode="lines+markers",
        name="Rate %", line=dict(color=LINE_COLOR, width=3), marker=dict(size=8),
        hovertemplate="Month: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack((trend_df["numerator"], trend_df["denominator"])),
    ))
    fig.update_layout(
        title=title, height=350,
        yaxis=dict(range=[0, 105], dtick=25, title="Coverage (%)"),
        xaxis=dict(title=""),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

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
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    csv_dl = pd.DataFrame(display_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", csv_dl, csv_name, "text/csv",
        key=_widget_key(f"nutr_{key_suffix}_dl", None),
    )

    if info_html:
        with st.expander("How this indicator is computed"):
            st.markdown(info_html, unsafe_allow_html=True)


def _render_feeding_distribution_chart(working_df, period_col, bg_color, text_color, facility_uids):
    """Render horizontal bar chart showing feeding distribution at discharge."""
    distribution = compute_feeding_distribution_data(working_df, facility_uids)

    if not distribution or sum(distribution.values()) == 0:
        st.caption("Feeding Distribution — No data")
        return

    total = sum(distribution.values())
    sorted_items = sorted(distribution.items(), key=lambda x: -x[1])

    fig = go.Figure(go.Bar(
        x=[item[1] for item in sorted_items],
        y=[item[0] for item in sorted_items],
        orientation="h",
        marker_color=[FEEDING_COLORS.get(item[0], LINE_COLOR) for item in sorted_items],
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Total: %{customdata[0]}<br>Percentage: %{customdata[1]}%<extra></extra>",
        customdata=[[total, round((item[1] / total * 100), 1)] for item in sorted_items],
    ))

    fig.update_layout(
        title="Percentage Distribution of Feeding Methods at Discharge",
        height=400,
        xaxis=dict(title="Count", dtick=1),
        yaxis=dict(title="Feeding Method"),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="nutr_feeding_dist_chart")

    display_rows = []
    for label, count in sorted_items:
        display_rows.append({
            "Feeding Method": label,
            "Count": count,
            "Percentage": f"{count / total * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_nutrition_trend_chart(
    df, period_col="period_display", title="Nutrition Dashboard",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs,
):
    """Main function to render the complete Nutrition dashboard."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    working_df = _prepare_period_df(df, date_range_filters)
    if working_df is None or working_df.empty:
        st.warning("No data available.")
        return

    if period_col not in working_df.columns:
        st.warning("Period column not found.")
        return

    st.subheader("Nutrition - Indicator Coverage Run Charts")

    tab_coverage, tab_qoc = st.tabs(["Indicator Coverage Run Charts", "Quality of Care"])

    with tab_coverage:
        col1, col2 = st.columns(2)

        with col1:
            _render_single_nutrition_chart(
                working_df, period_col, compute_breastmilk_on_dob_data,
                "Breastmilk on Day of Birth", None,
                "breastmilk_on_dob.csv",
                '<div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">'
                '<table style="width:100%; border-collapse:collapse;">'
                '<tr style="background-color:#1f77b4; color:white;">'
                '<th style="padding:8px; text-align:left;">Indicator</th>'
                '<th style="padding:8px; text-align:left;">Numerator</th>'
                '<th style="padding:8px; text-align:left;">Denominator</th>'
                '</tr>'
                '<tr style="background-color:#f0f8ff;">'
                '<td style="padding:8px;"><b>Breastmilk on Day of Birth</b></td>'
                '<td style="padding:8px;">Babies whose breast milk start date equals date of delivery</td>'
                '<td style="padding:8px;">Total admitted newborns</td>'
                '</tr>'
                '</table>'
                '<p style="margin-top:8px;">Measures the proportion of admitted newborns who received breastmilk on the same day they were born. The <b>If Yes, Date of initiation of breast milk feeding</b> is compared to the <b>Date of delivery</b>.</p>'
                '</div>',
                bg_color, text_color, facility_uids, "ind1",
            )

        with col2:
            _render_single_nutrition_chart(
                working_df, period_col, compute_exclusive_breastmilk_discharge_data,
                "Exclusive Breastmilk at Discharge", None,
                "exclusive_breastmilk_discharge.csv",
                '<div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">'
                '<table style="width:100%; border-collapse:collapse;">'
                '<tr style="background-color:#1f77b4; color:white;">'
                '<th style="padding:8px; text-align:left;">Indicator</th>'
                '<th style="padding:8px; text-align:left;">Numerator</th>'
                '<th style="padding:8px; text-align:left;">Denominator</th>'
                '</tr>'
                '<tr style="background-color:#f0f8ff;">'
                '<td style="padding:8px;"><b>Exclusive Breastmilk at Discharge</b></td>'
                '<td style="padding:8px;">Babies with feeding on discharge = Exclusive breastmilk (code 1)</td>'
                '<td style="padding:8px;">Admissions with a recorded type of feeding on discharge</td>'
                '</tr>'
                '</table>'
                '<p style="margin-top:8px;">Measures the proportion of discharged newborns who were exclusively on breastmilk at the time of discharge. Uses the <b>Type of feeding on discharge</b> field where code 1 = Exclusive breastmilk.</p>'
                '</div>',
                bg_color, text_color, facility_uids, "ind2",
            )

    with tab_qoc:
        col3, col4 = st.columns(2)

        with col3:
            _render_feeding_distribution_chart(working_df, period_col, bg_color, text_color, facility_uids)

        with col4:
            _render_single_nutrition_chart(
                working_df, period_col, compute_not_hypoglycemic_rate_data,
                "Not Hypoglycemic Rate", None,
                "not_hypoglycemic_rate.csv",
                '<div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">'
                '<table style="width:100%; border-collapse:collapse;">'
                '<tr style="background-color:#1f77b4; color:white;">'
                '<th style="padding:8px; text-align:left;">Indicator</th>'
                '<th style="padding:8px; text-align:left;">Numerator</th>'
                '<th style="padding:8px; text-align:left;">Denominator</th>'
                '</tr>'
                '<tr style="background-color:#f0f8ff;">'
                '<td style="padding:8px;"><b>Not Hypoglycemic Rate</b></td>'
                '<td style="padding:8px;">Babies with blood sugar > 2.5 mmol/L (unit=1) OR > 45 mg/dL (unit=2)</td>'
                '<td style="padding:8px;">Babies with documented blood sugar measurement</td>'
                '</tr>'
                '</table>'
                '<p style="margin-top:8px;">Measures the proportion of newborns with documented blood sugar who are NOT hypoglycemic. The indicator first checks the <b>blood sugar unit</b> field, then evaluates the blood sugar value using the correct threshold: <b>&gt; 2.5 mmol/L</b> for mmol/L or <b>&gt; 45 mg/dL</b> for mg/dL.</p>'
                '</div>',
                bg_color, text_color, facility_uids, "ind3",
            )

        with st.expander("How feeding distribution is computed"):
            st.markdown("""
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Feeding Distribution</b></td>
                <td style="padding:8px;">Number of babies in each feeding category</td>
                <td style="padding:8px;">Admissions with a recorded type of feeding on discharge</td>
            </tr>
            </table>
            <p style="margin-top:8px;">Shows the percentage distribution of all feeding methods at discharge. Categories: <b>Exclusive breastmilk</b> (1), <b>Formula only</b> (2), <b>Fortified breastmilk</b> (3), <b>Predominant breastmilk</b> (4), <b>Combination of breastmilk and formula</b> (5), <b>Unknown / Absconder</b> (6).</p>
            </div>
            """, unsafe_allow_html=True)


__all__ = [
    "compute_breastmilk_on_dob_data",
    "compute_exclusive_breastmilk_discharge_data",
    "compute_not_hypoglycemic_rate_data",
    "compute_feeding_distribution_data",
    "render_nutrition_trend_chart",
]
