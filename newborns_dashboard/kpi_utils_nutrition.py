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
BLOOD_SUGAR_UNIT_PREFIX = "what_units_is_blood_sugar_measured_in"
BLOOD_SUGAR_MMOL_PREFIX = "blood_sugar_concentration_on_admission_mmol_l_nicu_admission_careform"
BLOOD_SUGAR_MGDL_PREFIX = "blood_sugar_concentration_on_admission_mg_dl_nicu_admission_careform"

LINE_COLOR = "#1f77b4"

FEEDING_LABELS = {
    "1": "Exclusive breastmilk",
    "2": "Formula only",
    "3": "Fortified breastmilk",
    "4": "Predominant breastmilk",
    "5": "Combination of breastmilk and formula",
    "6": "Unknown - Absconder",
}

# Free-text aliases -> canonical label (case-insensitive matching)
_FEEDING_ALIASES = {
    "exclusive breastmilk": "Exclusive breastmilk",
    "exclusive breastfeeding": "Exclusive breastmilk",
    "exclusive breast feeding": "Exclusive breastmilk",
    "ebf": "Exclusive breastmilk",
    "e bf": "Exclusive breastmilk",
    "ebm": "Exclusive breastmilk",
    "breast": "Exclusive breastmilk",
    "breast feeding": "Exclusive breastmilk",
    "breastfeeding": "Exclusive breastmilk",
    "bf": "Exclusive breastmilk",
    "breastfeed": "Exclusive breastmilk",
    "breastmilk": "Exclusive breastmilk",
    "only breast feeding": "Exclusive breastmilk",
    "only ebf": "Exclusive breastmilk",
    "full exclusive bf": "Exclusive breastmilk",
    "only b.feeding": "Exclusive breastmilk",
    "until 6 months only breast feeding": "Exclusive breastmilk",
    "only breast feeding up to 6 month": "Exclusive breastmilk",
    "mothers breast": "Exclusive breastmilk",
    "mother's breast": "Exclusive breastmilk",
    "mbf": "Exclusive breastmilk",
    "formula only": "Formula only",
    "formula": "Formula only",
    "formula millk": "Formula only",
    "formula milk": "Formula only",
    "ff": "Formula only",
    "fortified breastmilk": "Fortified breastmilk",
    "fortified breastfeeding": "Fortified breastmilk",
    "fortified": "Fortified breastmilk",
    "predominant breastmilk": "Predominant breastmilk",
    "predominant breastfeeding": "Predominant breastmilk",
    "predominant": "Predominant breastmilk",
    "combination": "Combination of breastmilk and formula",
    "combination of breastmilk and formula": "Combination of breastmilk and formula",
    "mixed": "Combination of breastmilk and formula",
    "unknown": "Unknown - Absconder",
    "unknown - absconder": "Unknown - Absconder",
    "absconder": "Unknown - Absconder",
    "-----": "Unknown - Absconder",
    "n/a": "Unknown - Absconder",
}

FEEDING_COLORS = {
    "Exclusive breastmilk": "#2ca02c",
    "Formula only": "#ff7f0e",
    "Fortified breastmilk": "#1f77b4",
    "Predominant breastmilk": "#9467bd",
    "Combination of breastmilk and formula": "#8c564b",
    "Unknown - Absconder": "#7f7f7f",
}


_column_cache: dict = {}

def _any_version_value(df, col_prefix):
    """Return the first non-empty value across versioned columns matching col_prefix."""
    if col_prefix not in _column_cache:
        _column_cache[col_prefix] = [c for c in df.columns if c.startswith(col_prefix)]
    else:
        cached = _column_cache[col_prefix]
        if not all(c in df.columns for c in cached):
            _column_cache[col_prefix] = [c for c in df.columns if c.startswith(col_prefix)]
    cols = _column_cache[col_prefix]
    if not cols:
        return pd.Series("", index=df.index)
    result = pd.Series("", index=df.index)
    for col in cols:
        mask = result == ""
        col_empty = (df[col] == "") | df[col].isna()
        result = result.where(~mask | col_empty, df[col])
    return result.fillna("")


def _filter_by_facility(df, facility_uids):
    if facility_uids and "orgUnit" in df.columns:
        return df[df["orgUnit"].isin(facility_uids)]
    return df


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
    not_negative = ~feeding.astype(str).str.startswith("-")
    has_value = (feeding.ne("") & feeding.ne("N/A") & feeding.ne("nan") & feeding.notna() & not_negative)

    # Map feeding values to canonical labels (handles both codes and free text)
    unique_vals = feeding.unique()
    val_to_label = {v: _classify_feeding_value(v) for v in unique_vals}
    classified = feeding.map(val_to_label)
    is_exclusive = classified == FEEDING_LABELS["1"]

    numerator = int((has_value & is_exclusive).sum())
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

    bs_unit = pd.to_numeric(_any_version_value(working_df, BLOOD_SUGAR_UNIT_PREFIX), errors="coerce")
    bs_mmol = pd.to_numeric(_any_version_value(working_df, BLOOD_SUGAR_MMOL_PREFIX), errors="coerce")
    bs_mgdl = pd.to_numeric(_any_version_value(working_df, BLOOD_SUGAR_MGDL_PREFIX), errors="coerce")

    is_mmol = bs_unit == 1
    is_mgdl = bs_unit == 2

    has_mmol = is_mmol & bs_mmol.notna()
    has_mgdl = is_mgdl & bs_mgdl.notna()

    denominator = int(has_mmol.sum() + has_mgdl.sum())
    not_hypo = int((has_mmol & (bs_mmol > 2.5)).sum() + (has_mgdl & (bs_mgdl > 45)).sum())
    rate = (not_hypo / denominator * 100) if denominator > 0 else 0.0
    return not_hypo, denominator, rate


def _classify_feeding_value(val):
    """Classify a single feeding value into a canonical label."""
    val_str = str(val).strip()
    val_lower = val_str.lower()
    if val_str in FEEDING_LABELS:
        return FEEDING_LABELS[val_str]
    # Handle float-to-int conversion (e.g., "1.0" -> "1")
    if val_str:
        try:
            num = float(val_str)
            if num == int(num):
                int_s = str(int(num))
                if int_s in FEEDING_LABELS:
                    return FEEDING_LABELS[int_s]
        except (ValueError, TypeError):
            pass
    if val_lower in _FEEDING_ALIASES:
        return _FEEDING_ALIASES[val_lower]
    # Known missing-value markers — return as Unknown immediately
    if not val_str or val_lower in {"nan", "none", "n/a", "na", "-", "nil"}:
        return "Unknown - Absconder"
    for alias, canonical in _FEEDING_ALIASES.items():
        if alias in val_lower or val_lower in alias:
            return canonical
    return "Unknown - Absconder"


def compute_feeding_distribution_data(df, facility_uids=None):
    """Percentage distribution of feeding methods at discharge.

    Returns dict: {label: count} for each feeding category.
    Handles both coded values (1-6) and free-text entries.
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()

    feeding = _any_version_value(working_df, FEEDING_DISCHARGE_PREFIX)
    not_negative = ~feeding.astype(str).str.startswith("-")
    has_value = (feeding.ne("") & feeding.ne("N/A") & feeding.ne("nan") & feeding.notna() & not_negative)

    if not has_value.any():
        return {label: 0 for label in FEEDING_LABELS.values()}

    # Build a mapping from unique values to canonical labels (same approach
    # as compute_exclusive_breastmilk_discharge_data — uses .map() not .replace())
    unique_vals = feeding.unique()
    val_to_label = {v: _classify_feeding_value(v) for v in unique_vals}
    classified = feeding.map(val_to_label)

    # Filter and count only rows with a recorded feeding value
    valid = classified[has_value]
    counts = valid.value_counts().to_dict()

    result = {label: counts.get(label, 0) for label in FEEDING_LABELS.values()}
    return result


# ---------------------------------------------------------------------------
# Trend computation helpers (groupby-based, replaces per-period loop)
# ---------------------------------------------------------------------------

def _compute_breastmilk_trend(working_df, period_col, facility_uids):
    """Breastmilk on DOB — all periods at once via groupby."""
    if facility_uids and "orgUnit" in working_df.columns:
        pdf = working_df[working_df["orgUnit"].isin(facility_uids)]
    else:
        pdf = working_df

    bm_start = _any_version_value(pdf, BREAST_MILK_START_PREFIX)
    doDelivery = _any_version_value(pdf, DATE_OF_DELIVERY_PREFIX)
    bm_start_dt = pd.to_datetime(bm_start, errors="coerce")
    doDelivery_dt = pd.to_datetime(doDelivery, errors="coerce")

    same_day = bm_start_dt.notna().values & doDelivery_dt.notna().values & (bm_start_dt.values == doDelivery_dt.values)
    agg_df = pd.DataFrame({
        period_col: pdf[period_col].values,
        "numerator": same_day,
        "denominator": np.ones(len(pdf), dtype=int),
    })
    trend = agg_df.groupby(period_col, observed=False, sort=False).agg(
        numerator=("numerator", "sum"),
        denominator=("denominator", "sum"),
    )
    trend["rate"] = (trend["numerator"] / trend["denominator"] * 100).round(1)
    trend = trend.reset_index().rename(columns={period_col: "period"})
    return trend


def _compute_exclusive_breastmilk_trend(working_df, period_col, facility_uids):
    """Exclusive breastmilk at discharge — all periods at once via groupby."""
    if facility_uids and "orgUnit" in working_df.columns:
        pdf = working_df[working_df["orgUnit"].isin(facility_uids)]
    else:
        pdf = working_df

    feeding = _any_version_value(pdf, FEEDING_DISCHARGE_PREFIX)
    has_value = feeding.ne("") & feeding.ne("N/A") & feeding.ne("nan") & feeding.notna()

    # Map feeding values to canonical labels (handles codes "1"-"6" AND free text like "EBF", "BF")
    unique_vals = feeding.unique()
    val_to_label = {v: _classify_feeding_value(v) for v in unique_vals}
    classified = feeding.map(val_to_label)
    is_exclusive = classified == FEEDING_LABELS["1"]

    agg_df = pd.DataFrame({
        period_col: pdf[period_col].values,
        "numerator": has_value.values & is_exclusive.values,
        "denominator": has_value.values.astype(int),
    })
    trend = agg_df.groupby(period_col, observed=False, sort=False).agg(
        numerator=("numerator", "sum"),
        denominator=("denominator", "sum"),
    )
    trend = trend[trend["denominator"] > 0]
    trend["rate"] = (trend["numerator"] / trend["denominator"] * 100).round(1)
    trend = trend.reset_index().rename(columns={period_col: "period"})
    return trend


def _compute_not_hypoglycemic_trend(working_df, period_col, facility_uids):
    """Not hypoglycemic rate — all periods at once via groupby."""
    if facility_uids and "orgUnit" in working_df.columns:
        pdf = working_df[working_df["orgUnit"].isin(facility_uids)]
    else:
        pdf = working_df

    bs_unit = pd.to_numeric(_any_version_value(pdf, BLOOD_SUGAR_UNIT_PREFIX), errors="coerce")
    bs_mmol = pd.to_numeric(_any_version_value(pdf, BLOOD_SUGAR_MMOL_PREFIX), errors="coerce")
    bs_mgdl = pd.to_numeric(_any_version_value(pdf, BLOOD_SUGAR_MGDL_PREFIX), errors="coerce")

    is_mmol = bs_unit.values == 1
    is_mgdl = bs_unit.values == 2
    mmol_ok = is_mmol & bs_mmol.notna().values & (bs_mmol.values > 2.5)
    mgdl_ok = is_mgdl & bs_mgdl.notna().values & (bs_mgdl.values > 45)
    has_measure = is_mmol & bs_mmol.notna().values | is_mgdl & bs_mgdl.notna().values

    agg_df = pd.DataFrame({
        period_col: pdf[period_col].values,
        "numerator": mmol_ok | mgdl_ok,
        "denominator": has_measure,
    })
    trend = agg_df.groupby(period_col, observed=False, sort=False).agg(
        numerator=("numerator", "sum"),
        denominator=("denominator", "sum"),
    )
    trend = trend[trend["denominator"] > 0]
    trend["rate"] = (trend["numerator"] / trend["denominator"] * 100).round(1)
    trend = trend.reset_index().rename(columns={period_col: "period"})
    return trend


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_single_nutrition_chart(
    trend_df, title, target, csv_name, info_html,
    bg_color, text_color, key_suffix,
):
    """Helper: render one indicator's trend chart + table from pre-computed trend_df."""
    if trend_df.empty:
        st.caption(f"{title} — No data")
        return

    trend_df = trend_df[trend_df["denominator"] > 0]
    if trend_df.empty:
        st.info(f"No data to display for **{title}**.")
        return

    # Sort chronologically
    periods = sort_periods_chronologically(trend_df["period"].tolist())
    trend_df = trend_df.set_index("period").loc[periods].reset_index()

    overall_n = int(trend_df["numerator"].sum())
    overall_d = int(trend_df["denominator"].sum())
    overall_rate = round(overall_n / overall_d * 100, 1) if overall_d > 0 else 0

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


def _render_feeding_distribution_chart(working_df, bg_color, text_color, facility_uids):
    """Render horizontal bar chart showing feeding distribution at discharge."""
    distribution = compute_feeding_distribution_data(working_df, facility_uids)

    if not distribution or sum(distribution.values()) == 0:
        st.caption("Feeding Distribution — No data")
        return

    total = sum(distribution.values())
    # Only include categories with count > 0
    sorted_items = [(k, v) for k, v in sorted(distribution.items(), key=lambda x: -x[1]) if v > 0]

    if not sorted_items:
        st.caption("Feeding Distribution — No data")
        return

    percentages = [round(v / total * 100, 1) for _, v in sorted_items]

    fig = go.Figure(go.Bar(
        x=percentages,
        y=[item[0] for item in sorted_items],
        orientation="h",
        marker_color=[FEEDING_COLORS.get(item[0], LINE_COLOR) for item in sorted_items],
        hovertemplate="<b>%{y}</b><br>Percentage: %{x:.1f}%<br>Count: %{customdata[0]}<br>Total: %{customdata[1]}<extra></extra>",
        customdata=[[item[1], total] for item in sorted_items],
        text=[f"{p}%" for p in percentages],
        textposition="outside",
    ))

    fig.update_layout(
        title="Feeding Methods at Discharge",
        height=max(250, len(sorted_items) * 40 + 100),
        xaxis=dict(title="Percentage (%)", range=[0, 105]),
        yaxis=dict(title=""),
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
            "Numerator": count,
            "Denominator": total,
            "Rate (%)": f"{count / total * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    csv_dl = pd.DataFrame(display_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", csv_dl, "feeding_distribution.csv", "text/csv",
        key="nutr_feeding_dist_dl",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_nutrition_coverage_trend_chart(
    df, period_col="period_display", title="Nutrition Dashboard",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs,
):
    """Render the Indicator Coverage Run Charts section for Nutrition."""
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

    # Deduplicate: one row per baby
    if "tei_id" in working_df.columns:
        working_df = working_df.drop_duplicates(subset=["tei_id"], keep="first")

    col1, col2 = st.columns(2)

    with col1:
        trend_df = _compute_breastmilk_trend(working_df, period_col, facility_uids)
        _render_single_nutrition_chart(
            trend_df,
            "Breastmilk on Day of Birth", None,
            "breastmilk_on_dob.csv", None,
            bg_color, text_color, "ind1",
        )

    with col2:
        trend_df = _compute_exclusive_breastmilk_trend(working_df, period_col, facility_uids)
        _render_single_nutrition_chart(
            trend_df,
            "Exclusive Breastmilk at Discharge", None,
            "exclusive_breastmilk_discharge.csv", None,
            bg_color, text_color, "ind2",
        )

    with st.expander("How these indicators are computed"):
        st.markdown("""
        <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
        <table style="width:100%; border-collapse:collapse;">
        <tr style="background-color:#1f77b4; color:white;">
            <th style="padding:8px; text-align:left;">Indicator</th>
            <th style="padding:8px; text-align:left;">Numerator</th>
            <th style="padding:8px; text-align:left;">Denominator</th>
        </tr>
        <tr style="background-color:#f0f8ff;">
            <td style="padding:8px;"><b>Breastmilk on Day of Birth</b></td>
            <td style="padding:8px;">Babies whose breast milk start date equals date of delivery</td>
            <td style="padding:8px;">Total admitted newborns</td>
        </tr>
        <tr style="background-color:#ffffff;">
            <td style="padding:8px;"><b>Exclusive Breastmilk at Discharge</b></td>
            <td style="padding:8px;">Babies exclusively on breastmilk at discharge</td>
            <td style="padding:8px;">Admissions with a recorded type of feeding on discharge</td>
        </tr>
        </table>
        <p style="margin-top:8px;"><b>Breastmilk on Day of Birth</b> measures the proportion of admitted newborns who received breastmilk on the same day they were born. The <b>If Yes, Date of initiation of breast milk feeding</b> is compared to the <b>Date of delivery</b>.</p>
        <p style="margin-top:4px;"><b>Exclusive Breastmilk at Discharge</b> measures the proportion of discharged newborns who were exclusively on breastmilk.</p>
        </div>
        """, unsafe_allow_html=True)


def render_nutrition_qoc_trend_chart(
    df, period_col="period_display", title="Nutrition Dashboard",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs,
):
    """Render the Quality of Care section for Nutrition."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    working_df = _prepare_period_df(df, date_range_filters)
    if working_df is None or working_df.empty:
        st.warning("No data available.")
        return

    if period_col not in working_df.columns:
        st.warning("Period column not found.")
        return

    st.subheader("Nutrition - Quality of Care")

    # Deduplicate: one row per baby to avoid repeated _any_version_value calls on millions of rows
    if "tei_id" in working_df.columns:
        working_df = working_df.drop_duplicates(subset=["tei_id"], keep="first")

    col1, col2 = st.columns(2)

    with col1:
        _render_feeding_distribution_chart(working_df, bg_color, text_color, facility_uids)

    with col2:
        trend_df = _compute_not_hypoglycemic_trend(working_df, period_col, facility_uids)
        _render_single_nutrition_chart(
            trend_df,
            "Not Hypoglycemic Rate", None,
            "not_hypoglycemic_rate.csv", None,
            bg_color, text_color, "ind3",
        )

    with st.expander("How these indicators are computed"):
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
        <tr style="background-color:#ffffff;">
            <td style="padding:8px;"><b>Not Hypoglycemic Rate</b></td>
            <td style="padding:8px;">Babies with blood sugar above normal threshold</td>
            <td style="padding:8px;">Babies with documented blood sugar measurement</td>
        </tr>
        </table>
        <p style="margin-top:8px;"><b>Feeding Distribution</b> shows the percentage distribution of all feeding methods at discharge. Categories: Exclusive breastmilk, Formula only, Fortified breastmilk, Predominant breastmilk, Combination of breastmilk and formula, Unknown - Absconder.</p>
        <p style="margin-top:4px;"><b>Not Hypoglycemic Rate</b> measures the proportion of newborns with documented blood sugar who are NOT hypoglycemic.</p>
        </div>
        """, unsafe_allow_html=True)


def render_nutrition_coverage_comparison_chart(
    df, comparison_mode="facility", display_names=None, facility_uids=None,
    facilities_by_region=None, region_names=None, period_col="period_display",
    title="Nutrition Coverage Comparison", date_range_filters=None,
    bg_color="#FFFFFF", text_color=None, **kwargs
):
    """Comparison chart for Nutrition Coverage — multi-entity line charts."""
    if text_color is None:
        text_color = auto_text_color(bg_color)
    if df is None or df.empty:
        st.subheader(title); st.warning("No data."); return
    if period_col not in df.columns:
        st.warning("Period column not found."); return

    if comparison_mode == "facility":
        if not facility_uids or not display_names:
            st.warning("No facilities selected."); return
        entities = dict(zip(facility_uids, display_names))
    else:
        if not region_names:
            st.warning("No regions selected."); return
        entities = {r: r for r in region_names}

    periods = sort_periods_chronologically(df[period_col].unique())

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    sorted_entities = sorted(entities.keys(), key=lambda e: str(entities[e]).lower())
    entity_colors = {e: color_palette[i % len(color_palette)] for i, e in enumerate(sorted_entities)}

    def _filter_entity_df(base_df, eid):
        if comparison_mode == "facility":
            return base_df[base_df["orgUnit"] == eid]
        else:
            if not facilities_by_region or eid not in facilities_by_region:
                return None
            facs = [f[1] if isinstance(f, (list, tuple)) and len(f) > 1 else f for f in facilities_by_region[eid]]
            return base_df[base_df["orgUnit"].isin(facs)]

    comp_data = {}
    for eid, ename in entities.items():
        edf = _filter_entity_df(df, eid)
        if edf is None or edf.empty:
            continue
        working = edf.drop_duplicates(subset=["tei_id"], keep="first") if "tei_id" in edf.columns else edf
        bm_trend = _compute_breastmilk_trend(working, period_col, None)
        ebm_trend = _compute_exclusive_breastmilk_trend(working, period_col, None)
        comp_data[eid] = {"bm": bm_trend, "ebm": ebm_trend, "name": ename}

    if not comp_data:
        st.warning("No comparison data."); return

    st.subheader(title)
    tab1, tab2 = st.tabs(["Breastmilk on Day of Birth", "Exclusive Breastmilk at Discharge"])

    for tab, key, ylabel in [
        (tab1, "bm", "Rate (%)"),
        (tab2, "ebm", "Rate (%)"),
    ]:
        with tab:
            fig = go.Figure()
            for eid in sorted_entities:
                if eid not in comp_data:
                    continue
                trend = comp_data[eid][key]
                fig.add_trace(go.Scatter(
                    x=trend["period"], y=trend["rate"],
                    name=comp_data[eid]["name"], mode="lines+markers",
                    line=dict(color=entity_colors[eid], width=2),
                    hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
                    customdata=np.column_stack((trend["numerator"].values, trend["denominator"].values)),
                ))
            fig.update_layout(
                height=400,
                yaxis=dict(title=ylabel),
                xaxis=dict(title=""),
                paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                font_color=text_color, title_font_color=text_color,
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig, use_container_width=True, key=f"nut_{key}_comp")


def render_nutrition_qoc_comparison_chart(
    df, comparison_mode="facility", display_names=None, facility_uids=None,
    facilities_by_region=None, region_names=None, period_col="period_display",
    title="Nutrition QoC Comparison", date_range_filters=None,
    bg_color="#FFFFFF", text_color=None, **kwargs
):
    """Comparison chart for Nutrition QoC — multi-entity line chart for Not Hypoglycemic Rate."""
    if text_color is None:
        text_color = auto_text_color(bg_color)
    if df is None or df.empty:
        st.subheader(title); st.warning("No data."); return
    if period_col not in df.columns:
        st.warning("Period column not found."); return

    if comparison_mode == "facility":
        if not facility_uids or not display_names:
            st.warning("No facilities selected."); return
        entities = dict(zip(facility_uids, display_names))
    else:
        if not region_names:
            st.warning("No regions selected."); return
        entities = {r: r for r in region_names}

    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    sorted_entities = sorted(entities.keys(), key=lambda e: str(entities[e]).lower())
    entity_colors = {e: color_palette[i % len(color_palette)] for i, e in enumerate(sorted_entities)}

    def _filter_entity_df(base_df, eid):
        if comparison_mode == "facility":
            return base_df[base_df["orgUnit"] == eid]
        else:
            if not facilities_by_region or eid not in facilities_by_region:
                return None
            facs = [f[1] if isinstance(f, (list, tuple)) and len(f) > 1 else f for f in facilities_by_region[eid]]
            return base_df[base_df["orgUnit"].isin(facs)]

    comp_data = {}
    for eid, ename in entities.items():
        edf = _filter_entity_df(df, eid)
        if edf is None or edf.empty:
            continue
        working = edf.drop_duplicates(subset=["tei_id"], keep="first") if "tei_id" in edf.columns else edf
        nh_trend = _compute_not_hypoglycemic_trend(working, period_col, None)
        comp_data[eid] = {"nh": nh_trend, "name": ename}

    if not comp_data:
        st.warning("No comparison data."); return

    st.subheader(title)

    fig = go.Figure()
    for eid in sorted_entities:
        if eid not in comp_data:
            continue
        trend = comp_data[eid]["nh"]
        fig.add_trace(go.Scatter(
            x=trend["period"], y=trend["rate"],
            name=comp_data[eid]["name"], mode="lines+markers",
            line=dict(color=entity_colors[eid], width=2),
            hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack((trend["numerator"].values, trend["denominator"].values)),
        ))
    fig.update_layout(
        title="Not Hypoglycemic Rate", height=400,
        yaxis=dict(title="Rate (%)"),
        xaxis=dict(title=""),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, key="nut_nh_comp")


__all__ = [
    "compute_breastmilk_on_dob_data",
    "compute_exclusive_breastmilk_discharge_data",
    "compute_not_hypoglycemic_rate_data",
    "compute_feeding_distribution_data",
    "render_nutrition_coverage_trend_chart",
    "render_nutrition_qoc_trend_chart",
    "render_nutrition_coverage_comparison_chart",
    "render_nutrition_qoc_comparison_chart",
]
