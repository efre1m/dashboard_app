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
ANTIBIOTIC_PREFIX = "are_antibiotics_administered?_medication_sheet"
CSF_CULTURE_PREFIX = "csf_culture_for_suspected_meningitis_investigation_sheet"
BLOOD_CULTURE_PREFIX = "blood_culture_for_suspected_sepsis_investigation_sheet"
ANTIBIOTIC_MTEXT_PREFIX = "if_yes_select_antibiotics_medication_sheet"

LINE_COLOR = "#1f77b4"

# AWaRe classification
AWARE_ACCESS = {1, 2, 3, 7, 9, 10, 11, 12, 14, 17, 19}
AWARE_WATCH = {4, 5, 8, 13, 15, 16, 18, 20, 21}
AWARE_UNKNOWN = {6, 99}

AWARE_LABELS = {
    "Access": AWARE_ACCESS,
    "Watch": AWARE_WATCH,
    "Reserve": set(),
    "Unknown": AWARE_UNKNOWN,
}

AWARE_COLORS = {"Access": "#2ca02c", "Watch": "#ff7f0e", "Reserve": "#9467bd", "Unknown": "#999999"}

ANTIBIOTIC_NAMES = {
    1: "Benzathine Penicillin", 2: "Gentamicin", 3: "Amikacin",
    4: "Ceftriaxone", 5: "Ceftazidime", 6: "Piperazine",
    7: "Metronidazole", 8: "Vancomycin", 9: "Clindamycin",
    10: "Ampicillin", 11: "Amoxicillin", 12: "Flucloxacillin",
    13: "Meropenem", 14: "Crystalline Penicillin", 15: "Cefotaxime",
    16: "Levofloxacin", 17: "Cloxacillin", 18: "Ciprofloxacin",
    19: "Cefalexin", 20: "Cefixime", 21: "Tazobactam",
    99: "Other",
}


def _any_version_mask(df, col_prefix, *target_values):
    """Boolean mask: True where ANY column matching col_prefix equals any target_value."""
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    if not cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in cols:
        num = pd.to_numeric(df.get(col, pd.Series(dtype=str)), errors="coerce")
        for tv in target_values:
            mask = mask | (num == tv)
    return mask


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
    antibiotics_mask = _any_version_mask(working_df, ANTIBIOTIC_PREFIX, 1)
    for col in [REASON1_COL, REASON2_COL, REASON3_COL]:
        if col not in working_df.columns:
            working_df[col] = pd.NA
        else:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    is_sepsis = working_df.apply(_is_clinical_sepsis, axis=1)
    denominator = int(is_sepsis.sum())
    numerator = int((is_sepsis & antibiotics_mask).sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_culture_done_for_antibiotics_data(df, facility_uids=None):
    """Culture done for babies on antibiotics.

    Numerator: Babies who received antibiotics AND (blood culture or CSF culture done)
    Denominator: Babies who received antibiotics
    """
    working_df = _filter_by_facility(df, facility_uids)
    abx_mask = _any_version_mask(working_df, ANTIBIOTIC_PREFIX, 1)
    bld_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 1, 2, 3)
    csf_mask = _any_version_mask(working_df, CSF_CULTURE_PREFIX, 1, 2, 3)
    culture_mask = bld_mask | csf_mask

    denominator = int(abx_mask.sum())
    numerator = int((abx_mask & culture_mask).sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_blood_culture_results_recorded_data(df, facility_uids=None):
    """Blood culture results recorded.

    Numerator: Babies with blood culture done AND result positive or negative (1 or 2)
    Denominator: Babies with blood culture done (1, 2, or 3)
    """
    working_df = _filter_by_facility(df, facility_uids)
    bld_done_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 1, 2, 3)
    bld_recorded_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 1, 2)

    denominator = int(bld_done_mask.sum())
    numerator = int((bld_done_mask & bld_recorded_mask).sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_facilities_doing_blood_culture_data(df, facility_uids=None):
    """Facilities doing blood culture.

    Counts unique facilities per period. A facility counts as "did blood culture"
    if at least one baby under its care has a blood culture result recorded.

    Numerator: Number of unique facilities that did at least 1 blood culture test
    Denominator: Total number of unique facilities that reported data
    """
    working_df = _filter_by_facility(df, facility_uids)
    if "orgUnit" not in working_df.columns:
        return 0, 0, 0.0
    bld_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 1, 2, 3)
    working_df = working_df.copy()
    working_df["_bld_done"] = bld_mask
    fac_stats = working_df.groupby("orgUnit")["_bld_done"].max()
    denominator = int(len(fac_stats))
    numerator = int(fac_stats.sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def _widget_key(base_key, key_suffix=None):
    suffix = str(key_suffix).strip() if key_suffix is not None else ""
    return f"{base_key}_{suffix}" if suffix else base_key


def _render_single_infection_chart(
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
    has_any_denom = trend_df["denominator"].sum() > 0
    if not has_any_denom:
        st.caption(f"{title} — No blood culture tests done in this period")
        return

    fig = go.Figure()
    fig.add_hline(y=target, line_dash="dash", line_color="green", opacity=0.6,
                  annotation_text=f"Target: {target}%", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=trend_df["period"], y=trend_df["rate"], mode="lines+markers",
        name="Coverage %", line=dict(color=LINE_COLOR, width=3), marker=dict(size=8),
        hovertemplate="Month: %{x}<br>Coverage: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack((trend_df["numerator"], trend_df["denominator"])),
    ))
    fig.update_layout(
        title=title, height=350,
        yaxis=dict(range=[0, 105], dtick=25, title="Coverage (%)"),
        xaxis=dict(title=""), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=30),
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
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    csv_dl = pd.DataFrame(display_rows).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_dl, csv_name, "text/csv",
                       key=_widget_key(f"inf_{key_suffix}_dl", None))


def _render_facilities_blood_culture_stacked_bar(
    working_df, period_col, facility_uids,
    bg_color, text_color,
):
    """Stacked bar chart: facilities that did / did not do blood culture per period."""
    periods = sort_periods_chronologically(working_df[period_col].unique())
    rows = []
    for period in periods:
        pdf = working_df[working_df[period_col] == period]
        n, d, _ = compute_facilities_doing_blood_culture_data(pdf, facility_uids)
        did = n
        did_not = d - n
        rows.append({"period": period, "Did blood culture": did, "Did not do blood culture": did_not})
    if not rows:
        st.caption("Facilities Doing Blood Culture — No data")
        return

    chart_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["period"], y=chart_df["Did blood culture"],
        name="Did blood culture",
        marker_color="#2ca02c",  # green
        hovertemplate="Month: %{x}<br>Facilities: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=chart_df["period"], y=chart_df["Did not do blood culture"],
        name="Did not do blood culture",
        marker_color="#d62728",  # red
        hovertemplate="Month: %{x}<br>Facilities: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Facilities Doing Blood Culture",
        barmode="stack", height=350,
        yaxis=dict(tickformat="d", title="Number of facilities"),
        xaxis=dict(title=""),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    overall_did = int(chart_df["Did blood culture"].sum())
    overall_did_not = int(chart_df["Did not do blood culture"].sum())
    overall_rate = (overall_did / (overall_did + overall_did_not) * 100) if (overall_did + overall_did_not) > 0 else 0

    display_rows = []
    for _, row in chart_df.iterrows():
        total = row["Did blood culture"] + row["Did not do blood culture"]
        pct = (row["Did blood culture"] / total * 100) if total > 0 else 0
        display_rows.append({
            "Period": row["period"],
            "Did blood culture": int(row["Did blood culture"]),
            "Did not do blood culture": int(row["Did not do blood culture"]),
            "Total facilities": int(total),
            "Facilities doing culture (%)": f"{pct:.1f}%",
        })
    display_rows.append({
        "Period": "Overall",
        "Did blood culture": overall_did,
        "Did not do blood culture": overall_did_not,
        "Total facilities": overall_did + overall_did_not,
        "Facilities doing culture (%)": f"{overall_rate:.1f}%",
    })
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    csv_dl = pd.DataFrame(display_rows).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_dl, "facilities_doing_blood_culture.csv", "text/csv",
                       key="_fac_bld_dl")


def render_infection_coverage_trend_chart(
    df, period_col="period_display", title="",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs
):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    working_df = _prepare_period_df(df, date_range_filters)
    if working_df is None or working_df.empty:
        st.warning("No data available.")
        return

    if period_col not in working_df.columns:
        st.warning("Period column not found.")
        return

    st.subheader("Infection — Coverage Run Charts")

    abx_tab, bc_tab = st.tabs(["Antibiotics", "Blood Culture"])

    with abx_tab:
        st.markdown("**Antibiotic Management Indicators**")
        col1, col2 = st.columns(2)

        with col1:
            _render_single_infection_chart(
                working_df, period_col, compute_clinical_sepsis_data,
                "Antibiotics for Clinical Sepsis", 100,
                "antibiotics_clinical_sepsis.csv",
                "",  # no per-chart expander
                bg_color, text_color, facility_uids, "abx",
            )

        with col2:
            _render_single_infection_chart(
                working_df, period_col, compute_culture_done_for_antibiotics_data,
                "Culture done for babies on antibiotics", 100,
                "culture_done_for_antibiotics.csv",
                "",  # no per-chart expander
                bg_color, text_color, facility_uids, "culture",
            )

        with st.expander("ℹ️ How antibiotic indicators are computed"):
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
                <tr>
                    <td style="padding:8px;"><b>Culture Done for Babies on Antibiotics</b></td>
                    <td style="padding:8px;">Babies who received antibiotics AND had a blood or CSF culture test</td>
                    <td style="padding:8px;">Babies who received antibiotics</td>
                    <td style="padding:8px;">100%</td>
                </tr>
                </table>
                <p style="margin-top:8px;"><b>Clinical Sepsis Definition:</b> Infection discharge diagnosis = <b>Probable Sepsis</b> or <b>Culture-Positive Sepsis</b>, <b>OR</b> any admission reason = <b>Suspected Infection - Sepsis/Meningitis (Neonatal working diagnosis)</b>.</p>
                </div>
                """, unsafe_allow_html=True,
            )

    with bc_tab:
        st.markdown("**Blood Culture Indicators**")
        col3, col4 = st.columns(2)

        with col3:
            _render_single_infection_chart(
                working_df, period_col, compute_blood_culture_results_recorded_data,
                "Blood culture results recorded", 100,
                "blood_culture_results_recorded.csv",
                "",  # no per-chart expander
                bg_color, text_color, facility_uids, "bld_res",
            )

        with col4:
            single_facility = (
                facility_uids is not None
                and (isinstance(facility_uids, (list, tuple)) and len(facility_uids) == 1)
            )
            if single_facility:
                st.caption("Facilities Doing Blood Culture — not available for single-facility view")
            else:
                _render_facilities_blood_culture_stacked_bar(
                    working_df, period_col, facility_uids,
                    bg_color, text_color,
                )

        with st.expander("ℹ️ How blood culture indicators are computed"):
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
                    <td style="padding:8px;"><b>Blood Culture Results Recorded</b></td>
                    <td style="padding:8px;">Babies who had blood culture done AND had a positive or negative result recorded</td>
                    <td style="padding:8px;">Babies who had blood culture done</td>
                    <td style="padding:8px;">100%</td>
                </tr>
                <tr>
                    <td style="padding:8px;"><b>Facilities Doing Blood Culture</b></td>
                    <td style="padding:8px;">Number of unique facilities where at least one baby had a blood culture test (result = Not Done, Negative, Positive, or Unknown)</td>
                    <td style="padding:8px;">Total number of unique facilities that reported data in this period</td>
                    <td style="padding:8px;">100%</td>
                </tr>
                </table>
                <p style="margin-top:8px;"><b>Blood culture codes:</b> 0 = Not Done, 1 = Done - Culture Negative, 2 = Done - Culture Positive, 3 = Done but Unknown Result. Codes 1-3 count as "blood culture done".</p>
                <p style="margin-top:8px;"><b>Facilities Doing Blood Culture:</b> Each unique facility is counted once per period regardless of how many babies were tested. A facility with at least one baby having codes 1-3 counts as "did blood culture".</p>
                </div>
                """, unsafe_allow_html=True,
            )


def compute_aware_classification_data(df, facility_uids=None):
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()
    columns = [c for c in working_df.columns if c.startswith(ANTIBIOTIC_MTEXT_PREFIX)]
    if not columns:
        return {"Access": 0, "Watch": 0, "Reserve": 0, "Unknown": 0, "total": 0}

    counts = {"Access": 0, "Watch": 0, "Reserve": 0, "Unknown": 0}
    for col in columns:
        vals = working_df[col].dropna().astype(str)
        for val in vals:
            codes = [c.strip() for c in val.split(",") if c.strip().isdigit()]
            for code_str in codes:
                code = int(code_str)
                if code in AWARE_ACCESS:
                    counts["Access"] += 1
                elif code in AWARE_WATCH:
                    counts["Watch"] += 1
                elif code in AWARE_UNKNOWN:
                    counts["Unknown"] += 1
                else:
                    counts["Unknown"] += 1
    counts["total"] = sum(counts.values())
    return counts


def render_infection_qoc_trend_chart(
    df, period_col="period_display", title="Antibiotics Classification (AWaRe)",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs
):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    working_df = _prepare_period_df(df, date_range_filters)
    if working_df is None or working_df.empty:
        st.subheader(title); st.warning("No data available."); return

    if period_col not in working_df.columns:
        st.subheader(title); st.warning("Period column not found."); return

    periods = sorted(working_df[period_col].unique())
    periods = sort_periods_chronologically(periods)

    trend_data = []
    for period in periods:
        pdf = working_df[working_df[period_col] == period]
        counts = compute_aware_classification_data(pdf, facility_uids)
        total = counts["total"]
        if total > 0:
            trend_data.append({
                "period": period,
                "Access": counts["Access"], "Watch": counts["Watch"],
                "Reserve": counts["Reserve"], "Unknown": counts["Unknown"],
                "total": total,
            })

    if not trend_data:
        st.subheader(title); st.warning("No antibiotic data found."); return

    trend_df = pd.DataFrame(trend_data)

    fig = go.Figure()
    categories = ["Access", "Watch", "Reserve", "Unknown"]
    for cat in categories:
        pcts = trend_df[cat] / trend_df["total"] * 100
        hover_texts = []
        for i, row in trend_df.iterrows():
            cat_count = int(row[cat])
            total_count = int(row["total"])

            antibiotics_in_cat = {}
            pdf = working_df[working_df[period_col] == row["period"]]
            pdf = _filter_by_facility(pdf, facility_uids) if facility_uids else pdf
            columns = [c for c in pdf.columns if c.startswith(ANTIBIOTIC_MTEXT_PREFIX)]
            for col in columns:
                vals = pdf[col].dropna().astype(str)
                for val in vals:
                    codes = [c.strip() for c in val.split(",") if c.strip().isdigit()]
                    for code_str in codes:
                        code = int(code_str)
                        if cat == "Access" and code in AWARE_ACCESS:
                            antibiotics_in_cat[code] = antibiotics_in_cat.get(code, 0) + 1
                        elif cat == "Watch" and code in AWARE_WATCH:
                            antibiotics_in_cat[code] = antibiotics_in_cat.get(code, 0) + 1
                        elif cat == "Unknown" and (code in AWARE_UNKNOWN or code not in AWARE_ACCESS | AWARE_WATCH):
                            antibiotics_in_cat[code] = antibiotics_in_cat.get(code, 0) + 1
                        elif cat == "Reserve":
                            pass
            cat_counts_list = [(ANTIBIOTIC_NAMES.get(c, f"Code {c}"), cnt)
                               for c, cnt in sorted(antibiotics_in_cat.items(), key=lambda x: -x[1])]
            breakdown_html = "<br>".join(
                [f"&nbsp;&nbsp;{name}: {cnt} ({cnt/cat_count*100:.1f}% of {cat})"
                 for name, cnt in cat_counts_list]
            ) if cat_count > 0 else "&nbsp;&nbsp;None"
            hide = "&nbsp;" if len(str(total_count)) > 4 else ""
            hover_texts.append(
                f"<b>{cat}</b><br>"
                f"Numerator: {cat_count:,}<br>"
                f"Denominator: {total_count:,}<br>"
                f"Percentage: {pcts.iloc[i]:.1f}%<br>"
                f"<b>Antibiotics within {cat}:</b><br>{breakdown_html}"
            )
        fig.add_trace(go.Bar(
            x=trend_df["period"], y=pcts,
            name=cat, marker_color=AWARE_COLORS[cat],
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts,
        ))

    fig.update_layout(
        title="Antibiotics Classification (AWaRe)", height=450,
        barmode="stack",
        yaxis=dict(range=[0, 105], dtick=25, title="Percentage (%)"),
        xaxis=dict(title=""),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    overall_access = int(trend_df["Access"].sum())
    overall_watch = int(trend_df["Watch"].sum())
    overall_reserve = int(trend_df["Reserve"].sum())
    overall_unknown = int(trend_df["Unknown"].sum())
    overall_total = overall_access + overall_watch + overall_reserve + overall_unknown

    display_rows = []
    for _, row in trend_df.iterrows():
        display_rows.append({
            "Period": row["period"],
            "Access": f"{int(row['Access'])} ({row['Access']/row['total']*100:.1f}%)",
            "Watch": f"{int(row['Watch'])} ({row['Watch']/row['total']*100:.1f}%)",
            "Reserve": f"{int(row['Reserve'])} ({row['Reserve']/row['total']*100:.1f}%)",
            "Unknown": f"{int(row['Unknown'])} ({row['Unknown']/row['total']*100:.1f}%)",
            "Total": int(row["total"]),
        })
    display_rows.append({
        "Period": "Overall",
        "Access": f"{overall_access} ({overall_access/overall_total*100:.1f}%)" if overall_total > 0 else "0 (0%)",
        "Watch": f"{overall_watch} ({overall_watch/overall_total*100:.1f}%)" if overall_total > 0 else "0 (0%)",
        "Reserve": f"{overall_reserve} ({overall_reserve/overall_total*100:.1f}%)" if overall_total > 0 else "0 (0%)",
        "Unknown": f"{overall_unknown} ({overall_unknown/overall_total*100:.1f}%)" if overall_total > 0 else "0 (0%)",
        "Total": overall_total,
    })
    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "aware_classification.csv", "text/csv",
                       key=_widget_key("aware_dl", kwargs.get("key_suffix")))

    with st.expander("ℹ️ How AWaRe classification is calculated"):
        st.markdown(
            f"""
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Category</th>
                <th style="padding:8px; text-align:left;">Antibiotics</th>
                <th style="padding:8px; text-align:left;">Description</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b style="color:#2ca02c;">Access</b></td>
                <td style="padding:8px;">Benzathine Penicillin, Gentamicin, Amikacin, Metronidazole, Clindamycin, Ampicillin, Amoxicillin, Flucloxacillin, Crystalline Penicillin, Cloxacillin, Cefalexin</td>
                <td style="padding:8px;">First- and second-line antibiotics for common infections</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b style="color:#ff7f0e;">Watch</b></td>
                <td style="padding:8px;">Ceftriaxone, Ceftazidime, Vancomycin, Meropenem, Cefotaxime, Levofloxacin, Ciprofloxacin, Cefixime, Tazobactam</td>
                <td style="padding:8px;">Broad-spectrum antibiotics with higher resistance potential</td>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b style="color:#9467bd;">Reserve</b></td>
                <td style="padding:8px;">None</td>
                <td style="padding:8px;">Last-reserve antibiotics (None in current option set)</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b style="color:#d62728;">Unknown</b></td>
                <td style="padding:8px;">Piperazine, Other</td>
                <td style="padding:8px;">Unclassified antibiotics</td>
            </tr>
            </table>
            <p style="margin-top:10px; font-size:0.9em;"><b>Note:</b> Count is per antibiotic <b>administration</b>, not per baby. A baby receiving multiple antibiotics contributes multiple counts. The denominator is the total number of antibiotic selections in the time period.</p>
            </div>
            """, unsafe_allow_html=True,
        )


def render_infection_facility_comparison(
    df, comparison_mode="facility", display_names=None, facility_uids=None,
    facilities_by_region=None, region_names=None, period_col="period_display",
    title="Infection Comparison", bg_color="#FFFFFF", text_color=None,
    is_qoc=False, **kwargs
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
        if is_qoc:
            aware = compute_aware_classification_data(edf)
            t = aware["total"]
            comparison_data.append({
                "Entity": ename,
                "Access %": f"{aware['Access']/t*100:.1f}%" if t > 0 else "-",
                "Access N": aware["Access"],
                "Watch %": f"{aware['Watch']/t*100:.1f}%" if t > 0 else "-",
                "Watch N": aware["Watch"],
                "Reserve %": f"{aware['Reserve']/t*100:.1f}%" if t > 0 else "-",
                "Reserve N": aware["Reserve"],
                "Unknown %": f"{aware['Unknown']/t*100:.1f}%" if t > 0 else "-",
                "Unknown N": aware["Unknown"],
                "Total": t,
            })
        else:
            n1, d1, r1 = compute_clinical_sepsis_data(edf)
            n2, d2, r2 = compute_culture_done_for_antibiotics_data(edf)
            n3, d3, r3 = compute_blood_culture_results_recorded_data(edf)
            n4, d4, r4 = compute_facilities_doing_blood_culture_data(edf)
            comparison_data.append({
                "Entity": ename,
                "Antibiotics for Clinical Sepsis": f"{r1:.1f}%" if d1 > 0 else "-",
                "N/D (Sepsis)": f"{n1}/{d1}" if d1 > 0 else "-",
                "Culture Done for Antibiotics": f"{r2:.1f}%" if d2 > 0 else "-",
                "N/D (Culture)": f"{n2}/{d2}" if d2 > 0 else "-",
                "Blood Culture Results": f"{r3:.1f}%" if d3 > 0 else "-",
                "N/D (Results)": f"{n3}/{d3}" if d3 > 0 else "-",
                "Facilities Doing Blood Culture": f"{r4:.1f}%" if d4 > 0 else "-",
                "N/D (Facilities)": f"{n4}/{d4}" if d4 > 0 else "-",
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
    "compute_culture_done_for_antibiotics_data",
    "compute_blood_culture_results_recorded_data",
    "compute_facilities_doing_blood_culture_data",
    "render_infection_coverage_trend_chart",
    "render_infection_qoc_trend_chart",
    "render_infection_facility_comparison",
]
