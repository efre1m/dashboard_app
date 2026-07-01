# kpi_utils_jaundice.py - Jaundice & Phototherapy indicator computations and chart rendering

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
JAUNDICE_DIAGNOSIS_COL = "sub_categories_of_jaundice_pathological_n_discharge_care_form"
PHOTOTHERAPY_PREFIX = "phototherapy_administered?_medication_sheet"
BILIRUBIN_PREFIX = "bilirubin_tested?_nurse_followup_sheet"
TRANSFUSION_PREFIX = "transfusion_given?_medication_sheet"

LINE_COLOR = "#1f77b4"


def _any_version_mask(df, col_prefix, target_value=1):
    """Boolean mask: True where ANY column matching col_prefix equals target_value."""
    cols = [c for c in df.columns if c.startswith(col_prefix)]
    if not cols:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for col in cols:
        num = pd.to_numeric(df.get(col, pd.Series(dtype=str)), errors="coerce")
        mask = mask | (num == target_value)
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


def compute_phototherapy_coverage_data(df, facility_uids=None):
    working_df = _filter_by_facility(df, facility_uids)
    working_df[JAUNDICE_DIAGNOSIS_COL] = pd.to_numeric(
        working_df.get(JAUNDICE_DIAGNOSIS_COL, pd.Series(dtype=str)), errors="coerce"
    )
    phototherapy_mask = _any_version_mask(working_df, PHOTOTHERAPY_PREFIX, 1)
    denominator_mask = working_df[JAUNDICE_DIAGNOSIS_COL].isin([1, 2])
    numerator_mask = denominator_mask & phototherapy_mask
    denominator = int(denominator_mask.sum())
    numerator = int(numerator_mask.sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_bilirubin_measurement_data(df, facility_uids=None):
    working_df = _filter_by_facility(df, facility_uids)
    working_df[JAUNDICE_DIAGNOSIS_COL] = pd.to_numeric(
        working_df.get(JAUNDICE_DIAGNOSIS_COL, pd.Series(dtype=str)), errors="coerce"
    )
    bilirubin_mask = _any_version_mask(working_df, BILIRUBIN_PREFIX, 1)
    denominator_mask = working_df[JAUNDICE_DIAGNOSIS_COL].isin([1, 2])
    numerator_mask = denominator_mask & bilirubin_mask
    denominator = int(denominator_mask.sum())
    numerator = int(numerator_mask.sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def compute_exchange_transfusion_data(df, facility_uids=None):
    working_df = _filter_by_facility(df, facility_uids)
    working_df[JAUNDICE_DIAGNOSIS_COL] = pd.to_numeric(
        working_df.get(JAUNDICE_DIAGNOSIS_COL, pd.Series(dtype=str)), errors="coerce"
    )
    transfusion_mask = _any_version_mask(working_df, TRANSFUSION_PREFIX, 2)
    denominator_mask = working_df[JAUNDICE_DIAGNOSIS_COL] == 2
    numerator_mask = denominator_mask & transfusion_mask
    denominator = int(denominator_mask.sum())
    numerator = int(numerator_mask.sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    return numerator, denominator, rate


def _widget_key(base_key, key_suffix=None):
    suffix = str(key_suffix).strip() if key_suffix is not None else ""
    return f"{base_key}_{suffix}" if suffix else base_key


def render_jaundice_coverage_trend_chart(
    df, period_col="period_display", title="Phototherapy for Clinical Jaundice",
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
        n, d, r = compute_phototherapy_coverage_data(pdf, facility_uids)
        trend_data.append({"period": period, "numerator": n, "denominator": d, "rate": r})
    trend_df = pd.DataFrame(trend_data)
    if trend_df.empty:
        st.warning("No data."); return

    # Remove periods with zero denominator
    trend_df = trend_df[trend_df["denominator"] > 0].copy()
    if trend_df.empty:
        st.info("No data to display (no jaundice cases recorded)."); return

    st.subheader(f"{title} — Monthly")
    fig = go.Figure()
    fig.add_hline(y=90, line_dash="dash", line_color="green", opacity=0.6,
                  annotation_text="Target: 90%", annotation_position="top left")
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
    st.download_button("Download CSV", csv, "phototherapy_coverage.csv", "text/csv",
                       key=_widget_key("jaundice_cov_dl", kwargs.get("key_suffix")))

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
                <td style="padding:8px;"><b>Phototherapy for Clinical Jaundice</b></td>
                <td style="padding:8px;">Jaundice babies who received phototherapy</td>
                <td style="padding:8px;">Babies diagnosed with clinical jaundice (requiring phototherapy or exchange transfusion)</td>
                <td style="padding:8px;">90%</td>
            </tr>
            </table>
            </div>
            """, unsafe_allow_html=True,
        )


def render_jaundice_qoc_trend_chart(
    df, period_col="period_display", title="Quality of Care",
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

    qoc_tab1, qoc_tab2 = st.tabs(["Bilirubin Measurement", "Exchange Transfusion"])

    periods = sort_periods_chronologically(periods)

    with qoc_tab1:
        trend_data = []
        for period in periods:
            pdf = working_df[working_df[period_col] == period]
            n, d, r = compute_bilirubin_measurement_data(pdf, facility_uids)
            trend_data.append({"period": period, "numerator": n, "denominator": d, "rate": r})
        trend_df = pd.DataFrame(trend_data)
        # Remove periods with zero denominator
        trend_df = trend_df[trend_df["denominator"] > 0].copy()
        if not trend_df.empty:
            st.subheader("Bilirubin Measurement Rate — 100% Target")
            fig = go.Figure()
            fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.6,
                          annotation_text="Target: 100%", annotation_position="top left")
            fig.add_trace(go.Scatter(
                x=trend_df["period"], y=trend_df["rate"], mode="lines+markers",
                name="Rate %", line=dict(color=LINE_COLOR, width=3), marker=dict(size=8),
                hovertemplate="Month: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
                customdata=np.column_stack((trend_df["numerator"], trend_df["denominator"])),
            ))
            fig.update_layout(title="Bilirubin Measurement Rate", height=350,
                              yaxis=dict(range=[0, 105], dtick=25, title="Rate (%)"),
                              xaxis=dict(title=""), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                              font_color=text_color, title_font_color=text_color,
                              margin=dict(l=50, r=30, t=60, b=40))
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
            st.download_button("Download CSV", csv, "bilirubin_measurement_rate.csv", "text/csv",
                               key=_widget_key("bilirubin_dl", kwargs.get("key_suffix")))

    with qoc_tab2:
        trend_data = []
        for period in periods:
            pdf = working_df[working_df[period_col] == period]
            n, d, r = compute_exchange_transfusion_data(pdf, facility_uids)
            trend_data.append({"period": period, "numerator": n, "denominator": d, "rate": r})
        trend_df = pd.DataFrame(trend_data)
        # Remove periods with zero denominator
        trend_df = trend_df[trend_df["denominator"] > 0].copy()
        if not trend_df.empty:
            st.subheader("Exchange Transfusion Rate — 100% Target")
            fig = go.Figure()
            fig.add_hline(y=100, line_dash="dash", line_color="green", opacity=0.6,
                          annotation_text="Target: 100%", annotation_position="top left")
            fig.add_trace(go.Scatter(
                x=trend_df["period"], y=trend_df["rate"], mode="lines+markers",
                name="Rate %", line=dict(color=LINE_COLOR, width=3), marker=dict(size=8),
                hovertemplate="Month: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
                customdata=np.column_stack((trend_df["numerator"], trend_df["denominator"])),
            ))
            fig.update_layout(title="Exchange Transfusion Rate", height=350,
                              yaxis=dict(range=[0, 105], dtick=25, title="Rate (%)"),
                              xaxis=dict(title=""), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                              font_color=text_color, title_font_color=text_color,
                              margin=dict(l=50, r=30, t=60, b=40))
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
            st.download_button("Download CSV", csv, "exchange_transfusion_rate.csv", "text/csv",
                               key=_widget_key("transfusion_dl", kwargs.get("key_suffix")))

    with st.expander("ℹ️ How each indicator is computed"):
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
                <td style="padding:8px;"><b>Bilirubin Measurement Rate</b></td>
                <td style="padding:8px;">Jaundice babies who had bilirubin tested</td>
                <td style="padding:8px;">Babies diagnosed with clinical jaundice (requiring phototherapy or exchange transfusion)</td>
                <td style="padding:8px;">100%</td>
            </tr>
            <tr>
                <td style="padding:8px;"><b>Exchange Transfusion Rate</b></td>
                <td style="padding:8px;">Babies requiring exchange transfusion who actually received exchange blood transfusion</td>
                <td style="padding:8px;">Babies diagnosed with jaundice requiring exchange transfusion</td>
                <td style="padding:8px;">100%</td>
            </tr>
            <tr style="background-color:#fff3cd;">
                <td style="padding:8px;" colspan="4"><b>⚠️ Note:</b> Standard blood transfusion is not counted as success for this indicator. Only exchange blood transfusion counts.</td>
            </tr>
            </table>
            </div>
            """, unsafe_allow_html=True,
        )


def render_jaundice_facility_comparison(
    df, comparison_mode="facility", display_names=None, facility_uids=None,
    facilities_by_region=None, region_names=None, period_col="period_display",
    title="Jaundice & Phototherapy Comparison", bg_color="#FFFFFF", text_color=None, **kwargs
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

    if period_col not in df.columns:
        st.warning("Period column not found."); return

    periods = sort_periods_chronologically(df[period_col].unique())
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    sorted_entities = sorted(entities.keys(), key=lambda e: str(entities[e]).lower())
    entity_colors = {e: color_palette[i % len(color_palette)] for i, e in enumerate(sorted_entities)}

    comp_trends = {}
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
        pn, pd_, pr = compute_phototherapy_coverage_data(edf)
        bn, bd, br = compute_bilirubin_measurement_data(edf)
        en_, ed, er = compute_exchange_transfusion_data(edf)
        comparison_data.append({
            "Entity": ename,
            "Phototherapy Coverage": f"{pr:.1f}%" if pd_ > 0 else "-",
            "Phototherapy N/D": f"{pn}/{pd_}" if pd_ > 0 else "-",
            "Bilirubin Measurement": f"{br:.1f}%" if bd > 0 else "-",
            "Bilirubin N/D": f"{bn}/{bd}" if bd > 0 else "-",
            "Exchange Transfusion": f"{er:.1f}%" if ed > 0 else "-",
            "Exchange N/D": f"{en_}/{ed}" if ed > 0 else "-",
        })
        # Build per-period trend data for line charts
        rows = []
        for period in periods:
            pdf = edf[edf[period_col] == period]
            pn1, pd1, pr1 = compute_phototherapy_coverage_data(pdf)
            bn1, bd1, br1 = compute_bilirubin_measurement_data(pdf)
            en1, ed1, er1 = compute_exchange_transfusion_data(pdf)
            rows.append({
                "period": period,
                "photo_rate": pr1, "photo_num": pn1, "photo_den": pd1,
                "bili_rate": br1, "bili_num": bn1, "bili_den": bd1,
                "exchange_rate": er1, "exchange_num": en1, "exchange_den": ed1,
            })
        comp_trends[eid] = pd.DataFrame(rows)

    if not comparison_data:
        st.warning("No comparison data."); return

    st.subheader(title)

    if comp_trends:
        tab1, tab2, tab3 = st.tabs(["Phototherapy Coverage", "Bilirubin Measurement", "Exchange Transfusion"])
        for tab, key, nkey, dkey, ytitle in [
            (tab1, "photo_rate", "photo_num", "photo_den", "Rate (%)"),
            (tab2, "bili_rate", "bili_num", "bili_den", "Rate (%)"),
            (tab3, "exchange_rate", "exchange_num", "exchange_den", "Rate (%)"),
        ]:
            with tab:
                fig = go.Figure()
                for eid in sorted_entities:
                    if eid not in comp_trends:
                        continue
                    edf = comp_trends[eid]
                    fig.add_trace(go.Scatter(
                        x=edf["period"], y=edf[key],
                        name=entities[eid], mode="lines+markers",
                        line=dict(color=entity_colors[eid], width=2),
                        hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
                        customdata=np.column_stack((edf[nkey].values, edf[dkey].values)),
                    ))
                fig.update_layout(
                    height=400,
                    yaxis=dict(title=ytitle),
                    xaxis=dict(title=""),
                    paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    font_color=text_color, title_font_color=text_color,
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"jand_{key}_comp")

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    csv = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "jaundice_comparison.csv", "text/csv")


__all__ = [
    "JAUNDICE_DIAGNOSIS_COL", "PHOTOTHERAPY_PREFIX", "BILIRUBIN_PREFIX", "TRANSFUSION_PREFIX",
    "compute_phototherapy_coverage_data", "compute_bilirubin_measurement_data",
    "compute_exchange_transfusion_data",
    "render_jaundice_coverage_trend_chart", "render_jaundice_qoc_trend_chart",
    "render_jaundice_facility_comparison",
]
