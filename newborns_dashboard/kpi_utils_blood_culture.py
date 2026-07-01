# kpi_utils_blood_culture.py - Blood Culture Dashboard Implementation

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
BLOOD_CULTURE_PREFIX = "blood_culture_for_suspected_sepsis_investigation_sheet"  # Blood culture for suspected sepsis
ORGANISM_PREFIX = "culture_positive_organism_blood_investigation_sheet"           # Culture Positive Organism (Blood)
OTHER_ORGANISM_PREFIX = "if_other_full_species_genus_of_microorganism_investigation_sheet"  # If other - full species/genus of microorganism

LINE_COLOR = "#1f77b4"
CONTI_COLOR = "#ff7f0e"

# --- Distinct colors for each organism ---
ORGANISM_COLORS = {
    "Staphylococcus aureus": "#e6194b",
    "Klebsiella spp.": "#3cb44b",
    "Pseudomonas spp.": "#4363d8",
    "Escherichia coli": "#f58231",
    "Acinetobacter spp.": "#911eb4",
    "Group B Streptococcus": "#42d4f4",
    "Other gram-negatives": "#f032e6",
    "Other Streptococcus": "#bfef45",
    "Other Fungal spp.": "#fabed4",
    "Other": "#808080",
}

# --- Contaminant organism mapping (configurable) ---
CONTI_ORGANISMS = {
    "Staphylococcus aureus": True,
    "Other Streptococcus": True,
}

ORGANISM_NAMES = {
    1: "Staphylococcus aureus",
    2: "Klebsiella spp.",
    3: "Pseudomonas spp.",
    4: "Escherichia coli",
    5: "Acinetobacter spp.",
    6: "Group B Streptococcus",
    7: "Other gram-negatives",
    8: "Other Streptococcus",
    9: "Other Fungal spp.",
    99: "Other",
}
ORGANISM_CODES = {v: k for k, v in ORGANISM_NAMES.items()}


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


def _resolve_organism_code(row):
    """Resolve organism name from row data.
    
    Checks both primary (ORGANISM_PREFIX) and fallback (OTHER_ORGANISM_PREFIX)
    columns. If aCHOclZEx6o = 99, uses HvJ8H9tun4u value.
    Excludes aCHOclZEx6o = 11 (Not indicated).
    """
    # Check primary organism column
    org_col = None
    for col in row.index:
        if col.startswith(ORGANISM_PREFIX):
            org_col = col
            break
    if org_col is None:
        return None
    
    org_code = row[org_col]
    
    # Skip NaN or non-string values
    if pd.isna(org_code) or not isinstance(org_code, str):
        org_code = str(org_code).strip() if pd.notna(org_code) else None
        if org_code is None:
            return None
        # Convert numeric floats like "11.0" → "11"
        if org_code.endswith(".0"):
            org_code = org_code[:-2]
    
    # Skip if "Not indicated"
    if org_code == "11":
        return None
    
    # Handle special "Other" case where we use the other organims field
    if org_code == "99":
        other_col = None
        for col in row.index:
            if col.startswith(OTHER_ORGANISM_PREFIX):
                other_col = col
                break
        if other_col is not None and pd.notna(row.get(other_col)):
            return row[other_col]
        # Fall back to "Other" label
        return "Other"
    
    # Handle numeric codes (1-9)
    if org_code.isdigit():
        code = int(org_code)
        if code in ORGANISM_NAMES:
            return ORGANISM_NAMES[code]
    
    return None


def _widget_key(base_key, key_suffix=None):
    suffix = str(key_suffix).strip() if key_suffix is not None else ""
    return f"{base_key}_{suffix}" if suffix else base_key


def compute_positive_culture_rate_data(df, facility_uids=None):
    """Positive Blood Culture Rate Among All Cultures Done.
    
    NEST360 Definition: Proportion of blood cultures performed that yielded positive results.
    
    Numerator: Babies with blood culture = 2 (Positive)
    Denominator: Babies with blood culture = 1,2,3 (Any culture done, exclude 0)
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()
    done_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 1, 2, 3)
    positive_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 2)
    
    denominator = int(done_mask.sum())
    numerator = int(positive_mask.sum())
    rate = (numerator / denominator * 100) if denominator > 0 else 0.0
    
    return numerator, denominator, rate


def compute_probable_contaminant_rate_data(df, facility_uids=None):
    """Percent Probable Contaminants Among Positive Blood Cultures.
    
    NEST360 Definition: Proportion of positive blood cultures that are likely contaminants.
    
    Probable Contaminant Definition (initial implementation):
    - Staphylococcus aureus
    - Other Streptococcus
    
    Configurable via CONTI_ORGANISMS mapping.
    
    Numerator: Positive blood cultures with contaminant organisms
    Denominator: All positive blood cultures
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()
    positive_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 2)
    
    if not positive_mask.any():
        return 0, 0, 0.0
    
    working_df = working_df[positive_mask].copy()
    contaminated_count = 0
    
    for _, row in working_df.iterrows():
        org_name = _resolve_organism_code(row)
        if org_name and org_name in CONTI_ORGANISMS:
            contaminated_count += 1
    
    denominator = int(positive_mask.sum())
    rate = (contaminated_count / denominator * 100) if denominator > 0 else 0.0
    
    return contaminated_count, denominator, rate


def compute_microorganism_distribution_data(df, facility_uids=None):
    """Distribution of Microorganisms Identified in Positive Blood Cultures.
    
    Shows the proportional distribution of microorganisms identified among all 
    positive blood cultures.
    
    Chart Type: Horizontal Percentage Bar Chart
    Rankings microorganisms from most frequent to least frequent.
    
    Only includes: GwrkagnbTet = 2 (positive cultures)
    Excludes: GwrkagnbTet = 11 (Not indicated)
    
    Primary organism source: aCHOclZEx6o with HvJ8H9tun4u fallback for code 99
    
    Numerator: Number of positive cultures with specific microorganism
    Denominator: Total microorganisms identified in positive blood cultures
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()
    positive_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 2)
    
    if not positive_mask.any():
        return {}
    
    working_df = working_df[positive_mask].copy()
    working_df["_organism"] = working_df.apply(_resolve_organism_code, axis=1)
    
    # Filter out "Not indicated" and None values
    filtered_df = working_df[working_df["_organism"].notna()].copy()
    
    distribution = filtered_df["_organism"].value_counts().to_dict()
    return distribution


def compute_microorganism_trend_data(df, facility_uids=None, top_n=5, period_col="period_display"):
    """Monthly Trend of Microorganisms Identified in Positive Blood Cultures.
    
    Tracks how the prevalence of specific microorganisms changes over time.
    
    Chart Type: Multi-Series Run Chart
    
    Inclusion: Only include GwrkagnbTet = 2 (positive cultures)
    
    Organism determination:
    - Primary: aCHOclZEx6o
    - Fallback for code 99: HvJ8H9tun4u
    
    Exclusion: GwrkagnbTet = 11 (Not indicated)
    
    Numerator: Number of positive cultures with organism in period
    Denominator: Total positive organisms identified in same period
    
    Default: Display top 5 most common microorganisms
    """
    working_df = _filter_by_facility(df, facility_uids) if facility_uids else df.copy()
    positive_mask = _any_version_mask(working_df, BLOOD_CULTURE_PREFIX, 2)
    
    if not positive_mask.any():
        return pd.DataFrame()
    
    working_df = working_df[positive_mask].copy()
    working_df["_organism"] = working_df.apply(_resolve_organism_code, axis=1)
    working_df = working_df[working_df["_organism"].notna()].copy()
    
    if working_df.empty:
        return pd.DataFrame()
    
    # Use period_col if it already exists in the dataframe, otherwise assign
    if period_col not in working_df.columns:
        period_label = st.session_state.get("period_label", "Monthly")
        try:
            working_df = assign_period(working_df, "event_date", period_label)
        except Exception:
            pass
    
    if period_col not in working_df.columns:
        return pd.DataFrame()
    
    periods = sorted(working_df[period_col].unique())
    periods = sort_periods_chronologically(periods)
    
    # Get top N organisms by frequency
    all_organisms = working_df["_organism"].unique()
    if len(all_organisms) > top_n:
        organism_counts = working_df["_organism"].value_counts()
        top_organisms = organism_counts.head(top_n).index.tolist()
        working_df = working_df[working_df["_organism"].isin(top_organisms)]
    
    # Build trend data
    trend_data = []
    for period in periods:
        period_df = working_df[working_df[period_col] == period]
        if period_df.empty:
            continue
        period_total = len(period_df)
        for org in working_df["_organism"].unique():
            org_count = len(period_df[period_df["_organism"] == org])
            trend_data.append({
                "period": period,
                "organism": org,
                "count": org_count,
                "total": period_total,
                "percentage": (org_count / period_total * 100) if period_total > 0 else 0
            })
    
    trend_df = pd.DataFrame(trend_data)
    return trend_df


def _render_single_blood_culture_chart(
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

    # Remove periods with zero denominator (no data for that month)
    trend_df = trend_df[trend_df["denominator"] > 0].copy()
    if trend_df.empty:
        st.info(f"No data to display for **{title}** (no positive blood cultures recorded).")
        return

    overall_n = int(trend_df["numerator"].sum())
    overall_d = int(trend_df["denominator"].sum())
    overall_rate = (overall_n / overall_d * 100) if overall_d > 0 else 0

    fig = go.Figure()
    if target is not None:
        fig.add_hline(y=target, line_dash="dash", line_color="green", opacity=0.6,
                      annotation_text=f"Target: {target}%", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=trend_df["period"], y=trend_df["rate"], mode="lines+markers",
        name="Rate %", line=dict(color=LINE_COLOR, width=3), marker=dict(size=8),
        hovertemplate="Month: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
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
                       key=_widget_key(f"blood_{key_suffix}_dl", None))

    if info_html:
        with st.expander("ℹ How this indicator is computed"):
            st.markdown(info_html, unsafe_allow_html=True)


def _render_microorganism_distribution_chart(working_df, period_col, bg_color, text_color, facility_uids):
    """Render horizontal bar chart showing microorganism distribution."""
    distribution = compute_microorganism_distribution_data(working_df, facility_uids)
    
    if not distribution:
        st.caption("Microorganism Distribution — No data")
        return
    
    total_organisms = sum(distribution.values())
    sorted_items = sorted(distribution.items(), key=lambda x: -x[1])
    
    fig = go.Figure(go.Bar(
        x=[item[1] for item in sorted_items],
        y=[item[0] for item in sorted_items],
        orientation="h",
        marker_color=[ORGANISM_COLORS.get(item[0], LINE_COLOR) for item in sorted_items],
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Total: %{customdata[0]}<br>Percentage: %{customdata[1]}%<extra></extra>",
        customdata=[[total_organisms, round((item[1] / total_organisms * 100), 1)] for item in sorted_items],
    ))
    
    fig.update_layout(
        title="Microorganism Distribution in Positive Blood Cultures",
        height=400,
        xaxis=dict(title="Count", dtick=1),
        yaxis=dict(title="Organism"),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="bc_dist_chart")


def _render_microorganism_trend_chart(working_df, period_col, bg_color, text_color, facility_uids):
    """Render multi-series run chart showing microorganism trends over time."""
    trend_df = compute_microorganism_trend_data(working_df, facility_uids, period_col=period_col)
    
    if trend_df.empty:
        st.caption("Microorganism Trend — No data")
        return
    
    all_organisms = sorted(trend_df["organism"].unique().tolist())

    fig = go.Figure()
    for org in all_organisms:
        org_df = trend_df[trend_df["organism"] == org]
        if org_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=org_df["period"], y=org_df["percentage"],
            name=org,
            line=dict(color=ORGANISM_COLORS.get(org, LINE_COLOR)),
            mode="lines+markers",
            hovertemplate="<b>" + org + "</b><br>Month: %{x}<br>Count: %{customdata[0]}<br>Total: %{customdata[1]}<br>Percentage: %{y:.1f}%<extra></extra>",
            customdata=np.column_stack((org_df["count"], org_df["total"])),
        ))
    
    fig.update_layout(
        title="Microorganism Trend Over Time",
        height=400,
        yaxis=dict(range=[0, 105], dtick=25, title="Percentage (%)"),
        xaxis=dict(title=""),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=80),
        legend=dict(
            orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
            tracegroupgap=0,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key="bc_trend_chart")


def render_blood_culture_trend_chart(
    df, period_col="period_display", title="Blood Culture Dashboard",
    bg_color="#FFFFFF", text_color=None, facility_uids=None, date_range_filters=None, **kwargs
):
    """Main function to render the complete Blood Culture dashboard with all 4 indicators."""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    working_df = _prepare_period_df(df, date_range_filters)
    if working_df is None or working_df.empty:
        st.warning("No data available.")
        return

    if period_col not in working_df.columns:
        st.warning("Period column not found.")
        return

    st.subheader("Blood Culture - Indicator Coverage Run Charts")

    # Sub-tabs: Culture Rates | Microorganism Analysis
    tab_rates, tab_micro = st.tabs(["Culture Rates", "Microorganism Analysis"])

    with tab_rates:
        col1, col2 = st.columns(2)

        with col1:
            _render_single_blood_culture_chart(
                working_df, period_col, compute_positive_culture_rate_data,
                "Positive Blood Culture Rate", None,
                "positive_blood_culture_rate.csv", "",
                bg_color, text_color, facility_uids, "ind1",
            )

        with col2:
            _render_single_blood_culture_chart(
                working_df, period_col, compute_probable_contaminant_rate_data,
                "Probable Contaminants Rate", None,
                "probable_contaminant_rate.csv", "",
                bg_color, text_color, facility_uids, "ind2",
            )

        with st.expander("ℹ How these indicators are computed"):
            st.markdown("""
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Positive Blood Culture Rate</b></td>
                <td style="padding:8px;">Babies with blood culture done AND culture positive</td>
                <td style="padding:8px;">Babies with blood culture performed (any result)</td>
            </tr>
            <tr style="background-color:#ffffff;">
                <td style="padding:8px;"><b>Probable Contaminants Rate</b></td>
                <td style="padding:8px;">Positive blood cultures where the organism is classified as a probable contaminant</td>
                <td style="padding:8px;">All positive blood cultures</td>
            </tr>
            </table>
            <p style="margin-top:8px;"><b>Positive Blood Culture Rate</b> measures the proportion of blood cultures performed that yielded a positive culture result.</p>
            <p style="margin-top:4px;"><b>Probable Contaminants Rate</b> measures the proportion of positive blood cultures that are likely contaminants. <b>Probable Contaminants</b> are <i>Staphylococcus aureus</i> and <i>Other Streptococcus</i> — organisms commonly associated with skin flora contamination during blood collection.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_micro:
        col3, col4 = st.columns(2)

        with col3:
            _render_microorganism_distribution_chart(working_df, period_col, bg_color, text_color, facility_uids)

        with col4:
            _render_microorganism_trend_chart(working_df, period_col, bg_color, text_color, facility_uids)

        with st.expander("ℹ How these indicators are computed"):
            st.markdown("""
            <div style="background-color:#e8f4fd; padding:15px; border-radius:8px; border-left:4px solid #1f77b4;">
            <table style="width:100%; border-collapse:collapse;">
            <tr style="background-color:#1f77b4; color:white;">
                <th style="padding:8px; text-align:left;">Indicator</th>
                <th style="padding:8px; text-align:left;">Numerator</th>
                <th style="padding:8px; text-align:left;">Denominator</th>
            </tr>
            <tr style="background-color:#f0f8ff;">
                <td style="padding:8px;"><b>Microorganism Distribution</b></td>
                <td style="padding:8px;">Number of positive cultures with specific microorganism</td>
                <td style="padding:8px;">Total microorganisms identified in positive blood cultures</td>
            </tr>
            <tr style="background-color:#ffffff;">
                <td style="padding:8px;"><b>Microorganism Trend Over Time</b></td>
                <td style="padding:8px;">Number of positive cultures with organism in period</td>
                <td style="padding:8px;">Total positive organisms identified in same period</td>
            </tr>
            </table>
            <p style="margin-top:8px;">Shows the proportional distribution and monthly trend of microorganisms identified among all positive blood cultures. The organism is identified from the <b>Culture Positive Organism (Blood)</b> field. If the organism is not in the standard list, the free-text entry from <b>If other - full species/genus of microorganism</b> is used. Records with <b>Not indicated</b> are excluded.</p>
            </div>
            """, unsafe_allow_html=True)


__all__ = [
    "compute_positive_culture_rate_data",
    "compute_probable_contaminant_rate_data",
    "compute_microorganism_distribution_data",
    "compute_microorganism_trend_data",
    "render_blood_culture_trend_chart",
    "render_blood_culture_comparison_chart",
]


def render_blood_culture_comparison_chart(
    df, comparison_mode="facility", display_names=None, facility_uids=None,
    facilities_by_region=None, region_names=None, period_col="period_display",
    title="Blood Culture Comparison", bg_color="#FFFFFF", text_color=None, **kwargs
):
    """Comparison chart for Blood Culture — multi-entity line charts."""
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

    comp_data = {}
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
        entity_rows = []
        for period in periods:
            pdf = edf[edf[period_col] == period]
            n1, d1, r1 = compute_positive_culture_rate_data(pdf)
            n2, d2, r2 = compute_probable_contaminant_rate_data(pdf)
            entity_rows.append({
                "period": period,
                "positive_rate": r1, "positive_num": n1, "positive_den": d1,
                "contaminant_rate": r2, "contaminant_num": n2, "contaminant_den": d2,
            })
        comp_data[eid] = pd.DataFrame(entity_rows)

    if not comp_data:
        st.warning("No comparison data."); return

    st.subheader(title)
    tab1, tab2 = st.tabs(["Positive Blood Culture Rate", "Probable Contaminants Rate"])

    with tab1:
        fig = go.Figure()
        for eid in sorted_entities:
            if eid not in comp_data:
                continue
            edf = comp_data[eid]
            fig.add_trace(go.Scatter(
                x=edf["period"], y=edf["positive_rate"],
                name=entities[eid], mode="lines+markers",
                line=dict(color=entity_colors[eid], width=2),
                hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
                customdata=np.column_stack((edf["positive_num"].values, edf["positive_den"].values)),
            ))
        fig.update_layout(
            title="Positive Blood Culture Rate", height=400,
            yaxis=dict(range=[0, 105], dtick=25, title="Rate (%)"),
            xaxis=dict(title=""),
            paper_bgcolor=bg_color, plot_bgcolor=bg_color,
            font_color=text_color, title_font_color=text_color,
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, key="bc_comp_pos")

    with tab2:
        fig = go.Figure()
        for eid in sorted_entities:
            if eid not in comp_data:
                continue
            edf = comp_data[eid]
            fig.add_trace(go.Scatter(
                x=edf["period"], y=edf["contaminant_rate"],
                name=entities[eid], mode="lines+markers",
                line=dict(color=entity_colors[eid], width=2),
                hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Rate: %{y:.1f}%<br>Num: %{customdata[0]}<br>Den: %{customdata[1]}<extra></extra>",
                customdata=np.column_stack((edf["contaminant_num"].values, edf["contaminant_den"].values)),
            ))
        fig.update_layout(
            title="Probable Contaminants Rate", height=400,
            yaxis=dict(range=[0, 105], dtick=25, title="Rate (%)"),
            xaxis=dict(title=""),
            paper_bgcolor=bg_color, plot_bgcolor=bg_color,
            font_color=text_color, title_font_color=text_color,
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, key="bc_comp_cont")
