# kpi_utils_blood_culture.py - Blood Culture Dashboard Implementation

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import logging
import re

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

# --- Distinct colors for each organism (coded 1-9 and 99 baseline organisms) ---
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
    # --- Extended palette for common free-text organisms (code 99) ---
    "Enterococcus spp.": "#9A6324",
    "Hemolytic Streptococcus spp.": "#469990",
    "Citrobacter spp.": "#800000",
    "Streptococcus pneumoniae": "#aaffc3",
    "Salmonella spp.": "#dcbeff",
    "Serratia spp.": "#a9a9a9",
    "Proteus spp.": "#ffd8b1",
    "Candida spp.": "#fffac8",
    "Listeria spp.": "#000075",
    "Haemophilus spp.": "#008080",
    "Moraxella spp.": "#e6beff",
    "Stenotrophomonas spp.": "#aa6e28",
    "Burkholderia spp.": "#ffe119",
    "Neisseria spp.": "#46f0f0",
    "Enterobacter spp.": "#d2f53c",
}

# --- Extra color pool for organisms not in ORGANISM_COLORS (guarantee uniqueness) ---
_EXTRA_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#82E0AA", "#F1948A",
    "#85C1E9", "#F0B27A", "#C39BD3", "#76D7C4", "#F9E79F",
    "#AED6F1", "#A9DFBF", "#FDEBD0", "#D2B4DE", "#A3E4D7",
    "#FAD7A0", "#A9CCE3", "#A8D5B5", "#F5CBA7", "#D7BDE2",
]

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


# ---------------------------------------------------------------------------
# Organism name standardisation
# ---------------------------------------------------------------------------

# Compiled regex rules: (pattern, canonical_name)
# Rules are applied in order; first match wins.
# A canonical_name of None means the text should be excluded (returned as None).
_ORGANISM_NORM_RULES = [
    # --- Exclusion rules: recognised non-organism patterns ----------
    # Bare negative numbers: -1, -3, -1.0, -3.0
    (re.compile(r'^-\d+(?:\.\d+)?$', re.I), None),
    # Negative number with text: -1:Not recorded, -3:Not readable, 0:Done
    (re.compile(r'^-?\d+:', re.I), None),
    # General "not" + word patterns
    (re.compile(r'not\s+recorded', re.I), None),
    (re.compile(r'not\s+readable', re.I), None),
    (re.compile(r'not\s+done', re.I), None),
    (re.compile(r'no\s+growth', re.I), None),
    # --- Enterococcus variants: catches ENTEROCOUS, ENTEROCOCCUS SPP, Entrococous, Enterococcus Species, etc.
    #     Uses negative lookahead to avoid matching Enterobacter ---
    (re.compile(
        r'(?:\benteroc(?!bacter)[a-z]*\b|\bentr[oe]coc[a-z]*\b)',
        re.IGNORECASE,
    ), "Enterococcus spp."),
    # --- Hemolytic / Haemolytic Streptococcus ---
    (re.compile(r'h[ae]e?molytic\s+streptoc+oc+(?:ci|us|cus)?(?:\s+sp+\.?|\s+species?)?', re.IGNORECASE), "Hemolytic Streptococcus spp."),
    # --- Citrobacter ---
    (re.compile(r'citrobacter(?:\s+sp+\.?|\s+species?)?', re.IGNORECASE), "Citrobacter spp."),
    # --- Streptococcus pneumoniae / pneumo ---
    (re.compile(r'streptoc+oc+(?:us|cus)?\s+pneumon+(?:iae?)?', re.IGNORECASE), "Streptococcus pneumoniae"),
    # --- Staphylococcus aureus ---
    (re.compile(r'staphyloc+oc+(?:us|cus)?\s+aureus', re.IGNORECASE), "Staphylococcus aureus"),
    # --- Coagulase-negative Staphylococci ---
    (re.compile(r'(?:coagulase[-\s]*neg(?:ative)?\s+)?staph(?:yloc+oc+(?:us|ci|cus)?)?(?:\s+(?:coagulase[-\s]*neg(?:ative)?|cons|epidermidis|haemolyticus))?', re.IGNORECASE), None),  # pass-through; handle below if needed
    # --- Klebsiella ---
    (re.compile(r'klebsiell+a(?:\s+sp+\.?|\s+species?|\s+pneumon+(?:iae?)?)?', re.IGNORECASE), "Klebsiella spp."),
    # --- Pseudomonas ---
    (re.compile(r'pseudomonas(?:\s+sp+\.?|\s+species?|\s+aeruginosa)?', re.IGNORECASE), "Pseudomonas spp."),
    # --- E. coli ---
    (re.compile(r'(?:escherichia\s+coli|e\.\s*coli)', re.IGNORECASE), "Escherichia coli"),
    # --- Acinetobacter ---
    (re.compile(r'acinetobacter(?:\s+sp+\.?|\s+species?|\s+baumannii)?', re.IGNORECASE), "Acinetobacter spp."),
    # --- Group B Streptococcus / GBS ---
    (re.compile(r'(?:group\s+b\s+streptoc+oc+(?:us|cus)?|streptoc+oc+(?:us|cus)?\s+agalactiae|gbs)', re.IGNORECASE), "Group B Streptococcus"),
    # --- Salmonella ---
    (re.compile(r'salmonell+a(?:\s+sp+\.?|\s+species?|\s+typhi?)?', re.IGNORECASE), "Salmonella spp."),
    # --- Serratia ---
    (re.compile(r'serratia(?:\s+sp+\.?|\s+species?)?', re.IGNORECASE), "Serratia spp."),
    # --- Proteus ---
    (re.compile(r'proteus(?:\s+sp+\.?|\s+species?|\s+mirabilis)?', re.IGNORECASE), "Proteus spp."),
    # --- Candida / fungal ---
    (re.compile(r'candida(?:\s+sp+\.?|\s+species?|\s+albicans)?', re.IGNORECASE), "Candida spp."),
    # --- Enterobacter ---
    (re.compile(r'enterobacter(?:\s+sp+\.?|\s+species?|\s+cloacae)?', re.IGNORECASE), "Enterobacter spp."),
    # --- Haemophilus ---
    (re.compile(r'ha?emoph?ilus(?:\s+sp+\.?|\s+species?|\s+influenzae)?', re.IGNORECASE), "Haemophilus spp."),
    # --- Listeria ---
    (re.compile(r'listeria(?:\s+sp+\.?|\s+species?|\s+monocytogenes)?', re.IGNORECASE), "Listeria spp."),
    # --- Stenotrophomonas ---
    (re.compile(r'stenotrophomonas(?:\s+sp+\.?|\s+species?)?', re.IGNORECASE), "Stenotrophomonas spp."),
    # --- Burkholderia ---
    (re.compile(r'burkholderia(?:\s+sp+\.?|\s+species?)?', re.IGNORECASE), "Burkholderia spp."),
    # --- Neisseria ---
    (re.compile(r'neisseria(?:\s+sp+\.?|\s+species?|\s+meningitidis|\s+gonorrhoeae)?', re.IGNORECASE), "Neisseria spp."),
    # --- Moraxella ---
    (re.compile(r'moraxella(?:\s+sp+\.?|\s+species?)?', re.IGNORECASE), "Moraxella spp."),
]


def _standardize_organism_name(raw_name: str) -> str | None:
    """Normalize a free-text organism name to a canonical representation.

    Tries regex rules in order using re.search (partial match within the string).
    Returns the canonical name for the first match found.
    Returns *None* when a recognised non-organism pattern (e.g. "Not recorded",
    "Not readable") is matched — callers should treat None as "exclude".
    If no rule matches, the original stripped text is returned so unknown
    organisms still appear in charts rather than being silently dropped.
    """
    if not isinstance(raw_name, str):
        text = str(raw_name).strip()
        return text if text else None
    text = raw_name.strip()
    if not text:
        return None
    for pattern, canonical in _ORGANISM_NORM_RULES:
        if pattern.search(text):
            return canonical
    # No rule matched — return as-is so the organism still shows in charts
    return text


def get_organism_colors(organisms):
    """Return a dict mapping every organism name to a unique color.

    Colors from ORGANISM_COLORS are used for known organisms.  Any organism
    not in ORGANISM_COLORS receives the next available color from
    _EXTRA_COLORS (cycling if necessary), guaranteeing no two organisms
    share a color within the same chart render.
    """
    color_map = {}
    extra_iter = iter(_EXTRA_COLORS)
    used_colors = set(ORGANISM_COLORS.values())
    extra_pool = list(_EXTRA_COLORS)
    extra_idx = 0

    for org in organisms:
        if org in ORGANISM_COLORS:
            color_map[org] = ORGANISM_COLORS[org]
        else:
            # Find the next extra color not already used
            assigned = False
            while extra_idx < len(extra_pool):
                candidate = extra_pool[extra_idx]
                extra_idx += 1
                if candidate not in used_colors:
                    color_map[org] = candidate
                    used_colors.add(candidate)
                    assigned = True
                    break
            if not assigned:
                # All extra colors exhausted — cycle with a generated shade
                import hashlib
                h = int(hashlib.md5(org.encode()).hexdigest()[:6], 16)
                color_map[org] = "#{:06X}".format(h & 0xFFFFFF)
    return color_map


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
    columns. When code == 99 the free-text field is read and normalised via
    _standardize_organism_name before being returned, so spelling variants are
    collapsed into one canonical label.
    Excludes code 11 (Not indicated).
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

    # Normalise to a plain string code
    if pd.isna(org_code):
        return None
    org_code = str(org_code).strip()
    # Convert numeric floats like "11.0" -> "11"
    if org_code.endswith(".0"):
        org_code = org_code[:-2]

    # Skip if "Not indicated"
    if org_code == "11":
        return None

    # Handle special "Other" case — read and normalise the free-text field
    if org_code == "99":
        other_col = None
        for col in row.index:
            if col.startswith(OTHER_ORGANISM_PREFIX):
                other_col = col
                break
        if other_col is not None and pd.notna(row.get(other_col)):
            raw_text = str(row[other_col]).strip()
            if raw_text:
                return _standardize_organism_name(raw_text)
        # Fall back to "Other" label
        return "Other"

    # Handle numeric codes (1-9)
    if org_code.isdigit():
        code = int(org_code)
        if code in ORGANISM_NAMES:
            return ORGANISM_NAMES[code]

    # Non-numeric, non-empty text in the primary field — normalise it too
    if org_code and not org_code.isdigit():
        return _standardize_organism_name(org_code)

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


def compute_microorganism_trend_data(df, facility_uids=None, period_col="period_display"):
    """Monthly Trend of Microorganisms Identified in Positive Blood Cultures.

    Tracks how the prevalence of ALL identified microorganisms changes over time.
    Shows every organism present in the distribution — no top-N cap.

    Chart Type: Multi-Series Run Chart

    Inclusion: Only include blood_culture = 2 (positive cultures)
    Exclusion: code 11 (Not indicated)

    Numerator: Number of positive cultures with organism in period
    Denominator: Total positive organisms identified in same period
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

    # Use ALL organisms — no top-N cap — so the trend matches the distribution chart exactly
    all_organisms = working_df["_organism"].unique().tolist()

    # Build trend data
    trend_data = []
    for period in periods:
        period_df = working_df[working_df[period_col] == period]
        if period_df.empty:
            continue
        period_total = len(period_df)
        for org in all_organisms:
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

    # Build a consistent, unique color per organism
    color_map = get_organism_colors([item[0] for item in sorted_items])

    fig = go.Figure(go.Bar(
        x=[item[1] for item in sorted_items],
        y=[item[0] for item in sorted_items],
        orientation="h",
        marker_color=[color_map[item[0]] for item in sorted_items],
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Total: %{customdata[0]}<br>Percentage: %{customdata[1]}%<extra></extra>",
        customdata=[[total_organisms, round((item[1] / total_organisms * 100), 1)] for item in sorted_items],
    ))

    fig.update_layout(
        title="Microorganism Distribution in Positive Blood Cultures",
        height=max(350, 40 * len(sorted_items) + 80),
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

    # Build a consistent, unique color per organism (same palette as distribution chart)
    color_map = get_organism_colors(all_organisms)

    fig = go.Figure()
    for org in all_organisms:
        org_df = trend_df[trend_df["organism"] == org]
        if org_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=org_df["period"], y=org_df["percentage"],
            name=org,
            line=dict(color=color_map[org], width=2),
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate="<b>" + org + "</b><br>Month: %{x}<br>Count: %{customdata[0]}<br>Total: %{customdata[1]}<br>Percentage: %{y:.1f}%<extra></extra>",
            customdata=np.column_stack((org_df["count"], org_df["total"])),
        ))

    fig.update_layout(
        title="Microorganism Trend Over Time",
        height=420,
        yaxis=dict(range=[0, 105], dtick=25, title="Percentage (%)"),
        xaxis=dict(title=""),
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font_color=text_color, title_font_color=text_color,
        margin=dict(l=40, r=20, t=50, b=100),
        legend=dict(
            orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5,
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
