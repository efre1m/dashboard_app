import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt

# ---------------- Styling Helpers ----------------
def auto_text_color(bg):
    """Return black or white text depending on background brightness"""
    bg = bg.lstrip("#")
    try:
        r, g, b = int(bg[0:2],16), int(bg[2:4],16), int(bg[4:6],16)
        brightness = (r*299 + g*587 + b*114)/1000
        return "#000000" if brightness > 150 else "#ffffff"
    except Exception:
        return "#000000"

# ---------------- KPI Calculations ----------------
FP_ACCEPTANCE_UID = "Q1p7CxWGUoi"
FP_ACCEPTED_VALUES = {"sn2MGial4TT", "aB5By4ATx8M", "TAxj9iLvWQ0", "FyCtuLALNpY", "ejFYFZlmlwT"}
BIRTH_OUTCOME_UID = "wZig9cek3Gv"
ALIVE_CODE = "1"
STILLBIRTH_CODE = "2"
DELIVERY_MODE_UID = "z9wWxK7fw8W"
PNC_TIMING_UID = "z7Eb2yFLOBI"
PNC_EARLY_VALUES = {"tbiEfEybERM", "JVO0gl1FuqK"}

def compute_total_deliveries(df):
    if df is None or df.empty:
        return 0
    deliveries = df[(df["dataElement_uid"]=="lphtwP2ViZU") & (df["value"].notna())]
    if deliveries.empty:
        deliveries = df[(df["dataElement_uid"]==DELIVERY_MODE_UID) & (df["value"].notna())]
    return deliveries["tei_id"].nunique()

def compute_fp_acceptance(df):
    if df is None or df.empty:
        return 0
    return df[(df["dataElement_uid"]==FP_ACCEPTANCE_UID) & (df["value"].isin(FP_ACCEPTED_VALUES))]["tei_id"].nunique()

def compute_stillbirth_rate(df):
    if df is None or df.empty:
        return 0.0, 0, 0
    births = df[(df["dataElement_uid"]==BIRTH_OUTCOME_UID) & (df["value"].isin([ALIVE_CODE, STILLBIRTH_CODE]))]
    total_births = births["tei_id"].nunique()
    stillbirths = births[births["value"]==STILLBIRTH_CODE]["tei_id"].nunique()
    rate = (stillbirths / total_births * 1000) if total_births > 0 else 0.0
    return rate, stillbirths, total_births

def compute_early_pnc_coverage(df):
    if df is None or df.empty:
        return 0.0, 0, 0
    total_deliveries = compute_total_deliveries(df)
    early_pnc = df[(df["dataElement_uid"]==PNC_TIMING_UID) & (df["value"].isin(PNC_EARLY_VALUES))]["tei_id"].nunique()
    coverage = (early_pnc / total_deliveries * 100) if total_deliveries > 0 else 0.0
    return coverage, early_pnc, total_deliveries

def compute_kpis(df):
    """Compute all KPIs using a single dataframe"""
    total_deliveries = compute_total_deliveries(df)
    fp_acceptance = compute_fp_acceptance(df)
    ippcar = (fp_acceptance / total_deliveries * 100) if total_deliveries > 0 else 0.0
    stillbirth_rate, stillbirths, total_births = compute_stillbirth_rate(df)
    pnc_coverage, early_pnc, total_deliveries_pnc = compute_early_pnc_coverage(df)

    return {
        "total_deliveries": int(total_deliveries),
        "fp_acceptance": int(fp_acceptance),
        "ippcar": float(ippcar),
        "stillbirth_rate": float(stillbirth_rate),
        "stillbirths": int(stillbirths),
        "total_births": int(total_births),
        "pnc_coverage": float(pnc_coverage),
        "early_pnc": int(early_pnc),
        "total_deliveries_pnc": int(total_deliveries_pnc)
    }

# ---------------- Trend Symbol ----------------
def compute_trend_symbol(values):
    """Compute trend symbol and CSS class for a list of numeric values"""
    if not values or len(values) < 2:
        return "–", "trend-neutral"
    last, prev = values[-1], values[-2]
    if last > prev:
        return "▲", "trend-up"
    elif last < prev:
        return "▼", "trend-down"
    else:
        return "–", "trend-neutral"

# ---------------- Chart Rendering ----------------
def render_trend_chart(df, period_col, value_col, title, bg_color, text_color=None):
    """Render line or bar chart with trend symbols"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    is_categorical = not all(isinstance(x, (dt.date, dt.datetime)) for x in df[period_col]) if not df.empty else True

    # Choose chart type
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=["Line", "Bar"],
        index=0,
        horizontal=True,
        key=f"chart_type_{title}"
    ).lower()

    if chart_type == "line":
        fig = px.line(df, x=period_col, y=value_col, markers=True, line_shape="linear", title=title, height=400)
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
    else:
        fig = px.bar(df, x=period_col, y=value_col, text=value_col, labels={value_col: title}, title=title, height=400)

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis=dict(
            type='category' if is_categorical else None,
            tickangle=-45 if is_categorical else 0,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            rangemode='tozero',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.5)'
        )
    )

    if "Rate" in title or "%" in title:
        fig.update_layout(yaxis_tickformat=".1f")
    if any(keyword in title for keyword in ["Deliveries", "Acceptance"]):
        fig.update_layout(yaxis_tickformat=",")

    fig.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y}<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)
