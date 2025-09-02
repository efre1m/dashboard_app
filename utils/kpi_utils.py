import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt

def auto_text_color(bg):
    """Return black or white text depending on background brightness"""
    bg = bg.lstrip("#")
    try:
        r, g, b = int(bg[0:2],16), int(bg[2:4],16), int(bg[4:6],16)
        brightness = (r*299 + g*587 + b*114)/1000
        return "#000000" if brightness > 150 else "#ffffff"
    except Exception:
        return "#000000"

# KPI Constants
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
    delivery_events = df[(df["dataElement_uid"]=="lphtwP2ViZU") & (df["value"].notna())]
    if delivery_events.empty:
        delivery_events = df[(df["dataElement_uid"]==DELIVERY_MODE_UID) & (df["value"].notna())]
    return delivery_events["tei_id"].nunique()

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
    """Calculate Early PNC Coverage (PNC within 48 hours)"""
    if df is None or df.empty:
        return 0.0, 0, 0
    
    total_deliveries = compute_total_deliveries(df)
    early_pnc = df[
        (df["dataElement_uid"] == PNC_TIMING_UID) & 
        (df["value"].isin(PNC_EARLY_VALUES))
    ]["tei_id"].nunique()
    
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

def render_gauge_chart(value, title, bg_color, text_color):
    """Render a gauge chart for the latest value"""
    if "PNC Coverage" in title or "IPPCAR" in title:
        min_val, max_val = 0, 100
        threshold_vals = [50, 80]
        colorscale = [[0, "red"], [0.5, "yellow"], [1.0, "green"]]
    elif "Stillbirth Rate" in title:
        min_val, max_val = 0, 50
        threshold_vals = [15, 5]
        colorscale = [[0, "green"], [0.3, "yellow"], [1.0, "red"]]
    else:
        min_val, max_val = 0, 100
        threshold_vals = [50, 80]
        colorscale = [[0, "red"], [0.5, "yellow"], [1.0, "green"]]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': 0, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': text_color},
            'bar': {'color': "darkblue"},
            'bgcolor': bg_color,
            'borderwidth': 2,
            'bordercolor': text_color,
            'steps': [
                {'range': [min_val, threshold_vals[0]], 'color': 'lightcoral'},
                {'range': [threshold_vals[0], threshold_vals[1]], 'color': 'lightyellow'},
                {'range': [threshold_vals[1], max_val], 'color': 'lightgreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value}}))
    
    fig.update_layout(
        paper_bgcolor=bg_color,
        font={'color': text_color, 'family': "Arial"},
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_trend_chart(df, period_col, value_col, title, bg_color, text_color=None):
    """Render line or bar chart with trend symbols"""
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    chart_options = get_chart_options(title)
    chart_type = st.radio(
        f"📊 Chart type for {title}",
        options=chart_options,
        index=0,
        horizontal=True,
        key=f"chart_type_{title}"
    ).lower()

    df = df.copy()
    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    if chart_type == "gauge":
        if not df.empty:
            latest_value = df[value_col].iloc[-1] if len(df) > 0 else 0
            render_gauge_chart(latest_value, title, bg_color, text_color)
        return

    is_categorical = not all(isinstance(x, (dt.date, dt.datetime)) for x in df[period_col]) if not df.empty else True

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
        xaxis_title=period_col,
        yaxis_title=value_col,
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

    if len(df) > 1:
        last_value = df[value_col].iloc[-1]
        prev_value = df[value_col].iloc[-2]
        if last_value > prev_value:
            trend_symbol = "▲"
            trend_class = "trend-up"
        elif last_value < prev_value:
            trend_symbol = "▼"
            trend_class = "trend-down"
        else:
            trend_symbol = "–"
            trend_class = "trend-neutral"
        st.markdown(
            f'<p style="font-size:1.2rem; font-weight:600;">Latest Value: {last_value:.1f} <span class="{trend_class}">{trend_symbol}</span></p>',
            unsafe_allow_html=True
        )

    st.subheader(f"📋 {title} Summary Table")
    styled_table = df.rename(columns={value_col: "Value"}).style.set_table_attributes('class="summary-table"')
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

def get_chart_options(title):
    """Get appropriate chart options based on metric type"""
    if "IPPCAR" in title:
        return ["Line", "Bar"]
    elif "PNC Coverage" in title:
        return ["Line", "Gauge"]
    elif "Stillbirth Rate" in title:
        return ["Line", "Bar"]
    else:
        return ["Line", "Bar", "Gauge"]