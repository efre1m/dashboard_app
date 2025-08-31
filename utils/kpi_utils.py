import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
# ---------------- Styling ----------------
def auto_text_color(bg):
    bg = bg.lstrip("#")
    r, g, b = int(bg[0:2],16), int(bg[2:4],16), int(bg[4:6],16)
    brightness = (r*299 + g*587 + b*114)/1000
    return "#000000" if brightness > 150 else "#ffffff"

# ---------------- Maternal Complications Map ----------------
COMPLICATION_MAP = {
    "1": "Eclampsia",
    "2": "Postpartum Hemorrhage (PPH)",
    "3": "Antepartum Hemorrhage (APH)",
    "4": "PROM / Sepsis",
    "5": "Ruptured Uterus",
    "6": "Prolonged Labor",
    "7": "Repaired Uterus",
    "8": "Hysterectomy performed",
    "9": "Obstructed labor",
    "10": "Other maternal complication"
}

# ---------------- KPI Calculations ----------------
def compute_total_deliveries(delivery_df):
    if delivery_df.empty or "dataElement_uid" not in delivery_df.columns:
        return 0
    return delivery_df[
        (delivery_df["dataElement_uid"]=="lphtwP2ViZU") & (delivery_df["value"].notna())
    ]["tei_id"].nunique()

def compute_maternal_complications(delivery_df):
    if delivery_df.empty:
        return 0, pd.DataFrame(columns=["Complication","Count"])
    counts = {name: 0 for name in COMPLICATION_MAP.values()}
    for _, row in delivery_df.iterrows():
        if row.get("dataElement_uid") == "CJiTafFo0TS":  # Obstetric condition
            comp = COMPLICATION_MAP.get(str(row.get("value")))
            if comp:
                counts[comp] += 1
    df = pd.DataFrame(list(counts.items()), columns=["Complication","Count"])
    return df["Count"].sum(), df

def compute_maternal_deaths(delivery_df):
    """Count maternal deaths based on Condition of Discharge = 'Dead' (code 4)"""
    if delivery_df.empty:
        return 0
    deaths_df = delivery_df[
        (delivery_df["dataElement_uid"]=="TjQOcW6tm8k") &
        (delivery_df["value"]=="4")
    ]
    return deaths_df["tei_id"].nunique()

def compute_kpis(enrollments_df, delivery_df):
    total_admissions = enrollments_df["tei_id"].nunique() if not enrollments_df.empty else 0
    active_count = enrollments_df[enrollments_df["status"]=="ACTIVE"]["tei_id"].nunique() if not enrollments_df.empty else 0
    completed_count = enrollments_df[enrollments_df["status"]=="COMPLETED"]["tei_id"].nunique() if not enrollments_df.empty else 0

    total_deliveries = compute_total_deliveries(delivery_df)

    instrumental_deliveries = delivery_df[
        (delivery_df["dataElement_uid"]=="Y4AwETxqTeK") &
        (delivery_df["value"].astype(str).str.lower()=="true")
    ]["tei_id"].nunique() if not delivery_df.empty else 0
    idr = (instrumental_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0

    csection_deliveries = delivery_df[
        (delivery_df["dataElement_uid"]=="lphtwP2ViZU") & (delivery_df["value"]=="XxhH16ujEgj")
    ]["tei_id"].nunique() if not delivery_df.empty else 0
    csr = (csection_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0

    maternal_complications_total, maternal_complications_df = compute_maternal_complications(delivery_df)

    maternal_deaths = compute_maternal_deaths(delivery_df)

    return {
        "total_admissions": total_admissions,
        "active_count": active_count,
        "completed_count": completed_count,
        "total_deliveries": total_deliveries,
        "instrumental_deliveries": instrumental_deliveries,
        "idr": idr,
        "csection_deliveries": csection_deliveries,
        "csr": csr,
        "maternal_complications_total": maternal_complications_total,
        "maternal_complications_df": maternal_complications_df,
        "maternal_deaths": maternal_deaths
    }

# ---------------- General Graph Rendering ----------------
def render_trend_chart(df, period_col, value_col, title, bg_color, text_color=None, chart_type="line"):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # If dataframe is empty, create empty chart with proper layout
    if df.empty:
        # Create empty figure with proper layout
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title=period_col,
            yaxis_title=value_col,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color,
            title_font_color=text_color,
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader(f"📋 {title} Summary Table")
        st.info("No data available for this period.")
        return

    if chart_type=="line":
        fig = px.line(df, x=period_col, y=value_col, markers=True, line_shape="linear", title=title, height=400)
        fig.update_traces(line=dict(width=4), marker=dict(size=10))
    else:
        fig = px.bar(df, x=period_col, y=value_col, text=value_col, labels={value_col: title}, title=title, height=400)
        fig.update_traces(marker_color="purple")

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title=period_col,
        yaxis_title=value_col
    )

    for trace in fig.data:
        trace.textfont.color = text_color

    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"📋 {title} Summary Table")
    styled_table = df.rename(columns={value_col:"Value"}).style.set_table_attributes('class="summary-table"')
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

# ---------------- Maternal Complications Chart ----------------
def render_maternal_complications_chart(delivery_df, period_col, bg_color, text_color=None):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Handle empty delivery dataframe
    if delivery_df.empty:
        # Create empty figure with proper layout
        fig = go.Figure()
        fig.update_layout(
            title="Maternal Complications by Type",
            xaxis_title="Period",
            yaxis_title="Number of Cases",
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color,
            title_font_color=text_color,
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📋 Maternal Complications Summary")
        st.info("No maternal complication data found.")
        return

    mc_df = delivery_df[delivery_df["dataElement_uid"]=="CJiTafFo0TS"].copy()
    mc_df["Complication"] = mc_df["value"].astype(str).map(COMPLICATION_MAP)
    mc_df = mc_df.dropna(subset=["Complication"])
    
    if mc_df.empty:
        # Create empty figure with proper layout
        fig = go.Figure()
        fig.update_layout(
            title="Maternal Complications by Type",
            xaxis_title="Period",
            yaxis_title="Number of Cases",
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color,
            title_font_color=text_color,
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("📋 Maternal Complications Summary")
        st.info("No maternal complication data found.")
        return

    grouped = mc_df.groupby([period_col, "Complication"])["tei_id"].nunique().reset_index(name="Count")
    pivot_df = grouped.pivot(index=period_col, columns="Complication", values="Count").fillna(0)
    pivot_df = pivot_df[pivot_df.sum().sort_values(ascending=False).index]

    fig = px.bar(
        pivot_df,
        x=pivot_df.index,
        y=pivot_df.columns,
        text_auto=True,
        title="Maternal Complications by Type",
        height=450
    )

    fig.update_layout(
        barmode='stack',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        xaxis_title="Period",
        yaxis_title="Number of Cases",
        legend_title_text="Complication Type",
        legend=dict(font=dict(color=text_color))
    )

    for trace in fig.data:
        trace.textfont.color = text_color

    st.plotly_chart(fig, use_container_width=True)

    summary = mc_df.groupby("Complication")["tei_id"].nunique().reset_index(name="Count")
    summary = summary.sort_values("Count", ascending=False)

    st.subheader("📋 Maternal Complications Summary")
    styled_table = summary.style.set_table_attributes('class="summary-table"')
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)