import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Styling Helpers ----------------
def auto_text_color(bg):
    bg = bg.lstrip("#")
    try:
        r, g, b = int(bg[0:2],16), int(bg[2:4],16), int(bg[4:6],16)
        brightness = (r*299 + g*587 + b*114)/1000
        return "#000000" if brightness > 150 else "#ffffff"
    except Exception:
        return "#000000"

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
    if delivery_df is None or delivery_df.empty or "dataElement_uid" not in delivery_df.columns:
        return 0
    return delivery_df[
        (delivery_df["dataElement_uid"]=="lphtwP2ViZU") & (delivery_df["value"].notna())
    ]["tei_id"].nunique()

def compute_maternal_complications(delivery_df):
    if delivery_df is None or delivery_df.empty:
        return 0, pd.DataFrame(columns=["Complication","Count"])
    counts = {name: 0 for name in COMPLICATION_MAP.values()}
    for _, row in delivery_df.iterrows():
        if row.get("dataElement_uid") == "CJiTafFo0TS":
            comp = COMPLICATION_MAP.get(str(row.get("value")))
            if comp:
                counts[comp] += 1
    df = pd.DataFrame(list(counts.items()), columns=["Complication","Count"])
    return int(df["Count"].sum()), df

def compute_maternal_deaths(delivery_df):
    if delivery_df is None or delivery_df.empty:
        return 0
    deaths_df = delivery_df[
        (delivery_df.get("dataElement_uid")=="TjQOcW6tm8k") & (delivery_df.get("value")=="4")
    ]
    return int(deaths_df["tei_id"].nunique()) if not deaths_df.empty else 0

def compute_kpis(enrollments_df, delivery_df):
    total_admissions = int(enrollments_df["tei_id"].nunique()) if (enrollments_df is not None and not enrollments_df.empty and "tei_id" in enrollments_df.columns) else 0
    active_count = int(enrollments_df[enrollments_df.get("status")=="ACTIVE"]["tei_id"].nunique()) if (enrollments_df is not None and not enrollments_df.empty and "status" in enrollments_df.columns and "tei_id" in enrollments_df.columns) else 0
    completed_count = int(enrollments_df[enrollments_df.get("status")=="COMPLETED"]["tei_id"].nunique()) if (enrollments_df is not None and not enrollments_df.empty and "status" in enrollments_df.columns and "tei_id" in enrollments_df.columns) else 0

    total_deliveries = compute_total_deliveries(delivery_df)

    if delivery_df is None or delivery_df.empty:
        instrumental_deliveries = 0
    else:
        instrumental_deliveries = int(
            delivery_df[
                (delivery_df.get("dataElement_uid")=="Y4AwETxqTeK") & (delivery_df["value"].astype(str).str.lower()=="true")
            ]["tei_id"].nunique()
        )
    idr = (instrumental_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0

    if delivery_df is None or delivery_df.empty:
        csection_deliveries = 0
    else:
        csection_deliveries = int(
            delivery_df[
                (delivery_df.get("dataElement_uid")=="lphtwP2ViZU") & (delivery_df.get("value")=="XxhH16ujEgj")
            ]["tei_id"].nunique()
        )
    csr = (csection_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0

    maternal_complications_total, maternal_complications_df = compute_maternal_complications(delivery_df)
    maternal_deaths = compute_maternal_deaths(delivery_df)

    return {
        "total_admissions": total_admissions,
        "active_count": active_count,
        "completed_count": completed_count,
        "total_deliveries": total_deliveries,
        "instrumental_deliveries": instrumental_deliveries,
        "idr": float(idr),
        "csection_deliveries": csection_deliveries,
        "csr": float(csr),
        "maternal_complications_total": int(maternal_complications_total),
        "maternal_complications_df": maternal_complications_df,
        "maternal_deaths": int(maternal_deaths)
    }

# ---------------- General Graph Rendering ----------------
def render_trend_chart(df, period_col, value_col, title, bg_color, text_color=None, chart_type="line"):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    # Case 1: no dataframe or missing period col
    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    df = df.copy()
    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    # ✅ Removed the "sum == 0" check – we still plot zeros
    if df.empty:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return

    # Plot
    if chart_type == "line":
        fig = px.line(df, x=period_col, y=value_col, markers=True, line_shape="linear", title=title, height=400)
        fig.update_traces(line=dict(width=4), marker=dict(size=8))
    else:
        fig = px.bar(df, x=period_col, y=value_col, text=value_col, labels={value_col: title}, title=title, height=400)

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font_color=text_color,
        xaxis_title=period_col,
        yaxis_title=value_col
    )

    for trace in fig.data:
        try:
            trace.textfont.color = text_color
        except Exception:
            pass

    st.plotly_chart(fig, use_container_width=True)

    # Always show table (even with zeros)
    st.subheader(f"📋 {title} Summary Table")
    styled_table = df.rename(columns={value_col: "Value"}).style.set_table_attributes('class="summary-table"')
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)

# ---------------- Maternal Complications Chart ----------------
def render_maternal_complications_chart(delivery_df, period_col, bg_color, text_color=None):
    if text_color is None:
        text_color = auto_text_color(bg_color)

    if delivery_df is None or delivery_df.empty or period_col not in delivery_df.columns:
        st.subheader("Maternal Complications by Type")
        st.info("⚠️ No maternal complication data found.")
        return

    mc_df = delivery_df[delivery_df.get("dataElement_uid")=="CJiTafFo0TS"].copy()
    if mc_df.empty:
        st.subheader("Maternal Complications by Type")
        st.info("⚠️ No maternal complication data found.")
        return

    mc_df["Complication"] = mc_df["value"].astype(str).map(COMPLICATION_MAP)
    mc_df = mc_df.dropna(subset=["Complication"])

    # ✅ Ensure all complication types are represented, even with 0
    grouped = mc_df.groupby([period_col, "Complication"])["tei_id"].nunique().reset_index(name="Count")
    pivot_df = grouped.pivot(index=period_col, columns="Complication", values="Count").fillna(0)

    if pivot_df.empty:
        st.subheader("Maternal Complications by Type")
        st.info("⚠️ No maternal complication data found.")
        return

    pivot_df = pivot_df[pivot_df.sum().sort_values(ascending=False).index]

    fig = px.bar(
        pivot_df.reset_index(),
        x=period_col,
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
        try:
            trace.textfont.color = text_color
        except Exception:
            pass

    st.plotly_chart(fig, use_container_width=True)

    # Summary (with zeros for missing complications)
    summary = pivot_df.sum().reset_index()
    summary.columns = ["Complication","Count"]
    summary = summary.sort_values("Count", ascending=False)

    st.subheader("📋 Maternal Complications Summary")
    styled_table = summary.style.set_table_attributes('class="summary-table"')
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)