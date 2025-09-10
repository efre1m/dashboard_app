import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import streamlit as st

# ---------------- Utility ----------------
def auto_text_color(bg):
    """Return black or white text depending on background brightness"""
    bg = bg.lstrip("#")
    try:
        r, g, b = int(bg[0:2],16), int(bg[2:4],16), int(bg[4:6],16)
        brightness = (r*299 + g*587 + b*114)/1000
        return "#000000" if brightness > 150 else "#ffffff"
    except Exception:
        return "#000000"

# ---------------- KPI Constants ----------------
FP_ACCEPTANCE_UID = "Q1p7CxWGUoi"
FP_ACCEPTED_VALUES = {"sn2MGial4TT", "aB5By4ATx8M", "TAxj9iLvWQ0", "FyCtuLALNpY", "ejFYFZlmlwT"}

BIRTH_OUTCOME_UID = "wZig9cek3Gv"
ALIVE_CODE = "1"
STILLBIRTH_CODE = "2"

DELIVERY_MODE_UID = "z9wWxK7fw8W"
DELIVERY_TYPE_UID = "lphtwP2ViZU"
SVD_CODE = "1"
CSECTION_CODE = "2"

PNC_TIMING_UID = "z7Eb2yFLOBI"
PNC_EARLY_VALUES = {"tbiEfEybERM", "JVO0gl1FuqK"}

CONDITION_OF_DISCHARGE_UID = "TjQOcW6tm8k"
DEAD_CODE = "4"

# ---------------- KPI Computation Functions ----------------
def compute_total_deliveries(df, facility_uids=None):
    if df is None or df.empty:
        return 0
    
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
    
    deliveries = df[(df["dataElement_uid"]==DELIVERY_TYPE_UID) & df["value"].notna()]
    if deliveries.empty:
        deliveries = df[(df["dataElement_uid"]==DELIVERY_MODE_UID) & df["value"].notna()]
    return deliveries["tei_id"].nunique()

def compute_fp_acceptance(df, facility_uids=None):
    if df is None or df.empty:
        return 0
    
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
        
    return df[(df["dataElement_uid"]==FP_ACCEPTANCE_UID) & (df["value"].isin(FP_ACCEPTED_VALUES))]["tei_id"].nunique()

def compute_stillbirth_rate(df, facility_uids=None):
    if df is None or df.empty:
        return 0.0, 0, 0
    
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
        
    births = df[(df["dataElement_uid"]==BIRTH_OUTCOME_UID) & (df["value"].isin([ALIVE_CODE, STILLBIRTH_CODE]))]
    total_births = births["tei_id"].nunique()
    stillbirths = births[births["value"]==STILLBIRTH_CODE]["tei_id"].nunique()
    rate = (stillbirths / total_births * 1000) if total_births > 0 else 0.0
    return rate, stillbirths, total_births

def compute_early_pnc_coverage(df, facility_uids=None):
    if df is None or df.empty:
        return 0.0, 0, 0
    
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
        
    total_deliveries = compute_total_deliveries(df, facility_uids)
    early_pnc = df[(df["dataElement_uid"]==PNC_TIMING_UID) & (df["value"].isin(PNC_EARLY_VALUES))]["tei_id"].nunique()
    coverage = (early_pnc / total_deliveries * 100) if total_deliveries > 0 else 0.0
    return coverage, early_pnc, total_deliveries

def compute_maternal_death_rate(df, facility_uids=None):
    if df is None or df.empty:
        return 0.0, 0, 0
    
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
        
    dfx = df.copy()
    dfx["event_date"] = pd.to_datetime(dfx["event_date"], errors="coerce")
    dfx = dfx[dfx["event_date"].notna()]

    # Maternal deaths
    deaths_df = dfx[(dfx["dataElement_uid"]==CONDITION_OF_DISCHARGE_UID) & (dfx["value"]==DEAD_CODE)]
    maternal_deaths = deaths_df["tei_id"].nunique()

    # Live births (always compute)
    live_births = dfx[(dfx["dataElement_uid"]==BIRTH_OUTCOME_UID) & (dfx["value"]==ALIVE_CODE)]["tei_id"].nunique()

    # Compute rate (0 if no deaths)
    rate = (maternal_deaths / live_births * 100000) if live_births > 0 and maternal_deaths > 0 else 0.0
    
    return rate, maternal_deaths, live_births

def compute_csection_rate(df, facility_uids=None):
    if df is None or df.empty:
        return 0.0, 0, 0
    
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
        
    csection_deliveries = df[(df["dataElement_uid"]==DELIVERY_TYPE_UID) & (df["value"]==CSECTION_CODE)]["tei_id"].nunique()
    total_deliveries = df[(df["dataElement_uid"]==DELIVERY_TYPE_UID) & (df["value"].isin([SVD_CODE,CSECTION_CODE]))]["tei_id"].nunique()
    rate = (csection_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0
    return rate, csection_deliveries, total_deliveries

# ---------------- Master KPI Function ----------------
def compute_kpis(df, facility_uids=None):
    # Filter by facilities if specified
    if facility_uids:
        df = df[df["orgUnit"].isin(facility_uids)]
    
    total_deliveries = compute_total_deliveries(df, facility_uids)
    fp_acceptance = compute_fp_acceptance(df, facility_uids)
    ippcar = (fp_acceptance / total_deliveries * 100) if total_deliveries > 0 else 0.0
    stillbirth_rate, stillbirths, total_births = compute_stillbirth_rate(df, facility_uids)
    pnc_coverage, early_pnc, total_deliveries_pnc = compute_early_pnc_coverage(df, facility_uids)
    maternal_death_rate, maternal_deaths, live_births = compute_maternal_death_rate(df, facility_uids)
    csection_rate, csection_deliveries, total_deliveries_cs = compute_csection_rate(df, facility_uids)

    return {
        "total_deliveries": int(total_deliveries),
        "fp_acceptance": int(fp_acceptance),
        "ippcar": float(ippcar),
        "stillbirth_rate": float(stillbirth_rate),
        "stillbirths": int(stillbirths),
        "total_births": int(total_births),
        "pnc_coverage": float(pnc_coverage),
        "early_pnc": int(early_pnc),
        "total_deliveries_pnc": int(total_deliveries_pnc),
        "maternal_death_rate": float(maternal_death_rate),
        "maternal_deaths": int(maternal_deaths),
        "live_births": int(live_births),
        "csection_rate": float(csection_rate),
        "csection_deliveries": int(csection_deliveries),
        "total_deliveries_cs": int(total_deliveries_cs)
    }

# ---------------- Chart Functions ----------------
def render_gauge_chart(value, title, bg_color, text_color, numerator=None, denominator=None, numerator_name="Numerator", denominator_name="Denominator"):
    if "PNC Coverage" in title or "IPPCAR" in title or "C-Section Rate" in title:
        min_val, max_val = 0, 100
        steps = [{'range':[0,50],'color':'lightcoral'},{'range':[50,80],'color':'lightyellow'},{'range':[80,100],'color':'lightgreen'}]
    elif "Maternal Death Rate" in title:
        min_val,max_val = 0, max(500,value*1.2)
        steps = [{'range':[0,max_val*0.2],'color':'lightgreen'},{'range':[max_val*0.2,max_val*0.4],'color':'lightyellow'},{'range':[max_val*0.4,max_val],'color':'lightcoral'}]
    elif "Stillbirth Rate" in title:
        min_val,max_val = 0, max(50,value*1.2)
        steps = [{'range':[0,max_val*0.2],'color':'lightgreen'},{'range':[max_val*0.2,max_val*0.4],'color':'lightyellow'},{'range':[max_val*0.4,max_val],'color':'lightcoral'}]
    else: 
        return
    
    # Create custom hover text with numerator and denominator
    hover_text = f"Value: {value:.1f}"
    if numerator is not None and denominator is not None:
        hover_text += f"<br>{numerator_name}: {numerator}<br>{denominator_name}: {denominator}"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title,'font':{'size':20}},
        gauge={'axis':{'range':[min_val,max_val],'tickwidth':1,'tickcolor':text_color},
               'bar':{'color':'darkblue'},
               'bgcolor':bg_color,
               'borderwidth':2,'bordercolor':text_color,
               'steps':steps,
               'threshold':{'line':{'color':'red','width':4},'thickness':0.75,'value':value}}
    ))
    fig.update_layout(paper_bgcolor=bg_color,font={'color':text_color,'family':'Arial'},height=400)
    st.plotly_chart(fig,use_container_width=True)

def get_chart_options(title):
    if "PNC Coverage" in title or "IPPCAR" in title: 
        return ["Line","Gauge"]
    elif "Maternal Death Rate" in title: 
        return ["Line","Bar","Gauge"]
    elif "Stillbirth Rate" in title: 
        return ["Line","Bar"]
    elif "C-Section Rate" in title: 
        return ["Line","Bar","Gauge"]
    else: 
        return ["Line","Bar"]

def render_trend_chart(df, period_col, value_col, title, bg_color, text_color=None, facility_names=None, numerator_name="Numerator", denominator_name="Denominator", facility_uids=None):
    if text_color is None: 
        text_color = auto_text_color(bg_color)
    
    # Add facility name to title if provided
    if facility_names and "All Facilities" not in facility_names:
        if len(facility_names) == 1:
            title = f"{title} - {facility_names[0]}"
        else:
            title = f"{title} - {len(facility_names)} Facilities"
    
    if df is None or df.empty or period_col not in df.columns:
        st.subheader(title)
        st.info("⚠️ No data available for the selected period.")
        return
        
    chart_options = get_chart_options(title)
    
    # Create radio button with black text for the header using a proper label
    st.markdown(f'<p style="color:#000000; font-weight:700; font-size:16px;">📊 Chart type for {title}</p>', unsafe_allow_html=True)
    chart_type = st.radio(
        "chart_type_selector",
        options=chart_options, 
        index=0, 
        horizontal=True, 
        key=f"chart_type_{title}_{str(facility_uids)}",
        label_visibility="collapsed"
    ).lower()
    
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col],errors="coerce").fillna(0)
    
    if chart_type=="gauge":
        # Get the latest values for numerator and denominator
        latest_row = df.iloc[-1]
        numerator = latest_row.get(numerator_name, None)
        denominator = latest_row.get(denominator_name, None)
        render_gauge_chart(df[value_col].iloc[-1], title, bg_color, text_color, numerator, denominator, numerator_name, denominator_name)
        return
    is_categorical = not all(isinstance(x,(dt.date,dt.datetime)) for x in df[period_col]) if not df.empty else True
    
    # Create custom hover text with numerator and denominator if available
    hover_data = {}
    if numerator_name in df.columns and denominator_name in df.columns:
        hover_data = {numerator_name: True, denominator_name: True}
    
    if chart_type=="line":
        fig = px.line(df, x=period_col, y=value_col, markers=True, line_shape="linear", 
                     title=title, height=400, hover_data=hover_data)
        fig.update_traces(line=dict(width=3), marker=dict(size=7),
                         hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.1f}}<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>")
    else:
        fig = px.bar(df, x=period_col, y=value_col, text=value_col, labels={value_col:title}, 
                    title=title, height=400, hover_data=hover_data)
        fig.update_traces(hovertemplate=f"<b>%{{x}}</b><br>Value: %{{y:.1f}}<br>{numerator_name}: %{{customdata[0]}}<br>{denominator_name}: %{{customdata[1]}}<extra></extra>")
        
    fig.update_layout(
        paper_bgcolor=bg_color,plot_bgcolor=bg_color,font_color=text_color,title_font_color=text_color,
        xaxis_title=period_col,yaxis_title=value_col,
        xaxis=dict(type='category' if is_categorical else None,tickangle=-45 if is_categorical else 0,showgrid=True,gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(rangemode='tozero',showgrid=True,gridcolor='rgba(128,128,128,0.2)',zeroline=True,zerolinecolor='rgba(128,128,128,0.5)')
    )
    
    if "Rate" in title or "%" in title: 
        fig.update_layout(yaxis_tickformat=".1f")
    if any(k in title for k in ["Deliveries","Acceptance"]): 
        fig.update_layout(yaxis_tickformat=",")
        
    st.plotly_chart(fig,use_container_width=True)

    if len(df)>1:
        last_value = df[value_col].iloc[-1]
        prev_value = df[value_col].iloc[-2]
        trend_symbol = "▲" if last_value>prev_value else ("▼" if last_value<prev_value else "–")
        trend_class = "trend-up" if last_value>prev_value else ("trend-down" if last_value<prev_value else "trend-neutral")
        st.markdown(f'<p style="font-size:1.2rem;font-weight:600;">Latest Value: {last_value:.1f} <span class="{trend_class}">{trend_symbol}</span></p>',unsafe_allow_html=True)

    st.subheader(f"📋 {title} Summary Table")
    
    # Create a summary table with proper column names
    summary_df = df.copy().reset_index(drop=True)
    
    # Remove the index column and keep only relevant columns
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        summary_df = summary_df[[period_col, numerator_name, denominator_name, value_col]]
    else:
        summary_df = summary_df[[period_col, value_col]]
    
    # Calculate overall value using the same formula as individual periods
    if numerator_name in summary_df.columns and denominator_name in summary_df.columns:
        total_numerator = summary_df[numerator_name].sum()
        total_denominator = summary_df[denominator_name].sum()
        
        # Calculate overall value based on the KPI type
        if "IPPCAR" in title or "Coverage" in title or "C-Section Rate" in title:
            # Percentage-based KPIs
            overall_value = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
        elif "Stillbirth Rate" in title:
            # Rate per 1000 births
            overall_value = (total_numerator / total_denominator * 1000) if total_denominator > 0 else 0
        elif "Maternal Death Rate" in title:
            # Rate per 100,000 births
            overall_value = (total_numerator / total_denominator * 100000) if total_denominator > 0 else 0
        else:
            # Default to average if we don't recognize the KPI type
            overall_value = summary_df[value_col].mean() if not summary_df.empty else 0
        
        # Create overall row
        overall_row = pd.DataFrame({
            period_col: [f"Overall {title}"],
            numerator_name: [total_numerator],
            denominator_name: [total_denominator],
            value_col: [overall_value]
        })
        
        # Combine with original dataframe
        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    else:
        overall_value = summary_df[value_col].mean() if not summary_df.empty else 0
        overall_row = pd.DataFrame({
            period_col: [f"Overall {title}"],
            value_col: [overall_value]
        })
        summary_table = pd.concat([summary_df, overall_row], ignore_index=True)
    
    # Add row numbering starting from 1
    summary_table.insert(0, 'No', range(1, len(summary_table) + 1))
    
    # Format the table for display
    if numerator_name in summary_table.columns and denominator_name in summary_table.columns:
        styled_table = summary_table.style.format({
            value_col: "{:.1f}",
            numerator_name: "{:,.0f}",
            denominator_name: "{:,.0f}"
        }).set_table_attributes('class="summary-table"').hide(axis='index')
    else:
        styled_table = summary_table.style.format({
            value_col: "{:.1f}"
        }).set_table_attributes('class="summary-table"').hide(axis='index')
    
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)
    
    # Add download button for CSV
    csv = summary_table.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_summary.csv",
        mime="text/csv"
    )

def render_facility_comparison_chart(df, period_col, value_col, title, bg_color, text_color, facility_names, facility_uids, numerator_name, denominator_name):
    """Render a comparison chart showing each facility's performance over time"""
    
    # Group by period and facility to get values for each facility
    facility_comparison_data = []
    
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        # Filter data for this specific facility
        facility_df = df[df["orgUnit"] == facility_uid]
        
        if not facility_df.empty:
            # Calculate KPI for this facility for each period
            if "IPPCAR" in title:
                facility_period_data = facility_df.groupby(period_col, as_index=False).apply(
                    lambda x: pd.Series({
                        "value": (
                            x[(x["dataElement_uid"]=="Q1p7CxWGUoi") &
                              (x["value"].isin(["sn2MGial4TT","aB5By4ATx8M","TAxj9iLvWQ0",
                                                "FyCtuLALNpY","ejFYFZlmlwT"]))]["tei_id"].nunique()
                            / max(1, x[(x["dataElement_uid"]=="lphtwP2ViZU") & (x["value"].notna())]["tei_id"].nunique())
                        ) * 100
                    })
                ).reset_index(drop=True)
            elif "Stillbirth Rate" in title:
                facility_period_data = facility_df.groupby(period_col, as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, [facility_uid])["stillbirth_rate"]
                    })
                ).reset_index(drop=True)
            elif "PNC Coverage" in title:
                facility_period_data = facility_df.groupby(period_col, as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, [facility_uid])["pnc_coverage"]
                    })
                ).reset_index(drop=True)
            elif "Maternal Death Rate" in title:
                facility_period_data = facility_df.groupby(period_col, as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, [facility_uid])["maternal_death_rate"]
                    })
                ).reset_index(drop=True)
            elif "C-Section Rate" in title:
                facility_period_data = facility_df.groupby(period_col, as_index=False).apply(
                    lambda x: pd.Series({
                        "value": compute_kpis(x, [facility_uid])["csection_rate"]
                    })
                ).reset_index(drop=True)
            else:
                continue
            
            # Add facility name to the data
            facility_period_data["Facility"] = facility_name
            facility_comparison_data.append(facility_period_data)
    
    if not facility_comparison_data:
        st.info("⚠️ No data available for facility comparison.")
        return
    
    # Combine all facility data
    comparison_df = pd.concat(facility_comparison_data, ignore_index=True)
    
    # Create line chart with different colors for each facility
    fig = px.line(comparison_df, x=period_col, y="value", color="Facility", markers=True,
                 title=f"{title} - Facility Comparison", height=500,
                 hover_data={"Facility": True, "value": ":.1f"})
    
    fig.update_traces(line=dict(width=3), marker=dict(size=7),
                     hovertemplate="<b>%{x}</b><br>Facility: %{customdata[0]}<br>Value: %{y:.1f}<extra></extra>")
    
    fig.update_layout(
        paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color, title_font_color=text_color,
        xaxis_title=period_col, yaxis_title=value_col,
        xaxis=dict(tickangle=-45, showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(rangemode='tozero', showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(128,128,128,0.5)'),
        legend=dict(title="Facilities", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    if "Rate" in title or "%" in title: 
        fig.update_layout(yaxis_tickformat=".1f")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show facility comparison table
    st.subheader("📋 Facility Comparison Summary")
    
    # Create facility comparison table with numerator, denominator, and KPI value
    facility_table_data = []
    
    for facility_name, facility_uid in zip(facility_names, facility_uids):
        # Filter data for this specific facility
        facility_df = df[df["orgUnit"] == facility_uid]
        
        if not facility_df.empty:
            # Calculate KPI metrics for this facility
            kpi_data = compute_kpis(facility_df, [facility_uid])
            
            if "IPPCAR" in title:
                numerator = kpi_data["fp_acceptance"]
                denominator = kpi_data["total_deliveries"]
                kpi_value = kpi_data["ippcar"]
                numerator_label = "FP Accepted Clients"
                denominator_label = "Total Deliveries"
            elif "Stillbirth Rate" in title:
                numerator = kpi_data["stillbirths"]
                denominator = kpi_data["total_births"]
                kpi_value = kpi_data["stillbirth_rate"]
                numerator_label = "Stillbirths"
                denominator_label = "Total Births"
            elif "PNC Coverage" in title:
                numerator = kpi_data["early_pnc"]
                denominator = kpi_data["total_deliveries_pnc"]
                kpi_value = kpi_data["pnc_coverage"]
                numerator_label = "Early PNC Clients"
                denominator_label = "Total Deliveries"
            elif "Maternal Death Rate" in title:
                numerator = kpi_data["maternal_deaths"]
                denominator = kpi_data["live_births"]
                kpi_value = kpi_data["maternal_death_rate"]
                numerator_label = "Maternal Deaths"
                denominator_label = "Live Births"
            elif "C-Section Rate" in title:
                numerator = kpi_data["csection_deliveries"]
                denominator = kpi_data["total_deliveries_cs"]
                kpi_value = kpi_data["csection_rate"]
                numerator_label = "C-Section Deliveries"
                denominator_label = "Total Deliveries"
            else:
                continue
            
            facility_table_data.append({
                "Facility Name": facility_name,
                numerator_label: numerator,
                denominator_label: denominator,
                "KPI Value": kpi_value
            })
    
    if not facility_table_data:
        st.info("⚠️ No data available for facility comparison table.")
        return
    
    # Create DataFrame for the table
    facility_table_df = pd.DataFrame(facility_table_data)
    
    # Add overall row with aggregated values
    if "IPPCAR" in title:
        overall_numerator = sum(row[numerator_label] for row in facility_table_data)
        overall_denominator = sum(row[denominator_label] for row in facility_table_data)
        overall_value = (overall_numerator / overall_denominator * 100) if overall_denominator > 0 else 0
    elif "Stillbirth Rate" in title:
        overall_numerator = sum(row[numerator_label] for row in facility_table_data)
        overall_denominator = sum(row[denominator_label] for row in facility_table_data)
        overall_value = (overall_numerator / overall_denominator * 1000) if overall_denominator > 0 else 0
    elif "PNC Coverage" in title or "C-Section Rate" in title:
        overall_numerator = sum(row[numerator_label] for row in facility_table_data)
        overall_denominator = sum(row[denominator_label] for row in facility_table_data)
        overall_value = (overall_numerator / overall_denominator * 100) if overall_denominator > 0 else 0
    elif "Maternal Death Rate" in title:
        overall_numerator = sum(row[numerator_label] for row in facility_table_data)
        overall_denominator = sum(row[denominator_label] for row in facility_table_data)
        overall_value = (overall_numerator / overall_denominator * 100000) if overall_denominator > 0 else 0
    else:
        overall_numerator = sum(row[numerator_label] for row in facility_table_data)
        overall_denominator = sum(row[denominator_label] for row in facility_table_data)
        overall_value = sum(row["KPI Value"] for row in facility_table_data) / len(facility_table_data) if facility_table_data else 0
    
    # Add overall row
    overall_row = pd.DataFrame({
        "Facility Name": [f"Overall {title}"],
        numerator_label: [overall_numerator],
        denominator_label: [overall_denominator],
        "KPI Value": [overall_value]
    })
    
    facility_table_df = pd.concat([facility_table_df, overall_row], ignore_index=True)
    
    # Add row numbering starting from 1
    facility_table_df.insert(0, 'No', range(1, len(facility_table_df) + 1))
    
    # Format the table
    styled_table = facility_table_df.style.format({
        numerator_label: "{:,.0f}",
        denominator_label: "{:,.0f}",
        "KPI Value": "{:.1f}"
    }).set_table_attributes('class="summary-table"').hide(axis='index')
    
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)
    
    # Add download button for CSV
    csv = facility_table_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_facility_comparison.csv",
        mime="text/csv"
    )