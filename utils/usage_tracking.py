import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from utils.db import get_db_connection


@st.cache_data(ttl=3600)
def load_facility_code_mapping():
    """Load facility name -> facility code mapping from the local reference file."""
    mapping_path = Path(__file__).resolve().parent / "IMNID_facility_code.xlsx"
    if not mapping_path.exists():
        return {}

    try:
        ref_df = pd.read_excel(mapping_path)
    except Exception:
        return {}

    required_cols = {"facility_name", "facility_code"}
    if not required_cols.issubset(set(ref_df.columns)):
        return {}

    clean_df = ref_df[["facility_name", "facility_code"]].copy()
    clean_df["facility_name"] = clean_df["facility_name"].astype(str).str.strip()
    clean_df = clean_df[clean_df["facility_name"] != ""]

    def to_code_str(v):
        if pd.isna(v):
            return None
        as_num = pd.to_numeric(v, errors="coerce")
        if pd.notna(as_num):
            return str(int(as_num))
        v_str = str(v).strip()
        return v_str if v_str else None

    clean_df["facility_code"] = clean_df["facility_code"].apply(to_code_str)
    clean_df = clean_df[clean_df["facility_code"].notna()]

    return dict(
        zip(
            clean_df["facility_name"].str.lower(),
            clean_df["facility_code"],
        )
    )


def enrich_logs_with_facility_codes(logs_df):
    """Replace facility labels with facility_code and derive region code from first digit."""
    if logs_df is None or logs_df.empty:
        return logs_df

    code_map = load_facility_code_mapping()
    if not code_map:
        return logs_df

    df = logs_df.copy()
    name_norm = df["facility_name"].fillna("").astype(str).str.strip().str.lower()
    mapped_codes = name_norm.map(code_map)

    # Use mapped code where available; otherwise keep original facility label.
    original_name = df["facility_name"].fillna("Unknown Facility").astype(str).str.strip()
    df["facility_name"] = mapped_codes.fillna(original_name)

    # Region code is the first digit of facility_code for facility users.
    is_facility_user = df["role"].astype(str).str.lower() == "facility"
    mapped_code_str = mapped_codes.fillna("").astype(str).str.strip()
    region_code = mapped_code_str.str[0]
    region_code = region_code.where(mapped_code_str != "", None)
    df.loc[is_facility_user, "region_name"] = region_code[is_facility_user]

    return df


@st.cache_data(ttl=60)
def fetch_usage_logs():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT u.username, u.first_name, u.last_name, u.role,
               f.facility_name,
               COALESCE(r_user.region_name, r_fac.region_name) AS region_name,
               c.country_name,
               l.login_time, u.region_id, u.facility_id
        FROM login_logs l
        JOIN users u ON l.user_id = u.user_id
        LEFT JOIN facilities f ON u.facility_id = f.facility_id
        LEFT JOIN regions r_user ON u.region_id = r_user.region_id
        LEFT JOIN regions r_fac ON f.region_id = r_fac.region_id
        LEFT JOIN countries c ON u.country_id = c.country_id
        ORDER BY l.login_time DESC
    """, conn)
    conn.close()
    return df

def render_usage_tracking_shared(user_role, user_region_id=None):
    st.subheader("Usage Analytics & Tracking")
    
    logs = enrich_logs_with_facility_codes(fetch_usage_logs())
    if logs.empty:
        st.info("No login activity recorded yet.")
        return

    # Hierarchical Filtering Logic
    if user_role == 'admin':
        # Admin sees everything
        display_roles = ['national', 'regional', 'facility', 'admin']
    elif user_role == 'national':
        # National sees Regional and Facility
        display_roles = ['regional', 'facility']
    elif user_role == 'regional':
        # Regional sees Facility users in their region
        display_roles = ['facility']
        if user_region_id:
            logs = logs[logs['region_id'] == user_region_id]
    else:
        st.warning("Usage tracking is restricted for your role.")
        return

    filtered_logs = logs[logs['role'].str.lower().isin(display_roles)].copy()
    
    if filtered_logs.empty:
        st.info(f"No login activity found for {', '.join(display_roles)} users.")
        return

    # Tabs based on available roles for the current user
    tab_list = []
    if any(r in ['national', 'admin'] for r in display_roles): tab_list.append("Country/Admin")
    if 'regional' in display_roles: tab_list.append("Regional")
    if 'facility' in display_roles: tab_list.append("Facility")
    
    tabs = st.tabs(tab_list)
    
    # Helper for rendering summary-style table
    def render_styled_table(df, columns):
        st.markdown("""
        <style>
        .usage-table-container { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin: 0.5rem 0; border: 1px solid #e2e8f0; }
        .usage-table { width: 100%; border-collapse: collapse; font-size: 11px !important; }
        .usage-table thead tr { background: linear-gradient(135deg, #1e293b, #334155); }
        .usage-table th { color: white !important; padding: 6px 10px; text-align: left; font-weight: 600; font-size: 11px !important; border: none; }
        .usage-table td { padding: 4px 10px; border-bottom: 1px solid #f1f5f9; font-size: 11px !important; background-color: white; color: #334155; }
        .usage-table tbody tr:hover td { background-color: #f8fafc; }
        .usage-table td:first-child { font-weight: 600; color: #64748b; text-align: center; width: 30px; }
        .usage-table th:first-child { text-align: center; width: 30px; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="usage-table-container">', unsafe_allow_html=True)
        st.markdown(
            df[columns].style.set_table_attributes('class="usage-table"')
            .hide(axis="index")
            .to_html(),
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    tab_idx = 0
    
    # 1. Country/Admin Level
    if any(r in ['national', 'admin'] for r in display_roles):
        with tabs[tab_idx]:
            df = filtered_logs[filtered_logs['role'].str.lower().isin(['admin', 'national'])].copy()
            if not df.empty:
                col_chart, col_tbl = st.columns([3, 2])
                
                with col_chart:
                    df['login_time'] = pd.to_datetime(df['login_time'])
                    df['login_date'] = df['login_time'].dt.date
                    chart_df = df.groupby('login_date').size().reset_index(name='Logins')
                    fig = px.line(chart_df, x='login_date', y='Logins', title="Daily National & Admin Logins",
                                 markers=True, line_shape="spline", template="plotly_white")
                    fig.update_traces(line_color='#3b82f6')
                    fig.update_yaxes(tickmode='linear', tick0=0, dtick=1 if chart_df['Logins'].max() < 10 else None, 
                                     showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', nticks=10)
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_tbl:
                    df.insert(0, 'No', range(1, len(df) + 1))
                    if user_role == 'admin':
                        cols = ['No', 'username', 'first_name', 'last_name', 'role', 'login_time']
                    else:
                        cols = ['No', 'role', 'login_time']
                    render_styled_table(df, cols)
            else:
                st.info("No National/Admin activity found.")
        tab_idx += 1

    # 2. Regional Level (Line Chart per Region)
    if 'regional' in display_roles:
        with tabs[tab_idx]:
            df = filtered_logs[filtered_logs['role'].str.lower() == 'regional'].copy()
            if not df.empty:
                col_chart, col_tbl = st.columns([3, 2])
                
                with col_chart:
                    df['login_time'] = pd.to_datetime(df['login_time'])
                    df['login_date'] = df['login_time'].dt.date
                    # Group by date AND region for multi-line
                    chart_df = df.groupby(['login_date', 'region_name']).size().reset_index(name='Logins')
                    
                    fig = px.line(chart_df, x='login_date', y='Logins', color='region_name', 
                                 title="Regional Login Trends",
                                 markers=True, line_shape="spline", template="plotly_white")
                    fig.update_yaxes(tickmode='linear', tick0=0, dtick=1 if chart_df['Logins'].max() < 10 else None,
                                     showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', nticks=10)
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_tbl:
                    df.insert(0, 'No', range(1, len(df) + 1))
                    if user_role == 'admin':
                        cols = ['No', 'username', 'first_name', 'last_name', 'region_name', 'login_time']
                    else:
                        cols = ['No', 'region_name', 'login_time']
                    render_styled_table(df, cols)
            else:
                st.info("No Regional activity found.")
        tab_idx += 1

    # 3. Facility Level (Line Chart per Facility)
    if 'facility' in display_roles:
        with tabs[tab_idx]:
            df = filtered_logs[filtered_logs['role'].str.lower() == 'facility'].copy()
            if not df.empty:
                col_chart, col_tbl = st.columns([3, 2])
                
                with col_chart:
                    df['login_time'] = pd.to_datetime(df['login_time'])
                    df['login_date'] = df['login_time'].dt.date
                    # Group by date AND facility for multi-line
                    chart_df = df.groupby(['login_date', 'facility_name']).size().reset_index(name='Logins')
                    
                    fig = px.line(chart_df, x='login_date', y='Logins', color='facility_name', 
                                 title="Facility Login Trends",
                                 markers=True, line_shape="spline", template="plotly_white")
                    fig.update_yaxes(tickmode='linear', tick0=0, dtick=1 if chart_df['Logins'].max() < 10 else None,
                                     showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)', nticks=10)
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_tbl:
                    df.insert(0, 'No', range(1, len(df) + 1))
                    if user_role == 'admin':
                        cols = ['No', 'username', 'first_name', 'last_name', 'region_name', 'facility_name', 'login_time']
                    else:
                        cols = ['No', 'region_name', 'facility_name', 'login_time']
                    render_styled_table(df, cols)
            else:
                st.info("No Facility activity found.")
        tab_idx += 1
