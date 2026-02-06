# admin.py
import streamlit as st
import pandas as pd
import bcrypt
import time
import plotly.express as px
from utils.db import get_db_connection
from utils.usage_tracking import render_usage_tracking_shared

# ---------------- Styling ----------------
def apply_css():
    st.markdown("""
        <style>
        /* Premium Professional Background */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        /* Card-based layout for better depth */
        .form-container, [data-testid="stExpander"], .stDataFrame, .stPlotlyChart {
            background-color: white !important;
            padding: 1.5rem !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid #e2e8f0 !important;
            margin-bottom: 1.5rem !important;
        }

        /* Tab Navigation Enhancements */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 12px;
            background-color: rgba(255, 255, 255, 0.5);
            padding: 10px 15px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] { 
            height: 48px;
            padding: 0 24px; 
            border-radius: 8px;
            background-color: #f1f5f9;
            color: #64748b;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stTabs [aria-selected="true"] { 
            background-color: #2563eb !important; 
            color: white !important; 
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
            transform: translateY(-2px);
        }
        
        /* Typography & Spacing */
        h1, h2, h3 {
            color: #0f172a !important;
            font-weight: 800 !important;
            letter-spacing: -0.025em !important;
        }
        
        .stMarkdown p {
            color: #475569;
        }
        
        /* Sidebar Polish (if used) */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------- Optimized Data Fetchers ----------------
@st.cache_data(ttl=60)
def fetch_users():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT u.user_id, u.username, u.first_name, u.last_name, u.role,
               f.facility_name, r.region_name, c.country_name,
               u.facility_id, u.region_id, u.country_id
        FROM users u
        LEFT JOIN facilities f ON u.facility_id = f.facility_id
        LEFT JOIN regions r ON u.region_id = r.region_id
        LEFT JOIN countries c ON u.country_id = c.country_id
        ORDER BY u.username
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def fetch_facilities(region_id=None):
    conn = get_db_connection()
    query = """
        SELECT f.facility_id, f.facility_name, f.dhis2_uid, r.region_name, f.region_id
        FROM facilities f
        LEFT JOIN regions r ON f.region_id = r.region_id
    """
    params = []
    if region_id is not None:
        if region_id == 0:
            query += " WHERE f.region_id IS NULL"
        else:
            query += " WHERE f.region_id = %s"
            params.append(region_id)
    
    df = pd.read_sql(query + " ORDER BY f.facility_name", conn, params=params if params else None)
    conn.close()
    return df

@st.cache_data(ttl=60)
def fetch_regions():
    conn = get_db_connection()
    df = pd.read_sql("""
        SELECT r.region_id, r.region_name, r.dhis2_regional_uid, c.country_name, r.country_id
        FROM regions r
        LEFT JOIN countries c ON r.country_id = c.country_id
        ORDER BY r.region_name
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def fetch_countries():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM countries ORDER BY country_name", conn)
    conn.close()
    return df

# fetch_usage_logs moved to utils/usage_tracking.py

# ---------------- Generic CRUD Helpers ----------------
def run_query(query, params=()):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
        cur.close()
        conn.close()
        return True, "Executed successfully"
    except Exception as e:
        return False, str(e)

# ---------------- Management Components ----------------
def manage_users():
    st.subheader("User Accounts")
    
    u_tabs = st.tabs(["🔍 View Users", "➕ Add User", "📝 Edit User", "🗑️ Delete User"])
    
    with u_tabs[0]:
        # View Users
        search = st.text_input("🔍 Search Users", placeholder="Enter username or name...", key="view_users_search")
        df = fetch_users()
        if search:
            df = df[df.apply(lambda row: search.lower() in str(row).lower(), axis=1)]
        
        display_df = df.rename(columns={
            "facility_name": "Facility", "region_name": "Region", "country_name": "Country"
        })[["username", "first_name", "last_name", "role", "Facility", "Region", "Country"]]
        st.dataframe(display_df, use_container_width=True)

    with u_tabs[1]:
        # Add User
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        # Selectors OUTSIDE the form for reactivity
        role = st.selectbox("Role*", ["facility", "regional", "national", "admin"], key="add_user_role_select")
        
        col_L, col_R = st.columns(2)
        
        # Logic: Disable irrelevant fields
        is_nat = role == "national"
        is_reg = role == "regional"
        is_fac = role == "facility"
        
        # Country Select
        c_df = fetch_countries()
        country_id = col_L.selectbox("Assign Country*", list(c_df['country_id']), 
                                     format_func=lambda x: c_df[c_df['country_id']==x]['country_name'].values[0],
                                     disabled=not is_nat, key="add_user_c_sel")
        
        # Region Select
        r_df = fetch_regions()
        region_id = col_R.selectbox("Assign/Filter Region*", list(r_df['region_id']), 
                                    format_func=lambda x: r_df[r_df['region_id']==x]['region_name'].values[0],
                                    disabled=not (is_reg or is_fac), key="add_user_r_sel")
        
        # Facility Select (Filtered by Region)
        f_df = fetch_facilities(region_id) if is_fac and region_id else fetch_facilities()
        f_options = [None] + list(f_df['facility_id'])
        facility_id = st.selectbox("Assign Facility*", f_options, 
                                  format_func=lambda x: f_df[f_df['facility_id']==x]['facility_name'].values[0] if x else "Select Facility",
                                  disabled=not is_fac, key="add_user_f_sel")

        with st.form("add_user_credentials_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            username = c1.text_input("Username*")
            password = c1.text_input("Password*", type="password")
            fname = c2.text_input("First Name")
            lname = c2.text_input("Last Name")
            
            submit = st.form_submit_button("🚀 Create User Account", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Username and Password are required")
                elif is_nat and not country_id: st.error("Country required")
                elif is_reg and not region_id: st.error("Region required")
                elif is_fac and not facility_id: st.error("Facility required")
                else:
                    hash_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                    # Final ID cleaning
                    f_out = int(facility_id) if is_fac else None
                    r_out = int(region_id) if is_reg else None
                    c_out = int(country_id) if is_nat else None
                    
                    ok, msg = run_query("""
                        INSERT INTO users (username, password_hash, first_name, last_name, role, facility_id, region_id, country_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (username, hash_pw, fname, lname, role, f_out, r_out, c_out))
                    if ok:
                        st.success("User created successfully!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else: st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)

    with u_tabs[2]:
        # Edit User
        df = fetch_users()
        user_to_edit = st.selectbox("Select User to Edit", [None] + list(df['user_id']), 
                                   format_func=lambda x: df[df['user_id']==x]['username'].values[0] if x else "Select a User",
                                   key="edit_user_id_loader")
        
        if user_to_edit:
            row = df[df['user_id'] == user_to_edit].iloc[0]
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            # Interactive selections OUTSIDE the form
            new_role = st.selectbox("Role", ["facility", "regional", "national", "admin"], 
                                  index=["facility", "regional", "national", "admin"].index(row['role']),
                                  key=f"edit_role_{user_to_edit}")
            
            is_n = new_role == "national"
            is_r = new_role == "regional"
            is_f = new_role == "facility"
            
            e_col1, e_col2 = st.columns(2)
            
            # Country
            c_df = fetch_countries()
            c_list = list(c_df['country_id'])
            c_idx = c_list.index(row['country_id']) if row['country_id'] in c_list else 0
            new_c_id = e_col1.selectbox("Country", c_list, 
                                       format_func=lambda x: c_df[c_df['country_id']==x]['country_name'].values[0],
                                       index=c_idx, disabled=not is_n, key=f"edit_c_{user_to_edit}")
            
            # Region (Filter/Assign)
            r_df = fetch_regions()
            r_list = list(r_df['region_id'])
            r_idx = r_list.index(row['region_id']) if row['region_id'] in r_list else 0
            new_r_id = e_col2.selectbox("Region", r_list, 
                                       format_func=lambda x: r_df[r_df['region_id']==x]['region_name'].values[0],
                                       index=r_idx, disabled=not (is_r or is_f), key=f"edit_r_{user_to_edit}")
            
            # Facility (Filtered)
            f_df = fetch_facilities(new_r_id) if is_f else fetch_facilities()
            f_list = [None] + list(f_df['facility_id'])
            f_idx = f_list.index(row['facility_id']) if row['facility_id'] in f_list else 0
            new_f_id = st.selectbox("Facility", f_list, 
                                   format_func=lambda x: f_df[f_df['facility_id']==x]['facility_name'].values[0] if x else "None",
                                   index=f_idx, disabled=not is_f, key=f"edit_f_{user_to_edit}")

            with st.form(f"edit_user_creds_{user_to_edit}"):
                c1, c2 = st.columns(2)
                u = c1.text_input("Username", value=row['username'])
                fn = c1.text_input("First Name", value=row['first_name'] or "")
                ln = c2.text_input("Last Name", value=row['last_name'] or "")
                pw = st.text_input("New Password (blank to keep current)", type="password")
                
                if st.form_submit_button("💾 Save Changes", use_container_width=True):
                    # Clean values
                    f_out = int(new_f_id) if is_f and new_f_id else None
                    r_out = int(new_r_id) if is_r and new_r_id else None
                    c_out = int(new_c_id) if is_n and new_c_id else None
                    
                    update_q = "UPDATE users SET username=%s, first_name=%s, last_name=%s, role=%s, facility_id=%s, region_id=%s, country_id=%s"
                    params = [u, fn, ln, new_role, f_out, r_out, c_out]
                    if pw:
                        update_q += ", password_hash=%s"
                        params.append(bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode())
                    update_q += " WHERE user_id=%s"
                    params.append(int(user_to_edit))
                    
                    ok, msg = run_query(update_q, tuple(params))
                    if ok:
                        st.success("User updated!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else: st.error(msg)
            st.markdown("</div>", unsafe_allow_html=True)

    with u_tabs[3]:
        # Delete User
        df = fetch_users()
        user_to_del = st.selectbox("Select User to Delete", [None] + list(df['user_id']), 
                                  format_func=lambda x: df[df['user_id']==x]['username'].values[0] if x else "Select a User",
                                  key="del_user_select")
        
        if user_to_del:
            u_name = df[df['user_id']==user_to_del]['username'].values[0]
            st.warning(f"Are you sure you want to delete user '{u_name}'?")
            if st.button("Confirm Delete", type="primary"):
                ok, msg = run_query("DELETE FROM users WHERE user_id=%s", (user_to_del,))
                if ok: 
                    st.success("Deleted")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(msg)

def manage_facilities():
    st.subheader("Facilities")
    
    # Global Add Form
    with st.expander("➕ Add Facility", expanded=False):
        with st.form("add_facility_form_global"):
            r_df = fetch_regions()
            name = st.text_input("Facility Name*")
            uid = st.text_input("DHIS2 UID")
            rid = st.selectbox("Region*", list(r_df['region_id']), 
                               format_func=lambda x: r_df[r_df['region_id']==x]['region_name'].values[0])
            if st.form_submit_button("Add Facility"):
                ok, msg = run_query("INSERT INTO facilities (facility_name, dhis2_uid, region_id) VALUES (%s, %s, %s)", (name, uid, rid))
                if ok: st.cache_data.clear(); st.rerun()
    
    # Tabs by Region
    regions_df = fetch_regions()
    region_options = [("All", None)] + [(row['region_name'], row['region_id']) for _, row in regions_df.iterrows()]
    
    tabs = st.tabs([r[0] for r in region_options])
    for i, (r_name, r_id) in enumerate(region_options):
        with tabs[i]:
            df = fetch_facilities(r_id)
            st.dataframe(df.drop(columns=['region_id', 'facility_id']), use_container_width=True)
            
            for _, row in df.iterrows():
                with st.expander(f"Edit: {row['facility_name']}"):
                    # UNIQUE KEY for form: table_id_regionsuffix
                    with st.form(f"edit_fac_{row['facility_id']}_{i}"):
                        n = st.text_input("Name", value=row['facility_name'])
                        u = st.text_input("UID", value=row['dhis2_uid'] or "")
                        if st.form_submit_button("Update"):
                            run_query("UPDATE facilities SET facility_name=%s, dhis2_uid=%s WHERE facility_id=%s", (n, u, row['facility_id']))
                            st.cache_data.clear(); st.rerun()
                    if st.button("Delete", key=f"del_f_{row['facility_id']}_{i}"):
                        run_query("DELETE FROM facilities WHERE facility_id=%s", (row['facility_id'],))
                        st.cache_data.clear(); st.rerun()

def manage_regions():
    st.subheader("Regions")
    df = fetch_regions()
    st.dataframe(df.drop(columns=['region_id', 'country_id']), use_container_width=True)
    # Simplified Add/Edit...

def manage_usage_tracking():
    st.subheader("Usage Tracking")
    
    # Clear logs button
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("Clear All Tracking Logs", type="secondary", use_container_width=True):
            st.session_state.confirm_clear_logs = True
    
    if st.session_state.get('confirm_clear_logs', False):
        st.warning("Are you sure you want to delete all login logs? This cannot be undone.")
        c1, c2 = st.columns(2)
        if c1.button("Yes, Clear Everything", type="primary", use_container_width=True):
            ok, msg = run_query("DELETE FROM login_logs")
            if ok:
                st.success("All logs cleared!")
                st.session_state.confirm_clear_logs = False
                st.cache_data.clear()
                st.rerun()
        if c2.button("No, Keep Logs", use_container_width=True):
            st.session_state.confirm_clear_logs = False
            st.rerun()

    # Shared Tracking Component
    render_usage_tracking_shared('admin')

# ---------------- Main Render ----------------
def render():
    apply_css()
    st.title("Admin Control Center")
    
    menu = st.tabs(["Users", "Facilities", "Regions", "Countries", "Tracking"])
    
    with menu[0]: manage_users()
    with menu[1]: manage_facilities()
    with menu[2]: 
        add_region_form_simple()
        manage_regions()
    with menu[3]: 
        add_country_form_simple()
        st.dataframe(fetch_countries(), use_container_width=True)
    with menu[4]: manage_usage_tracking()

def add_region_form_simple():
    with st.expander("➕ Add Region", expanded=False):
        with st.form("add_region_form_simple"):
            c_df = fetch_countries()
            name = st.text_input("Region Name*")
            uid = st.text_input("DHIS2 UID")
            cid = st.selectbox("Country*", list(c_df['country_id']), format_func=lambda x: c_df[c_df['country_id']==x]['country_name'].values[0])
            if st.form_submit_button("Add Region"):
                run_query("INSERT INTO regions (region_name, country_id, dhis2_regional_uid) VALUES (%s, %s, %s)", (name, cid, uid))
                st.cache_data.clear(); st.rerun()

def add_country_form_simple():
    with st.expander("➕ Add Country", expanded=False):
        with st.form("add_country_form_simple"):
            name = st.text_input("Country Name*")
            uid = st.text_input("DHIS2 UID")
            if st.form_submit_button("Add Country"):
                run_query("INSERT INTO countries (country_name, dhis2_uid) VALUES (%s, %s)", (name, uid))
                st.cache_data.clear(); st.rerun()
