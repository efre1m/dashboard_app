# admin.py
import streamlit as st
import pandas as pd
import bcrypt
import time
from utils.db import get_db_connection

# ---------------- Apply CSS Styling ----------------
def apply_css_styling():
    st.markdown("""
    <style>
    /* Main container styling */
    .main-container {
        padding: 20px;
        background-color: #f8f9fa;
    }
    
    /* Table styling */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    
    .dataframe th {
        background-color: #4CAF50;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: bold;
    }
    
    .dataframe td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    
    .dataframe tr:hover {
        background-color: #e9f7e9;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Update button */
    .stButton button:first-child {
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    
    /* Delete button */
    .stButton button:last-child {
        background-color: #f44336;
        color: white;
        border: none;
    }
    
    /* Form container */
    .form-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Search input */
    .search-input {
        margin-bottom: 20px;
    }
    
    /* Success message */
    .stSuccess {
        border-radius: 4px;
        padding: 10px;
    }
    
    /* Warning message */
    .stWarning {
        border-radius: 4px;
        padding: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: bold;
        background-color: #e9f7e9;
        padding: 10px;
        border-radius: 4px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- Helper Functions ----------------
def fetch_table(table_name, search_column=None, search_value=None):
    """Fetch table data, optionally filtered by a search term"""
    conn = get_db_connection()
    cur = conn.cursor()
    query = f"SELECT * FROM {table_name}"
    params = ()
    if search_column and search_value:
        query += f" WHERE {search_column} ILIKE %s"
        params = (f"%{search_value}%",)
    cur.execute(query, params)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columns)

def fetch_table_multi_filter(table_name, search_columns=None, search_value=None):
    """Fetch table data with multi-column filtering"""
    if not search_value or not search_columns:
        return fetch_table(table_name)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Build the WHERE clause with multiple OR conditions
    where_conditions = " OR ".join([f"{col} ILIKE %s" for col in search_columns])
    query = f"SELECT * FROM {table_name} WHERE {where_conditions}"
    
    # Create parameters for each column
    params = [f"%{search_value}%"] * len(search_columns)
    
    cur.execute(query, params)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columns)

def execute_query(query, params=()):
    """Execute INSERT, UPDATE, DELETE queries"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    cur.close()
    conn.close()

def delete_row(table, row_id_column, row_id):
    execute_query(f"DELETE FROM {table} WHERE {row_id_column}=%s", (row_id,))
    st.success(f"Deleted {table} ID {row_id}")

def get_name_from_id(table_name, id_column, id_value, name_column):
    """Get a name from an ID in a table"""
    if not id_value or pd.isna(id_value):
        return None
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT {name_column} FROM {table_name} WHERE {id_column} = %s", (id_value,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None

# ---------------- Add Forms ----------------
def add_country_form():
    """Form to add a new country"""
    with st.expander("➕ Add New Country", expanded=False):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        country_name = st.text_input("Country Name", key="new_country_name")
        dhis2_uid = st.text_input("DHIS2 UID", key="new_country_dhis")
        if st.button("Add Country", key="add_country_btn"):
            if country_name:
                execute_query(
                    "INSERT INTO countries (country_name, dhis2_uid) VALUES (%s, %s)",
                    (country_name, dhis2_uid)
                )
                st.success(f"Country '{country_name}' added successfully!")
                st.rerun()
            else:
                st.warning("Country Name is required.")
        st.markdown('</div>', unsafe_allow_html=True)

def add_region_form():
    """Form to add a new region"""
    with st.expander("➕ Add New Region", expanded=False):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        country_df = fetch_table("countries")
        country_map = dict(zip(country_df["country_id"], country_df["country_name"]))
        if country_map:
            country_id = st.selectbox("Country", list(country_map.keys()), format_func=lambda x: country_map[x], key="region_country")
        else:
            st.warning("Add countries first!")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        region_name = st.text_input("Region Name", key="new_region_name")
        dhis2_uid = st.text_input("DHIS2 Regional UID", key="new_region_dhis")
        if st.button("Add Region", key="add_region_btn"):
            if region_name:
                execute_query(
                    "INSERT INTO regions (region_name, country_id, dhis2_regional_uid) VALUES (%s, %s, %s)",
                    (region_name, country_id, dhis2_uid)
                )
                st.success(f"Region '{region_name}' added successfully!")
                st.rerun()
            else:
                st.warning("Region Name is required.")
        st.markdown('</div>', unsafe_allow_html=True)

def add_facility_form():
    """Form to add a new facility"""
    with st.expander("➕ Add New Facility", expanded=False):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        region_df = fetch_table("regions")
        region_map = dict(zip(region_df["region_id"], region_df["region_name"]))
        if region_map:
            region_id = st.selectbox("Region", list(region_map.keys()), format_func=lambda x: region_map[x], key="facility_region")
        else:
            st.warning("Add regions first!")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        facility_name = st.text_input("Facility Name", key="new_facility_name")
        dhis2_uid = st.text_input("DHIS2 UID", key="new_facility_dhis")
        if st.button("Add Facility", key="add_facility_btn"):
            if facility_name:
                execute_query(
                    "INSERT INTO facilities (facility_name, region_id, dhis2_uid) VALUES (%s, %s, %s)",
                    (facility_name, region_id, dhis2_uid)
                )
                st.success(f"Facility '{facility_name}' added successfully!")
                st.rerun()
            else:
                st.warning("Facility Name is required.")
        st.markdown('</div>', unsafe_allow_html=True)

def add_user_form():
    """Form to add a new user"""
    with st.expander("➕ Add New User", expanded=False):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        username = st.text_input("Username", key="new_user_name")
        password = st.text_input("Password", type="password", key="new_user_pw")
        first_name = st.text_input("First Name", key="new_user_fn")
        last_name = st.text_input("Last Name", key="new_user_ln")
        role = st.selectbox("Role", ["facility","regional","national","admin"], key="new_user_role")

        # Optional associations
        facility_id = region_id = country_id = None
        if role == "facility":
            facility_df = fetch_table("facilities")
            if not facility_df.empty:
                facility_map = dict(zip(facility_df["facility_id"], facility_df["facility_name"]))
                facility_id = st.selectbox("Facility", list(facility_map.keys()), format_func=lambda x: facility_map[x], key="new_user_facility")
            else:
                st.warning("Add facilities first!")
        elif role == "regional":
            region_df = fetch_table("regions")
            if not region_df.empty:
                region_map = dict(zip(region_df["region_id"], region_df["region_name"]))
                region_id = st.selectbox("Region", list(region_map.keys()), format_func=lambda x: region_map[x], key="new_user_region")
            else:
                st.warning("Add regions first!")
        elif role == "national":
            country_df = fetch_table("countries")
            if not country_df.empty:
                country_map = dict(zip(country_df["country_id"], country_df["country_name"]))
                country_id = st.selectbox("Country", list(country_map.keys()), format_func=lambda x: country_map[x], key="new_user_country")
            else:
                st.warning("Add countries first!")

        if st.button("Add User", key="add_user_btn"):
            if username and password:
                hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                execute_query(
                    """INSERT INTO users (username,password_hash,first_name,last_name,role,facility_id,region_id,country_id)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (username, hashed_pw, first_name, last_name, role, facility_id, region_id, country_id)
                )
                st.success(f"User '{username}' added successfully!")
                st.rerun()
            else:
                st.warning("Username and Password are required.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Editable Table Functions ----------------
def user_editable_table(df):
    """Display and manage users table with editing capabilities"""
    # Create a display copy with proper column names and types
    display_df = df.copy()
    
    # Create new columns for display instead of modifying the original ones
    display_df['Facility'] = None
    display_df['Region'] = None
    display_df['Country'] = None
    
    # Fill display columns based on role
    for idx, row in display_df.iterrows():
        role = row['role']
        
        if role == "admin":
            display_df.at[idx, 'Facility'] = "Admin"
            display_df.at[idx, 'Region'] = "Admin"
            display_df.at[idx, 'Country'] = "Admin"
        elif role == "facility" and pd.notna(row['facility_id']):
            facility_name = get_name_from_id("facilities", "facility_id", row['facility_id'], "facility_name")
            display_df.at[idx, 'Facility'] = facility_name or "Unknown Facility"
        elif role == "regional" and pd.notna(row['region_id']):
            region_name = get_name_from_id("regions", "region_id", row['region_id'], "region_name")
            display_df.at[idx, 'Region'] = region_name or "Unknown Region"
        elif role == "national" and pd.notna(row['country_id']):
            country_name = get_name_from_id("countries", "country_id", row['country_id'], "country_name")
            display_df.at[idx, 'Country'] = country_name or "Unknown Country"
    
    # Remove original ID columns and password from display
    display_df = display_df.drop(columns=['facility_id', 'region_id', 'country_id', 'password_hash', 'user_id'], errors='ignore')
    
    # Reorder columns for better display
    column_order = ['username', 'first_name', 'last_name', 'role', 'Facility', 'Region', 'Country']
    display_df = display_df[[col for col in column_order if col in display_df.columns]]
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    # Add edit functionality for each row
    for idx, row in df.iterrows():
        # Initialize session state for delete confirmation
        delete_key = f"delete_confirm_{row['user_id']}"
        if delete_key not in st.session_state:
            st.session_state[delete_key] = False
            
        with st.expander(f"Edit User: {row['username']}", expanded=False):
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            # Create form for editing
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input("Username", value=row['username'], key=f"username_{row['user_id']}_{idx}")
                new_first_name = st.text_input("First Name", value=row['first_name'], key=f"first_name_{row['user_id']}_{idx}")
            with col2:
                new_last_name = st.text_input("Last Name", value=row['last_name'], key=f"last_name_{row['user_id']}_{idx}")
                new_role = st.selectbox("Role", ["facility","regional","national","admin"], 
                                       index=["facility","regional","national","admin"].index(row['role']), 
                                       key=f"role_{row['user_id']}_{idx}")
            
            # Role-dependent associations
            new_facility_id = new_region_id = new_country_id = None
            
            if new_role == "facility":
                facility_df = fetch_table("facilities")
                if not facility_df.empty:
                    facility_map = dict(zip(facility_df["facility_id"], facility_df["facility_name"]))
                    current_facility = row['facility_id'] if pd.notna(row['facility_id']) else None
                    new_facility_id = st.selectbox("Facility", options=[None] + list(facility_map.keys()),
                                                 format_func=lambda x: facility_map.get(x, "None"),
                                                 index=0 if current_facility is None else list(facility_map.keys()).index(current_facility) + 1,
                                                 key=f"facility_{row['user_id']}_{idx}")
            
            elif new_role == "regional":
                region_df = fetch_table("regions")
                if not region_df.empty:
                    region_map = dict(zip(region_df["region_id"], region_df["region_name"]))
                    current_region = row['region_id'] if pd.notna(row['region_id']) else None
                    new_region_id = st.selectbox("Region", options=[None] + list(region_map.keys()),
                                               format_func=lambda x: region_map.get(x, "None"),
                                               index=0 if current_region is None else list(region_map.keys()).index(current_region) + 1,
                                               key=f"region_{row['user_id']}_{idx}")
            
            elif new_role == "national":
                country_df = fetch_table("countries")
                if not country_df.empty:
                    country_map = dict(zip(country_df["country_id"], country_df["country_name"]))
                    current_country = row['country_id'] if pd.notna(row['country_id']) else None
                    new_country_id = st.selectbox("Country", options=[None] + list(country_map.keys()),
                                                format_func=lambda x: country_map.get(x, "None"),
                                                index=0 if current_country is None else list(country_map.keys()).index(current_country) + 1,
                                                key=f"country_{row['user_id']}_{idx}")
            
            # Password update field
            new_password = st.text_input("New Password (leave blank to keep current)", type="password", 
                                       key=f"password_{row['user_id']}_{idx}")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", key=f"update_user_{row['user_id']}_{idx}"):
                    # Prepare update query
                    update_fields = {
                        'username': new_username,
                        'first_name': new_first_name,
                        'last_name': new_last_name,
                        'role': new_role,
                        'facility_id': new_facility_id,
                        'region_id': new_region_id,
                        'country_id': new_country_id
                    }
                    
                    # Add password if changed
                    if new_password:
                        hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                        update_fields['password_hash'] = hashed_pw
                    
                    # Build query
                    set_clause = ", ".join([f"{k}=%s" for k in update_fields.keys()])
                    params = list(update_fields.values()) + [row['user_id']]
                    
                    execute_query(f"UPDATE users SET {set_clause} WHERE user_id=%s", params)
                    st.success(f"User '{new_username}' updated successfully!")
                    st.rerun()
            
            with col2:
                if st.button("Delete", key=f"delete_user_{row['user_id']}_{idx}"):
                    st.session_state[delete_key] = True
                
                if st.session_state[delete_key]:
                    st.warning(f"Are you sure you want to delete user '{row['username']}'?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Delete", key=f"confirm_delete_{row['user_id']}_{idx}"):
                            delete_row("users", "user_id", row['user_id'])
                            st.session_state[delete_key] = False
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_delete_{row['user_id']}_{idx}"):
                            st.session_state[delete_key] = False
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add new user form
    add_user_form()
    
    # Export button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Users CSV", csv, file_name="users.csv", mime="text/csv")

def entity_editable_table(df, table_name, id_column, add_function):
    """Display and manage entities table (facilities, regions, countries)"""
    # Create a display copy with names instead of IDs
    display_df = df.copy()
    
    # Replace IDs with names for display
    if table_name == "facilities":
        # For facilities, show region name instead of region_id
        display_df["Region"] = display_df["region_id"].apply(
            lambda x: get_name_from_id("regions", "region_id", x, "region_name") if pd.notna(x) else None
        )
        display_df = display_df.drop(columns=["region_id"], errors='ignore')
    elif table_name == "regions":
        # For regions, show country name instead of country_id
        display_df["Country"] = display_df["country_id"].apply(
            lambda x: get_name_from_id("countries", "country_id", x, "country_name") if pd.notna(x) else None
        )
        display_df = display_df.drop(columns=["country_id"], errors='ignore')
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    # Add edit functionality for each row
    for idx, row in df.iterrows():
        # Initialize session state for delete confirmation
        delete_key = f"delete_confirm_{table_name}_{row[id_column]}"
        if delete_key not in st.session_state:
            st.session_state[delete_key] = False
            
        with st.expander(f"Edit {table_name.capitalize()}: {row.iloc[1]}", expanded=False):
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            
            # Create form for editing
            updated_vals = {}
            for col in df.columns:
                if col != id_column:
                    # For foreign keys, show dropdowns
                    if col.endswith('_id'):
                        ref_table = col.replace('_id', 's')  # Convert facility_id to facilities
                        try:
                            ref_df = fetch_table(ref_table)
                            ref_map = dict(zip(ref_df[f"{ref_table[:-1]}_id"], ref_df[f"{ref_table[:-1]}_name"]))
                            current_val = row[col] if pd.notna(row[col]) else None
                            updated_vals[col] = st.selectbox(
                                col.replace('_id', '').title(), 
                                options=[None] + list(ref_map.keys()),
                                format_func=lambda x: ref_map.get(x, "None"),
                                index=0 if current_val is None else list(ref_map.keys()).index(current_val) + 1,
                                key=f"{table_name}_{row[id_column]}_{col}_{idx}"
                            )
                        except:
                            # Fallback to text input if reference table doesn't exist
                            updated_vals[col] = st.text_input(col, value=row[col], key=f"{table_name}_{row[id_column]}_{col}_{idx}")
                    else:
                        updated_vals[col] = st.text_input(col, value=row[col], key=f"{table_name}_{row[id_column]}_{col}_{idx}")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", key=f"update_{table_name}_{row[id_column]}_{idx}"):
                    set_clause = ", ".join([f"{k}=%s" for k in updated_vals.keys()])
                    params = list(updated_vals.values()) + [row[id_column]]
                    execute_query(f"UPDATE {table_name} SET {set_clause} WHERE {id_column}=%s", params)
                    st.success(f"{table_name.capitalize()} updated successfully!")
                    st.rerun()
            
            with col2:
                if st.button("Delete", key=f"delete_{table_name}_{row[id_column]}_{idx}"):
                    st.session_state[delete_key] = True
                
                if st.session_state[delete_key]:
                    st.warning(f"Are you sure you want to delete this {table_name[:-1]}?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Delete", key=f"confirm_delete_{table_name}_{row[id_column]}_{idx}"):
                            delete_row(table_name, id_column, row[id_column])
                            st.session_state[delete_key] = False
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_delete_{table_name}_{row[id_column]}_{idx}"):
                            st.session_state[delete_key] = False
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add new entity form
    add_function()
    
    # Export button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {table_name.capitalize()} CSV", csv, file_name=f"{table_name}.csv", mime="text/csv")

# ---------------- Search Configuration ----------------
def get_search_config(table_name):
    """Return search configuration for each table"""
    config = {
        "users": {
            "columns": ["username", "first_name", "last_name", "role"],
            "default_column": "username",
            "display_names": {
                "username": "Username",
                "first_name": "First Name",
                "last_name": "Last Name",
                "role": "Role"
            }
        },
        "facilities": {
            "columns": ["facility_name", "dhis2_uid"],
            "default_column": "facility_name",
            "display_names": {
                "facility_name": "Facility Name",
                "dhis2_uid": "DHIS2 UID"
            }
        },
        "regions": {
            "columns": ["region_name", "dhis2_regional_uid"],
            "default_column": "region_name",
            "display_names": {
                "region_name": "Region Name",
                "dhis2_regional_uid": "DHIS2 Regional UID"
            }
        },
        "countries": {
            "columns": ["country_name", "dhis2_uid"],
            "default_column": "country_name",
            "display_names": {
                "country_name": "Country Name",
                "dhis2_uid": "DHIS2 UID"
            }
        }
    }
    return config.get(table_name, {"columns": [], "default_column": None, "display_names": {}})

# ---------------- Admin Render ----------------
def render():
    apply_css_styling()
    st.title("🛠️ Admin Dashboard")
    
    # Initialize session state for search terms
    if 'search_terms' not in st.session_state:
        st.session_state.search_terms = {
            "users": "", "facilities": "", "regions": "", "countries": ""
        }
    
    if 'search_columns' not in st.session_state:
        st.session_state.search_columns = {
            "users": "username", "facilities": "facility_name", 
            "regions": "region_name", "countries": "country_name"
        }
    
    tabs = st.tabs(["Users", "Facilities", "Regions", "Countries"])

    # Users tab
    with tabs[0]:
        st.markdown('<div class="search-input">', unsafe_allow_html=True)
        search_config = get_search_config("users")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            filter_by = st.selectbox(
                "Filter By",
                options=["all"] + search_config["columns"],
                format_func=lambda x: "All Columns" if x == "all" else search_config["display_names"].get(x, x),
                key="users_filter_by"
            )
        with col2:
            search_input = st.text_input(
                "Search", 
                value=st.session_state.search_terms["users"],
                key="users_search_input"
            )
        with col3:
            st.write("")  # Spacer for alignment
            st.write("")  # Spacer for alignment
            if st.button("Search", key="users_search_btn"):
                st.session_state.search_terms["users"] = search_input
                st.session_state.search_columns["users"] = filter_by
            if st.button("Clear", key="users_clear_btn"):
                st.session_state.search_terms["users"] = ""
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fetch data based on search
        if st.session_state.search_terms["users"]:
            if st.session_state.search_columns["users"] == "all":
                df = fetch_table_multi_filter("users", search_config["columns"], st.session_state.search_terms["users"])
            else:
                df = fetch_table("users", st.session_state.search_columns["users"], st.session_state.search_terms["users"])
        else:
            df = fetch_table("users")
        
        user_editable_table(df)

    # Facilities tab
    with tabs[1]:
        st.markdown('<div class="search-input">', unsafe_allow_html=True)
        search_config = get_search_config("facilities")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            filter_by = st.selectbox(
                "Filter By",
                options=["all"] + search_config["columns"],
                format_func=lambda x: "All Columns" if x == "all" else search_config["display_names"].get(x, x),
                key="facilities_filter_by"
            )
        with col2:
            search_input = st.text_input(
                "Search", 
                value=st.session_state.search_terms["facilities"],
                key="facilities_search_input"
            )
        with col3:
            st.write("")  # Spacer for alignment
            st.write("")  # Spacer for alignment
            if st.button("Search", key="facilities_search_btn"):
                st.session_state.search_terms["facilities"] = search_input
                st.session_state.search_columns["facilities"] = filter_by
            if st.button("Clear", key="facilities_clear_btn"):
                st.session_state.search_terms["facilities"] = ""
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fetch data based on search
        if st.session_state.search_terms["facilities"]:
            if st.session_state.search_columns["facilities"] == "all":
                df = fetch_table_multi_filter("facilities", search_config["columns"], st.session_state.search_terms["facilities"])
            else:
                df = fetch_table("facilities", st.session_state.search_columns["facilities"], st.session_state.search_terms["facilities"])
        else:
            df = fetch_table("facilities")
        
        entity_editable_table(df, "facilities", "facility_id", add_facility_form)

    # Regions tab
    with tabs[2]:
        st.markdown('<div class="search-input">', unsafe_allow_html=True)
        search_config = get_search_config("regions")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            filter_by = st.selectbox(
                "Filter By",
                options=["all"] + search_config["columns"],
                format_func=lambda x: "All Columns" if x == "all" else search_config["display_names"].get(x, x),
                key="regions_filter_by"
            )
        with col2:
            search_input = st.text_input(
                "Search", 
                value=st.session_state.search_terms["regions"],
                key="regions_search_input"
            )
        with col3:
            st.write("")  # Spacer for alignment
            st.write("")  # Spacer for alignment
            if st.button("Search", key="regions_search_btn"):
                st.session_state.search_terms["regions"] = search_input
                st.session_state.search_columns["regions"] = filter_by
            if st.button("Clear", key="regions_clear_btn"):
                st.session_state.search_terms["regions"] = ""
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fetch data based on search
        if st.session_state.search_terms["regions"]:
            if st.session_state.search_columns["regions"] == "all":
                df = fetch_table_multi_filter("regions", search_config["columns"], st.session_state.search_terms["regions"])
            else:
                df = fetch_table("regions", st.session_state.search_columns["regions"], st.session_state.search_terms["regions"])
        else:
            df = fetch_table("regions")
        
        entity_editable_table(df, "regions", "region_id", add_region_form)

    # Countries tab
    with tabs[3]:
        st.markdown('<div class="search-input">', unsafe_allow_html=True)
        search_config = get_search_config("countries")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            filter_by = st.selectbox(
                "Filter By",
                options=["all"] + search_config["columns"],
                format_func=lambda x: "All Columns" if x == "all" else search_config["display_names"].get(x, x),
                key="countries_filter_by"
            )
        with col2:
            search_input = st.text_input(
                "Search", 
                value=st.session_state.search_terms["countries"],
                key="countries_search_input"
            )
        with col3:
            st.write("")  # Spacer for alignment
            st.write("")  # Spacer for alignment
            if st.button("Search", key="countries_search_btn"):
                st.session_state.search_terms["countries"] = search_input
                st.session_state.search_columns["countries"] = filter_by
            if st.button("Clear", key="countries_clear_btn"):
                st.session_state.search_terms["countries"] = ""
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fetch data based on search
        if st.session_state.search_terms["countries"]:
            if st.session_state.search_columns["countries"] == "all":
                df = fetch_table_multi_filter("countries", search_config["columns"], st.session_state.search_terms["countries"])
            else:
                df = fetch_table("countries", st.session_state.search_columns["countries"], st.session_state.search_terms["countries"])
        else:
            df = fetch_table("countries")
        
        entity_editable_table(df, "countries", "country_id", add_country_form)