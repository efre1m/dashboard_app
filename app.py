# app.py - FIXED WITH ALL SESSION STATE INITIALIZATIONS

import streamlit as st

# ====================== INITIALIZE ALL SESSION STATES ======================
# This MUST be at the VERY TOP before any imports

# Define ALL cache names used across the application
ALL_CACHES = [
    # Maternal caches
    "assisted_cache",
    "uterotonic_cache",
    "pph_cache",
    "missing_md_cache",
    "missing_bo_cache",
    "missing_cod_cache",
    "arv_cache",
    "svd_cache",
    "admitted_mothers_cache",
    "kpi_cache",
    # Newborn caches
    "kpi_cache_newborn",
    "kpi_cache_newborn_simplified",  # Added for kpi_utils_newborn_simplified.py
    "kpi_cache_newborn_v2",  # Added for kpi_utils_newborn_v2.py
]

# Initialize ALL caches
for cache_name in ALL_CACHES:
    if cache_name not in st.session_state:
        st.session_state[cache_name] = {}

# Initialize other essential session state variables
ESSENTIAL_STATES = {
    "filters": {},
    "period_label": "Monthly",
    "user": {},
    "authenticated": False,
    "data_loaded": False,
    "raw_data": None,
    "processed_data": None,
}

for key, default_value in ESSENTIAL_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ====================== NOW IMPORT OTHER MODULES ======================
# Now it's safe to import modules that might use session state

from components.login import login_component
from dashboards import facility, regional, national, admin
from utils.auth import logout

# ====================== STREAMLIT PAGE CONFIG ======================
st.set_page_config(
    page_title="IMNID Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("📊 IMNID Dashboard")

    # Check authentication
    if st.session_state.get("authenticated", False):
        # Logout button
        if st.button("Logout"):
            logout()
            st.rerun()  # Refresh app after logout
    else:
        st.info("🔑 Please log in")

# ====================== ROUTING BASED ON ROLE ======================
if not st.session_state.get("authenticated", False):
    login_component()  # Render login page if not authenticated
else:
    role = st.session_state["user"].get("role", "")

    if role == "facility":
        facility.render()
    elif role == "regional":
        regional.render()
    elif role == "national":
        national.render()
    elif role == "admin":
        # Admin dashboard with full CRUD for users, facilities, regions, countries
        admin.render()
    else:
        st.error("❌ Unauthorized role")
