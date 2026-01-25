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
    "episiotomy_cache",
    "antipartum_compl_cache",
    "vaccine_coverage_cache",
    "status_cache",
    # Newborn caches
    "kpi_cache_newborn",
    "kpi_cache_newborn_simplified",
    "kpi_cache_newborn_v2",
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
    "chatbot_context": {},
}

for key, default_value in ESSENTIAL_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ====================== NOW IMPORT OTHER MODULES ======================
# Now it's safe to import modules that might use session state

from components.login import login_component
from dashboards import facility, regional, national, admin
from utils.auth import logout
from components.chatbot import render_chatbot

# ====================== STREAMLIT PAGE CONFIG ======================
st.set_page_config(
    page_title="IMNID Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Ultra-Aggressive UI Scaling for "Fits all in one" look */
        html, body, [data-testid="stAppViewContainer"] {
            font-size: 11px !important;
            line-height: 1.2 !important;
        }

        /* Very Narrow Sidebar */
        [data-testid="stSidebar"] {
            min-width: 220px !important;
            max-width: 220px !important;
        }

        /* Minimize headings */
        h1, .main-header { font-size: 1.2rem !important; margin: 0 !important; font-weight: 800 !important; }
        h2, .section-header { font-size: 1.1rem !important; margin: 0 !important; }
        h3 { font-size: 1.0rem !important; margin: 0 !important; }
        
        /* Ultra-tight content padding */
        .main .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 99.5% !important;
        }

        /* Smallest possible metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
            margin-bottom: 0 !important;
        }
        [data-testid="stMetric"] {
            padding: 0px 5px !important;
            border-radius: 4px !important;
            background-color: #f8fafc;
        }

        /* Tiny Sidebar Widgets */
        section[data-testid="stSidebar"] .stWidgetLabel p {
            color: #0f172a !important;
            font-weight: 700;
            font-size: 10px !important;
            margin-bottom: 0 !important;
        }
        
        /* Micro Buttons */
        div.stButton > button {
            padding: 0.05rem 0.3rem !important;
            height: 20px !important;
            min-height: 0 !important;
            font-size: 10px !important;
            border-radius: 2px !important;
            line-height: 1 !important;
            white-space: normal !important;
        }

        /* Micro Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1px 4px !important;
            font-size: 10px !important;
            height: 24px !important;
        }
        
        /* No vertical margins between containers */
        .element-container {
            margin-bottom: 0.1rem !important;
        }
        
        /* Compact Selectboxes - Allow multiselect to expand */
        div[data-baseweb="select"], div[data-baseweb="select"] > div {
            min-height: 22px !important;
            height: auto !important;
            font-size: 11px !important;
        }
        
        /* Hide unnecessary dividers */
        hr {
            margin: 0.1rem 0 !important;
            opacity: 0.3;
        }

        /* Tighten Plotly Charts */
        .js-plotly-plot .plotly .modebar {
            transform: scale(0.7) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.title("📊 IMNID Dashboard")

    # Check authentication
    if st.session_state.get("authenticated", False):
        st.write("---")
        # Chat Bot Mode Toggle
        st.toggle("🤖 Chat Bot Mode", key="chatbot_mode")
        
        st.write("---")
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
    # Check if Chat Bot Mode is active
    if st.session_state.get("chatbot_mode", False):
        render_chatbot()
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
