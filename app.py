import streamlit as st
from components.login import login_component
from dashboards import facility, regional, national, admin
from utils.auth import logout

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="IMNID Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Sidebar ----------------
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

# ---------------- Routing Based on Role ----------------
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
