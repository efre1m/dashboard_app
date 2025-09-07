import streamlit as st
from components.login import login_component
from dashboards import facility, regional, national, admin
from utils.auth import logout, get_user_display_info

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="IMNID Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📊 IMNID Dashboard")

    # Check authentication
    if st.session_state.get("authenticated", False):
        user = st.session_state.get("user", {})
        st.success(f"✅ Logged in as: **{get_user_display_info(user)}**")
        st.caption(f"Role: `{user.get('role','N/A')}`")

        # Show facility info only if available
        if user.get("facility_name"):
            st.caption(f"Facility: `{user['facility_name']}`")
        if user.get("region_name"):
            st.caption(f"Region: `{user['region_name']}`")
        if user.get("country_id"):
            st.caption(f"Country ID: `{user['country_id']}`")

        # Logout button
        if st.button("Logout"):
            logout()
            st.experimental_rerun()  # Refresh app after logout
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
