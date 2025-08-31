import streamlit as st
from components.login import login_component
from dashboards import facility, regional, national
from utils.auth import logout

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
        st.success(f"✅ Logged in as: **{user.get('username','')}**")
        st.caption(f"Role: `{user.get('role','N/A')}`")
        st.caption(f"Facility: `{user.get('facility_name','N/A')}`")

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
        # Facility dashboard with KPI dropdown + styled graphs + Excel export
        facility.render()
    elif role == "regional":
        # Regional dashboard (to be updated with similar KPI dropdown & export)
        regional.render()
    elif role == "national":
        # National dashboard (to be updated similarly)
        national.render()
    else:
        st.error("❌ Unauthorized role")
