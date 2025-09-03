import streamlit as st
import pandas as pd
import json
from io import BytesIO
import zipfile
import logging
import concurrent.futures
from utils.data_service import fetch_program_data_for_user
from utils.kpi_utils import auto_text_color

logging.basicConfig(level=logging.INFO)
CACHE_TTL = 1800  # 30 minutes

# ---------------- Cache Wrapper ----------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_cached_data(user):
    """Fetch program data for user with caching in a thread-safe way."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(fetch_program_data_for_user, user)
        return future.result(timeout=180)

# ---------------- Render Function ----------------
def render():
    st.set_page_config(
        page_title="Maternal Health Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "refresh_trigger" not in st.session_state:
        st.session_state["refresh_trigger"] = False

    # ---------------- Load CSS ----------------
    try:
        with open("utils/facility.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # ---------------- Sidebar ----------------
    user = st.session_state.get("user", {})
    username = user.get("username", "Unknown User")
    role = user.get("role", "Unknown Role")
    region_name = user.get("region_name", "Unknown Region")

    st.sidebar.markdown(f"""
        <div class="user-info">
            <div>👤 Username: {username}</div>
            <div>🗺️ Region: {region_name}</div>
            <div>🛡️ Role: {role}</div>
        </div>
    """, unsafe_allow_html=True)

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.session_state["refresh_trigger"] = not st.session_state["refresh_trigger"]
        st.experimental_rerun()

    # ---------------- Fetch Data ----------------
    with st.spinner("Fetching maternal health data..."):
        try:
            dfs = fetch_cached_data(user)
        except concurrent.futures.TimeoutError:
            st.error("⚠️ DHIS2 data could not be fetched within 3 minutes.")
            return
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            return

    tei_df = dfs.get("tei", pd.DataFrame())
    enrollments_df = dfs.get("enrollments", pd.DataFrame())
    events_df = dfs.get("events", pd.DataFrame())
    raw_json = dfs.get("raw_json", [])

    # ---------------- Sidebar Export ----------------
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)

    col_exp1, col_exp2 = st.sidebar.columns(2)
    with col_exp1:
        if st.button("📥 Raw JSON"):
            st.download_button(
                "Download Raw JSON",
                data=json.dumps(raw_json, indent=2),
                file_name=f"{region_name}_raw.json",
                mime="application/json"
            )
    with col_exp2:
        if st.button("📊 Export CSV"):
            # Copy DataFrames and ensure orgUnit_name is present
            export_tei = tei_df.copy()
            export_enr = enrollments_df.copy()
            export_evt = events_df.copy()

            if "orgUnit_name" not in export_tei.columns and "tei_orgUnit" in export_tei.columns:
                export_tei["orgUnit_name"] = export_tei["tei_orgUnit"]

            if "orgUnit_name" not in export_enr.columns and "tei_orgUnit" in export_enr.columns:
                export_enr["orgUnit_name"] = export_enr["tei_orgUnit"]

            if "orgUnit_name" not in export_evt.columns and "orgUnit" in export_evt.columns:
                export_evt["orgUnit_name"] = export_evt["orgUnit"]

            buffer = BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("tei.csv", export_tei.to_csv(index=False).encode("utf-8"))
                zf.writestr("enrollments.csv", export_enr.to_csv(index=False).encode("utf-8"))
                zf.writestr("events.csv", export_evt.to_csv(index=False).encode("utf-8"))
            buffer.seek(0)

            st.download_button(
                "Download All DataFrames (ZIP)",
                data=buffer,
                file_name=f"{region_name}_dataframes.zip",
                mime="application/zip"
            )

    # ---------------- Main Page ----------------
    st.markdown(f'<div class="main-header">🏥 Maternal Health Dashboard - {region_name}</div>', unsafe_allow_html=True)

    if events_df.empty:
        st.markdown('<div class="no-data-warning">⚠️ No data available for this region.</div>', unsafe_allow_html=True)
        return

    st.success("✅ Maternal health data loaded successfully. KPI and charting logic can be added here.")

    # ---------------- Optional Debug ----------------
    # st.write("DEBUG: TEI DataFrame", tei_df.head())
    # st.write("DEBUG: Enrollments DataFrame", enrollments_df.head())
    # st.write("DEBUG: Events DataFrame", events_df.head())
