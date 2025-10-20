# utils/odk_dashboard.py
import streamlit as st
import pandas as pd
import logging
import time
from typing import Dict
from utils.odk_api import list_forms
from utils.data_service import fetch_odk_data_for_user
from utils.region_mapping import (
    get_odk_region_codes,
    get_region_name_from_database_id,
    get_odk_code_mapping_display,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# 🔥 OPTIMIZATION: Cache forms listing for 1 hour
@st.cache_data(ttl=3600, show_spinner=False)
def list_forms_cached():
    """Cached version of forms listing"""
    return list_forms()


def display_odk_dashboard(user: dict = None):
    """
    Display simplified forms dashboard with downloadable CSV files.
    Automatically loads data on first render.
    """
    st.markdown(
        """
    <style>
    .form-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .form-card h3 {
        color: white;
        margin: 0 0 10px 0;
    }
    .form-id {
        font-size: 0.8em;
        opacity: 0.9;
        font-family: monospace;
        background: rgba(255,255,255,0.2);
        padding: 2px 8px;
        border-radius: 10px;
        display: inline-block;
        margin-top: 5px;
    }
    .stats-badge {
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 0.8em;
        margin-right: 8px;
    }
    .refresh-info {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .auto-load-info {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .action-buttons-container {
        display: flex;
        gap: 10px;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 🔥 OPTIMIZATION: Get current user info directly from session state
    current_user = st.session_state.get("user", {})
    if not current_user:
        st.warning("🚪 Please log in to access data")
        return

    current_user_id = current_user.get("id", "anonymous")
    current_username = current_user.get("username", "anonymous")
    current_region_id = current_user.get("region_id")
    current_role = current_user.get("role", "anonymous")

    # 🔥 OPTIMIZATION: Create session state keys
    odk_data_key = f"odk_forms_data_{current_user_id}"
    last_refresh_key = f"last_odk_refresh_{current_user_id}"
    user_tracker_key = "current_odk_user"

    # 🔥 CRITICAL FIX: Check if user has changed
    current_user_info = f"{current_user_id}_{current_region_id}_{current_role}"

    if user_tracker_key not in st.session_state:
        st.session_state[user_tracker_key] = current_user_info
    else:
        previous_user_info = st.session_state[user_tracker_key]
        if previous_user_info != current_user_info:
            # 🔥 USER CHANGED - CLEAR ALL OLD DATA
            st.info(f"🔄 Loading data for {current_username}...")

            # Clear ALL session data
            for key in list(st.session_state.keys()):
                if key.startswith("odk_forms_data_") or key.startswith(
                    "last_odk_refresh_"
                ):
                    del st.session_state[key]

            st.session_state[user_tracker_key] = current_user_info
            st.rerun()

    # Initialize session state
    if last_refresh_key not in st.session_state:
        st.session_state[last_refresh_key] = None

    if odk_data_key not in st.session_state:
        st.session_state[odk_data_key] = {}

    # Header with action buttons
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<div class="main-header">📋 Integrated Mentorship Data</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**All available forms are automatically loaded below**")

    with col2:
        st.markdown('<div class="action-buttons-container">', unsafe_allow_html=True)

        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.session_state[odk_data_key] = {}
            st.session_state[last_refresh_key] = pd.Timestamp.now()
            st.rerun()

        # Download All button ALWAYS visible - no conditions
        if st.button("💾 Download All", use_container_width=True, type="secondary"):
            has_data = (
                st.session_state.get(odk_data_key)
                and len(st.session_state[odk_data_key]) > 0
            )
            if has_data:
                download_all_forms(st.session_state[odk_data_key])
            else:
                st.warning("No data available to download. Please refresh data first.")

        st.markdown("</div>", unsafe_allow_html=True)

    # 🔥 OPTIMIZATION: Load data only if needed
    if not st.session_state[odk_data_key]:
        with st.spinner("🔄 Loading forms data..."):
            try:
                # Fetch data once
                odk_data = fetch_odk_data_for_user(current_user)
                forms_data = odk_data.get("odk_forms", {})

                st.session_state[odk_data_key] = forms_data
                st.session_state[last_refresh_key] = pd.Timestamp.now()

                st.success(f"✅ Loaded {len(forms_data)} forms")

            except Exception as e:
                st.error(f"❌ Failed to load data: {str(e)}")

    # Show refresh info
    if st.session_state[last_refresh_key]:
        refresh_time = st.session_state[last_refresh_key].strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f'<div class="auto-load-info">🕒 Last refresh: {refresh_time}</div>',
            unsafe_allow_html=True,
        )

    # Display forms
    if st.session_state.get(odk_data_key) and len(st.session_state[odk_data_key]) > 0:
        display_forms_grid(st.session_state[odk_data_key])
    else:
        st.info("📭 No forms data available. Click 'Refresh Data' to try again.")


def display_forms_grid(forms_data: Dict[str, pd.DataFrame]):
    """Display all loaded forms in an attractive grid layout"""
    st.markdown(f"### 📁 Available Forms ({len(forms_data)})")

    consistent_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    cols = st.columns(2)

    for i, (form_id, df) in enumerate(forms_data.items()):
        # 🔥 OPTIMIZATION: Get display name from cached forms
        forms = list_forms_cached()
        form_metadata = next((f for f in forms if f.get("xmlFormId") == form_id), {})
        display_name = form_metadata.get("name", form_id)

        col = cols[i % 2]

        with col:
            display_form_card(form_id, df, consistent_color, i, display_name)


def display_form_card(
    form_id: str, df: pd.DataFrame, color: str, index: int, display_name: str
):
    """Display an individual form card with download button"""

    with st.container():
        st.markdown(
            f"""
        <div class="form-card" style="background: {color};">
            <h3>📄 {display_name}</h3>
            <div class="form-id">({form_id})</div>
            <div style="display: flex; gap: 8px; margin-bottom: 15px; margin-top: 10px;">
                <span class="stats-badge">📊 {len(df):,} records</span>
                <span class="stats-badge">📋 {len(df.columns)} columns</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=f"{form_id}.csv",
            mime="text/csv",
            key=f"download_{index}",
            use_container_width=True,
            type="primary",
        )


def download_all_forms(forms_data: Dict[str, pd.DataFrame]):
    """Create a zip file with all forms for download"""
    import zipfile
    import io

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for form_id, df in forms_data.items():
            csv_data = convert_df_to_csv(df)
            zip_file.writestr(f"{form_id}.csv", csv_data)

    zip_buffer.seek(0)

    st.download_button(
        label="💾 Download All as ZIP",
        data=zip_buffer,
        file_name="forms.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_zip",
    )


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string for download"""
    return df.to_csv(index=False, encoding="utf-8")
