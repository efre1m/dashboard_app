# utils/odk_dashboard.py
import streamlit as st
import pandas as pd
import logging
import time
from typing import Dict, List
from utils.odk_api import list_forms, fetch_form_csv, fetch_all_forms_as_dataframes
from utils.data_service import fetch_odk_data_for_user

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Cache ODK data for 30 minutes for automatic refresh
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odk_data_cached(_user: dict = None):
    """Cached version of ODK data fetching - auto-refresh every 30 minutes"""
    return fetch_odk_data_for_user(_user or {})


@st.cache_data(ttl=1800, show_spinner=False)
def list_forms_cached():
    """Cached version of forms listing"""
    return list_forms()


def display_odk_dashboard(user: dict = None):
    """
    Display simplified ODK forms dashboard with downloadable CSV files.
    Automatically loads data on first render.
    """
    # Add attractive styling
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
    .download-btn {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 8px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
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

    # Initialize session state for refresh tracking
    if "last_odk_refresh" not in st.session_state:
        st.session_state.last_odk_refresh = None

    # MAIN CONTENT - AUTO LOAD DATA FIRST
    with st.spinner("🔄 Loading ODK forms data..."):
        try:
            # Automatically fetch data (cached for 30 minutes)
            odk_data = fetch_odk_data_cached(user)
            forms_data = odk_data.get("odk_forms", {})

            # Store in session state for display
            st.session_state.odk_forms_data = forms_data

            # Update refresh time if this is a new load
            if not st.session_state.last_odk_refresh:
                st.session_state.last_odk_refresh = pd.Timestamp.now()

        except Exception as e:
            st.error(f"❌ Failed to load ODK data: {str(e)}")
            forms_data = {}

    # Header with action buttons on the right - AFTER data is loaded
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<div class="main-header">📋 ODK Forms Dashboard</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**All available ODK forms are automatically loaded below**")

    with col2:
        # Action buttons container
        st.markdown('<div class="action-buttons-container">', unsafe_allow_html=True)

        # Refresh data button - clears cache and reloads
        if st.button("🔄 Refresh", use_container_width=True, type="primary"):
            # Clear cache for fresh data
            st.cache_data.clear()
            st.session_state.last_odk_refresh = pd.Timestamp.now()
            st.rerun()

        # Download all button - check session state directly after loading data
        has_data = (
            st.session_state.get("odk_forms_data")
            and len(st.session_state.odk_forms_data) > 0
        )
        if has_data:
            if st.button("💾 Download All", use_container_width=True, type="secondary"):
                download_all_forms(st.session_state.odk_forms_data)

        st.markdown("</div>", unsafe_allow_html=True)

    # Show refresh info
    if st.session_state.last_odk_refresh:
        time_diff = (
            pd.Timestamp.now() - st.session_state.last_odk_refresh
        ).total_seconds() / 60
        refresh_info = f"Data automatically refreshes every 30 minutes. Last refresh: {st.session_state.last_odk_refresh.strftime('%Y-%m-%d %H:%M')}"

        if time_diff > 25:  # Warn if close to refresh time
            st.markdown(
                f"""
            <div class="refresh-info">
                ⚠️ {refresh_info} ({int(30-time_diff)} minutes until auto-refresh)
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="auto-load-info">
                ✅ {refresh_info}
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Display loaded forms or message
    if (
        st.session_state.get("odk_forms_data")
        and len(st.session_state.odk_forms_data) > 0
    ):
        display_forms_grid(st.session_state.odk_forms_data)
    else:
        st.info("📭 No ODK forms data available. Click 'Refresh' to try again.")


def display_forms_grid(forms_data: Dict[str, pd.DataFrame]):
    """Display all loaded forms in an attractive grid layout"""
    st.markdown(f"### 📁 Available Forms ({len(forms_data)})")

    # Single consistent color for all forms
    consistent_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

    # Create columns for the grid
    cols = st.columns(2)

    for i, (form_id, df) in enumerate(forms_data.items()):
        # Get form display name
        try:
            forms = list_forms_cached()
            form_metadata = next(
                (f for f in forms if f.get("xmlFormId") == form_id), {}
            )
            display_name = form_metadata.get("name", form_id)
        except:
            display_name = form_id

        col = cols[i % 2]  # Alternate between columns

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

        # Download button with downward arrow style
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=f"{form_id}.csv",  # Keep original form ID as filename
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
            zip_file.writestr(f"{form_id}.csv", csv_data)  # Keep original form IDs

    zip_buffer.seek(0)

    st.download_button(
        label="💾 Download All as ZIP",
        data=zip_buffer,
        file_name="odk_forms.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_zip",
    )


def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string for download"""
    return df.to_csv(index=False, encoding="utf-8")
