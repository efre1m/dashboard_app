import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
import requests
from utils.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionStatus:
    def __init__(self):
        self.last_online_check = None
        self.is_online = None
        self.last_successful_sync = None
        self.cache_age = None

    def is_local_server(self, base_url):
        """Check if this is a local server (localhost or local network)"""
        local_indicators = ["localhost", "127.0.0.1", "192.168.", "10.", "172."]
        return any(indicator in base_url for indicator in local_indicators)

    def test_dhis2_connection(self, user=None, timeout=10):
        """Simple connection test to DHIS2"""
        try:
            start_time = time.time()

            # Use provided user credentials or fall back to settings
            if (
                user
                and user.get("base_url")
                and user.get("username")
                and user.get("password")
            ):
                base_url = user.get("base_url")
                username = user.get("username")
                password = user.get("password")
            else:
                # Fall back to settings
                base_url = settings.DHIS2_BASE_URL
                username = settings.DHIS2_USERNAME
                password = settings.DHIS2_PASSWORD

            # Simple connection test
            response = requests.get(
                f"{base_url}/api/me", auth=(username, password), timeout=timeout
            )

            connection_ok = response.status_code == 200
            self.last_online_check = datetime.now()
            self.is_online = connection_ok

            if connection_ok:
                self.last_successful_sync = datetime.now()

            logger.info(
                f"Connection check: {connection_ok} (took {time.time() - start_time:.2f}s)"
            )
            return connection_ok

        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            self.is_online = False
            self.last_online_check = datetime.now()
            return False

    def check_connection(self, user=None, timeout=10):
        """Alias for test_dhis2_connection"""
        return self.test_dhis2_connection(user, timeout)

    def get_cache_status(self, events_data):
        """Simple cache status for non-technical users"""
        if events_data is None or events_data.empty:
            return "No data available"

        if "last_data_fetch" in st.session_state:
            fetch_time = st.session_state.last_data_fetch
            cache_age = datetime.now() - fetch_time
            self.cache_age = cache_age

            if cache_age < timedelta(minutes=5):
                return "Just updated"
            elif cache_age < timedelta(hours=1):
                return f"Updated {int(cache_age.total_seconds()/60)} minutes ago"
            elif cache_age < timedelta(hours=24):
                return f"Updated {int(cache_age.total_seconds()/3600)} hours ago"
            else:
                return f"Updated {int(cache_age.total_seconds()/86400)} days ago"

        return "Data loaded previously"


# Create global status instance
status_monitor = ConnectionStatus()


def render_connection_status(events_data=None, user=None):
    """Show simple connection status at the top of the page"""

    if events_data is None:
        events_data = st.session_state.get("cached_events_data", pd.DataFrame())

    if user is None:
        user = st.session_state.get("user", {})

    # Test connection
    is_connected = status_monitor.check_connection(user)

    # Get base URL to determine if it's local or internet
    base_url = user.get("base_url", settings.DHIS2_BASE_URL)
    is_local = status_monitor.is_local_server(base_url)

    cache_status = status_monitor.get_cache_status(events_data)

    # Show simple status banner at the top
    if is_connected:
        if is_local:
            st.success(f"✅ **Connected to Local System** - {cache_status}")
        else:
            st.success(f"✅ **Connected to Live System** - {cache_status}")
    else:
        if is_local:
            st.error(f"⚠️ **Local System Unavailable** - Showing previously saved data")
        else:
            st.error(f"⚠️ **No Internet Connection** - Showing previously saved data")


def update_last_sync_time():
    """Update the last successful sync time"""
    st.session_state.last_data_fetch = datetime.now()


def initialize_status_system():
    """Initialize the status system"""
    if "status_monitor" not in st.session_state:
        st.session_state.status_monitor = ConnectionStatus()

    if "last_data_fetch" not in st.session_state:
        st.session_state.last_data_fetch = datetime.now()


# Simple check functions
def is_online():
    return status_monitor.is_online if status_monitor.is_online is not None else False
