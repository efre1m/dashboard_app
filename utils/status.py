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
        self.check_interval = timedelta(minutes=10)  # Check every 10 minutes

    def is_local_server(self, base_url):
        """Check if this is a local server (localhost or local network)"""
        local_indicators = ["localhost", "127.0.0.1", "192.168.", "10.", "172."]
        return any(indicator in base_url for indicator in local_indicators)

    def should_check_connection(self):
        """Determine if we should check connection based on time interval"""
        if self.last_online_check is None:
            return True

        time_since_last_check = datetime.now() - self.last_online_check
        return time_since_last_check >= self.check_interval

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
        """Check connection only if enough time has passed since last check"""
        if self.should_check_connection():
            return self.test_dhis2_connection(user, timeout)
        else:
            # Return cached status
            time_since_check = datetime.now() - self.last_online_check
            minutes_since = int(time_since_check.total_seconds() / 60)
            logger.info(
                f"Using cached connection status (checked {minutes_since} minutes ago)"
            )
            return self.is_online

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

    def should_refresh_data(self):
        """Determine if we should refresh data based on cache age"""
        if "last_data_fetch" not in st.session_state:
            return True

        cache_age = datetime.now() - st.session_state.last_data_fetch
        return cache_age >= self.check_interval

    def get_time_until_next_check(self):
        """Get time remaining until next automatic check"""
        if self.last_online_check is None:
            return "Soon"

        next_check = self.last_online_check + self.check_interval
        time_remaining = next_check - datetime.now()

        if time_remaining.total_seconds() <= 0:
            return "Now"
        elif time_remaining.total_seconds() < 60:
            return f"{int(time_remaining.total_seconds())} seconds"
        else:
            return f"{int(time_remaining.total_seconds() / 60)} minutes"


# Create global status instance
status_monitor = ConnectionStatus()


def render_connection_status(events_data=None, user=None):
    """Show simple connection status at the top of the page"""

    if events_data is None:
        events_data = st.session_state.get("cached_events_data", pd.DataFrame())

    if user is None:
        user = st.session_state.get("user", {})

    # Test connection (will use cache if recent enough)
    is_connected = status_monitor.check_connection(user)

    # Get base URL to determine if it's local or internet
    base_url = user.get("base_url", settings.DHIS2_BASE_URL)
    is_local = status_monitor.is_local_server(base_url)

    cache_status = status_monitor.get_cache_status(events_data)
    time_until_next = status_monitor.get_time_until_next_check()

    # Show simple status banner at the top
    if is_connected:
        if is_local:
            st.success(
                f"✅ **Connected to Local System** - {cache_status} - Next check: {time_until_next}"
            )
        else:
            st.success(
                f"✅ **Connected to Live System** - {cache_status} - Next check: {time_until_next}"
            )
    else:
        if is_local:
            st.error(
                f"⚠️ **Local System Unavailable** - Showing previously saved data - Next check: {time_until_next}"
            )
        else:
            st.error(
                f"⚠️ **No Internet Connection** - Showing previously saved data - Next check: {time_until_next}"
            )


def update_last_sync_time():
    """Update the last successful sync time"""
    st.session_state.last_data_fetch = datetime.now()


def initialize_status_system():
    """Initialize the status system"""
    if "status_monitor" not in st.session_state:
        st.session_state.status_monitor = ConnectionStatus()

    if "last_data_fetch" not in st.session_state:
        st.session_state.last_data_fetch = datetime.now()


def should_refresh_data():
    """Check if data should be refreshed based on time interval"""
    return status_monitor.should_refresh_data()


# Simple check functions
def is_online():
    return status_monitor.is_online if status_monitor.is_online is not None else False
