# utils/auth.py
import streamlit as st
import bcrypt
from utils.db import get_db_connection

def _is_bcrypt_hash(value: str) -> bool:
    """Check if a string is a bcrypt hash."""
    return isinstance(value, str) and value.startswith("$2")

def authenticate_user(username: str, password: str):
    """
    Authenticate user against the database.
    Supports bcrypt + legacy plain-text password upgrade.
    Returns a user dictionary with facility, region, national, or admin info.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 
            u.user_id,
            u.username,
            u.password_hash,
            u.role,
            u.facility_id,
            u.region_id,
            u.country_id,
            f.facility_name,
            f.dhis2_uid AS facility_dhis_uid,
            r.region_name,
            r.dhis2_regional_uid
        FROM users u
        LEFT JOIN facilities f ON u.facility_id = f.facility_id
        LEFT JOIN regions r ON u.region_id = r.region_id
        WHERE u.username = %s
    """, (username,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return None

    (user_id, uname, stored_pw, role, facility_id, region_id, country_id,
     facility_name, facility_dhis_uid, region_name, region_dhis_uid) = row

    valid = False
    upgraded = False

    if _is_bcrypt_hash(stored_pw):
        valid = bcrypt.checkpw(password.encode("utf-8"), stored_pw.encode("utf-8"))
    else:
        valid = (password == stored_pw)
        if valid:
            # Upgrade plain-text password to bcrypt
            new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            cur.execute("UPDATE users SET password_hash = %s WHERE user_id = %s", (new_hash, user_id))
            conn.commit()
            upgraded = True

    cur.close()
    conn.close()

    if not valid:
        return None

    return {
        "user_id": user_id,
        "username": uname,
        "role": role,
        "facility_id": facility_id,
        "region_id": region_id,
        "country_id": country_id,
        "facility_name": facility_name,
        "facility_dhis_uid": facility_dhis_uid,
        "region_name": region_name,
        "region_dhis_uid": region_dhis_uid,
        "upgraded": upgraded,
    }


def logout():
    """Clear session state and query parameters."""
    for k in ["authenticated", "user", "page"]:
        if k in st.session_state:
            del st.session_state[k]
    st.query_params.clear()
    st.rerun()


def get_user_display_info(user: dict) -> str:
    """Return formatted user information for display."""
    role = user["role"]

    if role == "facility":
        return f"{user['username']} ({role} - {user['facility_name']})"
    elif role == "regional":
        return f"{user['username']} ({role} - {user['region_name']})"
    elif role == "national":
        return f"{user['username']} ({role})"
    elif role == "admin":
        return f"{user['username']} (admin)"
    return user["username"]
