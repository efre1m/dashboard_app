# utils/auth.py
import streamlit as st
import bcrypt
from utils.db import get_db_connection

def _is_bcrypt_hash(value: str) -> bool:
    return isinstance(value, str) and value.startswith("$2")

def authenticate_user(username: str, password: str):
    """
    Authenticate user against DB.
    Supports bcrypt + legacy plain-text upgrade.
    Returns user object with facility info if facility-level user.
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
            f.dhis2_uid AS facility_dhis_uid
        FROM users u
        LEFT JOIN facilities f
            ON u.facility_id = f.facility_id
        WHERE u.username = %s
    """, (username,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        return None

    user_id, uname, stored_pw, role, facility_id, region_id, country_id, facility_name, facility_dhis_uid = row

    valid = False
    upgraded = False

    if _is_bcrypt_hash(stored_pw):
        valid = bcrypt.checkpw(password.encode("utf-8"), stored_pw.encode("utf-8"))
    else:
        valid = (password == stored_pw)
        if valid:
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
        "upgraded": upgraded,
    }

def logout():
    for k in ["authenticated", "user", "page"]:
        if k in st.session_state:
            del st.session_state[k]
    st.query_params.clear()  # clear URL params
    st.rerun()
