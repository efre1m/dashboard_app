import streamlit as st
from utils.auth import get_user_profile, update_user_profile, change_user_password


def render_edit_profile(user: dict, view_state_key: str = "edit_profile_mode", key_prefix: str = "edit_profile"):
    """Render the Edit Profile page with scoped styling."""
    st.markdown(
        """
        <style>
        .edit-profile-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            text-align: center;
            margin-bottom: 0.25rem;
        }
        .edit-profile-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .edit-profile-card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .edit-profile-card-subtitle {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 1rem;
        }
        .edit-profile-divider {
            height: 1px;
            background: #e2e8f0;
            margin: 0.5rem 0 1rem 0;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%) !important;
        }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1100px !important;
            margin: 0 auto !important;
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stForm"] {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 22px 24px 26px 24px !important;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.08) !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stTextInput"] label {
            color: #334155 !important;
            font-weight: 600 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stTextInput"] div[data-baseweb="input"] {
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 10px !important;
            background: #ffffff !important;
            height: 46px !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button {
            background: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
            height: 44px !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.2px !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stFormSubmitButton"] button:hover {
            background: #1d4ed8 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stButton"] > button {
            background: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            height: 36px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        [data-testid="stAppViewContainer"] [data-testid="stButton"] > button:hover {
            background: #e2e8f0 !important;
        }

        @media (max-width: 900px) {
            [data-testid="stAppViewContainer"] .main .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='edit-profile-title'>Edit Profile</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='edit-profile-subtitle'>Update your account information and password</div>",
        unsafe_allow_html=True,
    )

    if st.button("Back to Dashboard", key=f"{key_prefix}_back"):
        st.session_state[view_state_key] = False
        st.rerun()

    user_id = user.get("user_id")
    if not user_id:
        st.error("User information is unavailable.")
        return

    profile = get_user_profile(user_id) or {}
    current_username = profile.get("username") or user.get("username", "")
    current_first_name = profile.get("first_name") or user.get("first_name", "")
    current_last_name = profile.get("last_name") or user.get("last_name", "")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='edit-profile-card-title'>User Information</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='edit-profile-card-subtitle'>Keep your details accurate for reports.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='edit-profile-divider'></div>", unsafe_allow_html=True)

        with st.form(f"{key_prefix}_profile_form"):
            first_name = st.text_input(
                "First Name",
                value=current_first_name or "",
                key=f"{key_prefix}_first_name",
            )
            last_name = st.text_input(
                "Last Name",
                value=current_last_name or "",
                key=f"{key_prefix}_last_name",
            )
            username = st.text_input(
                "Username",
                value=current_username or "",
                key=f"{key_prefix}_username",
            )

            submit_profile = st.form_submit_button("Update Profile", use_container_width=True)
            if submit_profile:
                if not username.strip():
                    st.error("Username cannot be empty.")
                else:
                    ok, msg = update_user_profile(
                        user_id=user_id,
                        username=username.strip(),
                        first_name=first_name.strip(),
                        last_name=last_name.strip(),
                    )
                    if ok:
                        st.session_state["user"]["username"] = username.strip()
                        st.session_state["user"]["first_name"] = first_name.strip()
                        st.session_state["user"]["last_name"] = last_name.strip()
                        st.success(msg)
                    else:
                        st.error(msg)

    with col2:
        st.markdown("<div class='edit-profile-card-title'>Update Password</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='edit-profile-card-subtitle'>Use a strong password to protect your account.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='edit-profile-divider'></div>", unsafe_allow_html=True)

        with st.form(f"{key_prefix}_password_form"):
            current_pw = st.text_input(
                "Existing Password",
                type="password",
                key=f"{key_prefix}_current_pw",
            )
            new_pw = st.text_input(
                "New Password",
                type="password",
                key=f"{key_prefix}_new_pw",
            )
            confirm_pw = st.text_input(
                "Confirm New Password",
                type="password",
                key=f"{key_prefix}_confirm_pw",
            )

            submit_pw = st.form_submit_button("Update Password", use_container_width=True)
            if submit_pw:
                if not current_pw or not new_pw or not confirm_pw:
                    st.error("All password fields are required.")
                elif new_pw != confirm_pw:
                    st.error("New passwords do not match.")
                else:
                    ok, msg = change_user_password(
                        user_id=user_id,
                        current_password=current_pw,
                        new_password=new_pw,
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
