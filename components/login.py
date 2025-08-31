import streamlit as st
from pathlib import Path
from utils.auth import authenticate_user  # make sure you have this util

def load_css():
    css_file = Path(__file__).parent.parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ CSS file not found, check path!")

def login_component():
    load_css()

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown("<h2 class='login-title'>Login to IMNID Dashboard</h2>", unsafe_allow_html=True)

    # collapsed labels → only placeholders show (no white label box)
    username = st.text_input(
        "Username",
        placeholder="Enter your username",
        key="login_user",
        label_visibility="collapsed"
    )
    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
        key="login_pass",
        label_visibility="collapsed"
    )

    if st.button("Sign In", use_container_width=True):
        user = authenticate_user(username, password)
        if user:
            st.session_state["authenticated"] = True
            st.session_state["user"] = user
            st.success(f"✅ Welcome {user['username']} ({user['role']})")
            st.rerun()
        else:
            st.markdown('<div class="login-error">❌ Invalid username or password</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
