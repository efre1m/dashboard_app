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

import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def login_component():
    load_css()

    logo_path = Path(__file__).parent.parent / "assets" / "logo.png"

    # Perfect side-by-side split using native columns and containers
    col_l, col_r = st.columns([1, 1], gap="large")
    
    with col_l:
        with st.container(border=True):
            if logo_path.exists():
                img_base64 = get_base64_of_bin_file(str(logo_path))
                st.markdown(f"""
                    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                        <img src="data:image/png;base64,{img_base64}" class="login-logo-img">
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Logo not found")
        
    with col_r:
        with st.container(border=True):
            st.markdown("<h2 class='login-title'>IMNID Dashboard</h2>", unsafe_allow_html=True)
            st.markdown("<p class='login-subtitle'>Enter your credentials to access the platform</p>", unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="Username", key="login_user", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", key="login_pass", label_visibility="collapsed")

            if st.button("Login", use_container_width=True):
                user = authenticate_user(username, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.success(f"Welcome back, {user['username']}")
                    st.rerun()
                else:
                    st.markdown('<div class="login-error">Invalid username or password</div>', unsafe_allow_html=True)
