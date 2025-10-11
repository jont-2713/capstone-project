import streamlit as st
import os
import numpy as np
import base64

#---Page Setup---
st.set_page_config(page_title="InstGator", layout="wide")


## ---------- Logo + Header ----------
logo_path = os.path.join(os.path.dirname(__file__), "Instegator logo.png")

st.markdown("""
    <style>
        .center-logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .center-logo {
            width: 400px; /* Adjust 350–500px to resize */
            max-width: 90%;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(255, 200, 100, 0.15);
            transition: transform 0.3s ease;
        }
        .center-logo:hover {
            transform: scale(1.02);
        }
        h2.title-sub {
            text-align: center;
            color: #f5f5f5;
            font-weight: 500;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Encode the image properly to base64 for display
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div class="center-logo-container">
            <img src="data:image/png;base64,{logo_base64}" class="center-logo" alt="Instagator Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2 class='title-sub'>🕵️‍♂️ The Social Media Detective</h2>", unsafe_allow_html=True)
else:
    st.title("🕵️‍♂️ Instagator")
    st.subheader("The Social Media Detective")

# ---------- Intro ----------
st.markdown("""
### Welcome, Investigator.
Instagator helps you **analyze social media sentiment** across text and images, 
providing AI-powered risk scoring to assist digital investigators.

Use the sidebar to:
- Scrape social media profiles under **Scraper**
- Explore findings under **Analytics**
- Manage and export your investigations
---
""")


st.subheader("Quick Actions")
col1, col2 = st.columns(2)
with col1:
    if st.button("Go to Scraper"):
        st.switch_page("Pages/1_Scraper.py")
with col2:
    if st.button("Open Post Browser"):
        st.switch_page("Pages/2_Post Browser.py")
