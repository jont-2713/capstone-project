import streamlit as st

st.set_page_config(page_title="InstGator", layout="wide")

st.markdown("""
# 🕵️‍♂️ Instagator Intelligence Dashboard
Welcome to **Instagator** — your investigation hub for analysing social media activity.  
Use the sidebar to:
- Scrape new profiles under **“Scraper”**
- Review and filter findings in **“Post Browser”**
- View Insights and visualisation of scraped contnet **data vis**

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
