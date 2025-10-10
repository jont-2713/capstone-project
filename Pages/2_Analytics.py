import streamlit as st


# pages/2_Analytics.py
import os
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Analytics")

st.title("Analytics Dashboard")

# ---------- Access shared session data ----------
S = st.session_state.get("scrape_state")
paths = st.session_state.get("storage_paths")

if not S or not S.get("users"):
    st.info("No scraped data available. Please run a scrape from the main page first.")
    st.stop()

runs = S["runs"]
users = [u for u in S["users"] if u in runs and runs[u].get("posts")]
if not users:
    st.warning("No valid posts cached. Try scraping again from the main page.")
    st.stop()

STATIC_DIR = (paths or {}).get("STATIC_DIR")

# ---------- Navigation ----------
col1, col2 = st.columns([2, 1])
selected_user = col1.selectbox("Select Profile", users, index=0)
limit = col2.slider("Max posts to display", 5, 50, 20)

data = runs[selected_user]
posts = list(data.get("posts", []))

# ---------- Filters ----------
st.subheader("Filters")

c1, c2 = st.columns(2)
sentiments = c1.multiselect(
    "Caption Sentiment",
    ["POSITIVE", "NEUTRAL", "NEGATIVE"],
    default=["POSITIVE", "NEUTRAL", "NEGATIVE"],
)

media_types = c2.multiselect(
    "Media Type",
    ["image", "video"],
    default=["image", "video"],
)

# ---------- Filter Logic ----------
def keep(item):
    lab = item.get("sentiment_label", "NEUTRAL")
    if lab not in sentiments:
        return False
    mt = "video" if item.get("is_video") else "image"
    if mt not in media_types:
        return False
    return True

filtered = [p for p in posts if keep(p)]

st.caption(f"Showing {min(len(filtered), limit)} of {len(filtered)} posts that match filters.")

# ---------- Display Results ----------
cols = st.columns(3)
for idx, item in enumerate(filtered[:limit]):
    with cols[idx % 3]:
        mt = "video" if item.get("is_video") else "image"
        shortcode = item.get("shortcode", "")
        taken_at = item.get("taken_at", "")[:19].replace("T", " ")

        st.markdown(f"**{shortcode}** · *{mt}*")
        st.caption(f"📅 {taken_at}")

        path = item.get("local_path")
        if mt == "video":
            if path and os.path.exists(path):
                st.video(path)
            else:
                st.info("Video not found")
        else:
            if path and os.path.exists(path):
                st.image(path)
            else:
                st.info("Image not found")

        caption = item.get("caption") or ""
        if caption:
            st.write(caption[:180] + ("…" if len(caption) > 180 else ""))

        lab = item.get("sentiment_label", "NEUTRAL")
        score = item.get("sentiment_score", 0)
        if lab == "POSITIVE":
            st.success(f"Caption sentiment: {lab} ({score:.2f})")
        elif lab == "NEGATIVE":
            st.error(f"Caption sentiment: {lab} ({score:.2f})")
        else:
            st.info(f"Caption sentiment: {lab} ({score:.2f})")

# ---------- Sidebar Debug Info ----------
if paths:
    st.sidebar.caption(f"📁 Shared Temp Folder: {STATIC_DIR}")
