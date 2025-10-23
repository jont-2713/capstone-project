import streamlit as st
import textwrap

from Pages.Utilities.utils_browser import (
    safe_risk, match_keywords, normalize_label, sentiment_sort_key,
    passes_sentiment_filter, TILE_CSS, LAB_MAP, RISK_MAP
)

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Post Browser", layout="wide")
st.title("Post Browser")

# -----------------------------------------------------------------------------
# Guards — ensure scrape_state exists
# -----------------------------------------------------------------------------
if "scrape_state" not in st.session_state:
    st.info("No scrape state found. Go to the Scraper page, run a scrape, then come back.")
    if st.button("Go to Scraper", key="go_scraper__no_state"):
        st.switch_page("Pages/1_Scraper.py")
    st.stop()

S = st.session_state.scrape_state
RUNS = S.get("runs", {})
USERS = [u for u in S.get("users", []) if u in RUNS] or list(RUNS.keys())

if not USERS or not RUNS:
    st.info("No cached data found. Please scrape some users first.")
    if st.button("Go to Scraper", key="go_scraper__no_cache"):
        st.switch_page("Pages/1_Scraper.py")
    st.stop()

# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
top_l, top_r = st.columns([3, 2])
with top_l:
    selected_user = st.selectbox("Select profile", USERS, index=0)
data = RUNS[selected_user]
posts = data.get("posts", [])

with top_r:
    st.caption(f"Last updated: {data.get('last_updated', '—')}")

# --- Risk, Keywords, Sentiment, Sorting Controls ---
f1, f2, f3 = st.columns([2, 3, 2])
with f1:
    min_risk, max_risk = st.slider("Risk range", 0, 100, (0, 100))
with f2:
    kw_input = st.text_input("Keywords (comma-separated)", placeholder="e.g. launch, mission, storm")
    match_mode = st.radio("Match mode", ["Any keyword", "All keywords"], horizontal=True)
with f3:
    cols_per_row = st.slider("Tiles per row", 2, 5, 3)

s1, s2, s3, s4 = st.columns([2, 3, 3, 3])
with s1:
    sent_mode = st.selectbox("Sentiment filter applies to…",
                             ["Either (OR)", "Both (AND)", "Image only", "Text only"])
with s2:
    allowed_img = st.multiselect("Image sentiment allowed",
                                 ["POSITIVE", "NEUTRAL", "NEGATIVE"],
                                 default=["POSITIVE", "NEUTRAL", "NEGATIVE"])
with s3:
    allowed_txt = st.multiselect("Caption sentiment allowed",
                                 ["POSITIVE", "NEUTRAL", "NEGATIVE"],
                                 default=["POSITIVE", "NEUTRAL", "NEGATIVE"])
with s4:
    sort_by = st.selectbox("Sort by", ["Risk", "Image Sentiment", "Caption Sentiment"])
    sort_order = st.radio("Order", ["Highest → Lowest", "Lowest → Highest"], horizontal=True)

# -----------------------------------------------------------------------------
# Filtering and Sorting
# -----------------------------------------------------------------------------
keywords = [k.strip() for k in (kw_input or "").split(",") if k.strip()]
require_all = match_mode == "All keywords"
reverse_sort = sort_order.startswith("Highest")

filtered = [
    p for p in posts
    if min_risk <= safe_risk(p) <= max_risk
    and match_keywords(p.get("caption", ""), keywords, require_all)
    and passes_sentiment_filter(p, sent_mode, allowed_img, allowed_txt)
]

if sort_by == "Risk":
    filtered.sort(key=safe_risk, reverse=reverse_sort)
elif sort_by == "Image Sentiment":
    filtered.sort(key=lambda x: sentiment_sort_key(x, "image"), reverse=reverse_sort)
else:
    filtered.sort(key=lambda x: sentiment_sort_key(x, "text"), reverse=reverse_sort)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
st.markdown(f"**{len(filtered)}** / {len(posts)} posts match your filters.")
m1, m2, m3, m4 = st.columns(4)
rc = data.get("risk_counts", {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
m1.metric("Avg Risk", f"{data.get('avg_risk', 0)}/100")
m2.metric("Low", rc.get("LOW", 0))
m3.metric("Medium", rc.get("MEDIUM", 0))
m4.metric("High", rc.get("HIGH", 0))
st.divider()

# -----------------------------------------------------------------------------
# CSS and Results
# -----------------------------------------------------------------------------
st.markdown(TILE_CSS, unsafe_allow_html=True)
st.subheader("Results")

if not filtered:
    st.info("No posts matched the current filters.")
    st.stop()

for start in range(0, len(filtered), cols_per_row):
    row_items = filtered[start:start + cols_per_row]
    cols = st.columns(len(row_items), gap="medium")

    for col, item in zip(cols, row_items):
        with col:
            st.markdown('<div class="tile-card">', unsafe_allow_html=True)
            short = item.get("shortcode", "?")
            taken = (item.get("taken_at") or "")[:19].replace("T", " ")
            st.markdown(f"**{short}** · *image* · {taken}")
            st.image(item.get("local_path"), width='stretch')

            cap = (item.get("caption") or "").strip()
            snippet = cap[:280] + ("…" if len(cap) > 280 else "")
            st.markdown(f'<div class="caption-box">{snippet}</div>', unsafe_allow_html=True)

            pred = item.get("prediction") or {}
            img_lab = normalize_label(pred.get("label"))
            img_conf = float(pred.get("confidence") or 0.0)
            cap_lab = normalize_label(item.get("sentiment_label"))
            cap_conf = float(item.get("sentiment_score") or 0.0)
            risk_score = int(item.get("risk_score") or 0)
            risk_band = (item.get("risk_band") or "LOW").upper()

            badges = f"""
            <div class="stats-row">
              <span class="badge {LAB_MAP.get(img_lab,'neu')}">Image: {img_lab} ({img_conf:.2f})</span>
              <span class="badge {LAB_MAP.get(cap_lab,'neu')}">Caption: {cap_lab} ({cap_conf:.2f})</span>
              <span class="badge {RISK_MAP.get(risk_band,'low')}">Risk: {risk_score}/100 ({risk_band})</span>
            </div>
            """
            st.markdown(textwrap.dedent(badges), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
