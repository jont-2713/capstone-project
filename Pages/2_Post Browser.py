import os
import streamlit as st
from typing import Optional, Union, List, Tuple

# ---------- Page ----------
st.set_page_config(page_title="Post Browser", layout="wide")
st.title("Post Browser")

# ---------- Guards ----------
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


# ---------- Helpers ----------
def _sentiment_callout(label: Optional[str], text: str) -> None:
    lab = (label or "").upper()
    if lab == "NEGATIVE":
        st.error(text)
    elif lab == "POSITIVE":
        st.success(text)
    else:
        st.warning(text)  # treat NEUTRAL/unknown as warning

def _risk_callout(score: Optional[Union[int, float]], band: Optional[str]) -> None:
    band = (band or "LOW").upper()
    s = int(score or 0)
    if band == "HIGH":
        st.error(f"Risk: {s}/100 ({band})")
    elif band == "MEDIUM":
        st.warning(f"Risk: {s}/100 ({band})")
    else:
        st.success(f"Risk: {s}/100 ({band})")

def _safe_risk(p: dict) -> int:
    return int(p.get("risk_score") or 0)

def _matches_keywords(caption: str, keywords: List[str], require_all: bool) -> bool:
    if not keywords:
        return True
    cap = (caption or "").lower()
    hits = [k for k in keywords if k and k.lower() in cap]
    return (len(hits) == len(keywords)) if require_all else (len(hits) > 0)

def _norm_label(label: Optional[str]) -> str:
    lab = (label or "").upper()
    if lab not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        lab = "NEUTRAL"
    return lab

# For sorting by sentiment: NEG > NEU > POS when "Highest - Lowest"
_SENTIMENT_WEIGHT = {"NEGATIVE": 2, "NEUTRAL": 1, "POSITIVE": 0}

def _sentiment_sort_key(item: dict, source: str) -> Tuple[int, float]:
    """
    Return a key to sort by sentiment.
    source: 'image' or 'text'
    Higher key sorts earlier if reverse=True.
    """
    if source == "image":
        pred = item.get("prediction") or {}
        lab = _norm_label(pred.get("label"))
        conf = float(pred.get("confidence") or 0.0)
    else:
        lab = _norm_label(item.get("sentiment_label"))
        conf = float(item.get("sentiment_score") or 0.0)
    return (_SENTIMENT_WEIGHT.get(lab, 1), conf)

def _passes_sentiment_filter(
    item: dict,
    mode: str,
    allowed_img: List[str],
    allowed_txt: List[str],
) -> bool:
    """
    mode: 'Image only' | 'Text only' | 'Both (AND)' | 'Either (OR)'
    allowed_*: list of allowed labels (POSITIVE/NEUTRAL/NEGATIVE)
    """
    # image
    pred = item.get("prediction") or {}
    img_lab = _norm_label(pred.get("label"))
    # text
    txt_lab = _norm_label(item.get("sentiment_label"))

    img_ok = (img_lab in allowed_img) if allowed_img else True
    txt_ok = (txt_lab in allowed_txt) if allowed_txt else True

    if mode == "Image only":
        return img_ok
    if mode == "Text only":
        return txt_ok
    if mode == "Both (AND)":
        return img_ok and txt_ok
    # Either (OR)
    return img_ok or txt_ok

# ---------- Controls ----------
top_l, top_r = st.columns([3, 2])
with top_l:
    selected_user = st.selectbox("Select profile", USERS, index=0)
data = RUNS[selected_user]
posts = data.get("posts", [])

with top_r:
    st.caption(f"Last updated: {data.get('last_updated','—')}")

# Row 1: risk + keywords
f1, f2, f3 = st.columns([2, 3, 2])
with f1:
    min_risk, max_risk = st.slider(
        "Risk range", 0, 100, (0, 100),
        help="Only show posts whose computed risk is within this range."
    )
with f2:
    kw = st.text_input("Keywords (comma-separated)", placeholder="e.g. launch,mission,storm")
    match_mode = st.radio("Match mode", ["Any keyword", "All keywords"], horizontal=True)
with f3:
    cols_per_row = st.slider("Tiles per row", 2, 5, 3)

# Row 2: sentiment filtering + sorting choice
s1, s2, s3, s4 = st.columns([2, 3, 3, 3])
with s1:
    sent_mode = st.selectbox(
        "Sentiment filter applies to…",
        ["Either (OR)", "Both (AND)", "Image only", "Text only"],
        index=0,
        help="Choose whether to filter by image sentiment, caption sentiment, or both."
    )
with s2:
    allowed_img = st.multiselect(
        "Image sentiment allowed",
        ["POSITIVE", "NEUTRAL", "NEGATIVE"],
        default=["POSITIVE", "NEUTRAL", "NEGATIVE"]
    )
with s3:
    allowed_txt = st.multiselect(
        "Caption sentiment allowed",
        ["POSITIVE", "NEUTRAL", "NEGATIVE"],
        default=["POSITIVE", "NEUTRAL", "NEGATIVE"]
    )
with s4:
    sort_by = st.selectbox(
        "Sort by",
        ["Risk", "Image Sentiment", "Caption Sentiment"],
        index=0
    )
    sort_order = st.radio("Order", ["Highest → Lowest", "Lowest → Highest"], horizontal=True)

# Parse filters
keywords = [k.strip() for k in (kw or "").split(",") if k.strip()]
require_all = (match_mode == "All keywords")
reverse_sort = sort_order.startswith("Highest")

# ---------- Filter ----------
filtered = []
for p in posts:
    r = _safe_risk(p)
    if r < min_risk or r > max_risk:
        continue
    if not _matches_keywords(p.get("caption", ""), keywords, require_all):
        continue
    if not _passes_sentiment_filter(p, sent_mode, allowed_img, allowed_txt):
        continue
    filtered.append(p)

# ---------- Sort ----------
if sort_by == "Risk":
    filtered.sort(key=_safe_risk, reverse=reverse_sort)
elif sort_by == "Image Sentiment":
    # Highest → Lowest means NEGATIVE first (by weight), then confidence
    filtered.sort(key=lambda it: _sentiment_sort_key(it, "image"), reverse=reverse_sort)
else:  # Caption Sentiment
    filtered.sort(key=lambda it: _sentiment_sort_key(it, "text"), reverse=reverse_sort)

# ---------- Summary ----------
st.markdown(f"**{len(filtered)}** / {len(posts)} posts match your filters.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg Risk", f"{data.get('avg_risk', 0)}/100")
rc = data.get("risk_counts", {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
m2.metric("Low", rc.get("LOW", 0))
m3.metric("Medium", rc.get("MEDIUM", 0))
m4.metric("High", rc.get("HIGH", 0))

st.divider()

# ---------- CSS for consistent tile heights ----------
st.markdown(
    """
<style>
div[data-testid="column"] > div:has(div[data-testid="stVerticalBlock"]) { height: 100%; }
div[data-testid="stVerticalBlock"] > div:first-child { height: 100%; }
div.tile-card {
  background-color: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 12px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: 0 0 8px rgba(0,0,0,0.25);
}
div.tile-card img { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Render tiles ----------
st.subheader("Results")

if not filtered:
    st.info("No posts matched the current filters.")
    st.stop()

cols = st.columns(cols_per_row, gap="medium")

for i, item in enumerate(filtered):
    col = cols[i % cols_per_row]
    with col:
        st.markdown('<div class="tile-card">', unsafe_allow_html=True)

        # Header
        kind = "video" if item.get("is_video") else "image"
        taken = (item.get("taken_at") or "")[:19].replace("T", " ")
        st.markdown(f"**{item.get('shortcode','?')}** · *{kind}* · {taken}")

        # Media
        if item.get("is_video"):
            st.video(item.get("local_path"))
        else:
            st.image(item.get("local_path"), use_container_width=True)

        # Caption snippet
        cap = item.get("caption") or ""
        if cap:
            st.write(cap[:180] + ("…" if len(cap) > 180 else ""))

        # --- SCORES (aligned & consistent) ---
        c_img, c_cap, c_risk = st.columns(3)

        # Image sentiment
        pred = item.get("prediction") or {}
        img_lab = _norm_label(pred.get("label"))
        img_conf = float(pred.get("confidence") or 0.0)
        with c_img:
            _sentiment_callout(img_lab, f"Image Sentiment: {img_lab or 'N/A'} ({img_conf:.2f})")

        # Caption sentiment
        cap_lab = _norm_label(item.get("sentiment_label"))
        cap_score = float(item.get("sentiment_score") or 0.0)
        with c_cap:
            _sentiment_callout(cap_lab, f"Caption Sentiment: {cap_lab or 'N/A'} ({cap_score:.2f})")

        # Risk
        with c_risk:
            _risk_callout(item.get("risk_score"), item.get("risk_band"))

        st.markdown('</div>', unsafe_allow_html=True)
