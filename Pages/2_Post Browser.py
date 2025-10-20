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

/* --- badge and sentiment colors --- */
/* --- badge and sentiment colors --- */
.stats-row {
  margin-top: auto;
  display: flex;
  flex-wrap: wrap;           /* allow badges to wrap to a new line */
  gap: 4px 6px;              /* row/column gap */
  align-items: center;
  height: auto;              /* don't clamp row height */
}

.badge {
  font-size: 16px;           /* keep requested size */
  line-height: 20px;         /* a touch of breathing room */
  padding: 2px 6px;
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.2);

  /* make each badge shrink if needed and avoid overflow */
  flex: 0 1 auto;            /* allow shrinking but don't force stretching */
  min-width: 0;              /* enable flexbox text truncation */
  max-width: 100%;
  white-space: nowrap;       /* keep each badge on one line */
  overflow: hidden;          /* hide overflowed text */
  text-overflow: ellipsis;   /* show … when truncated */
}

.badge.pos    { background-color:#21A36633; border-color:#21A366; }
.badge.neu    { background-color:#F1C40F33; border-color:#F1C40F; }
.badge.neg    { background-color:#E74C3C33; border-color:#E74C3C; }
.badge.low    { background-color:#21A36633; border-color:#21A366; }
.badge.medium { background-color:#F1C40F33; border-color:#F1C40F; }
.badge.high   { background-color:#E74C3C33; border-color:#E74C3C; }

</style>
""",
    unsafe_allow_html=True,
)

# ---------- Render tiles ----------
st.subheader("Results")

if not filtered:
    st.info("No posts matched the current filters.")
    st.stop()

# render in complete rows
for start in range(0, len(filtered), cols_per_row):
    row_items = filtered[start:start + cols_per_row]
    cols = st.columns(len(row_items), gap="medium")
    for col, item in zip(cols, row_items):
        with col:
            st.markdown('<div class="tile-card">', unsafe_allow_html=True)

            # Header
            short = item.get("shortcode", "?")
            taken = (item.get("taken_at") or "")[:19].replace("T", " ")
            st.markdown(f"**{short}** · *image* · {taken}")

            # Fixed-size image
            st.markdown('<div class="media-box">', unsafe_allow_html=True)
            st.image(item.get("local_path"), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

            # Caption
            cap = (item.get("caption") or "").strip()
            snippet = cap[:280] + ("…" if len(cap) > 280 else "")
            st.markdown(f'<div class="caption-box">{snippet}</div>', unsafe_allow_html=True)

            import textwrap

            # --- Badges row ---
            pred = item.get("prediction") or {}
            img_lab = _norm_label(pred.get("label"))
            img_conf = float(pred.get("confidence") or 0.0)
            cap_lab = _norm_label(item.get("sentiment_label"))
            cap_conf = float(item.get("sentiment_score") or 0.0)
            risk_score = int(item.get("risk_score") or 0)
            risk_band = (item.get("risk_band") or "LOW").upper()

            lab_map = {"POSITIVE": "pos", "NEUTRAL": "neu", "NEGATIVE": "neg"}
            risk_map = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}

            badges = textwrap.dedent(f"""
            <div class="stats-row">
            <span class="badge {lab_map.get(img_lab,'neu')}">Image: {img_lab} ({img_conf:.2f})</span>
            <span class="badge {lab_map.get(cap_lab,'neu')}">Caption: {cap_lab} ({cap_conf:.2f})</span>
            <span class="badge {risk_map.get(risk_band,'low')}">Risk: {risk_score}/100 ({risk_band})</span>
            </div>
            """)
            st.markdown(badges, unsafe_allow_html=True)


            st.markdown('</div>', unsafe_allow_html=True)
