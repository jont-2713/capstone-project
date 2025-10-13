import os
import json
import time
import requests
import streamlit as st
import instaloader
import re
from datetime import datetime
from datetime import timezone
import numpy as np
from PIL import Image
from tensorflow import keras
import tempfile
import shutil
import atexit
from typing import Optional, Tuple 

# ---------- style-------
st.set_page_config(page_title="Capstone")

# --------- Text Sentiment Model ------------
from Pages.sentiment import analyze_sentiment

# ---------- Image Sentiment Model ----------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "image-sentiment", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")

@st.cache_resource
def load_model_and_meta():
    model = keras.models.load_model(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    label_names = meta["label_names"]
    img_size = int(meta["img_size"])
    return model, label_names, img_size

model, LABEL_NAMES, IMG_SIZE = load_model_and_meta()

def preprocess_for_model(img_path: str, img_size: int) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict_image(img_path: str):
    x = preprocess_for_model(img_path, IMG_SIZE)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    pred_conf  = float(probs[pred_idx])
    return pred_label, pred_conf, probs.tolist()

# ---------- Setup ----------
L = instaloader.Instaloader()
L.context._session.cookies.set(
    "sessionid",
    "76921770876%3A697JYvfmJGdmjA%3A9%3AAYhTosD0CEvVeTTt8771fhJUTLcz0KH6TteKjbu5BQ"
)

st.title("Instagram Scraper")

# ----------  STATIC DIR (auto-deletes on exit) ----------
TEMP_ROOT = tempfile.mkdtemp(prefix="streamlit_static_")
STATIC_DIR = os.path.join(TEMP_ROOT, "Static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Input UI
# Old Lines - reinstate in case of error
# usernames = st.text_input("Enter Instagram usernames (comma-separated):")
# max_posts = st.slider("Max posts per user", 5, 500, 20)
@atexit.register
def _cleanup_temp_root():
    try:
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)
        print(f"[cleanup] removed {TEMP_ROOT}")
    except Exception as e:
        print(f"[cleanup] failed to remove {TEMP_ROOT}: {e}")

st.session_state["storage_paths"] = {
    "TEMP_ROOT": TEMP_ROOT,
    "STATIC_DIR": STATIC_DIR,
}

st.sidebar.caption(f"Temp folder (auto-delete): {STATIC_DIR}")

# ---------- Session State  ----------
def _init_state():
    if "scrape_state" not in st.session_state:
        st.session_state.scrape_state = {
            "last_input": "",
            "max_posts": 20,
            "users": [],
            "runs": {},  # username -> cached results
        }
_init_state()
S = st.session_state.scrape_state

# ---------- Input UI ----------
usernames_raw = st.text_area(
    "Enter Instagram usernames (comma, space, or newline-separated):",
    value=S["last_input"],
    key="usernames_raw_input",
    placeholder="e.g. nasa, natgeo\nsomeuser anotheruser"
)
S["last_input"] = st.session_state.get("usernames_raw_input", "")

max_posts = st.slider("Max posts per user", 1, 100, S["max_posts"], key="max_posts_slider")
S["max_posts"] = max_posts

def parse_usernames(s: str):
    parts = [p.strip() for chunk in s.replace(",", " ").split() for p in [chunk]]
    seen, out = set(), []
    for p in parts:
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out

# ---------- Helpers ----------
def safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".")).rstrip()

def download_file(url: str, out_path: str, timeout: int = 30):
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# === Save profile data for a fast header =================
def save_profile_header(profile, user_dir: str):
    """
    Persist a compact profile.json and profile.jpg so the header can render quickly
    and work on subsequent runs without re-scraping. Non-fatal if avatar download fails.
    """
    # Write small JSON with the essentials for the header
    profile_json = {
        "username": profile.username,
        "full_name": profile.full_name,
        "biography": profile.biography,
        "followers": profile.followers,
        "followees": profile.followees,
        "mediacount": profile.mediacount,
        "userid": profile.userid,
        "external_url": profile.external_url,
        "is_private": profile.is_private,
        "is_verified": profile.is_verified,
    }
    with open(os.path.join(user_dir, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile_json, f, ensure_ascii=False, indent=2)

    # Try to download the avatar once; if it fails, we still show stats/bio
    profile_pic_path = os.path.join(user_dir, "profile.jpg")
    try:
        download_file(profile.profile_pic_url, profile_pic_path)
    except Exception:
        profile_pic_path = None  # silently continue without an image

    return profile_json, profile_pic_path
# ================================================================================ 
# Decimal formatter for counts in summary
# === Compact count formatter: 67,165 -> "67k", 28,316,466 -> "28.3m" ==========
def fmt_compact(n):
    n = float(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n < 1_000:
        s = f"{n:,.0f}"                    # e.g., 532
    elif n < 1_000_000:
        s = f"{n/1_000:,.0f}k"             # e.g., 67k
    elif n < 1_000_000_000:
        s = f"{n/1_000_000:.1f}m"          # e.g., 28.3m
    elif n < 1_000_000_000_000:
        s = f"{n/1_000_000_000:.1f}b"      # e.g., 1.2b
    else:
        s = f"{n/1_000_000_000_000:.1f}t"  # e.g., 3.4t
    # drop trailing .0 for clean look (e.g., "12.0m" -> "12m")
    for unit in ("m","b","t"):
        if s.endswith(f".0{unit}"):
            s = s.replace(f".0{unit}", unit)
    return sign + s
# ==============================================================================


# ---------- Action ----------


if st.button("Download Posts"):
    if not usernames.strip():
# ---------- Risk scoring (0..100) ----------
def compute_risk(text_label: str, text_score: float,
                 img_label: Optional[str], img_conf: Optional[float]) -> Tuple[int, str]:
    """
    Heuristic:
      - Start at 50 (neutral baseline)
      - Text NEGATIVE raises risk up to +30, POSITIVE lowers up to -20
      - Image 'negative' raises up to +25, 'positive' lowers up to -15
      - Image 'neutral' adds a small bump
      - Clamp 0..100, then band to LOW/MEDIUM/HIGH
    """
    text_label = (text_label or "NEUTRAL").upper()
    text_score = float(text_score or 0.5)

    score = 50.0
    if text_label == "NEGATIVE":
        score += 30.0 * text_score        # 50..80
    elif text_label == "POSITIVE":
        score -= 20.0 * text_score        # 50..30
    else:
        score += 5.0                       # mild bump

    if img_label:
        il = img_label.lower()
        c = float(img_conf or 0.0)
        if il == "negative":
            score += 25.0 * c
        elif il == "positive":
            score -= 15.0 * c
        elif il == "neutral":
            score += 5.0 * c

    score = max(0.0, min(100.0, score))
    band = "LOW" if score < 33 else ("MEDIUM" if score <= 66 else "HIGH")
    return int(round(score)), band

# ---------- Actions ----------
colA, colB = st.columns([1, 1])
do_scrape = colA.button("Download / Update Posts")
do_clear  = colB.button("Clear cached results")

if do_clear:
    S["users"].clear()
    S["runs"].clear()
    st.success("Cleared cached results")

if do_scrape:
    users = parse_usernames(S["last_input"])
    if not users:
        st.warning("Please enter at least one username.")
    else:
        for u in users:
            if u not in S["users"]:
                S["users"].append(u)

        st.success(f"Scraping {len(users)} user(s): {', '.join(users)}")

        for username in users:
            try:
                # user_dir = os.path.join(STATIC_DIR, safe_filename(username))
                # os.makedirs(user_dir, exist_ok=True)

                # profile = instaloader.Profile.from_username(L.context, username)
                # st.markdown(f"### **User:** {profile.username}")
                # st.write(f"Followers: {profile.followers}")
                # st.write(f"Following: {profile.followees}")
                # st.write(f"Posts: {profile.mediacount}")

                user_dir = os.path.join(STATIC_DIR, safe_filename(username))
                os.makedirs(user_dir, exist_ok=True)

                with st.status(f"Fetching profile: {username}", expanded=True) as status:
                    profile = instaloader.Profile.from_username(L.context, username)
                    status.update(label=f"Fetched profile: {username}")

                # === NEW: persist a tiny profile bundle for the header (profile.json + profile.jpg)
                header_json, header_pic_path = save_profile_header(profile, user_dir)

                # === NEW: visual profile header (avatar + stats + bio) ==========================
                hc1, hc2 = st.columns([1, 4])
                with hc1:
                    if header_pic_path and os.path.exists(header_pic_path):
                        st.image(header_pic_path, width=96)  # small, fast to render
                with hc2:
                    st.markdown(f"### @{header_json['username']}")
                    st.markdown(
                        f"**{fmt_compact(header_json['mediacount'])}** posts &nbsp;&nbsp; "
                        f"**{fmt_compact(header_json['followers'])}** followers &nbsp;&nbsp; "
                        f"**{fmt_compact(header_json['followees'])}** following"
                        # f"**{header_json['mediacount']:,}** posts &nbsp;&nbsp; "
                        # f"**{header_json['followers']:,}** followers &nbsp;&nbsp; "
                        # f"**{header_json['followees']:,}** following"
                    )
                    if header_json.get("biography"):
                        st.caption(header_json["biography"])
                st.markdown("---")
                # ===============================================================================

        
                    
                # --- profile picture: get URL and download local copy ---\
                profile_pic_url_obj = getattr(profile, "profile_pic_url", None) or getattr(profile, "profile_pic_url_hd", None)
                profile_pic_url = None
                profile_pic_path = None
                if profile_pic_url_obj:
                    profile_pic_url = profile_pic_url_obj.url if hasattr(profile_pic_url_obj, "url") else str(profile_pic_url_obj)
                    try:
                        profile_pic_path = os.path.join(user_dir, "_profile.jpg")
                        download_file(profile_pic_url, profile_pic_path)
                        st.write(f"Saved profile image to: {profile_pic_path}")
                    except Exception as e:
                        st.warning(f"Could not download profile pic for {username}: {e}")
                        profile_pic_path = None

                posts = profile.get_posts()
                meta_out = []

                # ---- risk aggregates (per profile) ----
                risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
                risk_sum = 0

                total = min(S["max_posts"], profile.mediacount or S["max_posts"])
                prog = st.progress(0, text=f"Downloading up to {total} posts for @{profile.username}…")

                for i, post in enumerate(posts, start=1):

                    # # ==============================================
                    # # # Helper variables

                    caption = post.caption or ""
                    mentions = re.findall(r"@(\w+)", caption)


                    if i > S["max_posts"]:
                        break

                    is_video = getattr(post, "is_video", False)
                    media_url = getattr(post, "video_url", None) if is_video else post.url
                    if not media_url:
                        media_url = post.url
                        is_video = False

                    ext = "mp4" if is_video else "jpg"
                    shortcode = getattr(post, "shortcode", f"{username}_{i}")
                    filename = f"{shortcode}.{ext}"
                    out_path = os.path.join(user_dir, filename)

                    try:
                        download_file(media_url, out_path)
                    except Exception as dl_err:
                        st.error(f"Failed to download {username} post {i}: {dl_err}")
                        continue

                    prediction = None
                    if not is_video:
                        pred_label, pred_conf, probs_list = predict_image(out_path)
                        prediction = {"label": pred_label, "confidence": pred_conf, "probs": probs_list}

                    caption = post.caption or ""
                    sentiment = analyze_sentiment(caption)

                    # image sentiment parts (may be None for videos)
                    img_label = prediction["label"] if prediction else None
                    img_conf  = prediction["confidence"] if prediction else None

                    # --- compute risk ---
                    risk_score, risk_band = compute_risk(
                        text_label=sentiment["label"],
                        text_score=sentiment["score"],
                        img_label=img_label,
                        img_conf=img_conf
                    )
                    risk_sum += risk_score
                    risk_counts[risk_band] += 1

                    meta_out.append({
                        "username": profile.username,
                        "shortcode": shortcode,
                        "taken_at": datetime.fromtimestamp(post.date_utc.timestamp()).isoformat(),
                        "is_video": is_video,
                        "local_path": out_path,
                        "caption": caption,
                        "likes": getattr(post, "likes", None),
                        "comments": getattr(post, "comments", None),
                        "prediction": prediction,                 # image sentiment (if any)
                        "sentiment_label": sentiment["label"],    # text sentiment
                        "sentiment_score": sentiment["score"],
                        # --- Added extended metadata ---
                        # "date_utc": post.date_utc.replace(tzinfo=timezone.utc).isoformat(),
                        # "hashtags": hashtags,
                        "mentions": mentions,
                        # "media_urls": [n.url for n in post.get_sidecar_nodes()] if post.typename=="GraphSidecar" else [post.url],
                        # # "media_urls": media_urls,
                        "tagged_users": getattr(post, "tagged_users", [])
                        "risk_score": risk_score,                  # NEW
                        "risk_band": risk_band,                    # NEW
                    })

                    time.sleep(0.4)
                    prog.progress(min(i, total) / total, text=f"Downloaded {min(i, total)}/{total}")

                # write metadata to disk inside temp user dir
                meta_file = os.path.join(user_dir, "metadata.json")
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, indent=2, ensure_ascii=False)

                # cache this run in session_state for page nav restore
                S["runs"][username] = {
                    "username": profile.username,
                    "followers": profile.followers,
                    "followees": profile.followees,
                    "mediacount": profile.mediacount,
                    "risk_counts": risk_counts,                               # NEW
                    "avg_risk": int(round(risk_sum / max(1, len(meta_out)))) if meta_out else 0,  # NEW
                    "user_dir": user_dir,
                    "meta_file": meta_file,
                    "posts": meta_out,
                    "max_posts_used": S["max_posts"],
                    "last_updated": datetime.now().isoformat(timespec="seconds"),
                    "profile_pic_url": profile_pic_url,
                    "profile_pic_path": profile_pic_path,
                }

                st.success(f"Saved {len(meta_out)} posts for **{username}** → `{user_dir}`")

            except Exception as e:
                st.error(f"Failed for {username}: {e}")

# ---------- Render from cache  ----------
if S["users"]:
    st.markdown("### Results Preview")
    
    tabs = st.tabs(S["users"])
    for tab, username in zip(tabs, S["users"]):
        with tab:
            data = S["runs"].get(username)
            if not data:
                st.info("No cached data yet for this user. Click Download / Update Posts.")
                continue


                    # ---------- Display profile header ----------
            st.markdown(f"### {data['username']}")
            try:
                profile_pic_path = data.get("profile_pic_path")
                profile_pic_url = data.get("profile_pic_url")

                if profile_pic_path and os.path.exists(profile_pic_path):
                    st.image(profile_pic_path, width=120, caption=f"@{data['username']}")
                elif profile_pic_url:
                    if hasattr(profile_pic_url, "url"):
                        profile_pic_url = profile_pic_url.url
                    st.image(str(profile_pic_url), width=120, caption=f"@{data['username']}")
                else:
                    st.caption("(No profile picture found)")
            except Exception as e:
                st.caption(f"(Profile image unavailable: {e})")                 

            if st.button("Click to browse Posts"):
                st.switch_page("Pages/2_Post Browser.py")

            st.caption(f"Last updated: {data['last_updated']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Followers", f"{data['followers']:,}")
            col2.metric("Following", f"{data['followees']:,}")
            col3.metric("Posts", f"{data['mediacount']:,}")

            # ---- Risk summary ----
            st.markdown("#### Risk Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Risk", f"{data.get('avg_risk', 0)}/100")
            rc = data.get("risk_counts", {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
            c2.metric("Low", rc.get("LOW", 0))
            c3.metric("Medium", rc.get("MEDIUM", 0))
            c4.metric("High", rc.get("HIGH", 0))

            st.markdown("#### Recent items")
            for item in data["posts"][: min(len(data["posts"]), 10)]:
                with st.expander(f"{item['shortcode']} — {('video' if item['is_video'] else 'image')}", expanded=False):
                    if item["is_video"]:
                        st.video(item["local_path"])
                    else:
                        st.image(item["local_path"])
                        if item.get("prediction"):
                            st.write(f"**Image Sentiment:** {item['prediction']['label'].upper()} ({item['prediction']['confidence']:.2f})")


                    cap = item.get("caption") or ""
                    if cap:
                        st.write(cap[:200] + ("…" if len(cap) > 200 else ""))

                    # caption sentiment callout (context)
                    lab = item.get("sentiment_label")
                    score = item.get("sentiment_score")
                    if lab == "POSITIVE":
                        st.write(f"Caption Sentiment: {lab} ({score:.2f})")
                    elif lab == "NEGATIVE":
                        st.write(f"Caption Sentiment: {lab} ({score:.2f})")
                    else:
                        st.info(f"Caption Sentiment: {lab} ({score:.2f})")

                    # risk badge
                    r = item.get("risk_score")
                    rb = item.get("risk_band")
                    if r is not None and rb:
                        if rb == "HIGH":
                            st.error(f"Risk: {r}/100 ({rb})")
                        elif rb == "MEDIUM":
                            st.warning(f"Risk: {r}/100 ({rb})")
                        else:
                            st.success(f"Risk: {r}/100 ({rb})")

            st.caption(f"Metadata file: {data['meta_file']}")
else:
    st.info("No cached runs yet. Add usernames and click **Download / Update Posts**.")
