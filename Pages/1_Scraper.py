import os
import json
import time
import requests
import streamlit as st
import instaloader
from datetime import datetime
import numpy as np
from PIL import Image
from tensorflow import keras

# NEW: ephemeral storage bits
import tempfile
import shutil
import atexit

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
    "77091777356%3AGvYRV8iFJcEqAa%3A16%3AAYdkQngjsjwxNRcpQ64z-OKZVr9bX6krJ7y2Y1ZQwA"
)

st.title("Instagram Scraper with Instaloader")

# ---------- EPHEMERAL STATIC DIR (auto-deletes on exit) ----------
# Create a unique temp root and a "Static" subdir
TEMP_ROOT = tempfile.mkdtemp(prefix="streamlit_static_")
STATIC_DIR = os.path.join(TEMP_ROOT, "Static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Register cleanup when the Python process exits
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

# show where files live during runtime
st.sidebar.caption(f"📁 Temp folder (auto-delete): {STATIC_DIR}")

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
                user_dir = os.path.join(STATIC_DIR, safe_filename(username))
                os.makedirs(user_dir, exist_ok=True)

                with st.status(f"Fetching profile: {username}", expanded=True) as status:
                    profile = instaloader.Profile.from_username(L.context, username)
                    status.update(label=f"Fetched profile: {username}")

                posts = profile.get_posts()
                meta_out = []
                sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

                total = min(S["max_posts"], profile.mediacount or S["max_posts"])
                prog = st.progress(0, text=f"Downloading up to {total} posts for @{profile.username}…")

                for i, post in enumerate(posts, start=1):
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
                    sentiment_counts[sentiment["label"]] += 1

                    meta_out.append({
                        "username": profile.username,
                        "shortcode": shortcode,
                        "taken_at": datetime.fromtimestamp(post.date_utc.timestamp()).isoformat(),
                        "is_video": is_video,
                        "local_path": out_path,
                        "caption": caption,
                        "likes": getattr(post, "likes", None),
                        "comments": getattr(post, "comments", None),
                        "prediction": prediction,
                        "sentiment_label": sentiment["label"],
                        "sentiment_score": sentiment["score"],
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
                    "sentiment_counts": sentiment_counts,
                    "user_dir": user_dir,
                    "meta_file": meta_file,
                    "posts": meta_out,
                    "max_posts_used": S["max_posts"],
                    "last_updated": datetime.now().isoformat(timespec="seconds"),
                }

                st.success(f"Saved {len(meta_out)} posts for **{username}** → `{user_dir}`")

            except Exception as e:
                st.error(f"Failed for {username}: {e}")

# ---------- Render from cache (survives page navigation) ----------
if S["users"]:
    st.markdown("### Cached results")
    tabs = st.tabs(S["users"])
    for tab, username in zip(tabs, S["users"]):
        with tab:
            data = S["runs"].get(username)
            if not data:
                st.info("No cached data yet for this user. Click Download / Update Posts.")
                continue

            st.caption(f"Last updated: {data['last_updated']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Followers", f"{data['followers']:,}")
            col2.metric("Following", f"{data['followees']:,}")
            col3.metric("Posts", f"{data['mediacount']:,}")

            st.markdown("#### Overall Caption Sentiment")
            st.write(data["sentiment_counts"])

            st.markdown("#### Recent items")
            for item in data["posts"][: min(len(data["posts"]), 10)]:
                with st.expander(f"{item['shortcode']} — {('video' if item['is_video'] else 'image')}", expanded=False):
                    if item["is_video"]:
                        st.video(item["local_path"])
                    else:
                        st.image(item["local_path"])
                        if item["prediction"]:
                            st.write(f"**Prediction (Image):** {item['prediction']['label']} ({item['prediction']['confidence']:.2f})")

                    cap = item["caption"] or ""
                    if cap:
                        st.write(cap[:200] + ("…" if len(cap) > 200 else ""))
                    lab = item["sentiment_label"]
                    score = item["sentiment_score"]
                    if lab == "POSITIVE":
                        st.success(f"Caption Sentiment: {lab} ({score:.2f})")
                    elif lab == "NEGATIVE":
                        st.error(f"Caption Sentiment: {lab} ({score:.2f})")
                    else:
                        st.info(f"Caption Sentiment: {lab} ({score:.2f})")

            st.caption(f"Metadata file: {data['meta_file']}")
else:
    st.info("No cached runs yet. Add usernames and click **Download / Update Posts**.")
