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

st.title("Instagram Scraper with Instaloader")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "Static")

# Input UI
usernames = st.text_input("Enter Instagram usernames (comma-separated):")
max_posts = st.slider("Max posts per user", 5, 500, 20)

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
        st.warning("Please enter at least one username.")
    else:
        for username in [u.strip() for u in usernames.split(",") if u.strip()]:
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

                profile = instaloader.Profile.from_username(L.context, username)

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

                # (Optional) keep your original quick stats if you like (harmless duplication):
                # st.markdown(f"### **User:** {profile.username}")
                # st.write(f"Followers: {profile.followers}")
                # st.write(f"Following: {profile.followees}")
                # st.write(f"Posts: {profile.mediacount}")


                posts = profile.get_posts()
                meta_out = []
                sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

                for i, post in enumerate(posts, start=1):

                    # # ==============================================
                    # # # Helper variables

                    caption = post.caption or ""
                    # hashtags = re.findall(r"#(\w+)", caption)
                    mentions = re.findall(r"@(\w+)", caption)

                    # # loc = post.location
                    # # location_data = (
                    # #     {
                    # #         "name": getattr(loc, "name", None),
                    # #         "slug": getattr(loc, "slug", None),
                    # #         "lat": getattr(loc, "_lat", None),
                    # #         "lng": getattr(loc, "_lng", None),
                    # #     }
                    # #     if loc else None
                    # # )

                    # if post.typename == "GraphSidecar":
                    #     media_urls = [n.url for n in post.get_sidecar_nodes()]
                    # else:
                    #     media_urls = [post.video_url] if post.is_video and getattr(post, "video_url", None) else [post.url]
                    # # ==============================================


                    if i > max_posts:
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

                    # Image sentiment
                    if is_video:
                        st.video(out_path)
                    else:
                        pred_label, pred_conf, probs_list = predict_image(out_path)
                        st.image(out_path, caption=(post.caption or "")[:100] + "...")
                        st.write(f"**Prediction (Image):** {pred_label}")
                        prediction = {"label": pred_label}

                    # Text sentiment
                    caption = post.caption or ""
                    sentiment = analyze_sentiment(caption)
                    sentiment_counts[sentiment["label"]] += 1
                    if caption:
                        if sentiment["label"] == "POSITIVE":
                            st.success(f"Caption Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
                        elif sentiment["label"] == "NEGATIVE":
                            st.error(f"Caption Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")
                        else:
                            st.info(f"Caption Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")

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
                        # --- Added extended metadata ---
                        # "date_utc": post.date_utc.replace(tzinfo=timezone.utc).isoformat(),
                        # "hashtags": hashtags,
                        "mentions": mentions,
                        # "media_urls": [n.url for n in post.get_sidecar_nodes()] if post.typename=="GraphSidecar" else [post.url],
                        # # "media_urls": media_urls,
                        "tagged_users": getattr(post, "tagged_users", [])
                    })

                    time.sleep(0.5)

                # Save metadata
                meta_file = os.path.join(user_dir, "metadata.json")
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, indent=2, ensure_ascii=False)

                st.write("### Overall Caption Sentiment")
                st.write(sentiment_counts)

                st.success(f"Saved {len(meta_out)} posts for {username} into {user_dir}")
                st.caption(f"Metadata: {meta_file}")

            except Exception as e:
                st.error(f"Failed for {username}: {e}")
