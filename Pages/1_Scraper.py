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

# ---------- style-------
st.set_page_config(page_title="Capstone")

# ---------- Image Sentiment Model ----------
MODEL_DIR = os.path.expanduser(r"~\capstone-project\Pages\image-sentiment\model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
META_PATH  = os.path.join(MODEL_DIR, "metadata.json")

@st.cache_resource  # cache across reruns
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
    return np.expand_dims(arr, 0)  # (1, H, W, 3)

def predict_image(img_path: str):
    x = preprocess_for_model(img_path, IMG_SIZE)
    probs = model.predict(x, verbose=0)[0]        # shape: (num_classes,)
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    pred_conf  = float(probs[pred_idx])
    return pred_label, pred_conf, probs.tolist()


# ---------- Setup ----------
L = instaloader.Instaloader()
L.context._session.cookies.set("sessionid", "77091777356%3AGvYRV8iFJcEqAa%3A16%3AAYdkQngjsjwxNRcpQ64z-OKZVr9bX6krJ7y2Y1ZQwA")  # replace with your cookie/session

st.title("Instagram Scraper with Instaloader")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "Static")


# Input UI
usernames = st.text_input("Enter Instagram usernames (comma-separated):")
max_posts = st.slider("Max posts per user", 5, 100, 20)

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

# ---------- Action ----------
if st.button("Download Posts"):
    if not usernames.strip():
        st.warning("Please enter at least one username.")
    else:
        for username in [u.strip() for u in usernames.split(",") if u.strip()]:
            try:
                # Make per-user folder inside Static/
                user_dir = os.path.join(STATIC_DIR, safe_filename(username))
                os.makedirs(user_dir, exist_ok=True)

                profile = instaloader.Profile.from_username(L.context, username)

                st.markdown(f"### **User:** {profile.username}")
                st.write(f"Followers: {profile.followers}")
                st.write(f"Following: {profile.followees}")
                st.write(f"Posts: {profile.mediacount}")

                posts = profile.get_posts()
                meta_out = []

                for i, post in enumerate(posts, start=1):
                    if i > max_posts:
                        break

                    is_video = getattr(post, "is_video", False)
                    media_url = getattr(post, "video_url", None) if is_video else post.url
                    if not media_url:  # fallback if video url not available
                        media_url = post.url
                        is_video = False

                    ext = "mp4" if is_video else "jpg"
                    shortcode = getattr(post, "shortcode", f"{username}_{i}")
                    filename = f"{shortcode}.{ext}"
                    out_path = os.path.join(user_dir, filename)

                    # Download to Static/<username>/
                    try:
                        download_file(media_url, out_path)
                    except Exception as dl_err:
                        st.error(f"Failed to download {username} post {i}: {dl_err}")
                        continue

                    prediction = None

                    # Show & predict
                    if is_video:
                        st.video(out_path)
                    else:
                        # Run model prediction on the saved image
                        pred_label, pred_conf, probs_list = predict_image(out_path)
                        st.image(
                            out_path,
                            caption=(post.caption or "")[:100] + "...",
                            use_container_width=True
                        )
                        st.write(f"**Prediction:** {pred_label}")

                    # Collect metadata + prediction
                    prediction = {
                        "label": pred_label if not is_video else None,
                    }

                    meta_out.append({
                        "username": profile.username,
                        "shortcode": shortcode,
                        "taken_at": datetime.fromtimestamp(post.date_utc.timestamp()).isoformat(),
                        "is_video": is_video,
                        "local_path": out_path,
                        "caption": post.caption or "",
                        "likes": getattr(post, "likes", None),
                        "comments": getattr(post, "comments", None),
                        "prediction": prediction,            # <<— added
                    })

                    # Collect metadata
                    meta_out.append({
                        "username": profile.username,
                        "shortcode": shortcode,
                        "taken_at": datetime.fromtimestamp(post.date_utc.timestamp()).isoformat(),
                        "is_video": is_video,
                        "local_path": out_path,
                        "caption": post.caption or "",
                        "likes": getattr(post, "likes", None),
                        "comments": getattr(post, "comments", None),
                    })

                    time.sleep(0.5)  # be gentle to IG

                # Save metadata.json
                meta_file = os.path.join(user_dir, "metadata.json")
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, indent=2, ensure_ascii=False)

                st.success(f"Saved {len(meta_out)} posts for {username} into {user_dir}")
                st.caption(f"Metadata: {meta_file}")

            except Exception as e:
                st.error(f"Failed for {username}: {e}")
