import os
import json
import torch
import numpy as np
from PIL import Image
import streamlit as st


# === Model Paths ===
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "image-sentiment", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_mvsa.torchscript.pt")
META_PATH  = os.path.join(MODEL_DIR, "resnet50_mvsa.json")


@st.cache_resource
def load_image_sentiment_model():
    """Load TorchScript model + metadata (cached)."""
    # Load model
    model = torch.jit.load(MODEL_PATH, map_location="cpu").eval()

    # Load metadata 
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    label_names = meta.get("classes", ["Negative", "Neutral", "Positive"])
    img_size = int(meta.get("image_size", [224, 224])[0])
    mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    return model, label_names, img_size, mean, std


# Model and constants
MODEL, LABELS, IMG_SIZE, MEAN, STD = load_image_sentiment_model()


def preprocess_for_model(img_path: str) -> torch.Tensor:
    # Prepare image for TorchScript ResNet model
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1)) 
    return torch.from_numpy(arr).unsqueeze(0)  

def predict_image(img_path: str):
    # Run inference and return label, confidence, and probability list
    x = preprocess_for_model(img_path)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], float(probs[pred_idx]), probs.tolist()
