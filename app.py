"""Small Streamlit interface for the trained MRI classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "brain_tumor_mobilenetv2.keras"
CLASSES_PATH = ROOT / "artifacts" / "class_names.json"
IMAGE_SIZE = 224

st.set_page_config(page_title="Brain MRI Classifier | Learning Project", page_icon="🧠", layout="centered")
st.title("Brain MRI Classification")
st.caption("Deep-learning portfolio project · MobileNetV2 transfer learning")
st.warning(
    "Educational demonstration only. This model is not a medical device and must not "
    "be used to diagnose, rule out, or guide treatment for any condition."
)

if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
    st.info("No trained model was found. Download the dataset and run `python train.py` first.")
    st.stop()

@st.cache_resource
def load_classifier(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)

model = load_classifier(str(MODEL_PATH))
class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
upload = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp"])

if upload is not None:
    try:
        image = Image.open(upload).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")
        st.stop()
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Run model prediction", type="primary"):
        resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        batch = np.asarray(resized, dtype=np.float32)[None, ...]
        with st.spinner("Running inference..."):
            probabilities = model.predict(batch, verbose=0)[0]
        ranking = np.argsort(probabilities)[::-1]
        best = int(ranking[0])
        st.subheader("Model output")
        st.metric("Predicted dataset class", class_names[best], f"Model score: {probabilities[best]:.1%}")
        st.caption("The score is the model's softmax output, not calibrated diagnostic confidence.")
        st.write("Class scores")
        scores = pd.DataFrame({
            "Class": [class_names[int(i)] for i in ranking],
            "Score": [float(probabilities[int(i)]) for i in ranking],
        })
        st.bar_chart(scores, x="Class", y="Score")
