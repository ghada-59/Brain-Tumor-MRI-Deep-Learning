
"""Streamlit interface for the Brain MRI Classification learning project."""
from __future__ import annotations

import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.brain_tumor_dl.inference import preprocess_pil_image
from src.brain_tumor_dl.model_paths import resolve_mobilenet_path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = resolve_mobilenet_path(ROOT)
CLASSES_PATH = ROOT / "artifacts" / "class_names.json"
IMAGE_SIZE = 224

st.set_page_config(
    page_title="NeuroVision | Brain MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8fc 0%, #ffffff 55%);
    }
    [data-testid="stHeader"] {
        background: rgba(244, 248, 252, 0.9);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #102b46, #176b87);
        padding: 2rem 2.2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.3rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.15rem;
        margin: 0.3rem 0;
    }
    .hero p {
        color: #e0edf5;
        margin-bottom: 0;
    }
    .eyebrow {
        color: #a9e5ed;
        font-size: 0.78rem;
        letter-spacing: 0.13em;
        font-weight: 700;
        text-transform: uppercase;
    }
    .section-heading {
        color: #173650;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0.4rem 0 0.8rem 0;
    }
    .note {
        background: #edf5fb;
        border-left: 4px solid #287b9b;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        color: #24445c;
        font-size: 0.91rem;
    }
    .small-muted {
        color: #65798a;
        font-size: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce7f0;
        padding: 1rem;
        border-radius: 14px;
    }
    div.stButton > button[kind="primary"] {
        background: #176b87;
        border: 0;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Header ---------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Biomedical AI · Medical Imaging</div>
        <h1>NeuroVision</h1>
        <p>Brain MRI image classification using deep learning and MobileNetV2</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note">
        <strong>Educational project only.</strong> This application is not a
        medical device. Its predictions must not be used to diagnose or exclude
        a brain tumour, or to make treatment decisions. Only a qualified
        healthcare professional can interpret medical imaging in context.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("## 🧠 NeuroVision")
    st.caption("Brain MRI · Deep learning")
    st.divider()
    st.markdown("**Project information**")
    st.write("Architecture: MobileNetV2")
    st.write(f"Model input: {IMAGE_SIZE} × {IMAGE_SIZE} pixels")
    st.write("Interface: Streamlit")
    st.divider()
    st.markdown("**Workflow**")
    st.write("1. Upload an image")
    st.write("2. Preview the image")
    st.write("3. Run the trained model")
    st.write("4. Review the class scores")
    st.divider()
    st.caption("Portfolio / learning demonstration")

# --------------------------- Model availability ---------------------------
if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
    st.markdown("### Getting started")
    st.info(
        "The trained model files are not available yet. Prepare the dataset, "
        "run `python train.py`, and make sure training creates both expected "
        "files in the `artifacts` folder."
    )

    missing = []
    if not MODEL_PATH.is_file():
        missing.append(f"`{MODEL_PATH.relative_to(ROOT)}`")
    if not CLASSES_PATH.is_file():
        missing.append(f"`{CLASSES_PATH.relative_to(ROOT)}`")

    st.write("Missing file(s): " + ", ".join(missing))
    st.markdown("**Next steps**")
    st.code(
        "python check_dataset.py\n"
        "python train.py\n"
        "streamlit run app.py",
        language="bash",
    )
    st.stop()


@st.cache_resource
def load_classifier(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


try:
    model = load_classifier(str(MODEL_PATH))
    class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
except Exception as exc:
    st.error("The model or class-label file could not be loaded.")
    st.caption(f"Technical details: {exc}")
    st.stop()

if not isinstance(class_names, list) or not class_names:
    st.error("The class-label file must contain a non-empty JSON list.")
    st.stop()

# --------------------------- Main workspace ---------------------------
st.markdown('<div class="section-heading">MRI analysis workspace</div>',
            unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### 01 · Upload MRI")
        upload = st.file_uploader(
            "Choose an image to examine",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Accepted formats: JPG, JPEG, PNG and BMP.",
        )

        if upload is None:
            st.markdown(
                '<p class="small-muted">Your selected image will appear here.</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div style="background:#edf3f8; padding:2rem;
                            border-radius:14px; text-align:center;">
                    <div style="font-size:2.6rem;">🧠</div>
                    <strong style="color:#24445c;">MRI image preview</strong>
                    <p style="color:#65798a; margin-bottom:0;">
                        Upload an image to begin
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            image_signature = hashlib.sha256(upload.getvalue()).hexdigest()

            try:
                image = Image.open(upload).convert("RGB")
                st.image(image, caption="Uploaded MRI image", width="stretch")
                st.caption(
                    f"Image size: {image.width} × {image.height} pixels"
                    f" · File size: {upload.size / 1024:.1f} KB"
                )
            except (UnidentifiedImageError, OSError):
                st.error("The uploaded file could not be read as an image.")
                st.stop()

with right:

    with st.container(border=True):
        st.markdown("#### 02 · Model output")
        st.write(
            "Run inference to view the predicted dataset class and the "
            "model's scores for each class."
        )

        if upload is None:
            st.info("Upload an image to enable prediction.")
        else:
            run_prediction = st.button(
                "Run model prediction",
                type="primary",
                width="stretch",
            )

            if run_prediction:
                try:
                    batch = preprocess_pil_image(image, IMAGE_SIZE)

                    with st.spinner("Running model inference..."):
                        probabilities = np.asarray(
                            model.predict(batch, verbose=0)[0],
                            dtype=np.float64,
                        )

                    if (
                        probabilities.ndim != 1
                        or len(probabilities) != len(class_names)
                        or not np.all(np.isfinite(probabilities))
                    ):
                        st.error(
                            "The model output does not match the class-label "
                            "file. Check the training configuration."
                        )
                        st.stop()

                    if np.any(probabilities < 0) or np.any(probabilities > 1):
                        st.warning(
                            "The model outputs are outside the expected "
                            "probability range. Review the model's final layer."
                        )
                        st.stop()

                    st.session_state["mri_prediction"] = {
                        "scores": probabilities.tolist(),
                        "classes": class_names,
                        "filename": upload.name,
                        "image_signature": image_signature,
                    }
                except Exception as exc:
                    st.error("Prediction failed.")
                    st.caption(f"Technical details: {exc}")

            result = st.session_state.get("mri_prediction")

            if result is not None and result.get("image_signature") == image_signature:
                scores_array = np.asarray(result["scores"])
                best = int(np.argmax(scores_array))

                st.metric(
                    "Predicted dataset class",
                    result["classes"][best],
                    f"Model score: {scores_array[best]:.1%}",
                )
                st.caption(
                    "This is a model output, not calibrated diagnostic "
                    "confidence and not a clinical conclusion."
                )
            else:
                st.markdown(
                    '<p class="small-muted">Your results will appear here '
                    'after inference.</p>',
                    unsafe_allow_html=True,
                )

# --------------------------- Results ---------------------------
result = st.session_state.get("mri_prediction")

if upload is not None and result is not None and result.get("image_signature") == image_signature:
    st.divider()
    st.markdown("### Class score breakdown")
    st.write(
        "Scores are displayed in descending order. They describe the model's "
        "outputs for the supplied image."
    )

    scores_array = np.asarray(result["scores"])
    ranking = np.argsort(scores_array)[::-1]

    scores_df = pd.DataFrame({
        "Dataset class": [result["classes"][int(i)] for i in ranking],
        "Model score": [float(scores_array[int(i)]) for i in ranking],
    })

    st.bar_chart(scores_df, x="Dataset class", y="Model score")
    st.dataframe(
        scores_df.style.format({"Model score": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.markdown("### Visual explanation (Grad-CAM)")
    st.write(
        "Grad-CAM highlights image regions that influence the selected class score. "
        "The red rectangle is an approximate box around high-activation pixels; "
        "it is not a validated tumor boundary or an object-detection result."
    )
    if st.button("Generate Grad-CAM visualization", width="stretch"):
        try:
            from src.brain_tumor_dl.explainability import (
                compute_gradcam,
                make_explanation_views,
            )

            input_batch = preprocess_pil_image(image, IMAGE_SIZE)
            selected_class = int(np.argmax(scores_array))
            with st.spinner("Computing the visual explanation..."):
                heatmap = compute_gradcam(model, input_batch, selected_class)
                colored_map, overlay, box = make_explanation_views(image, heatmap)

            view_cols = st.columns(3)
            view_cols[0].image(image, caption="Original MRI", width="stretch")
            view_cols[1].image(colored_map, caption="Grad-CAM heatmap", width="stretch")
            view_cols[2].image(
                overlay,
                caption="Heatmap overlay + approximate red box",
                width="stretch",
            )
            if box is None:
                st.info("No stable high-activation region was found for a box.")
            else:
                st.caption(
                    f"Approximate activation box (pixels): x1={box[0]}, y1={box[1]}, "
                    f"x2={box[2]}, y2={box[3]}."
                )
            st.warning(
                "This visualization explains model activations only. It does not "
                "confirm that the highlighted area is a tumor and is not for clinical use."
            )
        except Exception as exc:
            st.error("Could not generate the Grad-CAM visualization.")
            st.caption(f"Technical details: {exc}")

st.divider()
st.markdown("### About this project")
a, b, c = st.columns(3)

with a:
    st.markdown("**Deep learning**")
    st.caption("MobileNetV2 transfer learning, subject to the training pipeline.")

with b:
    st.markdown("**Image processing**")
    st.caption("Images are resized to 224 × 224 before inference.")

with c:
    st.markdown("**Responsible use**")
    st.caption("Educational demonstration; not clinically validated.")

st.caption(
    "The class scores depend on the training data and model. "
    "Performance must be evaluated on held-out data before reporting metrics."
)
