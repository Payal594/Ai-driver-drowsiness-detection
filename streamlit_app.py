import streamlit as st
import torch
import timm
import json
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from app.utils import preprocess_frame


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load Labels
# ============================================================

with open("model/labels.json", "r") as f:
    labels = json.load(f)

num_classes = len(labels)


# ============================================================
# Download Model from Hugging Face
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="payal-preeti/driver-drowsiness-vit",
        filename="vit_final_model.pth"
    )

    # Create the same ViT architecture used during training
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )

    # Load trained weights
    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# ============================================================
# Load Model
# ============================================================

try:

    with st.spinner("Loading trained ViT model..."):
        model = load_model()

    st.success("✅ Model loaded successfully")

except Exception as e:

    st.error("❌ Failed to load model")

    st.exception(e)

    st.stop()


# ============================================================
# Application UI
# ============================================================

st.title("🚗 Driver Drowsiness Detection")
st.subheader("Vision Transformer (ViT)")

st.write(
    "Upload a driver's image to detect whether the driver "
    "is alert or showing signs of drowsiness."
)


# ============================================================
# Image Upload
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Driver Image",
        use_container_width=True
    )

    # Convert image to NumPy
    image_array = np.array(image)

    # Preprocess exactly as during inference
    input_tensor = preprocess_frame(image_array).to(device)

    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        pred = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[0][pred].item()


    # --------------------------------------------------------
    # Get Predicted Class
    # --------------------------------------------------------

    state = labels[str(pred)]


    # ========================================================
    # Display Prediction
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted State",
            state
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # Drowsiness Decision
    # ========================================================

    # Drowsy classes in your dataset:
    # Closed eyes and yawning
    #
    # Confidence threshold prevents weak predictions
    # from immediately being considered drowsiness.

    if state in ["Closed", "yawn"] and confidence >= 0.90:

        st.error(
            "⚠️ DRIVER APPEARS DROWSY!"
        )

        st.warning(
            f"Detected state: {state} "
            f"with {confidence * 100:.2f}% confidence."
        )

    else:

        st.success(
            "✅ DRIVER APPEARS ALERT"
        )


    # ========================================================
    # Probability Distribution
    # ========================================================

    st.divider()

    st.subheader("Class Probabilities")

    for i, class_name in labels.items():

        probability = probabilities[0][int(i)].item()

        st.write(
            f"**{class_name}**: "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)