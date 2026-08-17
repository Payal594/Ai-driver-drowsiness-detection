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
# Load Model from Hugging Face
# ============================================================

@st.cache_resource
def load_model():

    st.write("⏳ Loading trained ViT model...")

    model_path = hf_hub_download(
        repo_id="payal-preeti/driver-drowsiness-vit",
        filename="vit_final_model.pth"
    )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )

    state_dict = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


model = load_model()


# ============================================================
# Title
# ============================================================

st.title("🚗 Driver Drowsiness Detection")

st.write(
    "Upload a driver's image to detect the driver's current state."
)

st.info(
    "The model uses Vision Transformer (ViT) trained on four classes: "
    "Closed, Open, no_yawn, and yawn."
)


# ============================================================
# Upload Image
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Prediction
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Driver Image",
        use_container_width=True
    )


    # --------------------------------------------------------
    # Preprocessing
    # IMPORTANT:
    # Same preprocessing used during training
    # Resize(224,224) + ToTensor()
    # --------------------------------------------------------

    input_tensor = preprocess_frame(
        np.array(image)
    ).to(device)


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

        confidence = probabilities[
            0, pred
        ].item()


    state = labels[str(pred)]


    # ========================================================
    # Results
    # ========================================================

    st.subheader(
        f"Prediction: {state}"
    )

    st.write(
        f"Confidence: **{confidence * 100:.2f}%**"
    )


    # ========================================================
    # Drowsiness Decision
    # ========================================================

    if state in ["Closed", "yawn"] and confidence >= 0.90:

        st.error(
            "⚠️ Driver appears drowsy!"
        )

    else:

        st.success(
            "✅ Driver appears alert"
        )


    # ========================================================
    # Probability Distribution
    # ========================================================

    st.subheader("Class Probabilities")

    for i in range(num_classes):

        class_name = labels[str(i)]

        probability = probabilities[
            0, i
        ].item()

        st.write(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )