import cv2
import torch
import json
import timm
import os

from huggingface_hub import hf_hub_download

from utils import preprocess_frame
from alert import play_alert


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"🖥️ Using device: {device}")


# ============================================================
# Project Paths
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")


# ============================================================
# Load Labels
# ============================================================

labels_path = os.path.join(
    MODEL_DIR,
    "labels.json"
)

with open(labels_path, "r") as f:
    labels = json.load(f)

num_classes = len(labels)

print("Classes:", labels)


# ============================================================
# Download Trained Model from Hugging Face
# ============================================================

print("⏳ Loading trained ViT model...")

model_path = hf_hub_download(
    repo_id="payal-preeti/driver-drowsiness-vit",
    filename="vit_final_model.pth"
)

print("✅ Model downloaded/loaded from Hugging Face")


# ============================================================
# Create ViT Architecture
# ============================================================

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=num_classes
)


# ============================================================
# Load Trained Weights
# ============================================================

state_dict = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("✅ Trained ViT model loaded successfully")


# ============================================================
# Webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not open webcam.")

    exit()


# ============================================================
# Drowsiness Parameters
# ============================================================

DROWSY_THRESHOLD = 15

CONFIDENCE_THRESHOLD = 0.90

drowsy_counter = 0

alert_triggered = False


print()
print("🎥 Webcam started.")
print("Press 'q' to quit.")
print()


# ============================================================
# Main Loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("❌ Failed to read webcam frame.")

        break


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Preprocess
    #
    # IMPORTANT:
    # This must match training:
    #
    # Resize(224,224)
    # ToTensor()
    #
    # NO Normalize()
    # --------------------------------------------------------

    input_tensor = preprocess_frame(
        frame_rgb
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


    # --------------------------------------------------------
    # Get Predicted Class
    # --------------------------------------------------------

    state = labels[str(pred)]


    # --------------------------------------------------------
    # Drowsiness Detection
    # --------------------------------------------------------

    is_drowsy = (
        state in ["Closed", "yawn"]
        and confidence >= CONFIDENCE_THRESHOLD
    )


    if is_drowsy:

        drowsy_counter += 1

    else:

        drowsy_counter = 0

        # Reset alert when driver becomes alert
        alert_triggered = False


    # --------------------------------------------------------
    # Drowsiness Alert
    # --------------------------------------------------------

    if drowsy_counter >= DROWSY_THRESHOLD:

        cv2.putText(
            frame,
            "DROWSY ALERT!",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        # Play sound only once
        if not alert_triggered:

            play_alert()

            alert_triggered = True


    # --------------------------------------------------------
    # Display State
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"State: {state}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Display Confidence
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Display Drowsiness Counter
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Drowsy Frames: {drowsy_counter}/{DROWSY_THRESHOLD}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Display Webcam
    # --------------------------------------------------------

    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("🛑 Webcam stopped.")