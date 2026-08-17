# 🚗 AI Driver Drowsiness Detection using Vision Transformer

An AI-powered driver drowsiness detection system that uses a **Vision Transformer (ViT)** to classify a driver's facial state into four categories:

- 👁️ Open
- 😴 Closed
- 😊 no_yawn
- 🥱 yawn

The trained model is integrated with a **Streamlit web application** for easy image-based inference and is hosted online.

---

## 🌐 Live Demo

**Live Application:**  
https://payal594-ai-driver-drowsiness-detection-streamlit-app-tymuvv.streamlit.app/

---

## 📌 Project Overview

Driver drowsiness is one of the major causes of road accidents. This project aims to detect signs of driver fatigue by analyzing facial images and identifying eye and yawning states.

The system uses a pretrained **Vision Transformer (ViT-Base)** architecture that was fine-tuned on a driver drowsiness dataset.

The final system performs:

1. Image input
2. Image preprocessing
3. Vision Transformer inference
4. Four-class classification
5. Confidence estimation
6. Drowsiness decision
7. Web-based visualization using Streamlit

---

## 🧠 Model Architecture

The project uses:

**Vision Transformer (ViT-Base)**

Model:

```text
vit_base_patch16_224
