import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import boto3
import os

# --- CONFIGURATION ---
MODEL_PATH = "oil_spill_model.h5"
IMG_SIZE = (256, 256)
PIXEL_RESOLUTION_M = 10  # 10 meters per pixel

# --- AWS SNS CONFIGURATION ---
SNS_TOPIC_ARN = "arn:aws:sns:ap-southeast-2:061785322201:oil-spill-alerts"  # Replace with your actual SNS Topic ARN
AWS_REGION = "ap-southeast-2"

# Load credentials from environment or ~/.aws/credentials
sns_client = boto3.client("sns", region_name=AWS_REGION)

# --- LOAD MODEL ---
@st.cache_resource
def load_segmentation_model():
    return load_model(MODEL_PATH)

model = load_segmentation_model()


# --- SEVERITY CLASSIFICATION FUNCTION ---
def classify_spill(area_km2):
    if area_km2 < 1:
        return "Small", "🟡", "#ffff99"  # Yellow
    elif area_km2 < 10:
        return "Medium", "🟠", "#ffcc66"  # Orange
    else:
        return "Large", "🔴", "#ff6666"  # Red


# --- MASK VISUALIZATION FUNCTION ---
def overlay_mask(original, mask):
    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red overlay
    return overlay


# --- STREAMLIT UI ---
st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🛢 Oil Spill Detection & Severity Estimation")

uploaded_file = st.file_uploader("Upload a SAR image (Sentinel or PALSAR)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("📷 Original Image")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    # Convert PIL image to NumPy
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, IMG_SIZE)

    # Normalize and predict
    img_norm = img_resized.astype(np.float32) / 255.0
    input_image = np.expand_dims(img_norm, axis=0)
    pred_mask = model.predict(input_image)[0]

    # Convert prediction to binary mask
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    if binary_mask.shape[-1] == 1:
        binary_mask = binary_mask.squeeze()
    elif binary_mask.ndim == 3:
        binary_mask = binary_mask[:, :, 0]

    # Calculate area
    oil_pixels = np.sum(binary_mask)
    area_m2 = oil_pixels * (PIXEL_RESOLUTION_M ** 2)
    area_km2 = area_m2 / 1e6

    severity, emoji, color = classify_spill(area_km2)

    # Display mask
    st.subheader("🧠 Predicted Oil Spill Mask (in red)")
    overlay = overlay_mask(img_resized, binary_mask)
    st.image(overlay, use_column_width=True, caption="Red areas show detected oil spill")

    # Display results
    st.markdown("---")
    st.subheader("📊 Results")
    st.markdown(f"*Detected Oil Spill Pixels:* {oil_pixels}")
    st.markdown(f"*Estimated Spill Area:* {area_km2:.4f} km²")
    st.markdown(
        f"<div style='padding: 10px; border-radius: 10px; background-color: {color}; color: black;'>"
        f"<b>Severity Classification:</b> {emoji} {severity}"
        f"</div>",
        unsafe_allow_html=True
    )

    # --- SNS ALERT (for ALL spill sizes including small) ---
    if st.button("📤 Send AWS SNS Alert"):
        try:
            message = (
                f"🌊 Oil Spill Detected!\n\n"
                f"🧮 Pixels Detected: {oil_pixels}\n"
                f"📐 Area: {area_km2:.4f} km²\n"
                f"🟡 Severity: {severity}"
            )
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject="🚨 Oil Spill Detection Alert"
            )
            st.success("✅ AWS SNS Alert sent successfully!")
        except Exception as e:
            st.error(f"❌ Failed to send SNS Alert: {str(e)}")
