import os
import cv2
import boto3
import uuid
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
MODEL_PATH = "oil_spill_model.h5"
IMG_SIZE = (256, 256)
PIXEL_RESOLUTION_M = 10  # 10 meters per pixel
AWS_REGION = "ap-southeast-2"

# -------------------------------------------------------
# AWS SERVICE CLIENTS
# -------------------------------------------------------
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
sns_client = boto3.client("sns", region_name=AWS_REGION)
lambda_client = boto3.client("lambda", region_name=AWS_REGION)

# DynamoDB table (must exist)
DYNAMODB_TABLE = "OilSpillDetections"

# SNS topic (for manual alert option)
SNS_TOPIC_ARN = "arn:aws:sns:ap-southeast-2:061785322201:oil-spill-alerts"

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_segmentation_model():
    return load_model(MODEL_PATH)

model = load_segmentation_model()

# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------
def classify_spill(area_km2):
    """Classify spill severity by area."""
    if area_km2 < 1:
        return "Small", "🟡", "#ffff99"
    elif area_km2 < 10:
        return "Medium", "🟠", "#ffcc66"
    else:
        return "Large", "🔴", "#ff6666"


def overlay_mask(original, mask):
    """Overlay red mask on the original image."""
    overlay = original.copy()
    overlay[mask == 1] = [255, 0, 0]
    return overlay


def save_to_dynamodb(record):
    """Save detection results to DynamoDB."""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        table.put_item(Item=record)
        return True
    except Exception as e:
        st.error(f"DynamoDB Error: {e}")
        return False


def invoke_lambda(record_id):
    """Optionally invoke AWS Lambda manually."""
    try:
        payload = {"DetectionID": record_id}
        lambda_client.invoke(
            FunctionName="OilSpillAlertLambda",  # Replace with your Lambda function name
            InvocationType="Event",
            Payload=str(payload).encode("utf-8")
        )
        st.info("🔄 Lambda function triggered for processing.")
    except Exception as e:
        st.error(f"Lambda Trigger Error: {e}")


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("🛢 Real-Time Oil Spill Detection & Severity Estimation")

uploaded_file = st.file_uploader("Upload a SAR image (Sentinel or PALSAR)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("📷 Uploaded Image")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    # Preprocess
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, IMG_SIZE)
    img_norm = img_resized.astype(np.float32) / 255.0
    input_image = np.expand_dims(img_norm, axis=0)

    # Predict
    pred_mask = model.predict(input_image)[0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    if binary_mask.shape[-1] == 1:
        binary_mask = binary_mask.squeeze()

    # Calculate area
    oil_pixels = np.sum(binary_mask)
    area_m2 = oil_pixels * (PIXEL_RESOLUTION_M ** 2)
    area_km2 = area_m2 / 1e6
    severity, emoji, color = classify_spill(area_km2)

    # Display results
    st.subheader("🧠 Predicted Oil Spill Mask (in red)")
    overlay = overlay_mask(img_resized, binary_mask)
    st.image(overlay, use_column_width=True, caption="Red areas indicate detected oil spill")

    st.markdown("---")
    st.subheader("📊 Detection Summary")
    st.markdown(f"*Detected Oil Spill Pixels:* {oil_pixels}")
    st.markdown(f"*Estimated Spill Area:* {area_km2:.4f} km²")
    st.markdown(
        f"<div style='padding:10px;border-radius:10px;background-color:{color};color:black;'>"
        f"<b>Severity:</b> {emoji} {severity}</div>",
        unsafe_allow_html=True
    )

    # -------------------------------------------------------
    # SAVE DETECTION TO DYNAMODB
    # -------------------------------------------------------
    if st.button("💾 Save Detection Record"):
        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        record = {
            "DetectionID": record_id,
            "Timestamp": timestamp,
            "OilPixels": int(oil_pixels),
            "Area_km2": float(area_km2),
            "Severity": severity,
            "ImageName": uploaded_file.name,
        }
        if save_to_dynamodb(record):
            st.success("✅ Detection saved to DynamoDB successfully!")
            invoke_lambda(record_id)  # Optional Lambda trigger

    # -------------------------------------------------------
    # MANUAL SNS ALERT OPTION
    # -------------------------------------------------------
    if st.button("🚨 Send Manual SNS Alert"):
        try:
            message = (
                f"🌊 Oil Spill Detected!\n\n"
                f"🧮 Pixels: {oil_pixels}\n"
                f"📐 Area: {area_km2:.4f} km²\n"
                f"🟡 Severity: {severity}"
            )
            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject="🚨 Oil Spill Detection Alert"
            )
            st.success("📨 SNS Alert sent successfully!")
        except Exception as e:
            st.error(f"SNS Alert Error: {e}")
