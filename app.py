import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import cv2
from ultralytics import YOLO

st.set_page_config(
    page_title="Aerial Object Classification",
    page_icon="í¶…",
    layout="wide"
)

st.title("Aerial Object Classification & Detection")
st.markdown("### Bird vs Drone Detection System")
st.markdown("Upload an aerial image to classify it as **Bird** or **Drone**")

@st.cache_resource
def load_classification_model():
    models_available = {}
    if os.path.exists('models/transfer_model.h5'):
        models_available['EfficientNetB0 (Transfer Learning)'] = load_model('models/transfer_model.h5')
    if os.path.exists('models/custom_cnn.h5'):
        models_available['Custom CNN'] = load_model('models/custom_cnn.h5')
    return models_available

@st.cache_resource
def load_yolo_model():
    yolo_path = 'runs/detect/bird_drone/weights/best.pt'
    if os.path.exists(yolo_path):
        return YOLO(yolo_path)
    return None

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_class(model, img_array):
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    if confidence > 0.5:
        label = 'Drone'
        conf_score = confidence
    else:
        label = 'Bird'
        conf_score = 1 - confidence
    return label, conf_score

# Sidebar
st.sidebar.title("Settings")
task = st.sidebar.radio("Select Task", ["Classification", "Object Detection (YOLOv8)"])

# Load models
models = load_classification_model()
yolo_model = load_yolo_model()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    st.subheader("Prediction Results")
    if uploaded_file:
        if task == "Classification":
            if not models:
                st.error("No trained models found! Please train models first.")
            else:
                model_choice = st.selectbox("Select Model", list(models.keys()))
                if st.button("Classify Image", type="primary"):
                    with st.spinner("Analyzing..."):
                        img_array = preprocess_image(image)
                        label, confidence = predict_class(models[model_choice], img_array)
                        if label == 'Bird':
                            st.success(f"Prediction: BIRD")
                            st.metric("Confidence", f"{confidence:.2%}")
                            st.info("This is a Bird")
                        else:
                            st.error(f"Prediction: DRONE")
                            st.metric("Confidence", f"{confidence:.2%}")
                            st.warning("This is a Drone - Security Alert!")
                        st.progress(confidence)

        elif task == "Object Detection (YOLOv8)":
            if yolo_model is None:
                st.warning("YOLOv8 model not found. Please train YOLOv8 first.")
            else:
                if st.button("Detect Objects", type="primary"):
                    with st.spinner("Running YOLOv8 Detection..."):
                        img_array = np.array(image)
                        results = yolo_model(img_array)
                        annotated = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption='Detection Results', use_column_width=True)
                        boxes = results[0].boxes
                        if boxes:
                            st.write(f"Detected {len(boxes)} object(s)")
                            for box in boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                name = 'Bird' if cls == 0 else 'Drone'
                                st.write(f"- {name}: {conf:.2%} confidence")
                        else:
                            st.info("No objects detected")
    else:
        st.info("Please upload an image to get started")

# Model Performance Section
st.markdown("---")
st.subheader("Model Performance")
col3, col4, col5 = st.columns(3)

import json
with col3:
    if os.path.exists('logs/cnn_metrics.json'):
        with open('logs/cnn_metrics.json') as f:
            cnn_m = json.load(f)
        st.metric("Custom CNN Accuracy", f"{cnn_m['test_accuracy']:.2%}")

with col4:
    if os.path.exists('logs/transfer_metrics.json'):
        with open('logs/transfer_metrics.json') as f:
            tr_m = json.load(f)
        st.metric("EfficientNetB0 Accuracy", f"{tr_m['test_accuracy']:.2%}")

with col5:
    st.metric("Dataset Size", "3,319 Images")

# Show training plots
st.markdown("---")
st.subheader("Training History")
col6, col7 = st.columns(2)
with col6:
    if os.path.exists('logs/custom_cnn_history.png'):
        st.image('logs/custom_cnn_history.png', caption='Custom CNN Training History')
with col7:
    if os.path.exists('logs/transfer_history.png'):
        st.image('logs/transfer_history.png', caption='Transfer Learning Training History')

if os.path.exists('logs/model_comparison.png'):
    st.markdown("---")
    st.subheader("Model Comparison")
    st.image('logs/model_comparison.png', caption='Model Accuracy Comparison')
