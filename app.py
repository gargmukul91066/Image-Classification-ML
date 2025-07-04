import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Leaf Disease Classifier", page_icon="🌿", layout="centered")

# Class Labels 
class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Load Model 
@st.cache_resource
def load_model():
    try:
        st.info("📦 Loading model...")
        model = joblib.load("final_rf_model.pkl")
        st.success(" Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f" Failed to load model: {e}")
        st.stop()

model = load_model()

# Feature Extraction 
def extract_color_histogram(image, bins=(8, 8, 8)):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().reshape(1, -1)

# App UI 
st.markdown("<h1 style='text-align: center;'> Leaf Disease Classifier</h1>", unsafe_allow_html=True)

st.markdown("""
Upload a leaf image and the app will classify it into one of the following categories:
-  **Healthy**
-  **Multiple Diseases**
-  **Rust**
-  **Scab**
""")

# File Upload 
uploaded_file = st.file_uploader(" Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption=" Uploaded Leaf", use_column_width=True)

    features = extract_color_histogram(image)

    try:
        prediction = model.predict(features)[0]
        prediction_label = class_names[prediction]

        # Prediction Result
        st.success(f" Prediction: **{prediction_label.capitalize()}**")

        #  Confidence Bar Chart
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]

            st.subheader(" Prediction Confidence")

            # Plot confidence as bar chart
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(class_names, probs * 100, color="mediumseagreen")
            ax.set_ylabel("Confidence (%)")
            ax.set_ylim(0, 100)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)

            st.pyplot(fig)

    except Exception as e:
        st.error(f" Prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Made by Mukul Garg | GitHub:(https://github.com/gargmukul91066)")
