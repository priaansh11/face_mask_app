import os
# MANDATORY: Set these BEFORE any imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image

# Import TensorFlow after setting environment variables
import tensorflow as tf
from tensorflow import keras

# Verify Keras is accessible
st.sidebar.write(f"TensorFlow version: {tf.__version__}")

# 1. Load trained model
@st.cache_resource
def load_my_model():
    return keras.models.load_model("mask_detector.h5", compile=False)

try:
    model = load_my_model()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.error(f"TensorFlow version: {tf.__version__}")
    st.stop()

# 2. Class names
class_names = ["With Mask", "Without Mask", "Mask Worn Incorrectly"]

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check mask status")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- Image Processing ---
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    
    # Preprocess
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # --- Prediction ---
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array, verbose=0)
        
        # Unpacking prediction
        if isinstance(prediction, (list, tuple)):
            result = prediction[0]
        else:
            result = prediction
        pred_class = np.argmax(result)
        confidence = np.max(result)
    
    # --- UI Display ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_resized, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        label = class_names[pred_class]
        if label == "With Mask":
            st.success(f"**Status:** {label}")
        elif label == "Without Mask":
            st.error(f"**Status:** {label}")
        else:
            st.warning(f"**Status:** {label}")
            
        st.metric("Confidence", f"{confidence*100:.1f}%")
        
        # Show all probabilities
        with st.expander("View all probabilities"):
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {result[i]*100:.2f}%")
