import os
# MANDATORY: This must be the first thing in the script
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load trained model
@st.cache_resource
def load_my_model():
    # Use tf.keras specifically to ensure it pulls from the pinned tensorflow 2.15
    return tf.keras.models.load_model("mask_detector.h5", compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")
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
        prediction = model.predict(img_array)
        
        # Unpacking prediction
        if isinstance(prediction, (list, tuple)):
            result = prediction[0]
        else:
            result = prediction

        pred_class = np.argmax(result)
        confidence = np.max(result)

    # --- UI Display ---
    st.image(img_resized, caption="Uploaded Image")
    st.subheader("Prediction Result")
    
    label = class_names[pred_class]
    if label == "With Mask":
        st.success(f"**Status:** {label}")
    elif label == "Without Mask":
        st.error(f"**Status:** {label}")
    else:
        st.warning(f"**Status:** {label}")
        
    st.write(f"**Confidence:** {confidence:.2f}")
