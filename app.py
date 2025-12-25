import os
# This MUST be the first two lines to prevent Keras 3 from loading
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Face Mask Detector", page_icon="😷")

# 1. Load trained model (Inference mode)
@st.cache_resource
def load_my_model():
    # compile=False avoids errors with custom optimizers saved in the file
    return tf.keras.models.load_model("mask_detector.h5", compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. Class names
class_names = [
    "With Mask",
    "Without Mask",
    "Mask Worn Incorrectly"
]

st.title("😷 Face Mask Detection System")
st.write("Upload a photo to detect if a face mask is being worn correctly.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # --- Image Processing ---
    img = Image.open(uploaded_file).convert("RGB")
    
    # Show the image to the user
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Resize and normalize for the model (224x224)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Prediction ---
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array)
        
        # Robustly handle prediction output format
        if isinstance(prediction, list):
            result = prediction[0]
        else:
            result = prediction

        pred_class_idx = np.argmax(result)
        confidence = np.max(result)
        label = class_names[pred_class_idx]

    # --- Results Display ---
    st.divider()
    st.subheader("Results")
    
    if label == "With Mask":
        st.success(f"**Result:** {label}")
    elif label == "Without Mask":
        st.error(f"**Result:** {label}")
    else:
        st.warning(f"**Result:** {label}")
        
    st.write(f"**Confidence Level:** {confidence:.2%}")
