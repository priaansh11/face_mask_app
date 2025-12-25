import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model (inference mode)
model = tf.keras.models.load_model(
    "mask_detector.h5",
    compile=False
)

# Class names (same order as training)
class_names = [
    "With Mask",
    "Without Mask",
    "Mask Worn Incorrectly"
]

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check mask status")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))

    # Preprocess
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    _, class_pred = model.predict(img_array)

    pred_class = np.argmax(class_pred)
    confidence = np.max(class_pred)

    # Show image (NO bounding box)
    st.image(img_resized, caption="Uploaded Image")

    # Show result
    st.subheader("Prediction Result")
    st.write(f"**Class:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")