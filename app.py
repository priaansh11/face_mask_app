import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load trained model (inference mode)
# Using compile=False is the most robust way to avoid version mismatches
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("mask_detector.h5", compile=False)

model = load_my_model()

# 2. Class names (ensure these match your training labels exactly)
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
    # Load and Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    
    # Normalize to [0, 1] range as done during training
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Prediction Logic
    # We remove the "_, class_pred" unpacking to avoid ValueErrors
    predictions = model.predict(img_array)
    
    # Handle both single-output and multi-output model structures
    if isinstance(predictions, list):
        result = predictions[0]
    else:
        result = predictions

    pred_class = np.argmax(result)
    confidence = np.max(result)

    # 4. UI Display
    st.image(img_resized, caption="Processed Image (224x224)")
    
    st.subheader("Prediction Result")
    
    # Color-coded results for better UX
    if class_names[pred_class] == "With Mask":
        st.success(f"**Status:** {class_names[pred_class]}")
    elif class_names[pred_class] == "Without Mask":
        st.error(f"**Status:** {class_names[pred_class]}")
    else:
        st.warning(f"**Status:** {class_names[pred_class]}")
        
    st.write(f"**Confidence Score:** {confidence:.2f}")