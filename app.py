import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("mask_detector.h5", compile=False)

model = load_my_model()

class_names = ["With Mask", "Without Mask", "Mask Worn Incorrectly"]

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check mask status")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array, verbose=0)
        
        if isinstance(prediction, (list, tuple)):
            result = prediction[0]
        else:
            result = prediction
        pred_class = np.argmax(result)
        confidence = np.max(result)
    
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
