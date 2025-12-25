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

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "webp", "bmp"])

if uploaded_file is not None:
    try:
        # Read and validate image
        img = Image.open(uploaded_file)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img_resized = img.resize((224, 224), Image.LANCZOS)
        
        # Preprocess
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array, verbose=0)
            
            if isinstance(prediction, (list, tuple)):
                result = prediction[0]
            else:
                result = prediction
            
            # Debug: Show prediction shape and values
            st.write(f"Debug - Prediction shape: {result.shape}")
            st.write(f"Debug - Prediction values: {result}")
            
            pred_class = np.argmax(result)
            confidence = np.max(result)
            
            st.write(f"Debug - Predicted class index: {pred_class}")
            st.write(f"Debug - Number of classes: {len(class_names)}")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_resized, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Result")
            
            # Safe indexing
            if pred_class < len(class_names):
                label = class_names[pred_class]
                if label == "With Mask":
                    st.success(f"**Status:** {label}")
                elif label == "Without Mask":
                    st.error(f"**Status:** {label}")
                else:
                    st.warning(f"**Status:** {label}")
                    
                st.metric("Confidence", f"{confidence*100:.1f}%")
            else:
                st.error(f"Unexpected class index: {pred_class}. Model outputs {result.shape[0]} classes but only {len(class_names)} class names provided.")
            
            with st.expander("View all probabilities"):
                for i in range(len(result)):
                    if i < len(class_names):
                        st.write(f"{class_names[i]}: {result[i]*100:.2f}%")
                    else:
                        st.write(f"Class {i}: {result[i]*100:.2f}%")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image.")
