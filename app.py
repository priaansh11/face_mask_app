import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("mask_detector.h5", compile=False)

model = load_my_model()

# Only 3 classes needed
class_names = ["With Mask", "Without Mask", "Mask Worn Incorrectly"]

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check mask status")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

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
            
            # Take only first 3 classes, ignore the 4th output
            result = result[:3]
            
            pred_class = int(np.argmax(result))
            confidence = float(np.max(result))
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_resized, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Prediction Result")
            
            label = class_names[pred_class]
            
            if label == "With Mask":
                st.success(f"✅ {label}")
            elif label == "Without Mask":
                st.error(f"❌ {label}")
            else:
                st.warning(f"⚠️ {label}")
                
            st.metric("Confidence", f"{confidence*100:.1f}%")
            
            with st.expander("View all probabilities"):
                for i, class_name in enumerate(class_names):
                    st.write(f"{class_name}: {result[i]*100:.2f}%")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please try a different image (JPG or PNG format).")
