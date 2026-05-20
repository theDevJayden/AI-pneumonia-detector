import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="PneuScan AI - Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# Cache the model so it only loads ONCE when the server starts, not on every click
@st.cache_resource
def load_pneumonia_model():
    # Update this path to where your final production model is located
    model_path = 'pneumonia_final.keras'
    return tf.keras.models.load_model(model_path)

try:
    model = load_pneumonia_model()
    st.sidebar.success("🤖 AI Brain Active & Loaded!")
except Exception as e:
    st.sidebar.error("⚠️ Failed to load model. Check your path!")
    st.stop()

# -------------------------------------------------------------------------
# 2. IMAGE PREPROCESSING FUNCTION
# -------------------------------------------------------------------------
def preprocess_xray(image):
    # Match the exact training parameters: 224x224 target size
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    
    # Handle greyscale images safely by converting to 3 channels (RGB)
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[-1] == 4: # Handle PNG transparency layers if any
        img_array = img_array[:, :, :3]
        
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 224, 224, 3)
    
    # Crucial step: MobileNetV2 expected scaling (-1 to 1)
    processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return processed_img

# -------------------------------------------------------------------------
# 3. USER INTERFACE (UI)
# -------------------------------------------------------------------------
st.title("🫁 PneuScan AI Portal")
st.write("An advanced Deep Learning system designed to analyze chest X-ray scans for indications of Pneumonia.")
st.write("Visit my [GitHub Repository : theDevJayden](https://github.com/theDevJayden) for more projects.")
st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (JPEG / PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the raw image
    image = Image.open(uploaded_file)
    
    # Display layout split into columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded X-Ray Scan", use_container_width=True)
        
    with col2:
        st.subheader("Diagnostic Evaluation")
        
        # Run inference with a nice visual spinner
        with st.spinner("Analyzing structural lung density..."):
            processed_data = preprocess_xray(image)
            prediction_prob = model.predict(processed_data)[0][0]
            
        # Determine metrics based on our 0.5 classification threshold
        if prediction_prob > 0.5:
            confidence = prediction_prob * 100
            st.error(f"🚨 **Result: Pneumonia Detected**")
            st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
            st.warning("Recommendation: Please correlate these findings with clinical symptoms and consult a radiologist immediately.")
        else:
            confidence = (1 - prediction_prob) * 100
            st.success(f"✅ **Result: Normal / Clear**")
            st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
            st.info("Notice: No clear indicators of lung consolidation or dense infiltrates detected by the neural network.")