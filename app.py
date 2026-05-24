import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
from PIL import Image

# -------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="PneuScan AI - Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# Cache the models so they only load ONCE when the server starts
@st.cache_resource
def load_production_pipeline():
    supervised_path = 'pneumonia_final.keras'
    unsupervised_extractor_path = 'vgg16_feature_extractor_pneumonia.keras'
    pca_path = 'pca_model.pkl'
    kmeans_path = 'kmeans_model.pkl'
    
    # Load Neural Network Architectures
    mod1 = tf.keras.models.load_model(supervised_path)
    mod2 = tf.keras.models.load_model(unsupervised_extractor_path)
    
    # Load Unsupervised Pipeline States (.pkl)
    with open(pca_path, 'rb') as f:
        saved_pca = pickle.load(f)
    with open(kmeans_path, 'rb') as f:
        saved_kmeans = pickle.load(f)
        
    return mod1, mod2, saved_pca, saved_kmeans

try:
    model_supervised, feature_extractor, pca_model, kmeans_model = load_production_pipeline()
    st.sidebar.success("🤖 Stage 1 & Stage 2 AI Pipeline Active!")
except Exception as e:
    st.sidebar.error(f"⚠️ Missing pipeline files! Ensure all .keras and .pkl files are in the directory. Error: {e}")
    st.stop()

# -------------------------------------------------------------------------
# 2. IMAGE PREPROCESSING FUNCTIONS
# -------------------------------------------------------------------------
def preprocess_for_supervised(image):
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

def preprocess_for_unsupervised(image):
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.vgg16.preprocess_input(img_array)

def predict_pneumonia_cluster(image):
    """
    Extracts high-dimensional features, applies the exact PCA matrix, 
    and classifies it across the 4 trained pathogen spaces.
    """
    processed_data = preprocess_for_unsupervised(image)
    
    # 1. Get raw high-dimensional array from VGG16
    raw_features = feature_extractor.predict(processed_data)
    features_flattened = raw_features.reshape(1, -1)
    
    # 2. Project into the 50-component PCA space
    pca_features = pca_model.transform(features_flattened)
    
    # 3. Classify using the K-Means coordinate map (Returns 0, 1, 2, or 3)
    predicted_cluster = kmeans_model.predict(pca_features)[0]
    
    # Mapping definitions matching your friend's latest model setup
    cluster_mapping = {
        0: ("Bacterial Pneumonia", "High structural density with dense lobar infiltrates and localized consolidation patterns."),
        1: ("Atypical Pneumonia ", "Patchy opacity distribution displaying reticular density variations across interstitial walls."),
        2: ("Viral Pneumonia", "Diffuse bilateral ground-glass opacities outlining broad airway passages symmetrically."),
        3: ("Fungal Pneumonia", "Cavitary lesions or nodular infiltrate structures present across peripheral tissue distributions.")
    }
    
    return cluster_mapping.get(predicted_cluster, ("Unknown Sub-type", "Atypical lung density pattern detected."))

# -------------------------------------------------------------------------
# 3. USER INTERFACE (UI)
# -------------------------------------------------------------------------
st.title("🫁 PneuScan AI Portal")
st.write("An advanced Pipeline Deep Learning system designed to analyze chest X-ray scans for indications of Pneumonia.")
st.write("Visit my [GitHub Repository : theDevJayden](https://github.com/theDevJayden) for more projects.")
st.write("This application uses a two-stage diagnostic approach:")
st.write("1. Supervised Model (MobileNetV2): Initial screening for pneumonia presence.")
st.write("2. Unsupervised Sub-Clustering (VGG16 + PCA + KMeans): Detailed classification into specific pneumonia sub-types. Made by [Jojo](https://github.com/study-spec)")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (JPEG / PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded X-Ray Scan", use_container_width=True)
        
    with col2:
        st.subheader("Diagnostic Evaluation")
        
        # STAGE 1: Supervised Model Filter (MobileNetV2)
        with st.spinner("Analyzing structural lung density (Stage 1)..."):
            processed_data_sup = preprocess_for_supervised(image)
            prediction_prob = model_supervised.predict(processed_data_sup)[0][0]
            
        if prediction_prob > 0.5:
            confidence = prediction_prob * 100
            st.error(f"🚨 **Stage 1 Result: Pneumonia Detected**")
            st.metric(label="Supervised Confidence Score", value=f"{confidence:.2f}%")
            st.markdown("---")
            
            # STAGE 2: Run Unsupervised Sub-Clustering (VGG16 + PCA + KMeans)
            with st.spinner("Sub-clustering pneumonia pathogen characteristics (Stage 2)..."):
                cluster_title, cluster_desc = predict_pneumonia_cluster(image)
                
            st.info(f"🧬 **Stage 2 Sub-Type: {cluster_title}**")
            st.caption(f"*Pathology Profile: {cluster_desc}*")
            st.warning("Recommendation: Please correlate these findings with clinical symptoms and consult a radiologist immediately.")
            
        else:
            confidence = (1 - prediction_prob) * 100
            st.success(f"✅ **Stage 1 Result: Normal / Clear**")
            st.metric(label="Supervised Confidence Score", value=f"{confidence:.2f}%")
            st.info("Notice: No clear indicators of lung consolidation detected. Sub-clustering bypassed.")