<div align="center">
  <h1>🫁 Pneumonia Detection AI Portal</h1>
  <p><strong>A Dual-Stage Hybrid Deep Learning Pipeline using Supervised MobileNetV2 & Unsupervised VGG16 Sub-Clustering</strong></p>

  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Backend-TensorFlow-FF6F00?style=flat-square&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Unsupervised-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  
  <br><br>
  
  <p>Pneumonia Detection Website: <a href="https://ai-pneumonia-detector-thedevjayden.streamlit.app/" target="_blank">https://ai-pneumonia-detector-thedevjayden.streamlit.app/</a></p>
  
  <a href="https://ai-pneumonia-detector-thedevjayden.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20Web%20App-exploratory?style=for-the-badge&color=0078d4" alt="Live Website">
  </a>
</div>

<hr>

## 📌 Project Overview
This repository bridges the gap between academic machine learning research and clinical simulation deployment. The system transitions from a basic classifier into an **Advanced 2-Stage Diagnostic Pipeline**, integrating both supervised computer vision and unsupervised high-dimensional feature clustering to evaluate chest X-ray scans.

### 🧠 The Dual-Stage Pipeline Architecture
1. **Stage 1 (Supervised Screening):** Implements a fine-tuned **MobileNetV2** model trained with class-weight stabilization. It acts as a binary gatekeeper, scanning the image for general signs of lung consolidation with a verified **91.03% classification accuracy**.
2. **Stage 2 (Unsupervised Pathology Routing):** If Stage 1 detects pneumonia, the image is passed into a frozen **VGG16 Feature Extractor**. The high-dimensional features are reduced via **PCA (Principal Component Analysis)** and routed through a trained **K-Means Clustering** coordinate map to automatically segment the condition into 4 distinct pathogen profiles: **Bacterial, Atypical, Viral, or Fungal characteristics**.

### 📊 Dataset & Imbalance Management
* **Source:** [Kaggle Chest X-Ray (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **The Challenge:** The raw training dataset exhibits a severe class imbalance (significantly fewer healthy control scans than target pneumonia instances).
* **The Solution:** Instead of risky data duplication, the training pipeline incorporates a mathematical **Class Weights adjustment strategy**. By dynamically modifying the loss function penalty during the `.fit()` cycle, the network treats minority class misclassifications with higher severity, optimizing diagnostic sensitivity.

<hr>

## 📂 Project Architecture

```text
📂 Repo
├── 📄 app.py                           # Multi-stage Streamlit UI Dashboard Portal
├── 📄 Pneumonia_Supervised.ipynb       # MobileNetV2 EDA, training, and balance optimization workflow
├── 📄 Pneumonia_Unsupervised.ipynb     # VGG16 feature extraction, PCA reduction, and K-Means training
├── 📄 pneumonia_final.keras           # Serialized Stage 1 Production Weights (MobileNetV2)
├── 📄 vgg16_feature_extractor_pneumonia.keras # Serialized Stage 2 Feature Extractor Weights (VGG16)
├── 📄 pca_model.pkl                    # Serialized 50-component dimensional reduction transformer
├── 📄 kmeans_model.pkl                 # Serialized 4-cluster geometric mathematical layout map
├── 📄 requirements.txt                 # Explicit production cloud dependency manifest
└── 📄 LICENSE                          # MIT Open Source documentation
