<div align="center">
  <h1>🫁 Pneumonia Detection AI Portal</h1>
  <p><strong>An End-to-End Deep Learning System using Fine-Tuned MobileNetV2</strong></p>

  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Backend-TensorFlow-FF6F00?style=flat-square&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  
  <br><br>
  
  <a href="https://ai-pneumonia-detector-thedevjayden.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20Web%20App-exploratory?style=for-the-badge&color=0078d4" alt="Live Website">
  </a>
</div>

<hr>

## 📌 Project Overview
This repository bridges the gap between academic machine learning research and interactive deployment. Utilizing <strong>Transfer Learning</strong> via a fine-tuned <strong>MobileNetV2</strong> architecture, the core system analyzes chest X-ray scans to identify structural indicators of Pneumonia with a verified <strong>91.03% classification accuracy</strong>.

### 📊 Dataset & Imbalance Management
* **Source:** [Kaggle Chest X-Ray (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **The Challenge:** The raw training dataset exhibits a severe class imbalance (significantly fewer healthy control scans than target pneumonia instances).
* **The Solution:** Instead of risky data duplication, the training pipeline incorporates a mathematical <strong>Class Weights adjustment strategy</strong>. By dynamically modifying the loss function penalty during the `.fit()` cycle, the network is forced to treat minority class misclassifications with higher severity, optimizing diagnostic sensitivity.

<hr>

## 📂 Project Architecture

```text
📂 Repo
├── 📄 app.py                     # Interactive Streamlit UI dashboard
├── 📄 Pnuemonia_Supervised.ipynb # Comprehensive training, EDA, & optimization workflow
├── 📄 pneumonia_final.keras       # Serialized production-ready model weights
├── 📄 requirements.txt           # Explicit cloud dependency manifest
├── 📄 LICENSE                    # MIT Open Source documentation
