import streamlit as st

st.set_page_config(page_title="How I Did It", layout="centered")
st.title("How I Did It")

st.markdown("""
Project Overview
This project focuses on detecting brain tumors from MRI images using a deep learning approach paired with explainable AI (XAI) techniques.

Model
- Architecture: DenseNet121
- Type:Transfer learning (pre-trained on ImageNet)
- Fine-tuning: Last classifier layer replaced and trained on a medical MRI dataset

Dataset
- Source: Public MRI brain tumor dataset
- Classes:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Size: 7023 labeled MRI images
- Preprocessing: Resized to 384x384, grayscale conversion, normalization

Training Details
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Epochs: 40
- Validation Accuracy: ~96.67%

Explainability
To interpret model decisions, we applied:
- Grad-CAM: Highlights important regions influencing predictions
- Grad-CAM++: A refined version of Grad-CAM for improved localization

Tools & Libraries
- PyTorch, torchvision
- OpenCV
- PIL (Pillow)
- pytorch-grad-cam
- Streamlit

GUI
- Built with Streamlit for real-time visualization and interaction
- Upload images and view model predictions with visual saliency maps
- Options to save generated CAM outputs for further review

---

Future Improvements
- Add Guided Backpropagation integration
- Expand dataset for better generalization
- Deploy model online via Streamlit Cloud or Hugging Face Spaces

For feedback or contributions, feel free to reach out.
""")
