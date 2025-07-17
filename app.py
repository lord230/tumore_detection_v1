import os
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from Models.model import densenet_model


st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)


st.title("Brain Tumor Detection - GradCAM")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet_model(num_classes=4).to(device)
    model.load_state_dict(torch.load("F:/PROJECTS/Tumor GradCam/Resnet/CheckPoints/DenseNet121 block 3 4/densenet121_model96.67.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
st.sidebar.header("Upload MRI Image")
file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
if file:
    image = Image.open(file).convert("RGB")
    resized_image = image.resize((384, 384))
    input_tensor = transform(resized_image).unsqueeze(0).to(device)
    rgb_np = np.array(resized_image) / 255.0
    rgb_np = np.float32(rgb_np)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = np.argmax(probs)
        confidence = probs[pred_class] * 100

    # Prediction Text
    st.subheader("Prediction")
    if class_names[pred_class].lower() == "no tumor":
        st.success("No Tumor Detected")
    else:
        st.warning(f"Predicted Tumor: {class_names[pred_class]} ({confidence:.2f}%)")

    for idx, prob in enumerate(probs):
        st.write(f"{class_names[idx]}: {prob*100:.2f}%")

 
    cam = GradCAM(model=model, target_layers=[model.features.denseblock4], use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    cam_img = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
    cam_img_resized = cv2.resize(cam_img, (384, 384))

  
    cam_pp = GradCAMPlusPlus(model=model, target_layers=[model.features.denseblock4], use_cuda=torch.cuda.is_available())
    grayscale_cam_pp = cam_pp(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    cam_img_pp = show_cam_on_image(rgb_np, grayscale_cam_pp, use_rgb=True)
    cam_img_pp_resized = cv2.resize(cam_img_pp, (384, 384))


    st.subheader("Visualization")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(resized_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(cam_img_resized, caption="Grad-CAM", use_container_width=True)
    with col3:
        st.image(cam_img_pp_resized, caption="Grad-CAM++", use_container_width=True)

    save_dir = "streamlit_saved_outputs"
    os.makedirs(save_dir, exist_ok=True)
    if st.button("Save Results"):
        name = os.path.splitext(file.name)[0]
        Image.fromarray(np.uint8(rgb_np * 255)).save(os.path.join(save_dir, f"{name}_original.png"))
        cv2.imwrite(os.path.join(save_dir, f"{name}_gradcam.png"), cv2.cvtColor(cam_img_resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{name}_gradcampp.png"), cv2.cvtColor(cam_img_pp_resized, cv2.COLOR_RGB2BGR))
        st.success(f"Results saved to `{save_dir}`")
