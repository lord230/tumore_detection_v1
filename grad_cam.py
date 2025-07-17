import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from dataset import val_loader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Models.model import densenet_model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "F:/PROJECTS/Tumor GradCam/Resnet/CheckPoints/DenseNet121 block 3 4/densenet121_model96.67.pth"
TARGET_LAYER = "features.denseblock4"
SAVE_DIR = f"F:/PROJECTS/Tumor GradCam/Resnet/results/{TARGET_LAYER.replace('.', '_')}"

os.makedirs(SAVE_DIR, exist_ok=True)


model = densenet_model(num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


target_layers = [model.features.denseblock4]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())


def apply_gradcam(image_tensor, original_image_np, index, label):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_tensor)
    pred_class = outputs.argmax(dim=1).item()

    grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    cam_image = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)

    save_path = os.path.join(SAVE_DIR, f"gradcam_{index}_label_{label}_pred_{pred_class}_layer_denseblock4.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    for idx, (images, labels) in enumerate(val_loader):
        for i in range(images.size(0)):
            img_tensor = images[i]
            label = labels[i].item()

            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * std + mean).clip(0, 1)

            apply_gradcam(img_tensor, img_np, index=f"{idx}_{i}", label=label)
