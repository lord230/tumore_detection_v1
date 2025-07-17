import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from torchvision import transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from Models.model import densenet_model

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = densenet_model(num_classes=4).to(device)
model.load_state_dict(torch.load("F:/PROJECTS/Tumor GradCam/Resnet/CheckPoints/DenseNet121 block 3 4/densenet121_model96.67.pth", map_location=device))
model.eval()

# Constants
IMAGE_SIZE = 384
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class TumorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Brain Tumor Detection - GradCAM")
        self.master.geometry("1400x800")
        self.master.configure(bg="#121212")
        self.image_paths = []
        self.current_index = 0
        self.zoom_scale = 1.0

        self.setup_style()
        self.setup_ui()
        self.master.bind("<Configure>", lambda e: self.update_images())


    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TButton",
                        background="#1f1f1f",
                        foreground="white",
                        font=("Segoe UI", 11),
                        borderwidth=1,
                        focusthickness=3,
                        focuscolor="none")
        style.map("TButton",
                  background=[("active", "#333333")],
                  foreground=[("active", "white")])

    def setup_ui(self):
        # Top bar
        self.top_frame = tk.Frame(self.master, bg="#121212")
        self.top_frame.pack(pady=10)

        ttk.Button(self.top_frame, text="Select Folder", command=self.select_folder).pack(side="left", padx=10)
        ttk.Button(self.top_frame, text="← Prev", command=self.show_previous).pack(side="left", padx=10)
        ttk.Button(self.top_frame, text="Next →", command=self.show_next).pack(side="left", padx=10)

        # Image display
        self.image_frame = tk.Frame(self.master, bg="#121212")
        self.image_frame.pack(pady=10)

        self.canvas_frames = []
        self.img_canvases = []
        self.tk_images = [None, None, None]
        self.generated_images = [None, None, None]
        self.image_titles = ['Original', 'Grad-CAM', 'Grad-CAM++']

        for i in range(3):
            frame = tk.Frame(self.image_frame, bg="#121212")
            frame.grid(row=0, column=i, padx=20)

            tk.Label(frame, text=self.image_titles[i], font=("Segoe UI", 12, "bold"), bg="#121212", fg="white").pack()

            canvas = tk.Canvas(frame, width=IMAGE_SIZE, height=IMAGE_SIZE, bg="#1e1e1e", highlightthickness=0)
            canvas.pack()
            # canvas.bind("<MouseWheel>", self.mouse_zoom)

            self.img_canvases.append(canvas)

            ttk.Button(frame, text="Save", command=lambda i=i: self.save_image(i)).pack(pady=10)

        # Prediction label
        self.prediction_label = tk.Label(self.master, text="", font=("Segoe UI", 14),
                                         bg="#121212", fg="white")
        self.prediction_label.pack(pady=10)

        # Resize event
        self.master.bind("<Configure>", self.on_resize)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image_paths.sort()
        self.current_index = 0
        if self.image_paths:
            self.load_image()

    def show_previous(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def show_next(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()

    def mouse_zoom(self, event):
        # Identify which canvas was scrolled
        canvas_index = self.img_canvases.index(event.widget)
        canvas = self.img_canvases[canvas_index]

        # Get mouse position relative to canvas
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)

        # Adjust zoom
        old_scale = self.zoom_scale
        if event.delta > 0:
            self.zoom_scale *= 1.1
        else:
            self.zoom_scale /= 1.1
        self.zoom_scale = max(0.1, min(self.zoom_scale, 5.0))

        # Calculate new image size
        new_size = int(IMAGE_SIZE * self.zoom_scale)

        # Calculate new offset to keep zoom centered at mouse
        dx = (x / old_scale) * self.zoom_scale - x
        dy = (y / old_scale) * self.zoom_scale - y

        canvas.xview_scroll(int(dx), "units")
        canvas.yview_scroll(int(dy), "units")

        self.update_images(focus_index=canvas_index, center=(x, y))

    def update_images(self):
        total_width = self.master.winfo_width() - 200
        column_width = total_width // 3
        img_size = min(column_width, self.master.winfo_height() - 250)

        for i in range(3):
            img_array = self.generated_images[i]
            if img_array is None:
                continue

            img = Image.fromarray(np.uint8(img_array * 255) if img_array.max() <= 1 else np.uint8(img_array))
            resized = img.resize((img_size, img_size), Image.LANCZOS)
            self.tk_images[i] = ImageTk.PhotoImage(resized)

            canvas = self.img_canvases[i]
            canvas.delete("all")
            canvas.config(width=img_size, height=img_size)

            # Center the image in the canvas
            x_center = (img_size - resized.width) // 2
            y_center = (img_size - resized.height) // 2
            canvas.create_image(x_center, y_center, anchor='nw', image=self.tk_images[i])





    def load_image(self):
        image_path = self.image_paths[self.current_index]
        self.filename = os.path.splitext(os.path.basename(image_path))[0]

        image = Image.open(image_path).convert("RGB")
        resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        rgb_np = np.array(resized_image) / 255.0
        rgb_np = np.float32(rgb_np)
        input_tensor = transform(resized_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            pred_class = np.argmax(probs)
            confidence = probs[pred_class] * 100

        if class_names[pred_class].lower() == "no tumor":
            self.prediction_label.config(text="No Tumor Detected", fg="#00ff88")
        else:
            self.prediction_label.config(
                text=f"Predicted Tumor: {class_names[pred_class]} ({confidence:.2f}%)", fg="#ff4444"
            )

        cam = GradCAM(model=model, target_layers=[model.features.denseblock4], use_cuda=torch.cuda.is_available())
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_img = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

        cam_pp = GradCAMPlusPlus(model=model, target_layers=[model.features.denseblock4], use_cuda=torch.cuda.is_available())
        grayscale_cam_pp = cam_pp(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_img_pp = show_cam_on_image(rgb_np, grayscale_cam_pp, use_rgb=True)

        self.generated_images = [
            np.array(resized_image),
            cam_img,
            cam_img_pp
        ]   
        self.update_images()

    

    def save_image(self, index):
        if self.generated_images[index] is not None:
            from tkinter import filedialog
            default_name = f"{self.filename}_{self.image_titles[index].lower().replace(' ', '').replace('-', '')}.png"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if not file_path:
                return
            img = self.generated_images[index]
            if img.max() <= 1:
                img = np.uint8(img * 255)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
            messagebox.showinfo("Saved", f"{self.image_titles[index]} saved to:\n{file_path}")


    def on_resize(self, event):
        if event.widget == self.master:
            self.update_images()

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorApp(root)
    root.mainloop()
