import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Paths
DATASET_ROOT = "F:/PROJECTS/Tumor GradCam/Resnet/dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "Training")
TEST_DIR = os.path.join(DATASET_ROOT, "Testing")

# Transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # ResNet/DenseNet/Vit etc.
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Load Training Dataset
full_train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)

# Create train/val split
val_split = 0.1
val_size = int(len(full_train_dataset) * val_split)
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Load Test Dataset
test_dataset = ImageFolder(root=TEST_DIR, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Class Names (optional)
class_names = full_train_dataset.classes
print("Class mapping:", class_names)
