# Created By LORD 
import torch
import torch.nn as nn
import torch.optim as optim
from Models.model import densenet_model,densenet_model_v3_4
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


model_save_path = "F:/PROJECTS/Tumor GradCam/Resnet/CheckPoints/DenseNet121 block 3 4/"
model = densenet_model_v3_4(num_classes=4).to(device)  

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

def train_and_test(train_loader, val_loader, test_loader, num_epochs=40):
    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        val_acc, val_loss = evaluate(val_loader)
        scheduler.step(val_loss)

        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        test_acc, _ = test(test_loader)
        print(f"Test Acc: {test_acc:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_name = f"densenet121_model{best_val_acc:.2f}.pth"
            torch.save(model.state_dict(), model_save_path + save_name)
            print(f"Saved: {model_save_path}{save_name}")

def evaluate(loader):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss_total / len(loader)
    acc = 100 * correct / total
    return acc, avg_loss

def test(loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    report = classification_report(
        all_labels, all_preds,
        target_names=["glioma", "meningioma", "notumor", "pituitary"],
        zero_division=1
    )
    return acc, report
