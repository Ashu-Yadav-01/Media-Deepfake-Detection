import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 1. Image Dataset Loader and Preprocessing
class ImageDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_files)

# 2. Transform
img_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Model Definition
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = base.fc.in_features
        base.fc = nn.Identity()
        self.features = base
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feats = self.features(x)
        out = self.classifier(feats)
        return out

# 4. Training Function
def train_model(model, train_loader, val_loader, epochs=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                pred_class = torch.argmax(outputs, 1).cpu().numpy()
                val_preds.extend(pred_class)
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

# ------------------------
# MAIN TRAINING SCRIPT
# ------------------------

if __name__ == "__main__":

    real_dir = "C:/Deepfake Detection/dataset/real"
    fake_dir = "C:/Deepfake Detection/dataset/fake"

    image_files = []
    labels = []

    for img_name in os.listdir(real_dir):
        image_files.append(os.path.join(real_dir, img_name))
        labels.append(0)

    for img_name in os.listdir(fake_dir):
        image_files.append(os.path.join(fake_dir, img_name))
        labels.append(1)

    print(f"Total images: {len(image_files)} (Real: {labels.count(0)}, Fake: {labels.count(1)})")

    dataset = ImageDataset(image_files, labels, transform=img_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = ImageClassifier()
    train_model(model, train_loader, val_loader, epochs=8, lr=1e-4)

    torch.save(model.state_dict(), "image_classifier.pth")
    print("✅ Training complete. Model saved as 'image_classifier.pth'")
