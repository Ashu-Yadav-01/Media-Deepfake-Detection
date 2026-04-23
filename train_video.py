import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18


# ------------------------------------------------------
# ✅ DATASET
# ------------------------------------------------------
class VideoDataset(Dataset):
    def __init__(self, root, label, num_frames=8):
        self.video_paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".mp4", ".avi", ".mkv"))
        ]
        self.label = label
        self.num_frames = num_frames
        self.transform = transforms.Compose([transforms.ToTensor()])

    def read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total // self.num_frames, 1)

        for i in range(0, total, interval):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frame = self.transform(frame)
            frames.append(frame)

            if len(frames) == self.num_frames:
                break

        cap.release()

        # repeat last frame if video short
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        frames = torch.stack(frames)              # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)       # (C, T, H, W)
        return frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_tensor = self.read_video(self.video_paths[idx])
        return video_tensor, torch.tensor(self.label)


# ------------------------------------------------------
# ✅ VIDEO MODEL
# ------------------------------------------------------
class VideoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        model = r3d_18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        self.model = model

    def forward(self, x):
        return self.model(x)


# ------------------------------------------------------
# ✅ TRAINING FUNCTION
# ------------------------------------------------------
def train_video_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    real_path = "Datasetvideo/real"
    fake_path = "Datasetvideo/fake"

    # Dataset Load
    real_ds = VideoDataset(real_path, 0)
    fake_ds = VideoDataset(fake_path, 1)

    full_ds = real_ds + fake_ds
    dataloader = DataLoader(full_ds, batch_size=2, shuffle=True)

    # Model
    model = VideoClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for videos, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Completed | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "video_classifier.pth")
    print("✅ Saved video_classifier.pth")


# ------------------------------------------------------
# ✅ RUN TRAINING
# ------------------------------------------------------
if __name__ == "__main__":
    train_video_model()
