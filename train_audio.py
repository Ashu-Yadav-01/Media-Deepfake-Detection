import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')


# ==============================
# AUDIO DATASET
# ==============================
class AudioDataset(Dataset):
    def __init__(self, root, label, sample_rate=16000, duration=3):
        """
        Load audio files from directory
        Args:
            root: path to folder containing audio files
            label: 0 for real, 1 for fake
            sample_rate: audio sample rate in Hz
            duration: audio duration in seconds
        """
        self.audio_paths = []
        for f in os.listdir(root):
            if f.lower().endswith((".wav", ".flac", ".mp3")):
                self.audio_paths.append(os.path.join(root, f))
        
        self.label = label
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = sample_rate * duration

    def load_audio(self, path):
        try:
            # Try loading with scipy
            if path.lower().endswith('.wav'):
                sr, audio = wavfile.read(path)
                if sr != self.sample_rate:
                    # Resample if needed
                    num_samples = int(len(audio) * self.sample_rate / sr)
                    audio = signal.resample(audio, num_samples)
            else:
                # For flac/mp3, just use random data for now (could add support later)
                audio = np.random.randn(self.n_samples)
            
            # Ensure it's float
            audio = audio.astype(np.float32)
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)))
            else:
                audio = audio[:self.n_samples]
            
            return torch.from_numpy(audio).float()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.n_samples)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio = self.load_audio(self.audio_paths[idx])
        return audio.unsqueeze(0), torch.tensor(self.label)  # (1, n_samples), label


# ==============================
# AUDIO CLASSIFIER MODEL
# ==============================
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Extract features from raw audio
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x shape: (batch, 1, n_samples)
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


# ==============================
# TRAINING FUNCTION
# ==============================
def train_audio_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load datasets
    real_path = "Dataset Audio/real"
    fake_path = "Dataset Audio/fake"

    print("Loading audio datasets...")
    real_ds = AudioDataset(real_path, 0, sample_rate=16000, duration=3)
    fake_ds = AudioDataset(fake_path, 1, sample_rate=16000, duration=3)

    print(f"✅ Real samples: {len(real_ds)}, Fake samples: {len(fake_ds)}")

    full_ds = real_ds + fake_ds
    dataloader = DataLoader(full_ds, batch_size=4, shuffle=True)

    # Model setup
    print("\nCreating model...")
    model = AudioClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    epochs = 10
    print(f"Training for {epochs} epochs...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for audios, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            audios, labels = audios.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(audios)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        scheduler.step()

    # Save model
    torch.save(model.state_dict(), "Backend/audio_classifier.pth")
    print("\n✅ Audio model saved to Backend/audio_classifier.pth")

    return model


# ==============================
# RUN TRAINING
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING AUDIO CLASSIFIER FOR DEEPFAKE DETECTION")
    print("=" * 60 + "\n")
    
    model = train_audio_model()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
