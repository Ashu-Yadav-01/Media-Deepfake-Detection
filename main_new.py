"""
Deepfake Detection Inference (Image, Video, Audio)
Lazy loading version to avoid torch._dynamo issues
"""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Defer ALL torch imports to avoid dynamo issues
import cv2
import numpy as np
from PIL import Image
import soundfile as sf

# Global model variables
model = None
audio_model = None
device = None
torch = None  # Will be set when models are loaded

# ==============================
# LAZY MODEL LOADING WITH EVERYTHING
# ==============================
print("✅ Initializing models...")

def load_models():
    global model, audio_model, device

    if model is not None:
        return  # Already loaded

    print("Loading PyTorch models...")
    import torch
    import torch.nn as nn
    
    global torch  # Make torch available globally
    torch = torch

    # ==============================
    # AUDIO CLASSIFIER
    # ==============================
    class AudioClassifier(nn.Module):
        def __init__(self):
            super().__init__()
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
            feats = self.features(x)
            feats = feats.view(feats.size(0), -1)
            return self.classifier(feats)


    # ==============================
    # IMAGE CLASSIFIER - CNN
    # ==============================
    class ImageClassifier(nn.Module):
        """CNN-based classifier for deepfake detection"""
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )

        def forward(self, x):
            feats = self.features(x)
            feats = feats.view(feats.size(0), -1)
            return self.classifier(feats)

    global device
    device = torch.device('cpu')

    # Image model
    model = ImageClassifier()
    model_path = "Backend/image_classifier.pth"
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("✅ Image model weights loaded")
        except Exception as e:
            print(f"⚠️  Image model load error: {e}")
    else:
        print(f"⚠️  Image model not found: {model_path}")

    model.to(device)
    model.eval()

    # Audio model
    audio_model = AudioClassifier()
    audio_model_path = "Backend/audio_classifier.pth"
    if os.path.exists(audio_model_path):
        try:
            state_dict = torch.load(audio_model_path, map_location=device)
            audio_model.load_state_dict(state_dict, strict=False)
            print("✅ Audio model weights loaded")
        except Exception as e:
            print(f"⚠️  Audio model load error: {e}")
    else:
        print(f"⚠️  Audio model not found: {audio_model_path}")

    audio_model.to(device)
    audio_model.eval()
    print("✅ Models ready for inference\n")


# ==============================
# PREDICTION FUNCTIONS
# ==============================
def predict_image(img_path):
    """Predict if image is real or fake"""
    try:
        # Load models if not already loaded
        if model is None:
            load_models()

        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        pred = 0 if probs[0] > probs[1] else 1
        label = "REAL" if pred == 0 else "FAKE"
        confidence = max(probs) * 100

        return label, confidence
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def predict_audio(audio_path, sample_rate=16000, duration=3):
    """Predict if audio is real or fake"""
    try:
        # Load models if not already loaded
        if audio_model is None:
            load_models()

        # Load audio using soundfile (supports FLAC, WAV, etc.)
        audio, sr = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != sample_rate:
            try:
                from scipy import signal
                num_samples = int(len(audio) * sample_rate / sr)
                audio = signal.resample(audio, num_samples)
            except ImportError:
                # Simple downsampling if scipy not available
                ratio = sample_rate / sr
                if ratio < 1:  # Downsampling
                    step = int(1 / ratio)
                    audio = audio[::step]
                # Skip upsampling

        n_samples = sample_rate * duration

        # Normalize
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Pad or trim
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        else:
            audio = audio[:n_samples]

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = audio_model(audio_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        pred = 0 if probs[0] > probs[1] else 1
        label = "REAL" if pred == 0 else "FAKE"
        confidence = max(probs) * 100

        return label, confidence
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None, None


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("  DEEPFAKE DETECTION SYSTEM - IMAGE, VIDEO & AUDIO")
    print("=" * 60 + "\n")

    real_folder = "Dataset/real"
    fake_folder = "Dataset/fake"
    real_audio_folder = "Dataset Audio/real"
    fake_audio_folder = "Dataset Audio/fake"

    # Test IMAGE
    print("🖼️  IMAGE CLASSIFICATION TEST")
    print("-" * 60)
    test_images = []
    if os.path.exists(real_folder):
        for f in os.listdir(real_folder)[:2]:
            test_images.append((os.path.join(real_folder, f), "REAL"))

    if os.path.exists(fake_folder):
        for f in os.listdir(fake_folder)[:2]:
            test_images.append((os.path.join(fake_folder, f), "FAKE"))

    if test_images:
        correct = 0
        for img_path, gt_label in test_images:
            label, conf = predict_image(img_path)
            if label:
                status = "✅" if label == gt_label else "❌"
                filename = os.path.basename(img_path)
                print(f"  {status} {filename}: {label} ({conf:.1f}%)")
                if label == gt_label:
                    correct += 1
        acc = (correct / len(test_images)) * 100
        print(f"  📊 Image Accuracy: {correct}/{len(test_images)} ({acc:.1f}%)\n")

    # Test AUDIO
    print("🔊 AUDIO CLASSIFICATION TEST")
    print("-" * 60)
    test_audios = []
    if os.path.exists(real_audio_folder):
        for f in os.listdir(real_audio_folder)[:2]:
            if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                test_audios.append((os.path.join(real_audio_folder, f), "REAL"))

    if os.path.exists(fake_audio_folder):
        for f in os.listdir(fake_audio_folder)[:2]:
            if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg')):
                test_audios.append((os.path.join(fake_audio_folder, f), "FAKE"))

    if test_audios:
        correct = 0
        for audio_path, gt_label in test_audios:
            label, conf = predict_audio(audio_path)
            if label:
                status = "✅" if label == gt_label else "❌"
                filename = os.path.basename(audio_path)
                print(f"  {status} {filename}: {label} ({conf:.1f}%)")
                if label == gt_label:
                    correct += 1
        acc = (correct / len(test_audios)) * 100
        print(f"  📊 Audio Accuracy: {correct}/{len(test_audios)} ({acc:.1f}%)\n")
    else:
        print("  ⚠️  No audio files found for testing\n")

    print("=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)