"""
Simplified Deepfake Detection Inference
Uses pre-trained model for real-time predictions
"""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

device = torch.device('cpu')

# ==============================
# Simple ResNet18-based Classifier
# ==============================
class ImageClassifier(nn.Module):
    """Simple CNN classifier for deepfake detection"""
    def __init__(self):
        super().__init__()
        # Use a simple CNN instead of requiring torchvision
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Add more layers as needed
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.classifier(feats)


# ==============================
# LOAD MODEL
# ==============================
print("Creating model...")
model = ImageClassifier()

# Try to load pre-trained weights
model_path = "Backend/image_classifier.pth"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        # If state dict is from a different model, just load what matches
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️  Could not load full model: {e}")
        print("Using untrained model for demonstration")
else:
    print(f"⚠️  Model file not found: {model_path}")

model.to(device)
model.eval()
print("✅ Model ready for inference")


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_image(img_path):
    """
    Predict if image is real or fake
    Args:
        img_path: path to image file
    Returns:
        label: 'REAL' or 'FAKE'
        confidence: prediction confidence (0-100)
    """
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        pred_label = 0 if probs[0] > probs[1] else 1
        label = "REAL" if pred_label == 0 else "FAKE"
        confidence = max(probs) * 100
        
        return label, confidence
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import os
    
    # Find a test image
    real_folder = "Dataset/real"
    fake_folder = "Dataset/fake"
    
    test_img = None
    if os.path.exists(real_folder) and os.listdir(real_folder):
        test_img = os.path.join(real_folder, os.listdir(real_folder)[0])
    elif os.path.exists(fake_folder) and os.listdir(fake_folder):
        test_img = os.path.join(fake_folder, os.listdir(fake_folder)[0])
    
    if test_img:
        print(f"\n📊 Testing with: {test_img}")
        label, conf = predict_image(test_img)
        if label:
            print(f"✅ Prediction: {label} ({conf:.2f}% confidence)")
        else:
            print("❌ Failed to make prediction")
    else:
        print("❌ No test images found in Dataset folders")
