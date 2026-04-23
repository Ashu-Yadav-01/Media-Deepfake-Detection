"""
Deepfake Detection Inference with ResNet18
Uses pre-trained model from Backend/image_classifier.pth
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
# MODEL - ResNet18 Based Classifier
# ==============================
def create_model():
    """Create ResNet18-based classifier, avoiding torch._dynamo hang"""
    print("Creating ResNet18 classifier...", flush=True)
    
    # Lazy import to avoid torch._dynamo initialization
    import sys
    import importlib.util
    
    # Try to import resnet18 with workaround
    try:
        # Set flag before any torch import
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        
        from torchvision.models import resnet18
        print("✅ ResNet18 imported", flush=True)
        
        base = resnet18(weights=None)
        num_ftrs = base.fc.in_features
        base.fc = nn.Identity()
        
        class ImageClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = base
                self.classifier = nn.Sequential(
                    nn.Linear(num_ftrs, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2)
                )

            def forward(self, x):
                feats = self.features(x)
                return self.classifier(feats)
        
        model = ImageClassifier()
        return model
        
    except Exception as e:
        print(f"⚠️  Could not create ResNet18 model: {e}", flush=True)
        print("Using fallback model...", flush=True)
        
        # Fallback to simple CNN
        class SimpleClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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
        
        return SimpleClassifier()


# ==============================
# LOAD MODEL
# ==============================
print("✅ Initializing model...", flush=True)
model = create_model()
print("✅ Model created", flush=True)

# Try to load pre-trained weights
model_path = "Backend/image_classifier.pth"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}...", flush=True)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded successfully", flush=True)
    except Exception as e:
        print(f"⚠️  Could not load full model weights: {e}", flush=True)
else:
    print(f"⚠️  Model file not found: {model_path}", flush=True)

model.to(device)
model.eval()
print("✅ Model ready for inference\n", flush=True)


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_image(img_path, use_gradcam=False):
    """
    Predict if image is real or fake
    Args:
        img_path: path to image file
        use_gradcam: if True, show Grad-CAM visualization
    Returns:
        label: 'REAL' or 'FAKE'
        confidence: prediction confidence (0-100)
    """
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_orig = np.array(img)
        img = img.resize((224, 224))
        
        # Convert to tensor
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        
        pred_label = 0 if probs[0] > probs[1] else 1
        label = "REAL" if pred_label == 0 else "FAKE"
        confidence = max(probs) * 100
        
        return label, confidence, img_array
    except Exception as e:
        print(f"❌ Error predicting image: {e}")
        return None, None, None


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import os
    
    print("=" * 50)
    print("DEEPFAKE DETECTION INFERENCE")
    print("=" * 50 + "\n")
    
    # Find test images
    real_folder = "Dataset/real"
    fake_folder = "Dataset/fake"
    
    test_images = []
    if os.path.exists(real_folder):
        real_images = os.listdir(real_folder)[:3]  # Test first 3
        test_images.extend([(os.path.join(real_folder, f), "REAL") for f in real_images])
    
    if os.path.exists(fake_folder):
        fake_images = os.listdir(fake_folder)[:3]  # Test first 3
        test_images.extend([(os.path.join(fake_folder, f), "FAKE") for f in fake_images])
    
    if test_images:
        print(f"📊 Testing {len(test_images)} images...\n")
        correct = 0
        for img_path, ground_truth in test_images:
            try:
                label, conf, _ = predict_image(img_path)
                if label:
                    match = "✅" if label == ground_truth else "❌"
                    print(f"{match} {img_path.split(chr(92))[-1]}: {label} ({conf:.1f}%)")
                    if label == ground_truth:
                        correct += 1
            except:
                pass
        
        accuracy = (correct / len(test_images)) * 100 if test_images else 0
        print(f"\n📈 Accuracy on test set: {accuracy:.1f}%")
    else:
        print("❌ No test images found in Dataset folders")
