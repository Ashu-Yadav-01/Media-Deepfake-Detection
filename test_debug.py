import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cpu')

print("Step 1: Creating ImageClassifier...")
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
        return self.classifier(feats)

model = ImageClassifier()
print("✅ Model created")

print("Step 2: Loading model state dict...")
try:
    model.load_state_dict(torch.load(
        "Backend/image_classifier.pth",
        map_location=device
    ))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

model.to(device)
model.eval()
print("✅ Model ready for inference")
