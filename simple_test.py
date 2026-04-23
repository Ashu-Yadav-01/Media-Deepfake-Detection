#!/usr/bin/env python3
import sys
print("1. Starting", flush=True)
sys.stdout.flush()

print("2. Importing torch", flush=True)
import torch
print("3. Torch imported", flush=True)

print("4. Importing nn", flush=True)
import torch.nn as nn
print("5. nn imported", flush=True)

print("6. Importing models", flush=True)
from torchvision import models
print("7. models imported", flush=True)

print("8. Creating model", flush=True)
base = models.resnet18(weights=None)
print("9. Base model created", flush=True)

print("10. Setting up classifier", flush=True)
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = base
        self.model.fc = nn.Linear(512, 2)

model = ImageClassifier()
print("11. Model created successfully", flush=True)

print("12. Loading state dict", flush=True)
try:
    state = torch.load("Backend/image_classifier.pth", map_location='cpu')
    print(f"13. Loaded state: {list(state.keys())[:3]}", flush=True)
    model.load_state_dict(state)
    print("14. State dict loaded", flush=True)
except Exception as e:
    print(f"ERROR loading state: {e}", flush=True)

print("15. Test complete", flush=True)
