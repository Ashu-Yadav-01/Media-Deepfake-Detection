import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

print("1. Testing os.environ...")

print("2. Testing torch...")
import torch
print("✅ torch imported")

print("3. Testing torch.nn...")
import torch.nn as nn
print("✅ torch.nn imported")

print("4. Testing numpy...")
import numpy as np
print("✅ numpy imported")

print("5. Testing PIL...")
from PIL import Image
print("✅ PIL imported")

print("6. Testing soundfile...")
import soundfile as sf
print("✅ soundfile imported")

print("All imports successful!")