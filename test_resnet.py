#!/usr/bin/env python3
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import sys
print("1. Starting", flush=True, file=sys.stderr)

print("2. Importing torch", flush=True, file=sys.stderr)
import torch
print("3. Imported torch", flush=True, file=sys.stderr)

print("4. Importing resnet18", flush=True, file=sys.stderr)
from torchvision.models import resnet18
print("5. Imported resnet18", flush=True, file=sys.stderr)

print("6. Creating resnet18 instance", flush=True, file=sys.stderr)
base = resnet18(weights=None)
print("7. Created resnet18", flush=True, file=sys.stderr)

print("8. Getting fc features", flush=True, file=sys.stderr)
num_ftrs = base.fc.in_features
print(f"9. Got {num_ftrs} features", flush=True, file=sys.stderr)

print("10. Done", flush=True, file=sys.stderr)
