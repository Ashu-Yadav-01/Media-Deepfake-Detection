# 🎤 Audio Deepfake Detection Integration

## Overview
This project now includes **audio classification** alongside image and video deepfake detection. The audio model can detect synthetic/deepfake voices with high accuracy.

## Files Added/Modified

### 1. **train_audio.py** (NEW)
- Trains the audio classification model
- Uses `scipy` for audio loading (FLAC/WAV support)
- Dataset: `Dataset Audio/real` and `Dataset Audio/fake`
- Outputs trained model to: `Backend/audio_classifier.pth`
- **To train**: `python train_audio.py`

### 2. **main.py** (UPDATED)
- Added `AudioClassifier` class
- Added `predict_audio()` function
- Tests audio predictions on sample files
- Works without librosa (uses scipy instead)

### 3. **Backend/app.py** (UPDATED)
- Added `/predict_audio` FastAPI endpoint
- Audio model loading
- Supports WAV format uploads
- Returns: `{"prediction": "REAL/FAKE", "confidence": "XX.XX%"}`

## Audio Model Architecture

```
AudioClassifier(
  Conv1d (1 → 64)  → BatchNorm → ReLU → MaxPool
  Conv1d (64 → 128) → BatchNorm → ReLU → MaxPool  
  Conv1d (128 → 256) → BatchNorm → ReLU → AdaptiveAvgPool
  Linear (256 → 128 → 2)
)
```

## Dataset Structure
```
Dataset Audio/
├── real/        (620 FLAC files)
└── fake/        (675 FLAC files)
```

## Training Details
- **Epochs**: 10
- **Batch Size**: 4
- **Learning Rate**: 0.001 (with StepLR scheduler)
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Audio Duration**: 3 seconds
- **Sample Rate**: 16 kHz

## Usage

### Train Audio Model
```bash
python train_audio.py
```

### Use All 3 Classifiers (Image, Video, Audio)
```bash
python main.py
```

### Backend API Endpoints

#### Audio Prediction
```bash
curl -X POST "http://localhost:8000/predict_audio" \
  -F "file=@audio.wav"
```

Response:
```json
{
  "prediction": "REAL",
  "confidence": "92.45%"
}
```

#### Image Prediction
```bash
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@image.jpg"
```

#### Video Prediction
```bash
curl -X POST "http://localhost:8000/predict_video" \
  -F "file=@video.mp4"
```

## Integration with Frontend

Add this to your frontend for audio upload:

```javascript
const formData = new FormData();
formData.append('file', audioFile);

const response = await fetch('/predict_audio', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Prediction: ${result.prediction} (${result.confidence})`);
```

## Model Weights
- **Image**: `Backend/image_classifier.pth`
- **Video**: `Backend/video_classifier.pth`
- **Audio**: `Backend/audio_classifier.pth` (trained after running `train_audio.py`)

## Dependencies
- torch
- torchvision
- scipy
- numpy
- tqdm
- pillow
- opencv-python
- fastapi
- uvicorn (for backend)

## Performance Metrics
After training, check the console output for:
- Per-epoch loss
- Per-epoch accuracy
- Final test results

## Notes
- Audio files should be at least 3 seconds long (shorter files are padded, longer ones are trimmed)
- FLAC files are automatically detected and loaded
- WAV files are also supported
- Model works without librosa (uses scipy for compatibility)
