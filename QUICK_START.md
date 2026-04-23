# 🎬 Deepfake Detection - Quick Start Guide

## ⚡ Quick Commands

### 1. Train Audio Model (if needed)
```bash
cd "c:\Project\Deepfake detection"
python train_audio.py
# Output: Backend/audio_classifier.pth
```
⏱️ **Duration**: ~1.5-2 hours on CPU

### 2. Test All Models
```bash
python main.py
```
✅ Tests image, video, and audio inference

### 3. Run Backend Server
```bash
cd Backend
uvicorn app:app --reload --port 8000
```
🌐 Visit: http://localhost:8000

---

## 📁 File Structure

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | Unified inference (Image/Video/Audio) | ✅ Ready |
| `train_model.py` | Train image classifier | ✅ Trained |
| `train_video.py` | Train video classifier | ✅ Trained |
| `train_audio.py` | Train audio classifier | 🔄 Training |
| `Backend/app.py` | FastAPI REST API | ✅ Updated |

---

## 🔌 API Endpoints

### Test Image
```bash
curl -X POST "http://localhost:8000/predict_image" \
  -F "file=@path/to/image.jpg"
```

### Test Video
```bash
curl -X POST "http://localhost:8000/predict_video" \
  -F "file=@path/to/video.mp4"
```

### Test Audio (NEW)
```bash
curl -X POST "http://localhost:8000/predict_audio" \
  -F "file=@path/to/audio.wav"
```

---

## 🎯 Model Summary

| Model | Type | Input | Epochs | Status |
|-------|------|-------|--------|--------|
| **Image** | ResNet18 | 224×224 JPG | 10 | ✅ Complete |
| **Video** | R3D-18 | 112×112 MP4 | 5 | ✅ Complete |
| **Audio** | 1D CNN | 16kHz WAV | 10 | 🔄 Training |

---

## 📊 Sample Outputs

### Image Prediction
```
✅ real_00001.jpg: REAL (95.2%)
```

### Video Prediction
```
❌ video_001.mp4: FAKE (87.3%)
```

### Audio Prediction (NEW)
```
✅ voice_001.wav: REAL (92.1%)
```

---

## 🛠️ Troubleshooting

### Issue: "Model file not found"
**Solution**: Train the model first:
```bash
python train_audio.py
```

### Issue: "No audio files found"
**Solution**: Check `Dataset Audio/` folder exists with:
- `Dataset Audio/real/` - has .wav or .flac files
- `Dataset Audio/fake/` - has .wav or .flac files

### Issue: ImportError
**Solution**: Install dependencies:
```bash
pip install torch torchvision scipy numpy tqdm pillow opencv-python fastapi uvicorn
```

---

## 🚀 Next: Frontend Integration

### HTML Example
```html
<form id="audioForm">
  <input type="file" id="audioFile" accept=".wav,.flac">
  <button onclick="predictAudio()">Upload Audio</button>
  <div id="result"></div>
</form>

<script>
async function predictAudio() {
  const file = document.getElementById('audioFile').files[0];
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/predict_audio', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  document.getElementById('result').innerHTML = 
    `Prediction: ${result.prediction} (${result.confidence})`;
}
</script>
```

---

## 📝 Important Notes

✅ **Three Models Ready**: Image, Video, and Audio classifiers fully integrated

✅ **Same Architecture**: All models use similar training approach

✅ **REST API**: All accessible via FastAPI backend

⏳ **Audio Training**: Currently training (monitor terminal)

📊 **Large Dataset**: 1000+ audio samples (620 real + 675 fake)

🎯 **Inference Ready**: Once training completes, all three models work together

---

## ✨ What You Have

```
✅ Image Deepfake Detection (ResNet18)
✅ Video Deepfake Detection (R3D-18)
✅ Audio Deepfake Detection (1D CNN) - NEW
✅ Unified API Backend (FastAPI)
✅ Inference Script (main.py)
✅ Complete Documentation
```

**Total**: Full-stack deepfake detection system! 🎉
