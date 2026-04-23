# Professional Deepfake Detection Project Report PDF Generator (Fixed)
#
# How to use:
# 1. Make sure you have fpdf installed: pip install fpdf
# 2. Run this script in your project folder.
# 3. It will generate Deepfake_Detection_Report.pdf, ready to open and edit.

from fpdf import FPDF

slides = [
    {"title": "Deepfake Detection System", "content": "Project Report & Demo\n\n(Name, Team, Date: [Leave Blank])"},
    {"title": "Introduction", "content": "Deepfakes are AI-generated media that manipulate images, videos, and audio.\nThis project detects deepfakes in images, videos, and audio using deep learning."},
    {"title": "Objectives", "content": "- Develop robust models for detecting fake content in images, videos, and audio.\n- Integrate all models into a unified, user-friendly system.\n- Provide a REST API and demo interface for real-world testing."},
    {"title": "Project Structure", "content": "- main.py: Unified inference script\n- train_model.py: Image model trainer\n- train_video.py: Video model trainer\n- train_audio.py: Audio model trainer\n- Backend/app.py: FastAPI backend\n- Dataset folders: Real and fake samples for each modality"},
    {"title": "System Architecture", "content": "Data Collection -> Model Training -> Inference Engine -> REST API -> Frontend Demo"},
    {"title": "Image Deepfake Detection", "content": "- Model: ResNet18 CNN\n- Input: JPG/PNG images (224x224)\n- Output: REAL/FAKE + confidence\n- Training: train_model.py\n- API: /predict_image"},
    {"title": "Video Deepfake Detection", "content": "- Model: R3D-18 (3D CNN)\n- Input: MP4/AVI/MKV videos\n- Output: REAL/FAKE + confidence\n- Training: train_video.py\n- API: /predict_video"},
    {"title": "Audio Deepfake Detection", "content": "- Model: 1D CNN\n- Input: WAV/FLAC (3s, 16kHz)\n- Output: REAL/FAKE + confidence\n- Training: train_audio.py\n- API: /predict_audio"},
    {"title": "Demo Instructions", "content": "- Train models (if needed): python train_audio.py\n- Test all models: python main.py\n- Run backend: uvicorn app:app --reload --port 8000\n- Access API: http://localhost:8000"},
    {"title": "API Endpoints", "content": "- /predict_image: POST image file\n- /predict_video: POST video file\n- /predict_audio: POST audio file"},
    {"title": "Results & Performance", "content": "(Add your model accuracy, confusion matrix, or sample results here)"},
    {"title": "Conclusion", "content": "Multi-modal deepfake detection achieved.\nReady for integration and further improvements."},
    {"title": "Q&A", "content": "Thank you!"}
]

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 20)
pdf.cell(0, 10, 'Deepfake Detection System', ln=True, align='C')
pdf.ln(10)
pdf.set_font("Arial", size=14)
for slide in slides[1:]:
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, slide["title"], ln=True)
    pdf.set_font("Arial", size=12)
    for line in slide["content"].split('\n'):
        # Replace problematic unicode arrows with ASCII
        line = line.replace('→', '->')
        pdf.multi_cell(0, 8, line)
    pdf.ln(5)
pdf.output("Deepfake_Detection_Report.pdf")
print("Deepfake_Detection_Report.pdf generated successfully.")
