# CNN-LSTM Hybrid Model for Sign-Language Gesture Recognition

An end-to-end deep learning system and accessibility communication application that translates sign-language video clips into text captions and synthesized speech.

---

## 🌟 Key Features

1. **Spatial Feature Extraction (CNN)**:
   - 4-stage convolutional neural network extracting spatial descriptors from individual video frames.
   - Optional transfer learning support using `MobileNetV2` for edge deployment.
2. **Temporal Kinematic Modeling (Bidirectional LSTM)**:
   - 2-layer BiLSTM capturing sequential hand movement dynamics across time ($T=16$).
3. **Temporal Self-Attention Aggregation**:
   - Dynamically weights informative gesture transition frames over static pauses.
4. **Data Preprocessing & Augmentation Pipeline**:
   - Uniform temporal frame sub-sampling for variable-length video clips.
   - Built-in synthetic dataset generator for immediate verification.
5. **Accessibility Communication Web App (Streamlit)**:
   - Real-time video clip translation & webcam inference.
   - High-contrast visual caption banner.
   - Browser-native Text-to-Speech (TTS) voice synthesizer.
   - Conversation sentence builder and transcript exporter.
   - Temporal attention frame explainability.

---

## 📁 Repository Structure

```
sign_language_cnn_lstm/
├── config.py             # Hyperparameters, paths, and model dimensions
├── model.py              # CNN + BiLSTM + Attention deep learning model
├── dataset.py            # Video loading, sampling, augmentations & synthetic generator
├── train.py              # Training loop with early stopping and checkpointing
├── evaluate.py           # Evaluation metrics, confusion matrix, top-1/top-3 report
├── app.py                # Accessibility Streamlit application
├── utils.py              # Profiling and ONNX export utilities
├── requirements.txt      # Python dependencies
├── data/
│   └── raw_videos/       # Video dataset organized into class folders
├── checkpoints/          # Saved model weights (.pth)
└── outputs/              # Evaluation plots and training curves
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate to the project directory and install dependencies:
```bash
cd C:\Users\JSidd\.gemini\antigravity\scratch\sign_language_cnn_lstm
pip install -r requirements.txt
```

### 2. Dataset Preparation
Organize your sign language video clips under `data/raw_videos/`:
```
data/raw_videos/
├── Hello/
│   ├── clip1.mp4
│   └── clip2.mp4
├── Thank You/
├── Yes/
├── No/
└── ...
```
> **Tip:** If you do not have video files yet, the system can automatically generate synthetic gesture video clips using the `--synthetic` flag.

### 3. Model Training
Run training with default settings (or generate synthetic data automatically):
```bash
python train.py --epochs 30 --batch_size 8 --synthetic
```

Key arguments:
- `--epochs`: Number of training epochs (default: `35`)
- `--batch_size`: Batch size (default: `8`)
- `--lr`: Learning rate (default: `1e-4`)
- `--backbone`: `custom` or `mobilenet_v2`
- `--synthetic`: Generate synthetic videos if data directory is empty

### 4. Model Evaluation
Evaluate the trained checkpoint on the test split:
```bash
python evaluate.py
```
This generates:
- Top-1 & Top-3 Accuracy
- Per-class Precision, Recall, and F1-Score
- `outputs/confusion_matrix.png`
- `outputs/evaluation_metrics.json`

### 5. Launch the Accessibility Communication App
Run the interactive Streamlit communication interface:
```bash
streamlit run app.py
```
The app will open in your web browser, allowing you to:
- Upload any recorded gesture video or use your webcam.
- View translated text captions in real time.
- Listen to synthesized audio playback.
- Accumulate signs into complete conversational sentences.
