"""
Accessibility Communication App for Sign Language Gesture Recognition.
Features:
- Video File Gesture Recognition & Real-Time Webcam Inference
- Accessibility High-Contrast Caption Banner
- Browser-native Text-to-Speech (TTS) voice synthesizer
- Interactive Sentence Builder & Conversation Log
- Temporal Attention Frame Explainability
"""

import os
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms

import config
from model import CNNLSTMModel
from dataset import sample_frames_from_video

# Streamlit Page Config
st.set_page_config(
    page_title="SignComm - Gesture Accessibility App",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Accessibility Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .caption-card {
        background-color: #0F172A;
        color: #38BDF8;
        padding: 24px;
        border-radius: 12px;
        border: 2px solid #38BDF8;
        text-align: center;
        margin: 15px 0;
    }
    .caption-title {
        font-size: 1.1rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .caption-text {
        font-size: 2.8rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: 0.5px;
    }
    .confidence-badge {
        font-size: 1.2rem;
        color: #34D399;
        font-weight: 600;
    }
    .sentence-box {
        background-color: #F1F5F9;
        border-left: 6px solid #2563EB;
        padding: 16px;
        border-radius: 6px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #0F172A;
        min-height: 60px;
    }
</style>
""", unsafe_allow_html=True)


def speak_text_js(text: str):
    """Generates an HTML/JS snippet to speak text using browser Web Speech API."""
    if not text:
        return
    clean_text = text.replace('"', '\\"').replace("'", "\\'")
    js_code = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)


@st.cache_resource
def load_trained_model():
    """Loads the model checkpoint with fallback mock weights if none exists yet."""
    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
    device = torch.device(config.DEVICE)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        classes = checkpoint.get("classes", config.DEFAULT_CLASSES)
        backbone = checkpoint.get("backbone", "custom")
        model = CNNLSTMModel(
            num_classes=len(classes),
            backbone=backbone,
            feature_dim=config.CNN_FEATURE_DIM,
            lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
            lstm_layers=config.LSTM_LAYERS,
            bidirectional=config.LSTM_BIDIRECTIONAL
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, classes, device, True
    else:
        # Initialized fallback model for immediate UI testing
        classes = config.DEFAULT_CLASSES
        model = CNNLSTMModel(
            num_classes=len(classes),
            backbone="custom",
            feature_dim=config.CNN_FEATURE_DIM,
            lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
            lstm_layers=config.LSTM_LAYERS,
            bidirectional=config.LSTM_BIDIRECTIONAL
        ).to(device)
        model.eval()
        return model, classes, device, False


def preprocess_frames(frames: np.ndarray) -> torch.Tensor:
    normalize = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    tensor_list = []
    for f in frames:
        tensor = torch.from_numpy(f.transpose((2, 0, 1))).float() / 255.0
        tensor = normalize(tensor)
        tensor_list.append(tensor)
    return torch.stack(tensor_list, dim=0).unsqueeze(0)  # (1, T, C, H, W)


def predict_gesture(model, frames: np.ndarray, classes, device):
    tensor_input = preprocess_frames(frames).to(device)
    with torch.no_grad():
        logits, attn_weights = model(tensor_input, return_attention=True)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        attn = attn_weights.cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    predicted_class = classes[top_idx]
    confidence = float(probs[top_idx])

    top3_indices = np.argsort(probs)[::-1][:3]
    top3_predictions = [(classes[i], float(probs[i])) for i in top3_indices]

    return predicted_class, confidence, top3_predictions, attn


def main():
    # Session state initialization
    if "sentence" not in st.session_state:
        st.session_state["sentence"] = []
    if "last_spoken" not in st.session_state:
        st.session_state["last_spoken"] = ""
    if "trigger_speech" not in st.session_state:
        st.session_state["trigger_speech"] = False

    # Header
    st.markdown('<div class="main-header">🤟 SignComm — Sign Language Accessibility App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time video gesture translation into text captions and synthesized speech</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ System Configuration")
        model, classes, device, is_trained = load_trained_model()

        if is_trained:
            st.success("✅ Trained Model Loaded (best_model.pth)")
        else:
            st.warning("⚠️ Using Initialized Architecture (Run train.py to train on real data)")

        st.info(f"**Target Classes ({len(classes)}):**\n" + ", ".join(classes))

        input_mode = st.radio("Choose Input Mode:", ["📁 Upload Gesture Video", "📹 Real-time Webcam Feed"])
        confidence_threshold = st.slider("Confidence Threshold:", min_value=0.3, max_value=0.95, value=0.6, step=0.05)
        auto_tts = st.checkbox("🔊 Auto-speak Recognized Captions", value=True)

        st.markdown("---")
        st.markdown("### ℹ️ About the Model")
        st.markdown("""
        - **Spatial Extractor:** CNN (2D Frames)
        - **Temporal Sequencer:** BiLSTM (16 Frames)
        - **Aggregation:** Temporal Self-Attention
        - **Output:** Multi-class Captioning
        """)

    # Main columns
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("🎥 Video Input")

        if input_mode == "📁 Upload Gesture Video":
            uploaded_file = st.file_uploader("Upload a sign language gesture clip (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])

            if uploaded_file is not None:
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                st.video(tmp_path)

                if st.button("🚀 Translate Gesture Clip", type="primary", use_container_width=True):
                    with st.spinner("Extracting sequence and analyzing temporal features..."):
                        try:
                            frames = sample_frames_from_video(tmp_path, sequence_length=config.SEQUENCE_LENGTH)
                            pred_class, conf, top3, attn = predict_gesture(model, frames, classes, device)

                            # Update session state
                            if conf >= confidence_threshold:
                                st.session_state["sentence"].append(pred_class)
                                if auto_tts:
                                    st.session_state["last_spoken"] = pred_class
                                    st.session_state["trigger_speech"] = True

                            # Store last result for display
                            st.session_state["last_pred"] = {
                                "class": pred_class,
                                "conf": conf,
                                "top3": top3,
                                "attn": attn,
                                "frames": frames
                            }
                        except Exception as e:
                            st.error(f"Error processing video: {e}")
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

        elif input_mode == "📹 Real-time Webcam Feed":
            st.info("Click 'Start Real-time Buffer' to capture gestures directly from your camera.")
            run_webcam = st.toggle("Start Camera Feed")

            camera_placeholder = st.empty()
            status_placeholder = st.empty()

            if run_webcam:
                cap = cv2.VideoCapture(0)
                frame_buffer = []

                if not cap.isOpened():
                    st.error("Could not access webcam. Please check camera permissions.")
                else:
                    status_placeholder.info("Capturing live video buffer...")
                    stop_button = st.button("Stop Recognition")

                    while run_webcam and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        resized_frame = cv2.resize(rgb_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                        frame_buffer.append(resized_frame)

                        # Draw current status overlay
                        display_frame = frame.copy()
                        cv2.putText(
                            display_frame, f"Buffer: {len(frame_buffer)}/{config.SEQUENCE_LENGTH}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                        )
                        camera_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

                        if len(frame_buffer) >= config.SEQUENCE_LENGTH:
                            frames_array = np.array(frame_buffer[-config.SEQUENCE_LENGTH:])
                            pred_class, conf, top3, attn = predict_gesture(model, frames_array, classes, device)

                            if conf >= confidence_threshold:
                                if len(st.session_state["sentence"]) == 0 or st.session_state["sentence"][-1] != pred_class:
                                    st.session_state["sentence"].append(pred_class)
                                    if auto_tts:
                                        st.session_state["last_spoken"] = pred_class
                                        st.session_state["trigger_speech"] = True

                            st.session_state["last_pred"] = {
                                "class": pred_class,
                                "conf": conf,
                                "top3": top3,
                                "attn": attn,
                                "frames": frames_array
                            }
                            # Slide window by 8 frames to avoid immediate duplicate triggers
                            frame_buffer = frame_buffer[8:]

                    cap.release()

    with col_right:
        st.subheader("💬 Accessibility Caption Output")

        # Display last recognized caption
        if "last_pred" in st.session_state:
            res = st.session_state["last_pred"]
            pred_class = res["class"]
            conf = res["conf"]

            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-title">Detected Sign Gesture</div>
                <div class="caption-text">{pred_class}</div>
                <div class="confidence-badge">Confidence: {conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 Candidates
            st.markdown("#### Prediction Probabilities:")
            df_preds = pd.DataFrame(res["top3"], columns=["Gesture", "Probability"])
            df_preds["Probability (%)"] = df_preds["Probability"] * 100
            st.bar_chart(df_preds.set_index("Gesture")["Probability (%)"])

            # Temporal Attention Explainability
            with st.expander("🔍 Temporal Attention Frame Importance"):
                st.caption("Highlights which frames in the 16-frame gesture sequence influenced the classification most:")
                attn_weights = res["attn"]
                cols = st.columns(4)
                # Show top 4 key frames with highest attention
                top_frame_indices = np.argsort(attn_weights)[::-1][:4]
                for idx, f_idx in enumerate(top_frame_indices):
                    with cols[idx]:
                        st.image(res["frames"][f_idx], caption=f"Frame {f_idx+1} (w={attn_weights[f_idx]:.2f})", use_container_width=True)
        else:
            st.markdown("""
            <div class="caption-card">
                <div class="caption-title">Detected Sign Gesture</div>
                <div class="caption-text">—</div>
                <div class="confidence-badge">Awaiting Video Input</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📝 Conversation Sentence Builder")
        full_sentence = " ".join(st.session_state["sentence"])

        if full_sentence:
            st.markdown(f'<div class="sentence-box">{full_sentence}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sentence-box" style="color: #94A3B8;">Accumulated sign language conversation will appear here...</div>', unsafe_allow_html=True)

        # Control buttons
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            if st.button("🔊 Speak Sentence", use_container_width=True):
                if full_sentence:
                    st.session_state["last_spoken"] = full_sentence
                    st.session_state["trigger_speech"] = True

        with b_col2:
            if st.button("🧹 Clear All", use_container_width=True):
                st.session_state["sentence"] = []
                st.rerun()

        with b_col3:
            st.download_button(
                label="💾 Export Text",
                data=full_sentence,
                file_name="sign_translation_transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Trigger Text-To-Speech if requested
    if st.session_state.get("trigger_speech", False):
        speak_text_js(st.session_state["last_spoken"])
        st.session_state["trigger_speech"] = False


if __name__ == "__main__":
    main()
