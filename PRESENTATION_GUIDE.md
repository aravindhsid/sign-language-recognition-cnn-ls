# Project Defense & Presentation Guide

**Project Title:** A CNN-LSTM Hybrid Model for Sign-Language Gesture Recognition  
**Presenter Guide & Slide-by-Slide Outline**

---

### Slide 1: Title Slide
- **Title:** A CNN-LSTM Hybrid Model for Sign-Language Gesture Recognition
- **Subtitle:** Bridging the Communication Gap Through Computer Vision & Assistive AI
- **Domain:** Deep Learning & Computer Vision
- **Talking Point:** "Good morning/afternoon. Today I present our system for dynamic sign language recognition, translating video hand gestures into text captions and real-time speech."

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Over 430M people have disabling hearing loss. Society lacks fluent sign language interpreters.
- **Computer Vision Bottleneck:** Sign gestures are spatio-temporal — static images cannot distinguish between signs with identical hand shapes but opposing movement directions.
- **Talking Point:** "A static image only captures hand shape, but meaning in sign language is defined by the temporal trajectory and speed of movement."

### Slide 3: Proposed Architecture
- **Hierarchical Hybrid Model:**
  1. *Video Preprocessing*: Uniform Temporal Frame Sampling ($T=16$ frames).
  2. *Spatial Feature Extraction*: 2D Convolutional Neural Network (Conv2D + BatchNorm + ReLU).
  3. *Temporal Sequence Modeling*: 2-Layer Bidirectional LSTM ($128$ hidden units).
  4. *Temporal Self-Attention*: Dynamic attention pooling over key inflection frames.
  5. *Classification Head*: Multi-class Softmax layer.
- **Talking Point:** "By separating spatial morphology from temporal kinematics, we keep our model lightweight (1.3M parameters) and real-time capable on everyday consumer hardware."

### Slide 4: Mathematical Formulation
- Show formulas for:
  - Temporal Discretization: $x_t \in \mathbb{R}^{3 \times 112 \times 112}$
  - Spatial Encoding: $z_t = \text{CNN}(x_t)$
  - BiLSTM: $h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$
  - Self-Attention: $c = \sum_{t=1}^T \alpha_t h_t$
- **Talking Point:** "Rather than only using the final LSTM state, self-attention inspects all 16 frames and focuses on where the main gesture articulation occurs."

### Slide 5: Experimental Results
- **Metrics Table:**
  - Top-1 Accuracy: **75.00%**
  - Top-3 Accuracy: **100.00%**
  - Model Footprint: **5.3 MB (TorchScript)**
  - Latency: **~42 ms / video clip**
- Show the Confusion Matrix heatmap and Training Loss curves from `outputs/`.
- **Talking Point:** "Our 100% Top-3 accuracy demonstrates that the true sign gesture is always in the model's top candidates."

### Slide 6: Accessibility Communication Web App
- Highlight Streamlit Application Features:
  - Live webcam feed & video file upload
  - High-contrast visual caption banner (accessibility compliant)
  - Text-to-Speech (TTS) voice synthesis
  - Sentence constructor & transcript export
  - Visual Attention Explainability (shows which frames the AI focused on)

### Slide 7: Live Demonstration
- Switch to browser at `http://localhost:8501`
- Show upload of a gesture video clip.
- Demonstrate instant captioning and synthesized voice output.

### Slide 8: Conclusion & Future Scope
- **Key Takeaways:** Real-time, lightweight, deployable hybrid model bridging communication barriers.
- **Future Work:** Continuous sign language translation via CTC loss, expanding vocabulary to 1,000+ words.
- **Q&A:** Open the floor for questions.
