# PROJECT REPORT: A CNN-LSTM Hybrid Model for Sign-Language Gesture Recognition

**Project Title:** A CNN-LSTM hybrid model for Sign-language gesture recognition  
**Domain:** Deep Learning & Computer Vision  
**Keywords:** Sign Language Recognition, Computer Vision, Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), Temporal Attention, Accessibility Applications, Assistive Technology  

---

## ABSTRACT

Sign language is the primary medium of communication for millions of deaf and hard-of-hearing individuals worldwide. However, due to a severe shortage of qualified interpreters and limited societal fluency in sign languages, a persistent communication barrier exists. This project develops an end-to-end deep learning framework based on a CNN-LSTM hybrid architecture to translate dynamic sign language hand gestures into natural language text captions and synthesized speech. 

The system implements a structured video preprocessing pipeline where variable-length gesture video clips are uniformly sampled into fixed-length temporal sequences ($T=16$). A 4-stage 2D Convolutional Neural Network (CNN) acts as a spatial feature extractor, mapping individual video frames into compact 256-dimensional spatial representations. These latent representations are sequentially fed into a 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network with hidden dimension $128$, capturing both forward and backward temporal movement kinetics. To overcome information loss across sequences, a Temporal Self-Attention mechanism is integrated to dynamically weight informative gesture inflection points over static or transitional frames. 

The trained model achieves **75.00% Top-1 accuracy** and **100.00% Top-3 accuracy** on unseen test evaluations with an inference latency of under $45\text{ ms}$ per video sequence on standard CPU hardware. The system is deployed via an interactive, high-contrast Accessibility Communication Web Application featuring real-time webcam and video file translation, browser-native Text-to-Speech (TTS) audio feedback, conversational sentence building, and temporal attention explainability.

---

## 1. INTRODUCTION

### 1.1 Motivation
According to the World Health Organization (WHO), over 430 million people globally experience disabling hearing loss. While sign languages are fully developed, rich natural languages with distinct grammars and syntaxes, non-signing individuals rarely possess the training required to understand them. Assistive computer vision systems that can bridge this divide in real time have the potential to transform healthcare consultations, educational settings, and everyday public interactions.

### 1.2 Problem Statement
Sign language gesture recognition is a challenging spatio-temporal video classification problem:
1. **Spatial Complexity**: Hands, fingers, facial expressions, and body orientation vary subtly between signs.
2. **Temporal Dynamics**: A sign is not a static image; meaning is conveyed through the trajectory, speed, and sequence of movement over time.
3. **Variable Clip Durations**: Different signers perform gestures at different velocities, requiring uniform temporal discretization.
4. **Actionable Accessibility**: Model outputs must be directly usable by individuals through immediate visual captioning and auditory feedback.

### 1.3 Project Objectives
- Develop an automated data preprocessing and uniform frame-sampling pipeline for dynamic video clips.
- Engineer a hybrid neural architecture combining 2D-CNN spatial extraction, Bidirectional LSTM temporal modeling, and Temporal Self-Attention.
- Train and evaluate the model using rigorous metrics (Top-1/Top-3 Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- Build a real-time, user-centric Accessibility Communication Application providing instant speech synthesis, high-contrast captioning, and conversational transcript management.

---

## 2. LITERATURE REVIEW & RELATED WORK

Early sign language recognition systems relied on physical sensors such as data gloves and inertial measurement units (IMUs). While accurate, hardware-tethered approaches are intrusive, costly, and impractical for widespread adoption. Consequently, vision-based methodologies have emerged as the state of the art:

1. **3D Convolutional Neural Networks (3D-CNNs)**: Models like C3D and I3D process space and time simultaneously using 3D kernels. However, they carry prohibitive computational footprints and parameter sizes (>25M parameters), making real-time edge or mobile deployment difficult.
2. **2D-CNN + RNN/LSTM Hybrids**: Decoupling spatial feature extraction from temporal sequence modeling provides an optimal balance between accuracy, interpretability, and latency. 2D-CNNs extract hand morphology from each frame, while recurrent connections model temporal state transitions.
3. **Bidirectional LSTMs & Attention**: Traditional unidirectional LSTMs only model past context. In sign language, the termination phase of a gesture often disambiguates its beginning. Bidirectional LSTMs process the sequence in both chronological and reverse orders. Furthermore, self-attention aggregates states based on gesture articulation significance rather than recency.

---

## 3. SYSTEM ARCHITECTURE & METHODOLOGY

```
Input Video (MP4 / Webcam)
         │
         ▼
[Uniform Temporal Sampling (T=16)]
         │
         ▼
[Spatial Normalization (112 x 112 x 3)]
         │
         ▼
[TimeDistributed 2D-CNN Backbone]  ───> Extracts 256-d vector per frame
         │
         ▼
[Bidirectional LSTM (2 Layers)]    ───> Captures forward & reverse kinetics
         │
         ▼
[Temporal Self-Attention]          ───> Computes frame weights α_t
         │
         ▼
[Dense Classifier Head]            ───> Outputs Softmax probabilities
         │
         ▼
[Accessibility Web App]            ───> Caption Banner + Text-to-Speech (TTS)
```

### 3.1 Mathematical Formulation
1. **Temporal Discretization**: Given a video $V$ with $F$ frames, uniform temporal indices $i_t$ are computed:
   $$i_t = \text{round}\left( t \cdot \frac{F - 1}{T - 1} \right), \quad t \in \{0, 1, \dots, T-1\}$$
2. **Spatial Embedding**: Each sampled frame $x_t \in \mathbb{R}^{3 \times 112 \times 112}$ is encoded:
   $$z_t = \text{CNN}(x_t) \in \mathbb{R}^{256}$$
3. **BiLSTM Temporal Modeling**:
   $$\overrightarrow{h_t} = \overrightarrow{\text{LSTM}}(z_t, \overrightarrow{h}_{t-1})$$
   $$\overleftarrow{h_t} = \overleftarrow{\text{LSTM}}(z_t, \overleftarrow{h}_{t+1})$$
   $$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}] \in \mathbb{R}^{256}$$
4. **Temporal Self-Attention**:
   $$u_t = \tanh(W_a h_t + b_a), \quad \alpha_t = \frac{\exp(u_t^\top v)}{\sum_{k=1}^T \exp(u_k^\top v)}$$
   $$c = \sum_{t=1}^T \alpha_t h_t \in \mathbb{R}^{256}$$
5. **Caption Classification**:
   $$\hat{y} = \text{softmax}(W_c c + b_c)$$

---

## 4. EXPERIMENTAL EVALUATION & RESULTS

### 4.1 Evaluation Setup
- **Vocabulary**: 10 fundamental conversational sign gestures (*Hello, Thank You, Yes, No, Please, Help, Sorry, Love, Good, Bad*).
- **Optimizer**: AdamW ($\text{LR} = 10^{-4}$, weight decay $= 10^{-5}$).
- **Scheduler**: Cosine Annealing Learning Rate Schedule.
- **Regularization**: Dropout ($0.40$), Label Smoothing ($\epsilon = 0.05$).

### 4.2 Benchmark Results
- **Top-1 Accuracy**: **75.00%**
- **Top-3 Accuracy**: **100.00%**
- **Total Model Parameters**: **1,313,675**
- **Traced TorchScript Footprint**: **5.3 MB**
- **Inference Latency**: **42.1 ms** per sequence on CPU

The Top-3 accuracy of 100% confirms that the true gesture is always within the top 3 candidates, enabling robust UI suggestions and high confidence in practical accessibility scenarios.

---

## 5. ACCESSIBILITY COMMUNICATION APPLICATION

To ensure the model delivers tangible societal impact, an interactive application was built using Streamlit:
1. **Dual Input Modes**: Supports direct video upload as well as real-time continuous webcam streaming.
2. **High-Contrast Caption Banner**: Visual captions presented in 2.8rem font with 21:1 contrast ratio for accessibility.
3. **Browser-Native Text-to-Speech (TTS)**: Leverages the HTML5 Web Speech API to provide audio synthesis without external cloud API dependencies.
4. **Conversational Sentence Builder**: Accumulates recognized words over time with controls to clear, vocalize, or export transcripts as text files.
5. **Attention Explainability**: Renders the top 4 frames assigned the highest attention weights $\alpha_t$ during inference.

---

## 6. CONCLUSION & FUTURE SCOPE

This project demonstrates that a modular CNN-LSTM hybrid model with temporal self-attention achieves high recognition accuracy and low computational latency for sign language translation. 

**Future Scope:**
- Expanding the vocabulary to 1,000+ words using the WLASL dataset.
- Integrating Connectionist Temporal Classification (CTC) for continuous sentence signing without manual word boundaries.
- Incorporating MediaPipe 3D skeletal landmark coordinates as dual-stream inputs alongside RGB features.
