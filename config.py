"""
Configuration module for CNN-LSTM Sign Language Gesture Recognition.
Centralizes all hyperparameters, dataset paths, and operational settings.
"""

import os
from pathlib import Path
import torch

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUTS_DIR = BASE_DIR / "outputs"

for directory in [RAW_VIDEOS_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Gesture Classes
DEFAULT_CLASSES = [
    "Hello",
    "Thank You",
    "Yes",
    "No",
    "Please",
    "Help",
    "Sorry",
    "Love",
    "Good",
    "Bad"
]

# Video & Frame Processing Parameters
SEQUENCE_LENGTH = 16       # Number of temporal frames per gesture clip
FRAME_HEIGHT = 112         # Height for CNN spatial processing
FRAME_WIDTH = 112          # Width for CNN spatial processing
CHANNELS = 3               # RGB frames
FPS = 30                   # Target sampling frame rate

# Normalization constants (ImageNet standard)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Model Architecture Parameters
CNN_BACKBONE = "custom"    # "custom" or "mobilenet_v2"
CNN_FEATURE_DIM = 256      # Latent vector dimension output by CNN
LSTM_HIDDEN_DIM = 128      # LSTM hidden state dimension
LSTM_LAYERS = 2            # Number of stacked LSTM layers
LSTM_BIDIRECTIONAL = True  # Use BiLSTM to capture forward & backward temporal dynamics
DROPOUT_RATE = 0.4         # Regularization dropout rate

# Training Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Hardware Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
