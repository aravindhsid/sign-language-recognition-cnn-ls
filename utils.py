"""
Utility functions for video inspection, model profiling, and format conversions.
"""

import json
from pathlib import Path
import torch
import cv2
import numpy as np

import config
from model import CNNLSTMModel


def export_to_torchscript(checkpoint_path: Path, output_path: Path):
    """
    Exports trained PyTorch model to optimized TorchScript for lightweight deployment
    with no external runtime dependencies.
    """
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    backbone = checkpoint.get("backbone", "custom")

    model = CNNLSTMModel(
        num_classes=len(classes),
        backbone=backbone,
        feature_dim=config.CNN_FEATURE_DIM,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_layers=config.LSTM_LAYERS,
        bidirectional=config.LSTM_BIDIRECTIONAL
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(output_path))
    print(f"TorchScript model exported successfully to {output_path}")


def export_to_onnx(checkpoint_path: Path, output_onnx_path: Path):
    """
    Exports trained PyTorch model to ONNX for lightweight real-time edge deployment.
    """
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    backbone = checkpoint.get("backbone", "custom")

    model = CNNLSTMModel(
        num_classes=len(classes),
        backbone=backbone,
        feature_dim=config.CNN_FEATURE_DIM,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_layers=config.LSTM_LAYERS,
        bidirectional=config.LSTM_BIDIRECTIONAL
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["video_frames"],
        output_names=["gesture_logits"],
        dynamic_axes={"video_frames": {0: "batch_size"}, "gesture_logits": {0: "batch_size"}}
    )
    print(f"ONNX model exported successfully to {output_onnx_path}")


def summarize_model():
    """Prints layer parameter counts and tensor transformations."""
    model = CNNLSTMModel(num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("      CNN-LSTM SIGN LANGUAGE HYBRID MODEL SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Input Shape:          (Batch, {config.SEQUENCE_LENGTH}, 3, {config.FRAME_HEIGHT}, {config.FRAME_WIDTH})")
    print(f"Spatial Backbone:     Custom 4-block Conv2D (dim={config.CNN_FEATURE_DIM})")
    print(f"Temporal Layer:       BiLSTM (hidden={config.LSTM_HIDDEN_DIM}, layers={config.LSTM_LAYERS})")
    print("Attention Layer:      Temporal Self-Attention")
    print("Classifier Head:      LayerNorm -> Dense -> ReLU -> Dropout -> Linear")
    print("=" * 60)


if __name__ == "__main__":
    summarize_model()
