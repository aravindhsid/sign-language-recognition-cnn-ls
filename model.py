"""
CNN-LSTM Hybrid Architecture for Sign Language Gesture Recognition.
Combines 2D CNN spatial frame feature extraction with Bidirectional LSTM
temporal sequence modeling and self-attention temporal aggregation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class TemporalAttention(nn.Module):
    """
    Self-attention layer that calculates attention weights across the temporal dimension,
    enabling the network to focus on the key informative movement frames.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_out shape: (Batch, Seq_Len, Hidden_Dim)
        attn_scores = self.attention(lstm_out)  # (Batch, Seq_Len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (Batch, Seq_Len, 1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1)  # (Batch, Hidden_Dim)
        return context_vector, attn_weights.squeeze(-1)


class CustomSpatialCNN(nn.Module):
    """
    Custom 4-block 2D Convolutional Neural Network designed to extract spatial features
    from sign language video frames without external weight dependencies.
    """
    def __init__(self, feature_dim: int = 256, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))           # 14 -> 1x1
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 3, H, W)
        x = self.features(x)
        return self.fc(x)


class MobileNetSpatialCNN(nn.Module):
    """
    MobileNetV2 based spatial feature extractor for enhanced transfer learning.
    """
    def __init__(self, feature_dim: int = 256, pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base.last_channel, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)


class CNNLSTMModel(nn.Module):
    """
    Complete End-to-End CNN-LSTM Hybrid Model:
    1. Spatial CNN maps each frame in the sequence to a spatial embedding.
    2. Bidirectional LSTM captures temporal trajectories and gesture kinematics.
    3. Temporal Attention computes a dynamic weighted representation of key gesture frames.
    4. Classifier head outputs class probabilities for translation to text captions.
    """
    def __init__(
        self,
        num_classes: int,
        backbone: str = "custom",
        feature_dim: int = 256,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.4
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.backbone_type = backbone

        # Spatial Feature Extractor
        if backbone == "mobilenet_v2":
            self.cnn = MobileNetSpatialCNN(feature_dim=feature_dim, dropout_rate=dropout_rate / 2)
        else:
            self.cnn = CustomSpatialCNN(feature_dim=feature_dim, dropout_rate=dropout_rate / 2)

        # Temporal Sequence Modeling
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # Temporal Attention Aggregation
        self.attention = TemporalAttention(hidden_dim=lstm_out_dim)

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.LayerNorm(lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim // 2, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Seq_Len, Channels, Height, Width)
            return_attention: If True, returns temporal attention weights
        Returns:
            logits: (Batch, Num_Classes)
            attention_weights (optional): (Batch, Seq_Len)
        """
        batch_size, seq_len, c, h, w = x.size()

        # Step 1: Reshape to pass all frames simultaneously through 2D CNN
        frames = x.contiguous().view(batch_size * seq_len, c, h, w)
        spatial_features = self.cnn(frames)  # (Batch * Seq_Len, feature_dim)

        # Step 2: Reshape back to temporal sequence for LSTM
        temporal_seq = spatial_features.view(batch_size, seq_len, -1)  # (Batch, Seq_Len, feature_dim)

        # Step 3: Temporal sequential processing
        lstm_out, _ = self.lstm(temporal_seq)  # (Batch, Seq_Len, lstm_out_dim)

        # Step 4: Attention-weighted temporal pooling
        context_vector, attn_weights = self.attention(lstm_out)  # (Batch, lstm_out_dim), (Batch, Seq_Len)

        # Step 5: Caption classification
        logits = self.classifier(context_vector)  # (Batch, Num_Classes)

        if return_attention:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    # Test model shape verification
    model = CNNLSTMModel(num_classes=10, backbone="custom")
    dummy_input = torch.randn(2, 16, 3, 112, 112)
    output, attn = model(dummy_input, return_attention=True)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output.shape}")
    print(f"Attention weights shape: {attn.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
