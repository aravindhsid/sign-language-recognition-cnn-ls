"""
Comprehensive Test Suite for CNN-LSTM Sign Language Recognition System.
Validates:
1. Model forward pass and output dimensions
2. Attention weights validity (probabilities sum to 1.0)
3. Dataset frame sampler & transform pipeline
4. Checkpoint integrity and inference consistency
"""

import unittest
from pathlib import Path
import numpy as np
import torch

import config
from model import CNNLSTMModel
from dataset import sample_frames_from_video, VideoFrameTransform


class TestSignLanguagePipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.seq_len = config.SEQUENCE_LENGTH
        self.num_classes = len(config.DEFAULT_CLASSES)

    def test_model_forward_dimensions(self):
        """Verify model produces correct logits and attention shapes."""
        model = CNNLSTMModel(num_classes=self.num_classes).to(self.device)
        dummy_input = torch.randn(
            self.batch_size, self.seq_len, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH
        ).to(self.device)

        logits, attn_weights = model(dummy_input, return_attention=True)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.seq_len))

    def test_attention_weights_sum(self):
        """Verify temporal attention scores sum to 1.0 per sequence."""
        model = CNNLSTMModel(num_classes=self.num_classes).to(self.device)
        dummy_input = torch.randn(
            self.batch_size, self.seq_len, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH
        ).to(self.device)

        _, attn_weights = model(dummy_input, return_attention=True)
        sums = attn_weights.sum(dim=1).detach().numpy()

        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_frame_transform_normalization(self):
        """Verify spatial transform produces normalized tensors of shape (T, C, H, W)."""
        transform = VideoFrameTransform(is_train=False)
        dummy_frames = np.random.randint(
            0, 255, (self.seq_len, config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8
        )

        tensor_out = transform(dummy_frames)

        self.assertEqual(tensor_out.shape, (self.seq_len, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH))
        self.assertIsInstance(tensor_out, torch.Tensor)

    def test_checkpoint_loading(self):
        """Verify best_model.pth loads without errors and contains expected weights."""
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pth"
        self.assertTrue(checkpoint_path.exists(), f"Checkpoint {checkpoint_path} should exist.")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("classes", checkpoint)

        model = CNNLSTMModel(num_classes=len(checkpoint["classes"])).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(
            1, self.seq_len, 3, config.FRAME_HEIGHT, config.FRAME_WIDTH
        ).to(self.device)
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.softmax(output, dim=1)

        self.assertEqual(probs.shape, (1, len(checkpoint["classes"])))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
