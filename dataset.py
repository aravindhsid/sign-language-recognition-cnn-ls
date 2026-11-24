"""
Dataset module for Sign Language Gesture Recognition.
Handles video loading, temporal uniform frame sampling, data augmentation,
and PyTorch DataLoader construction. Also includes a synthetic dataset generator
for immediate testing and verification.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import config


class VideoFrameTransform:
    """
    Applies consistent spatial augmentations across all frames in a video sequence.
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: numpy array of shape (T, H, W, C) in RGB format, uint8 [0, 255]
        Returns:
            tensor of shape (T, C, H, W) normalized
        """
        t, h, w, c = frames.shape

        # Random horizontal flip probability (0.5 during training if desired,
        # note: set to 0 if gestures are strictly one-handed / handedness-sensitive)
        apply_hflip = self.is_train and (random.random() > 0.5)

        # Random brightness and contrast adjustment factor
        brightness_factor = random.uniform(0.85, 1.15) if self.is_train else 1.0

        transformed_frames = []
        for i in range(t):
            frame = frames[i].astype(np.float32)

            if apply_hflip:
                frame = cv2.flip(frame, 1)

            if self.is_train:
                frame = np.clip(frame * brightness_factor, 0.0, 255.0)

            # Convert to [0, 1] tensor
            tensor_frame = torch.from_numpy(frame.transpose((2, 0, 1))).float() / 255.0
            tensor_frame = self.normalize(tensor_frame)
            transformed_frames.append(tensor_frame)

        return torch.stack(transformed_frames, dim=0)  # (T, C, H, W)


def sample_frames_from_video(
    video_path: str,
    sequence_length: int = config.SEQUENCE_LENGTH,
    target_size: Tuple[int, int] = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
) -> np.ndarray:
    """
    Extracts a fixed number of uniformly sampled frames from a video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames <= 0:
        # Fallback reading sequentially if frame count not reported
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No valid frames read from {video_path}")

        # Uniform sub-sampling or repeating
        indices = np.linspace(0, len(frames) - 1, sequence_length).astype(int)
        sampled = [frames[idx] for idx in indices]
        return np.array(sampled, dtype=np.uint8)

    # Frame indices spaced uniformly
    frame_indices = np.linspace(0, max(0, total_frames - 1), sequence_length).astype(int)
    selected_frames = {}

    for idx in set(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            selected_frames[idx] = frame

    cap.release()

    # Fill any gaps if a frame read failed
    result = []
    last_valid_frame = None
    for idx in frame_indices:
        if idx in selected_frames:
            last_valid_frame = selected_frames[idx]
            result.append(last_valid_frame)
        elif last_valid_frame is not None:
            result.append(last_valid_frame)
        else:
            result.append(np.zeros((*target_size[::-1], 3), dtype=np.uint8))

    return np.array(result, dtype=np.uint8)


class VideoGestureDataset(Dataset):
    """
    PyTorch Dataset for sign language video clips.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        sequence_length: int = config.SEQUENCE_LENGTH,
        is_train: bool = True
    ):
        self.samples = samples
        self.sequence_length = sequence_length
        self.transform = VideoFrameTransform(is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        try:
            frames = sample_frames_from_video(video_path, sequence_length=self.sequence_length)
        except Exception:
            # Fallback zero tensor in case of corrupted video
            frames = np.zeros(
                (self.sequence_length, config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
                dtype=np.uint8
            )

        tensor_frames = self.transform(frames)
        return tensor_frames, label


def scan_dataset(root_dir: Path, class_names: List[str]) -> List[Tuple[str, int]]:
    """
    Scans directory structured as:
    root_dir/
      Class_A/
        video1.mp4
        video2.avi
      Class_B/
        ...
    """
    samples = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for class_name, label_idx in class_to_idx.items():
        class_folder = root_dir / class_name
        if not class_folder.exists():
            continue

        for file_path in class_folder.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                samples.append((str(file_path), label_idx))

    return samples


def get_dataloaders(
    data_dir: Path,
    class_names: List[str],
    batch_size: int = config.BATCH_SIZE,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Constructs train, validation, and test DataLoaders.
    """
    all_samples = scan_dataset(data_dir, class_names)
    if not all_samples:
        raise FileNotFoundError(
            f"No video files found in {data_dir}. Ensure folders match classes: {class_names}"
        )

    random.seed(seed)
    random.shuffle(all_samples)

    total_len = len(all_samples)
    train_end = int(total_len * config.TRAIN_RATIO)
    val_end = int(total_len * (config.TRAIN_RATIO + config.VAL_RATIO))

    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]

    # Fallback if dataset is very small
    if len(val_samples) == 0:
        val_samples = train_samples
    if len(test_samples) == 0:
        test_samples = val_samples

    train_ds = VideoGestureDataset(train_samples, is_train=True)
    val_ds = VideoGestureDataset(val_samples, is_train=False)
    test_ds = VideoGestureDataset(test_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names


def generate_synthetic_dataset(
    output_dir: Path,
    classes: List[str],
    samples_per_class: int = 5,
    num_frames: int = 24,
    size: Tuple[int, int] = (160, 160)
):
    """
    Generates synthetic gesture videos simulating hand trajectory and motion
    patterns. Enables verifying pipeline without downloading external datasets.
    """
    print(f"Generating synthetic sign gesture dataset in {output_dir}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for class_idx, class_name in enumerate(classes):
        class_folder = output_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for s in range(samples_per_class):
            video_path = class_folder / f"sample_{s+1}.mp4"
            out = cv2.VideoWriter(str(video_path), fourcc, 20.0, size)

            # Different movement frequency and trajectory per class
            phase = class_idx * (2 * np.pi / max(1, len(classes)))
            center_x, center_y = size[0] // 2, size[1] // 2

            for f in range(num_frames):
                frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                # Background subtle texture
                frame[:] = (30, 30, 35)

                # Simulated hand position moving along class-specific path
                t = f / num_frames
                radius_mod = 35 + 10 * np.sin(2 * np.pi * t + phase)
                x = int(center_x + radius_mod * np.cos(2 * np.pi * t + phase))
                y = int(center_y + radius_mod * np.sin(2 * np.pi * t + phase))

                # Draw simulated hand (palm + fingers)
                color = (
                    int(150 + 80 * np.sin(phase)),
                    int(180 + 50 * np.cos(phase)),
                    240
                )
                cv2.circle(frame, (x, y), 18, color, -1)
                for finger_offset in [(-12, -22), (-4, -26), (4, -26), (12, -22)]:
                    cv2.circle(frame, (x + finger_offset[0], y + finger_offset[1]), 5, color, -1)

                # Draw gesture label subtly in frame corner
                cv2.putText(
                    frame, f"Sign: {class_name}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                )
                out.write(frame)

            out.release()

    print(f"Synthetic dataset generation complete: {len(classes)} classes, {samples_per_class} videos each.")


if __name__ == "__main__":
    generate_synthetic_dataset(config.RAW_VIDEOS_DIR, config.DEFAULT_CLASSES, samples_per_class=3)
