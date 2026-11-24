"""
Training pipeline for CNN-LSTM Sign Language Gesture Recognition.
Features:
- AdamW optimizer with Cosine Annealing learning rate schedule
- Early stopping & best model checkpointing
- Loss & Accuracy visualization curve generation
- Metadata and class index export for deployment
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import get_dataloaders, generate_synthetic_dataset
from model import CNNLSTMModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (videos, labels) in enumerate(loader):
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in recurrent connections
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / max(1, total)
    val_acc = correct / max(1, total)
    return val_loss, val_acc


def plot_training_history(history, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")


def run_training(args):
    device = torch.device(config.DEVICE)
    print(f"Executing training on compute device: {device}")

    data_dir = Path(args.data_dir)
    # Check if dataset exists; if not or if requested, generate synthetic data
    if args.synthetic or not any(data_dir.iterdir() if data_dir.exists() else []):
        print("Dataset directory is empty or --synthetic specified.")
        generate_synthetic_dataset(data_dir, config.DEFAULT_CLASSES, samples_per_class=args.samples_per_class)

    # Prepare DataLoaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=data_dir,
        class_names=config.DEFAULT_CLASSES,
        batch_size=args.batch_size
    )
    print(f"Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # Instantiate Model
    model = CNNLSTMModel(
        num_classes=len(classes),
        backbone=args.backbone,
        feature_dim=config.CNN_FEATURE_DIM,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_layers=config.LSTM_LAYERS,
        bidirectional=config.LSTM_BIDIRECTIONAL,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nStarting Training Loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpoint Best Model (or initial model at epoch 1)
        if val_acc > best_val_acc or epoch == 1:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            best_model_path = config.CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": classes,
                "backbone": args.backbone
            }, best_model_path)
            print(f"  -> Checkpointed model to {best_model_path.name} (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes. Best Val Accuracy: {best_val_acc*100:.2f}%")

    # Save class indices mapping
    class_map_path = config.CHECKPOINT_DIR / "class_indices.json"
    with open(class_map_path, "w") as f:
        json.dump({str(i): name for i, name in enumerate(classes)}, f, indent=4)
    print(f"Saved class index mapping to {class_map_path}")

    # Generate curves plot
    plot_path = config.OUTPUTS_DIR / "training_curves.png"
    plot_training_history(history, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Sign Language Recognition Model")
    parser.add_argument("--data_dir", type=str, default=str(config.RAW_VIDEOS_DIR), help="Directory of gesture videos")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--backbone", type=str, default=config.CNN_BACKBONE, choices=["custom", "mobilenet_v2"])
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data before training")
    parser.add_argument("--samples_per_class", type=int, default=6, help="Synthetic samples per class")
    args = parser.parse_args()

    run_training(args)
