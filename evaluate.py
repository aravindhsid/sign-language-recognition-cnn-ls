"""
Model Evaluation and Performance Metrics for CNN-LSTM Sign Language Recognition.
Computes:
- Top-1 & Top-3 Accuracy
- Classification Report (Precision, Recall, F1-Score per class)
- Confusion Matrix Heatmap visualization
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
from dataset import get_dataloaders
from model import CNNLSTMModel


def plot_confusion_matrix(cm, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Sign Language Recognition - Confusion Matrix",
        ylabel="True Gesture Label",
        xlabel="Predicted Caption Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")


def evaluate_model(checkpoint_path: Path, data_dir: Path):
    device = torch.device(config.DEVICE)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    backbone = checkpoint.get("backbone", "custom")

    print(f"Loading checkpoint trained with {backbone} backbone across {len(classes)} classes.")

    model = CNNLSTMModel(
        num_classes=len(classes),
        backbone=backbone,
        feature_dim=config.CNN_FEATURE_DIM,
        lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
        lstm_layers=config.LSTM_LAYERS,
        bidirectional=config.LSTM_BIDIRECTIONAL,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        class_names=classes,
        batch_size=config.BATCH_SIZE
    )

    all_preds = []
    all_targets = []
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            logits = model(videos)
            probs = torch.softmax(logits, dim=1)

            # Top-1
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # Top-3
            if logits.size(1) >= 3:
                _, top3_preds = torch.topk(probs, k=3, dim=1)
                for i in range(labels.size(0)):
                    if labels[i] in top3_preds[i]:
                        top3_correct += 1
            else:
                top3_correct += (preds == labels).sum().item()

            total += labels.size(0)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    top1_acc = (all_preds == all_targets).mean()
    top3_acc = top3_correct / max(1, total)

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION SUMMARY REPORT          ")
    print("=" * 55)
    print(f"Total Test Samples: {total}")
    print(f"Top-1 Accuracy:     {top1_acc * 100:.2f}%")
    print(f"Top-3 Accuracy:     {top3_acc * 100:.2f}%\n")

    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0
    )
    print("Detailed Classification Report:")
    print(report)

    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))
    cm_path = config.OUTPUTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, classes, cm_path)

    # Save metrics JSON
    metrics_path = config.OUTPUTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "top1_accuracy": float(top1_acc),
            "top3_accuracy": float(top3_acc),
            "total_samples": int(total),
            "classes": classes
        }, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN-LSTM model")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pth"))
    parser.add_argument("--data_dir", type=str, default=str(config.RAW_VIDEOS_DIR))
    args = parser.parse_args()

    evaluate_model(Path(args.checkpoint), Path(args.data_dir))
