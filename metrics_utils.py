"""
Helper functions for printing & saving metrics.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from config import BASE_DIR


def print_classification_report(y_true, y_pred, target_names=None, title: str = ""):
    if title:
        print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(
    y_true,
    y_pred,
    target_names,
    title: str,
    save_path: Optional[Path] = None,
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    if save_path is None:
        save_path = BASE_DIR / f"{title.replace(' ', '_')}_cm.png"

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[metrics_utils] Saved confusion matrix to {save_path}")
