"""
Metrics module for SIGNN.
Author: Charlotte Cambier van Nooten
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import pandas as pd
from pathlib import Path


def calculate_classification_metrics(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    y_prob: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        average: Averaging strategy for multi-class ('binary', 'weighted', 'macro', 'micro')

    Returns:
        Dictionary containing various classification metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_prob is not None:
        y_prob = np.array(y_prob)

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    if average == "binary":
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
    else:
        metrics["precision"] = (
            precision.tolist() if hasattr(precision, "tolist") else precision
        )
        metrics["recall"] = recall.tolist() if hasattr(recall, "tolist") else recall
        metrics["f1"] = f1.tolist() if hasattr(f1, "tolist") else f1

    # Balanced accuracy (good for imbalanced datasets)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Confusion matrix components
    if len(np.unique(y_true)) == 2:  # Binary classification
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        # Derived metrics
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["ppv"] = (
            tp / (tp + fp) if (tp + fp) > 0 else 0.0
        )  # Positive predictive value
        metrics["npv"] = (
            tn / (tn + fn) if (tn + fn) > 0 else 0.0
        )  # Negative predictive value

        # Probability-based metrics
        if y_prob is not None:
            try:
                if y_prob.ndim == 1:
                    # Single probability values
                    prob_positive = y_prob
                elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    # Two-class probabilities
                    prob_positive = y_prob[:, 1]
                else:
                    prob_positive = y_prob

                metrics["roc_auc"] = roc_auc_score(y_true, prob_positive)
                metrics["avg_precision"] = average_precision_score(
                    y_true, prob_positive
                )

            except (ValueError, IndexError):
                # Handle cases where ROC AUC cannot be computed
                pass

    return metrics


def compute_confusion_matrix(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    normalize: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix with optional normalization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        labels: Class labels for display

    Returns:
        Tuple of (confusion_matrix, class_labels)
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Default labels if not provided
    if labels is None:
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            labels = ["not-n-1", "n-1"]
        else:
            labels = [f"Class {i}" for i in unique_labels]

    return cm, labels


def calculate_per_class_metrics(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each class individually.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class (optional)

    Returns:
        Dictionary with per-class metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Get unique classes
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    if class_names is None:
        if len(classes) == 2 and set(classes) == {0, 1}:
            class_names = ["not-n-1", "n-1"]
        else:
            class_names = [f"Class {i}" for i in classes]

    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class_metrics = {}
    for i, (cls, name) in enumerate(zip(classes, class_names)):
        per_class_metrics[name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }

    return per_class_metrics


def analyze_predictions_by_grid(
    dataset,
    sample_indices: List[int],
    predictions: List[int],
    true_labels: List[int],
) -> Dict[int, Dict[str, Any]]:
    """
    Analyze prediction performance by individual grids.

    Args:
        dataset: Dataset containing grid information
        sample_indices: List of sample indices
        predictions: Model predictions
        true_labels: True labels

    Returns:
        Dictionary with per-grid analysis
    """
    grid_results = defaultdict(
        lambda: {"correct": 0, "total": 0, "n1_correct": 0, "n1_total": 0}
    )

    pred_idx = 0
    for sample_idx in sample_indices:
        data = dataset[sample_idx]

        if data.edge_index.shape[1] == 0:
            continue

        num_edges = len(data.y)
        sample_preds = predictions[pred_idx : pred_idx + num_edges]
        sample_labels = true_labels[pred_idx : pred_idx + num_edges]

        grid_id = data.grid_id

        # Overall accuracy
        grid_results[grid_id]["correct"] += np.sum(
            np.array(sample_preds) == np.array(sample_labels)
        )
        grid_results[grid_id]["total"] += num_edges

        # n-1 class accuracy
        n1_mask = np.array(sample_labels) == 1
        if np.any(n1_mask):
            grid_results[grid_id]["n1_correct"] += np.sum(
                np.array(sample_preds)[n1_mask] == np.array(sample_labels)[n1_mask]
            )
            grid_results[grid_id]["n1_total"] += np.sum(n1_mask)

        pred_idx += num_edges

    # Calculate accuracies
    grid_accuracies = {}
    for grid_id, results in grid_results.items():
        overall_acc = (
            results["correct"] / results["total"] if results["total"] > 0 else 0
        )
        n1_acc = (
            results["n1_correct"] / results["n1_total"]
            if results["n1_total"] > 0
            else 0
        )
        grid_accuracies[grid_id] = {
            "overall_accuracy": overall_acc,
            "n1_accuracy": n1_acc,
            "total_edges": results["total"],
            "n1_edges": results["n1_total"],
        }

    return grid_accuracies


def calculate_imbalance_metrics(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Calculate metrics specifically for imbalanced datasets.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with imbalance-specific metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)

    for cls, count in zip(unique, counts):
        metrics[f"class_{cls}_proportion"] = count / total

    # Imbalance ratio
    if len(unique) == 2:
        majority_count = max(counts)
        minority_count = min(counts)
        metrics["imbalance_ratio"] = majority_count / minority_count

    # Balanced accuracy
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Geometric mean of sensitivity and specificity
    if len(unique) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["geometric_mean"] = np.sqrt(sensitivity * specificity)

    return metrics


def plot_training_history(
    train_losses: List[float],
    train_accs: List[float],
    val_losses: List[float],
    val_accs: List[float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training History",
) -> None:
    """
    Plot training history including losses and accuracies.

    Args:
        train_losses: Training losses per epoch
        train_accs: Training accuracies per epoch
        val_losses: Validation losses per epoch
        val_accs: Validation accuracies per epoch
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, "b-", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs, val_accs, "r-", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    # Get confusion matrix
    norm_mode = "true" if normalize else None
    cm, labels = compute_confusion_matrix(
        y_true, y_pred, normalize=norm_mode, labels=class_names
    )

    # Create plot
    plt.figure(figsize=figsize)

    # Choose format string based on normalization
    fmt = ".2f" if normalize else "d"

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_prob: Union[List, np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
) -> float:
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels
        y_prob: Prediction probabilities for positive class
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        AUC score
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")

    plt.show()

    return auc_score


def plot_precision_recall_curve(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_prob: Union[List, np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
) -> float:
    """
    Plot precision-recall curve for binary classification.

    Args:
        y_true: True binary labels
        y_prob: Prediction probabilities for positive class
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        Average precision score
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"Precision-Recall curve (AP = {avg_precision:.3f})",
    )

    # Baseline (random classifier)
    pos_ratio = np.mean(y_true)
    plt.axhline(
        y=pos_ratio,
        color="navy",
        linestyle="--",
        lw=2,
        label=f"Random classifier (AP = {pos_ratio:.3f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Precision-Recall curve saved to {save_path}")

    plt.show()

    return avg_precision


def generate_classification_report(
    y_true: Union[List, np.ndarray, torch.Tensor],
    y_pred: Union[List, np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate and optionally save a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
        save_path: Path to save the report

    Returns:
        Classification report as string
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Default class names
    if class_names is None:
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            class_names = ["not-n-1", "n-1"]
        else:
            class_names = [f"Class {i}" for i in unique_labels]

    # Generate report
    report = classification_report(y_true, y_pred, target_names=class_names)

    # Add additional metrics
    additional_metrics = calculate_classification_metrics(y_true, y_pred)

    report += f"\nAdditional Metrics:\n"
    report += f"Balanced Accuracy: {additional_metrics['balanced_accuracy']:.4f}\n"
    report += f"Matthews Correlation Coefficient: {additional_metrics['mcc']:.4f}\n"

    if len(np.unique(y_true)) == 2:
        report += f"Specificity: {additional_metrics['specificity']:.4f}\n"
        report += f"Sensitivity: {additional_metrics['sensitivity']:.4f}\n"

    # Save if path provided
    if save_path:
        with open(save_path, "w") as f:
            f.write("SIGNN Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        print(f"Classification report saved to {save_path}")

    return report


def calculate_confidence_statistics(
    probabilities: Union[List, np.ndarray, torch.Tensor],
    predictions: Union[List, np.ndarray, torch.Tensor],
    true_labels: Union[List, np.ndarray, torch.Tensor],
) -> Dict[str, Any]:
    """
    Calculate statistics about prediction confidence.

    Args:
        probabilities: Prediction probabilities [N, num_classes]
        predictions: Predicted class labels
        true_labels: True class labels

    Returns:
        Dictionary with confidence statistics
    """
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Maximum probability (confidence)
    if probabilities.ndim == 2:
        confidences = np.max(probabilities, axis=1)
    else:
        confidences = probabilities

    # Correctness
    correct = predictions == true_labels

    stats = {
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "mean_confidence_correct": float(np.mean(confidences[correct]))
        if np.any(correct)
        else 0.0,
        "mean_confidence_incorrect": float(np.mean(confidences[~correct]))
        if np.any(~correct)
        else 0.0,
        "high_confidence_correct": float(np.sum((confidences > 0.9) & correct)),
        "high_confidence_incorrect": float(np.sum((confidences > 0.9) & ~correct)),
        "low_confidence_total": float(np.sum(confidences < 0.6)),
    }

    # Confidence bins analysis
    bins = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bin_accuracies = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == len(bins) - 2:  # Last bin includes 1.0
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])

        if np.any(mask):
            bin_accuracy = np.mean(correct[mask])
            bin_count = np.sum(mask)
        else:
            bin_accuracy = 0.0
            bin_count = 0

        bin_accuracies.append(bin_accuracy)
        bin_counts.append(int(bin_count))

    stats["confidence_bins"] = {
        "bin_edges": bins.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }

    return stats
