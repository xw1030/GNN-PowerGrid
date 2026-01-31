"""
Evaluation module for SIGNN.
Author: Charlotte Cambier van Nooten
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
from pathlib import Path

from .metrics import calculate_classification_metrics


class ModelEvaluator:
    """
    Comprehensive model evaluator for power grid n-1 classification.

    Provides detailed evaluation including per-grid analysis, confusion matrices,
    ROC curves, and performance breakdowns by different criteria.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize model evaluator.

        Args:
            model: Trained neural network model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device

    def evaluate_samples(
        self, dataset, sample_indices: List[int], return_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on given samples.

        Args:
            dataset: Dataset containing the samples
            sample_indices: List of sample indices to evaluate
            return_predictions: Whether to return individual predictions

        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_grid_ids = []
        all_scenarios = []

        with torch.no_grad():
            for sample_idx in sample_indices:
                data = dataset[sample_idx]

                # Skip empty graphs
                if data.edge_index.shape[1] == 0:
                    continue

                # Move to device
                data.x = data.x.to(self.device)
                data.edge_index = data.edge_index.to(self.device)
                data.edge_attr = data.edge_attr.to(self.device)
                data.y = data.y.to(self.device)

                # Forward pass
                logits = self.model(data.x, data.edge_index, data.edge_attr)

                if logits.size(0) == 0:
                    continue

                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_grid_ids.extend([data.grid_id] * len(data.y))
                all_scenarios.extend([data.scenario] * len(data.y))

        # Calculate metrics
        metrics = calculate_classification_metrics(all_labels, all_predictions)

        # Add probability-based metrics
        if len(all_probabilities) > 0:
            probs_array = np.array(all_probabilities)
            if probs_array.shape[1] == 2:  # Binary classification
                positive_probs = probs_array[:, 1]
                metrics["roc_auc"] = roc_auc_score(all_labels, positive_probs)
                metrics["avg_precision"] = average_precision_score(
                    all_labels, positive_probs
                )

        # Prepare results
        results = {
            "metrics": metrics,
            "num_samples": len(all_labels),
            "num_grids": len(set(all_grid_ids)),
            "num_scenarios": len(set(all_scenarios)),
        }

        if return_predictions:
            results.update(
                {
                    "predictions": all_predictions,
                    "probabilities": all_probabilities,
                    "labels": all_labels,
                    "grid_ids": all_grid_ids,
                    "scenarios": all_scenarios,
                }
            )

        return results

    def evaluate_by_grid(
        self, dataset, sample_indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluate model performance per grid.

        Args:
            dataset: Dataset containing samples
            sample_indices: Sample indices to evaluate

        Returns:
            Dictionary with per-grid evaluation results
        """
        # Get full evaluation results
        results = self.evaluate_samples(
            dataset, sample_indices, return_predictions=True
        )

        if "predictions" not in results:
            return {}

        # Group by grid
        grid_results = defaultdict(
            lambda: {"predictions": [], "labels": [], "scenarios": []}
        )

        for pred, label, grid_id, scenario in zip(
            results["predictions"],
            results["labels"],
            results["grid_ids"],
            results["scenarios"],
        ):
            grid_results[grid_id]["predictions"].append(pred)
            grid_results[grid_id]["labels"].append(label)
            grid_results[grid_id]["scenarios"].append(scenario)

        # Calculate per-grid metrics
        per_grid_metrics = {}
        for grid_id, data in grid_results.items():
            if len(data["labels"]) > 0:
                metrics = calculate_classification_metrics(
                    data["labels"], data["predictions"]
                )
                metrics["num_edges"] = len(data["labels"])
                metrics["scenarios"] = list(set(data["scenarios"]))
                per_grid_metrics[grid_id] = metrics

        return per_grid_metrics

    def evaluate_by_scenario(
        self, dataset, sample_indices: List[int]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model performance per scenario.

        Args:
            dataset: Dataset containing samples
            sample_indices: Sample indices to evaluate

        Returns:
            Dictionary with per-scenario evaluation results
        """
        # Get full evaluation results
        results = self.evaluate_samples(
            dataset, sample_indices, return_predictions=True
        )

        if "predictions" not in results:
            return {}

        # Group by scenario
        scenario_results = defaultdict(
            lambda: {"predictions": [], "labels": [], "grid_ids": []}
        )

        for pred, label, grid_id, scenario in zip(
            results["predictions"],
            results["labels"],
            results["grid_ids"],
            results["scenarios"],
        ):
            scenario_results[scenario]["predictions"].append(pred)
            scenario_results[scenario]["labels"].append(label)
            scenario_results[scenario]["grid_ids"].append(grid_id)

        # Calculate per-scenario metrics
        per_scenario_metrics = {}
        for scenario, data in scenario_results.items():
            if len(data["labels"]) > 0:
                metrics = calculate_classification_metrics(
                    data["labels"], data["predictions"]
                )
                metrics["num_edges"] = len(data["labels"])
                metrics["grids"] = list(set(data["grid_ids"]))
                per_scenario_metrics[scenario] = metrics

        return per_scenario_metrics

    def comprehensive_evaluation(
        self, dataset, sample_indices: List[int], save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation with all analysis types.

        Args:
            dataset: Dataset to evaluate on
            sample_indices: Sample indices to evaluate
            save_dir: Directory to save plots and reports (optional)

        Returns:
            Comprehensive evaluation results
        """
        # Overall evaluation
        overall_results = self.evaluate_samples(dataset, sample_indices)

        # Per-grid evaluation
        per_grid_results = self.evaluate_by_grid(dataset, sample_indices)

        # Per-scenario evaluation
        per_scenario_results = self.evaluate_by_scenario(dataset, sample_indices)

        # Combine all results
        comprehensive_results = {
            "overall": overall_results,
            "per_grid": per_grid_results,
            "per_scenario": per_scenario_results,
            "summary": self._generate_summary(
                overall_results, per_grid_results, per_scenario_results
            ),
        }

        # Generate visualizations if save directory provided
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            self._create_evaluation_plots(
                overall_results, per_grid_results, per_scenario_results, save_dir
            )

            self._save_evaluation_report(comprehensive_results, save_dir)

        return comprehensive_results

    def _generate_summary(
        self,
        overall: Dict[str, Any],
        per_grid: Dict[int, Dict[str, Any]],
        per_scenario: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "overall_accuracy": overall["metrics"]["accuracy"],
            "overall_f1": overall["metrics"]["f1"],
            "num_grids_evaluated": len(per_grid),
            "num_scenarios_evaluated": len(per_scenario),
        }

        if per_grid:
            grid_accuracies = [metrics["accuracy"] for metrics in per_grid.values()]
            summary.update(
                {
                    "grid_accuracy_mean": np.mean(grid_accuracies),
                    "grid_accuracy_std": np.std(grid_accuracies),
                    "grid_accuracy_min": np.min(grid_accuracies),
                    "grid_accuracy_max": np.max(grid_accuracies),
                }
            )

        if per_scenario:
            scenario_accuracies = [
                metrics["accuracy"] for metrics in per_scenario.values()
            ]
            summary.update(
                {
                    "scenario_accuracy_mean": np.mean(scenario_accuracies),
                    "scenario_accuracy_std": np.std(scenario_accuracies),
                    "scenario_accuracy_min": np.min(scenario_accuracies),
                    "scenario_accuracy_max": np.max(scenario_accuracies),
                }
            )

        return summary

    def _create_evaluation_plots(
        self,
        overall: Dict[str, Any],
        per_grid: Dict[int, Dict[str, Any]],
        per_scenario: Dict[str, Dict[str, Any]],
        save_dir: Path,
    ):
        """Create evaluation visualization plots."""
        try:
            # Confusion matrix
            if "predictions" in overall and "labels" in overall:
                self._plot_confusion_matrix(
                    overall["labels"],
                    overall["predictions"],
                    save_dir / "confusion_matrix.png",
                )

            # Per-grid performance
            if per_grid:
                self._plot_per_grid_performance(
                    per_grid, save_dir / "per_grid_performance.png"
                )

            # Per-scenario performance
            if per_scenario:
                self._plot_per_scenario_performance(
                    per_scenario, save_dir / "per_scenario_performance.png"
                )

            # ROC curve if probabilities available
            if "probabilities" in overall and "labels" in overall:
                self._plot_roc_curve(
                    overall["labels"],
                    np.array(overall["probabilities"])[:, 1],
                    save_dir / "roc_curve.png",
                )

        except Exception as e:
            print(f"Warning: Could not create plots: {e}")

    def _plot_confusion_matrix(
        self, labels: List[int], predictions: List[int], save_path: Path
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["not-n-1", "n-1"],
            yticklabels=["not-n-1", "n-1"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_per_grid_performance(
        self, per_grid: Dict[int, Dict[str, Any]], save_path: Path
    ):
        """Plot per-grid performance."""
        grid_ids = list(per_grid.keys())
        accuracies = [per_grid[gid]["accuracy"] for gid in grid_ids]
        f1_scores = [per_grid[gid]["f1"] for gid in grid_ids]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.bar(range(len(grid_ids)), accuracies)
        ax1.set_title("Per-Grid Accuracy")
        ax1.set_xlabel("Grid Index")
        ax1.set_ylabel("Accuracy")
        ax1.set_xticks(range(0, len(grid_ids), max(1, len(grid_ids) // 10)))

        # F1 score plot
        ax2.bar(range(len(grid_ids)), f1_scores)
        ax2.set_title("Per-Grid F1 Score")
        ax2.set_xlabel("Grid Index")
        ax2.set_ylabel("F1 Score")
        ax2.set_xticks(range(0, len(grid_ids), max(1, len(grid_ids) // 10)))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_per_scenario_performance(
        self, per_scenario: Dict[str, Dict[str, Any]], save_path: Path
    ):
        """Plot per-scenario performance."""
        scenarios = list(per_scenario.keys())
        accuracies = [per_scenario[scenario]["accuracy"] for scenario in scenarios]
        f1_scores = [per_scenario[scenario]["f1"] for scenario in scenarios]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy plot
        ax1.bar(scenarios, accuracies)
        ax1.set_title("Per-Scenario Accuracy")
        ax1.set_xlabel("Scenario")
        ax1.set_ylabel("Accuracy")
        ax1.tick_params(axis="x", rotation=45)

        # F1 score plot
        ax2.bar(scenarios, f1_scores)
        ax2.set_title("Per-Scenario F1 Score")
        ax2.set_xlabel("Scenario")
        ax2.set_ylabel("F1 Score")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_roc_curve(
        self, labels: List[int], probabilities: np.ndarray, save_path: Path
    ):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_evaluation_report(self, results: Dict[str, Any], save_dir: Path):
        """Save comprehensive evaluation report."""
        report_path = save_dir / "evaluation_report.txt"

        with open(report_path, "w") as f:
            f.write("SIGNN Power Grid n-1 Classification Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # Overall metrics
            f.write("Overall Performance:\n")
            f.write("-" * 20 + "\n")
            overall_metrics = results["overall"]["metrics"]
            for metric, value in overall_metrics.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            f.write(f"Total samples: {results['overall']['num_samples']}\n")
            f.write(f"Total grids: {results['overall']['num_grids']}\n")
            f.write(f"Total scenarios: {results['overall']['num_scenarios']}\n\n")

            # Summary statistics
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            summary = results["summary"]
            for key, value in summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            f.write("\n")

            # Per-grid breakdown
            if results["per_grid"]:
                f.write("Per-Grid Performance (Top 10 and Bottom 10):\n")
                f.write("-" * 45 + "\n")

                # Sort grids by accuracy
                grid_items = list(results["per_grid"].items())
                grid_items.sort(key=lambda x: x[1]["accuracy"], reverse=True)

                # Top 10
                f.write("Top 10 Grids:\n")
                for grid_id, metrics in grid_items[:10]:
                    f.write(
                        f"Grid {grid_id}: Acc={metrics['accuracy']:.3f}, "
                        f"F1={metrics['f1']:.3f}, Edges={metrics['num_edges']}\n"
                    )

                # Bottom 10
                f.write("\nBottom 10 Grids:\n")
                for grid_id, metrics in grid_items[-10:]:
                    f.write(
                        f"Grid {grid_id}: Acc={metrics['accuracy']:.3f}, "
                        f"F1={metrics['f1']:.3f}, Edges={metrics['num_edges']}\n"
                    )
                f.write("\n")

            # Per-scenario breakdown
            if results["per_scenario"]:
                f.write("Per-Scenario Performance:\n")
                f.write("-" * 25 + "\n")
                for scenario, metrics in results["per_scenario"].items():
                    f.write(
                        f"{scenario}: Acc={metrics['accuracy']:.3f}, "
                        f"F1={metrics['f1']:.3f}, Edges={metrics['num_edges']}\n"
                    )


class MetricsCalculator:
    """
    Utility class for calculating various performance metrics.

    Provides static methods for computing different evaluation metrics
    commonly used in classification tasks.
    """

    @staticmethod
    def calculate_binary_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive binary classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        # Additional derived metrics
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
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                metrics["avg_precision"] = average_precision_score(y_true, y_prob)
            except ValueError:
                # Handle cases where ROC AUC cannot be computed
                pass

        return metrics

    @staticmethod
    def calculate_class_imbalance_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics specifically for imbalanced datasets.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of imbalance-specific metrics
        """
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)

        metrics = {}
        for cls, count in zip(unique, counts):
            metrics[f"class_{cls}_proportion"] = count / total

        # Balanced accuracy (accounts for class imbalance)
        from sklearn.metrics import balanced_accuracy_score

        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

        # Matthews Correlation Coefficient (good for imbalanced data)
        from sklearn.metrics import matthews_corrcoef

        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

        return metrics


class PerformanceAnalyzer:
    """
    Advanced performance analysis for power grid models.

    Provides in-depth analysis of model performance including
    failure case analysis, robustness evaluation, and
    performance trends across different conditions.
    """

    def __init__(self, evaluator: ModelEvaluator):
        """
        Initialize performance analyzer.

        Args:
            evaluator: Model evaluator instance
        """
        self.evaluator = evaluator

    def analyze_failure_cases(
        self, dataset, sample_indices: List[int], threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyze cases where model predictions fail.

        Args:
            dataset: Dataset to analyze
            sample_indices: Sample indices to analyze
            threshold: Confidence threshold for failure analysis

        Returns:
            Analysis of failure cases
        """
        results = self.evaluator.evaluate_samples(
            dataset, sample_indices, return_predictions=True
        )

        if "predictions" not in results:
            return {}

        # Identify failure cases
        failures = []
        for i, (pred, label, prob, grid_id, scenario) in enumerate(
            zip(
                results["predictions"],
                results["labels"],
                results["probabilities"],
                results["grid_ids"],
                results["scenarios"],
            )
        ):
            if pred != label:
                confidence = max(prob)
                failures.append(
                    {
                        "index": i,
                        "predicted": pred,
                        "actual": label,
                        "confidence": confidence,
                        "grid_id": grid_id,
                        "scenario": scenario,
                        "high_confidence": confidence > threshold,
                    }
                )

        # Analyze failure patterns
        failure_analysis = {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(results["labels"]),
            "high_confidence_failures": sum(
                1 for f in failures if f["high_confidence"]
            ),
            "failures_by_grid": defaultdict(int),
            "failures_by_scenario": defaultdict(int),
        }

        for failure in failures:
            failure_analysis["failures_by_grid"][failure["grid_id"]] += 1
            failure_analysis["failures_by_scenario"][failure["scenario"]] += 1

        return {"analysis": failure_analysis, "failure_cases": failures}

    def analyze_performance_vs_graph_size(
        self, dataset, sample_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze how performance varies with graph size.

        Args:
            dataset: Dataset to analyze
            sample_indices: Sample indices to analyze

        Returns:
            Performance vs graph size analysis
        """
        # Get per-grid results
        per_grid_results = self.evaluator.evaluate_by_grid(dataset, sample_indices)

        # Get graph sizes
        size_performance = []
        for grid_id, metrics in per_grid_results.items():
            # Find a sample from this grid to get size
            for sample_idx in sample_indices:
                data = dataset[sample_idx]
                if data.grid_id == grid_id:
                    size_performance.append(
                        {
                            "grid_id": grid_id,
                            "num_nodes": data.num_nodes,
                            "num_edges": data.edge_index.shape[1] // 2,  # Undirected
                            "accuracy": metrics["accuracy"],
                            "f1": metrics["f1"],
                        }
                    )
                    break

        # Analyze correlations
        if size_performance:
            df = pd.DataFrame(size_performance)

            # Compute correlations
            node_acc_corr = df["num_nodes"].corr(df["accuracy"])
            edge_acc_corr = df["num_edges"].corr(df["accuracy"])
            node_f1_corr = df["num_nodes"].corr(df["f1"])
            edge_f1_corr = df["num_edges"].corr(df["f1"])

            return {
                "correlations": {
                    "nodes_vs_accuracy": node_acc_corr,
                    "edges_vs_accuracy": edge_acc_corr,
                    "nodes_vs_f1": node_f1_corr,
                    "edges_vs_f1": edge_f1_corr,
                },
                "size_data": df.to_dict("records"),
            }

        return {}
