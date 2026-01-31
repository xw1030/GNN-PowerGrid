"""
Training utilities and modules for SIGNN.
Author: Charlotte Cambier van Nooten
"""

from .trainer import (
    MultiGridTrainer,
    Trainer,
    TrainingConfig,
)

from .evaluation import (
    ModelEvaluator,
    MetricsCalculator,
    PerformanceAnalyzer,
)

from .utils import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TrainingLogger,
)

from .metrics import (
    calculate_classification_metrics,
    compute_confusion_matrix,
    analyze_predictions_by_grid,
    plot_training_history,
)

__all__ = [
    # Training classes
    "MultiGridTrainer",
    "Trainer",
    "TrainingConfig",
    # Evaluation
    "ModelEvaluator",
    "MetricsCalculator",
    "PerformanceAnalyzer",
    # Utilities
    "EarlyStopping",
    "LearningRateScheduler",
    "ModelCheckpoint",
    "TrainingLogger",
    # Metrics functions
    "calculate_classification_metrics",
    "compute_confusion_matrix",
    "analyze_predictions_by_grid",
    "plot_training_history",
]
