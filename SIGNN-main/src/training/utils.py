"""
Training utilities for SIGNN
Author: Charlotte Cambier van Nooten
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from typing import Optional, Union, Dict, Any, List
import numpy as np
import logging
from datetime import datetime


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.

    This helps prevent overfitting by monitoring a validation metric and stopping
    training when it hasn't improved for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best_weights: bool = True,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics that should be maximized, 'min' for minimized
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.best_weights = None
        self.stopped_epoch = 0

    def step(self, metric: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop based on current metric.

        Args:
            metric: Current validation metric value
            model: Model to save weights from (optional)

        Returns:
            True if training should stop, False otherwise
        """
        if self._is_improvement(metric):
            self.best_metric = metric
            self.wait = 0

            # Save best weights
            if model is not None and self.restore_best_weights:
                self.best_weights = {
                    name: param.clone().detach()
                    for name, param in model.named_parameters()
                }
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            return True

        return False

    def _is_improvement(self, metric: float) -> bool:
        """Check if current metric is an improvement."""
        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        else:
            return metric < self.best_metric - self.min_delta

    def restore_best_weights_to_model(self, model: nn.Module):
        """Restore best weights to model."""
        if self.best_weights is not None:
            for name, param in model.named_parameters():
                if name in self.best_weights:
                    param.data.copy_(self.best_weights[name])


class ModelCheckpoint:
    """
    Model checkpointing utility for saving best models during training.

    Automatically saves model weights when validation metric improves,
    and can optionally save periodic checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        monitor: str = "val_acc",
        mode: str = "max",
        save_best_only: bool = True,
        filename: str = "best_model.pth",
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'max' for metrics to maximize, 'min' for minimize
            save_best_only: Whether to only save when metric improves
            filename: Filename for best model checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename = filename

        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.best_model_path = self.checkpoint_dir / filename

    def step(
        self,
        metric: float,
        model: nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save checkpoint if metric improved.

        Args:
            metric: Current metric value
            model: Model to save
            epoch: Current epoch number
            optimizer: Optimizer state to save (optional)
            additional_info: Additional information to save (optional)

        Returns:
            True if model was saved as new best, False otherwise
        """
        is_best = self._is_improvement(metric)

        if is_best or not self.save_best_only:
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metric": metric,
                "monitor": self.monitor,
            }

            if optimizer is not None:
                checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

            if additional_info is not None:
                checkpoint_data.update(additional_info)

            if is_best:
                self.best_metric = metric
                torch.save(checkpoint_data, self.best_model_path)

        return is_best

    def _is_improvement(self, metric: float) -> bool:
        """Check if current metric is an improvement."""
        if self.mode == "max":
            return metric > self.best_metric
        else:
            return metric < self.best_metric


class LearningRateScheduler:
    """
    Custom learning rate scheduler with multiple scheduling strategies.

    Provides various learning rate scheduling options beyond PyTorch's
    built-in schedulers, including warm-up, cosine annealing with restarts,
    and adaptive scheduling based on training progress.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = "cosine_with_warmup",
        **kwargs,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            schedule_type: Type of scheduling ('cosine_with_warmup', 'polynomial', 'exponential_decay')
            **kwargs: Additional arguments for specific schedulers
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

        # Store scheduler-specific parameters
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.total_steps = kwargs.get("total_steps", 1000)
        self.min_lr = kwargs.get("min_lr", 0.0)
        self.decay_rate = kwargs.get("decay_rate", 0.96)
        self.decay_steps = kwargs.get("decay_steps", 100)
        self.power = kwargs.get("power", 1.0)

    def step(self):
        """Update learning rate for current step."""
        self.current_step += 1

        if self.schedule_type == "cosine_with_warmup":
            lr = self._cosine_with_warmup()
        elif self.schedule_type == "polynomial":
            lr = self._polynomial_decay()
        elif self.schedule_type == "exponential_decay":
            lr = self._exponential_decay()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _cosine_with_warmup(self) -> float:
        """Cosine annealing with linear warmup."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            return self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )

    def _polynomial_decay(self) -> float:
        """Polynomial learning rate decay."""
        progress = min(self.current_step / self.total_steps, 1.0)
        return (self.initial_lr - self.min_lr) * (
            1 - progress
        ) ** self.power + self.min_lr

    def _exponential_decay(self) -> float:
        """Exponential learning rate decay."""
        return self.initial_lr * (
            self.decay_rate ** (self.current_step // self.decay_steps)
        )


class TrainingLogger:
    """
    Comprehensive training logger for tracking training progress and metrics.

    Logs training metrics, model information, and system statistics to both
    console and file outputs with configurable verbosity levels.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """
        Args:
            log_dir: Directory for log files (optional)
            experiment_name: Name for this experiment (optional)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.experiment_name = (
            experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Setup logger
        self.logger = logging.getLogger(f"SIGNN.{self.experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler if log directory provided
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                self.log_dir / f"{self.experiment_name}.log"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Metrics tracking
        self.metrics_history = {}
        self.start_time = None

    def log_experiment_start(self, config: Dict[str, Any]):
        """Log experiment configuration and start time."""
        self.start_time = time.time()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")

        # Save config to file if log directory exists
        if self.log_dir:
            config_path = self.log_dir / f"{self.experiment_name}_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Log to console/file
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")

    def log_model_info(self, model: nn.Module):
        """Log model architecture and parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def log_experiment_end(self, final_metrics: Dict[str, float]):
        """Log experiment completion and final results."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(f"Experiment completed in {duration:.1f} seconds")

        self.logger.info("Final metrics:")
        for key, value in final_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")

        # Save metrics history to file
        if self.log_dir:
            metrics_path = self.log_dir / f"{self.experiment_name}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

    def save_metrics_plot(self, save_path: Optional[Union[str, Path]] = None):
        """Save training metrics plot."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Training Metrics - {self.experiment_name}")

            # Loss plot
            if (
                "train_loss" in self.metrics_history
                and "val_loss" in self.metrics_history
            ):
                axes[0, 0].plot(self.metrics_history["train_loss"], label="Train Loss")
                axes[0, 0].plot(self.metrics_history["val_loss"], label="Val Loss")
                axes[0, 0].set_title("Loss")
                axes[0, 0].set_xlabel("Epoch")
                axes[0, 0].legend()
                axes[0, 0].grid(True)

            # Accuracy plot
            if (
                "train_acc" in self.metrics_history
                and "val_acc" in self.metrics_history
            ):
                axes[0, 1].plot(self.metrics_history["train_acc"], label="Train Acc")
                axes[0, 1].plot(self.metrics_history["val_acc"], label="Val Acc")
                axes[0, 1].set_title("Accuracy")
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].legend()
                axes[0, 1].grid(True)

            # Learning rate plot
            if "learning_rate" in self.metrics_history:
                axes[1, 0].plot(self.metrics_history["learning_rate"])
                axes[1, 0].set_title("Learning Rate")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_yscale("log")
                axes[1, 0].grid(True)

            # F1 score plot if available
            if "train_f1" in self.metrics_history and "val_f1" in self.metrics_history:
                axes[1, 1].plot(self.metrics_history["train_f1"], label="Train F1")
                axes[1, 1].plot(self.metrics_history["val_f1"], label="Val F1")
                axes[1, 1].set_title("F1 Score")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            else:
                axes[1, 1].axis("off")

            plt.tight_layout()

            if save_path is None and self.log_dir:
                save_path = self.log_dir / f"{self.experiment_name}_metrics.png"

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Saved metrics plot to {save_path}")

            plt.close()

        except ImportError:
            self.logger.warning("matplotlib not available, skipping metrics plot")


class GradientClipper:
    """
    Gradient clipping utility with multiple clipping strategies.

    Provides various gradient clipping methods to prevent exploding gradients
    during training of neural networks.
    """

    def __init__(self, clip_type: str = "norm", clip_value: float = 1.0):
        """
        Args:
            clip_type: Type of clipping ('norm', 'value', 'adaptive')
            clip_value: Clipping threshold
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.gradient_history = []

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters.

        Args:
            model: Model whose gradients to clip

        Returns:
            Gradient norm before clipping
        """
        if self.clip_type == "norm":
            return torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            return self._compute_gradient_norm(model)
        elif self.clip_type == "adaptive":
            return self._adaptive_clip(model)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")

    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def _adaptive_clip(self, model: nn.Module) -> float:
        """Adaptive gradient clipping based on gradient history."""
        current_norm = self._compute_gradient_norm(model)
        self.gradient_history.append(current_norm)

        # Keep only recent history
        if len(self.gradient_history) > 100:
            self.gradient_history = self.gradient_history[-100:]

        if len(self.gradient_history) > 10:
            # Adaptive threshold based on gradient statistics
            mean_norm = np.mean(self.gradient_history[-10:])
            std_norm = np.std(self.gradient_history[-10:])
            adaptive_threshold = mean_norm + 2 * std_norm

            if current_norm > adaptive_threshold:
                torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_threshold)

        return current_norm


def set_seed(seed: int = 42):
    """
    Set random seed for reproducible results.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def get_device_info() -> Dict[str, Any]:
    """
    Get device information for training.

    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()

    return info
