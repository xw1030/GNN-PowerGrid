"""
Training module for SIGNN
Author: Charlotte Cambier van Nooten
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from ..models.losses import create_loss_function
from ..data.subgraph import ego_nets_plus_subgraphs, pad_center_marker
from .evaluation import ModelEvaluator
from .utils import EarlyStopping, ModelCheckpoint, TrainingLogger
from .metrics import calculate_classification_metrics


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    # Model and data
    model_type: str = "multi_grid_classifier"
    batch_size: int = 32

    # Training hyperparameters
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-5

    # Loss function
    loss_type: str = (
        "weighted_focal"  # 'cross_entropy', 'focal', 'weighted_focal', 'class_balanced'
    )
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25

    # Optimization
    optimizer_type: str = "adam"  # 'adam', 'adamw', 'sgd'
    scheduler_type: str = "reduce_on_plateau"  # 'reduce_on_plateau', 'step', 'cosine'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Regularization
    gradient_clip_val: Optional[float] = 1.0
    dropout: float = 0.3

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Checkpointing
    save_best_model: bool = True
    save_checkpoint_every: int = 10
    checkpoint_dir: str = "output/checkpoints"

    # Logging
    log_interval: int = 10
    validate_every: int = 1

    # Device
    device: str = "auto"  # 'auto', 'cuda', 'cpu'

    # Data splits (by grid)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Class imbalance handling
    use_class_weights: bool = True
    oversample_minority: bool = False

    # Subgraph (ESAN-style ego_nets_plus)
    use_subgraph: bool = False
    subgraph_num_hops: int = 2
    subgraph_max_centers_per_sample: Optional[int] = None  # None = all nodes; set to limit
    subgraph_chunk_size: int = 500  # subgraphs per backward (gradient accumulation), larger = more GPU use

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.learning_rate < 1
        assert self.num_epochs > 0
        assert 0 <= self.dropout < 1
        assert self.train_ratio + self.val_ratio + self.test_ratio <= 1.0

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Base trainer class for neural network models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model to train
            config: Training configuration
            logger: Optional logger for training progress
        """
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_metric = float("-inf")
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_utilities()

    def _setup_optimizer(self):
        """Setup optimizer based on configuration."""
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
            )
        elif self.config.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs
            )
        else:
            self.scheduler = None

    def _setup_utilities(self):
        """Setup training utilities."""
        # Early stopping
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                mode="max",
            )
        else:
            self.early_stopping = None

        # Model checkpointing
        if self.config.save_best_model:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            self.checkpoint = ModelCheckpoint(
                checkpoint_dir=checkpoint_dir, monitor="val_acc", mode="max"
            )
        else:
            self.checkpoint = None

        # Training logger
        self.training_logger = TrainingLogger()

    def train_epoch(self, dataloader, criterion) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            criterion: Loss function

        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._batch_to_device(batch)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.edge_attr)

            if logits.size(0) == 0:  # Skip empty batches
                continue

            loss = criterion(logits, batch.y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)

            # Log batch progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.debug(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, avg_acc

    def validate_epoch(self, dataloader, criterion) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            dataloader: Validation data loader
            criterion: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._batch_to_device(batch)

                logits = self.model(batch.x, batch.edge_index, batch.edge_attr)

                if logits.size(0) == 0:  # Skip empty batches
                    continue

                loss = criterion(logits, batch.y)
                total_loss += loss.item()

                # Collect predictions
                pred = logits.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())

        # Calculate metrics
        metrics = calculate_classification_metrics(all_targets, all_predictions)
        metrics["loss"] = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0

        return metrics

    def _batch_to_device(self, batch):
        """Move batch data to device."""
        batch.x = batch.x.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        batch.edge_attr = batch.edge_attr.to(self.device)
        batch.y = batch.y.to(self.device)
        return batch

    def fit(
        self,
        train_loader,
        val_loader,
        criterion=None,
        class_counts: Optional[torch.Tensor] = None,
    ):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (created automatically if None)
            class_counts: Class sample counts for loss weighting
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # Setup loss function
        if criterion is None:
            criterion = create_loss_function(
                self.config.loss_type,
                class_counts=class_counts,
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
            )

        criterion = criterion.to(self.device)

        start_time = time.time()

        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Training
                train_loss, train_acc = self.train_epoch(train_loader, criterion)

                # Validation
                if epoch % self.config.validate_every == 0:
                    val_metrics = self.validate_epoch(val_loader, criterion)
                    val_loss = val_metrics["loss"]
                    val_acc = val_metrics["accuracy"]
                else:
                    val_loss = val_acc = 0.0
                    val_metrics = {}

                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_acc)
                    else:
                        self.scheduler.step()

                # Record history
                self.training_history["train_loss"].append(train_loss)
                self.training_history["train_acc"].append(train_acc)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_acc"].append(val_acc)
                self.training_history["learning_rate"].append(
                    self.optimizer.param_groups[0]["lr"]
                )

                # Log progress
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch + 1:3d}/{self.config.num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # Save checkpoint
                if self.checkpoint is not None and val_acc > 0:
                    is_best = self.checkpoint.step(val_acc, self.model, epoch)
                    if is_best:
                        self.best_metric = val_acc

                # Early stopping
                if self.early_stopping is not None and val_acc > 0:
                    if self.early_stopping.step(val_acc):
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                # Save periodic checkpoint
                if epoch % self.config.save_checkpoint_every == 0:
                    self._save_checkpoint(epoch)

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")

        # Load best model if available
        if self.checkpoint is not None and self.checkpoint.best_model_path.exists():
            checkpoint = torch.load(self.checkpoint.best_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Handle case where checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)
            self.logger.info("Loaded best model weights")

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "config": self.config,
        }

        checkpoint_path = (
            Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.training_history = checkpoint["training_history"]

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


class MultiGridTrainer(Trainer):
    """
    Specialized trainer for multi-grid power system datasets.

    This trainer handles the specific requirements of training on
    multiple power grids with proper data splitting and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize multi-grid trainer.

        Args:
            model: Neural network model
            dataset: Multi-grid dataset
            config: Training configuration
            logger: Optional logger
        """
        super().__init__(model, config, logger)

        self.dataset = dataset
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        self._prepare_data_splits()
        self.evaluator = ModelEvaluator(model, self.device)

    def _prepare_data_splits(self):
        """Prepare train/validation/test splits by grid."""
        # Split by grids to prevent data leakage
        grid_indices = list(range(len(self.dataset.grid_ids)))
        np.random.shuffle(grid_indices)

        n_grids = len(grid_indices)
        n_train_grids = int(self.config.train_ratio * n_grids)
        n_val_grids = int(self.config.val_ratio * n_grids)

        train_grid_indices = grid_indices[:n_train_grids]
        val_grid_indices = grid_indices[n_train_grids : n_train_grids + n_val_grids]
        test_grid_indices = grid_indices[n_train_grids + n_val_grids :]

        # Convert to sample indices
        for i, (grid_id, scenario) in enumerate(self.dataset.samples):
            grid_idx = self.dataset.grid_ids.index(grid_id)

            if grid_idx in train_grid_indices:
                self.train_samples.append(i)
            elif grid_idx in val_grid_indices:
                self.val_samples.append(i)
            else:
                self.test_samples.append(i)

        self.logger.info(
            f"Data split: {len(self.train_samples)} train, "
            f"{len(self.val_samples)} val, {len(self.test_samples)} test samples"
        )
        self.logger.info(
            f"Grid split: {len(train_grid_indices)} train, "
            f"{len(val_grid_indices)} val, {len(test_grid_indices)} test grids"
        )

    def train_epoch_multi_grid(self, criterion) -> Tuple[float, float]:
        """Train epoch with individual graph processing."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Shuffle training samples
        np.random.shuffle(self.train_samples)

        for sample_idx in self.train_samples:
            data = self.dataset[sample_idx]

            # Skip empty graphs
            if data.edge_index.shape[1] == 0:
                continue

            # Move to device
            data.x = data.x.to(self.device)
            data.edge_index = data.edge_index.to(self.device)
            data.edge_attr = data.edge_attr.to(self.device)
            data.y = data.y.to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_subgraph:
                subgraphs = ego_nets_plus_subgraphs(
                    data,
                    num_hops=self.config.subgraph_num_hops,
                    max_centers=self.config.subgraph_max_centers_per_sample,
                )
                if not subgraphs:
                    continue
                self.optimizer.zero_grad()
                chunk_size = max(1, self.config.subgraph_chunk_size)
                N = data.num_nodes
                for chunk_start in range(0, len(subgraphs), chunk_size):
                    chunk = subgraphs[chunk_start : chunk_start + chunk_size]
                    # 批成一张图：同结构子图拼成一大图，一次前向提高 GPU 利用率
                    xs, eis, eas, ys = [], [], [], []
                    for x_plus, ei, ea, y_sub in chunk:
                        if y_sub.size(0) == 0:
                            continue
                        xs.append(x_plus)
                        eis.append(ei)
                        eas.append(ea)
                        ys.append(y_sub)
                    if not xs:
                        continue
                    # 大图：节点 [0..N-1], [N..2N-1], ... ；边索引按子图偏移
                    batch_x = torch.cat(xs, dim=0).to(self.device)
                    batch_ei = torch.cat([eis[i] + i * N for i in range(len(xs))], dim=1).to(self.device)
                    batch_ea = torch.cat(eas, dim=0).to(self.device)
                    batch_y = torch.cat(ys, dim=0).to(self.device)
                    logits = self.model(batch_x, batch_ei, batch_ea)
                    chunk_loss = criterion(logits, batch_y)
                    total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                    total_samples += batch_y.size(0)
                    total_loss += chunk_loss.item()
                    chunk_loss.backward()
                if self.config.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()
            else:
                x_in = pad_center_marker(data.x, device=self.device)
                logits = self.model(x_in, data.edge_index, data.edge_attr)
                loss = criterion(logits, data.y)
                loss.backward()
                if self.config.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.optimizer.step()
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                total_correct += (pred == data.y).sum().item()
                total_samples += data.y.size(0)

        n_batches = max(1, len(self.train_samples))
        avg_loss = total_loss / n_batches
        avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, avg_acc

    def validate_epoch_multi_grid(self, samples, criterion) -> Dict[str, float]:
        """Validate epoch with individual graph processing."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sample_idx in samples:
                data = self.dataset[sample_idx]

                if data.edge_index.shape[1] == 0:
                    continue

                data.x = data.x.to(self.device)
                data.edge_index = data.edge_index.to(self.device)
                data.edge_attr = data.edge_attr.to(self.device)
                data.y = data.y.to(self.device)

                x_in = pad_center_marker(data.x, device=self.device)
                logits = self.model(x_in, data.edge_index, data.edge_attr)
                loss = criterion(logits, data.y)

                total_loss += loss.item()

                pred = logits.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(data.y.cpu().numpy())

        # Calculate metrics
        metrics = calculate_classification_metrics(all_targets, all_predictions)
        metrics["loss"] = total_loss / len(samples) if len(samples) > 0 else 0.0

        return metrics

    def fit_multi_grid(self):
        """
        Training loop for multi-grid dataset.
        """
        # Calculate class weights from training data
        class_counts = self._calculate_class_counts()

        # Create loss function
        criterion = create_loss_function(
            self.config.loss_type,
            class_counts=class_counts,
            gamma=self.config.focal_gamma,
            alpha=self.config.focal_alpha,
        ).to(self.device)

        self.logger.info("Starting multi-grid training...")
        self.logger.info(f"Class distribution: {class_counts}")

        start_time = time.time()

        try:
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Training
                train_loss, train_acc = self.train_epoch_multi_grid(criterion)

                # Validation
                val_metrics = self.validate_epoch_multi_grid(
                    self.val_samples, criterion
                )
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_acc)
                    else:
                        self.scheduler.step()

                # Record history
                self.training_history["train_loss"].append(train_loss)
                self.training_history["train_acc"].append(train_acc)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_acc"].append(val_acc)
                self.training_history["learning_rate"].append(
                    self.optimizer.param_groups[0]["lr"]
                )

                # Log progress
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch {epoch + 1:3d}/{self.config.num_epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # Checkpointing and early stopping
                if self.checkpoint is not None:
                    is_best = self.checkpoint.step(val_acc, self.model, epoch)
                    if is_best:
                        self.best_metric = val_acc

                if self.early_stopping is not None:
                    if self.early_stopping.step(val_acc):
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        total_time = time.time() - start_time
        self.logger.info(f"Multi-grid training completed in {total_time:.1f}s")

    def _calculate_class_counts(self) -> torch.Tensor:
        """Calculate class counts from training samples."""
        class_counts = torch.zeros(2)

        for sample_idx in self.train_samples:
            data = self.dataset[sample_idx]
            if len(data.y) > 0:
                unique, counts = torch.unique(data.y, return_counts=True)
                for cls, count in zip(unique, counts):
                    class_counts[cls] += count

        return class_counts

    def evaluate_test_set(self) -> Dict[str, Any]:
        """Evaluate model on test set."""
        if len(self.test_samples) == 0:
            return {}

        self.logger.info("Evaluating on test set...")

        # Load best model if available
        if self.checkpoint and self.checkpoint.best_model_path.exists():
            checkpoint = torch.load(self.checkpoint.best_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Handle case where checkpoint is just the state dict
                self.model.load_state_dict(checkpoint)

        # Use dummy criterion for evaluation
        criterion = nn.CrossEntropyLoss()

        # Get test metrics
        test_metrics = self.validate_epoch_multi_grid(self.test_samples, criterion)

        # Additional analysis
        test_results = self.evaluator.comprehensive_evaluation(
            self.dataset, self.test_samples
        )

        # Combine results
        final_results = {
            **test_metrics,
            **test_results,
            "num_test_samples": len(self.test_samples),
            "num_test_grids": len(
                set(self.dataset[i].grid_id for i in self.test_samples)
            ),
        }

        self.logger.info(f"Test Results:")
        self.logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
        self.logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        # INSERT_YOUR_CODE
        # Calculate Matthews Correlation Coefficient (MCC) for the test set
        from sklearn.metrics import matthews_corrcoef

        # Aggregate predictions and targets for MCC
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for sample_idx in self.test_samples:
                data = self.dataset[sample_idx]
                if data.edge_index.shape[1] == 0:
                    continue
                data.x = pad_center_marker(data.x, device=self.device)
                preds = self.model(data.x.to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device))
                pred_labels = preds.argmax(dim=1).cpu().numpy()
                all_predictions.extend(pred_labels)
                all_targets.extend(data.y.cpu().numpy())

        if len(all_predictions) > 0:
            mcc = matthews_corrcoef(all_targets, all_predictions)
            final_results["mcc"] = mcc
            self.logger.info(f"  MCC: {mcc:.4f}")
        else:
            final_results["mcc"] = None
            self.logger.info("  MCC: N/A (no predictions available)")
        return final_results
