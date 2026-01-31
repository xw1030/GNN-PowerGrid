"""
Loss functions
Author: Charlotte Cambier van Nooten
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.

    The Focal Loss is designed to down-weight easy examples and focus
    on hard negatives. This is particularly useful for the highly imbalanced
    n-1 classification task where ~98.8% of samples are n-1.

    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor for rare class (typically between 0.25 and 1.0)
            gamma: Focusing parameter (typically 2.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions tensor with shape [N, C] where C = number of classes
            targets: Ground truth labels with shape [N]

        Returns:
            Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class-specific weighting.

    Combines the benefits of weighted cross-entropy and focal loss
    for extreme class imbalance scenarios.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Class weights tensor with shape [C] or None for automatic weighting
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha: Optional[Tensor] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute weighted focal loss.

        Args:
            inputs: Predictions tensor [N, C]
            targets: Ground truth labels [N]

        Returns:
            Computed weighted focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    This loss re-weights samples based on the effective number of samples
    per class, which is particularly useful when the class imbalance is extreme.

    Reference: Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019).
    Class-balanced loss based on effective number of samples. CVPR, 2019.
    """

    def __init__(
        self,
        samples_per_class: Tensor,
        beta: float = 0.9999,
        loss_type: str = "focal",
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            samples_per_class: Number of samples per class [C]
            beta: Hyperparameter controlling the re-weighting strength
            loss_type: Type of base loss ('focal', 'cross_entropy')
            gamma: Focusing parameter for focal loss
            reduction: Reduction method
        """
        super().__init__()

        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # Normalize

        self.register_buffer("weights", weights)
        self.loss_type: str = loss_type
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute class-balanced loss.

        Args:
            inputs: Predictions tensor [N, C]
            targets: Ground truth labels [N]

        Returns:
            Computed class-balanced loss
        """
        if self.loss_type == "focal":
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss

            # Apply class weights
            weights_t = self.weights[targets]  # type: ignore
            loss = weights_t * focal_loss
        else:  # cross_entropy
            loss = F.cross_entropy(
                inputs,
                targets,
                weight=self.weights,
                reduction="none",  # type: ignore
            )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.

    This loss encourages larger margins for minority classes by incorporating
    class frequency information into the margin calculation.

    Reference: Cao, K., Wei, C., Gaidon, A., Arechiga, N., & Ma, T. (2019).
    Learning imbalanced datasets with label-distribution-aware margin loss. NeurIPS, 2019.
    """

    def __init__(
        self,
        cls_num_list: list,
        max_m: float = 0.5,
        s: float = 30.0,
        reduction: str = "mean",
    ):
        """
        Args:
            cls_num_list: List of number of samples for each class
            max_m: Maximum margin
            s: Scaling factor
            reduction: Reduction method
        """
        super().__init__()

        # Calculate per-class margins
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(cls_num_list)))
        m_list = m_list * (max_m / torch.max(m_list))

        self.register_buffer("m_list", m_list)
        self.s = s
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute LDAM loss.

        Args:
            inputs: Predictions tensor [N, C]
            targets: Ground truth labels [N]

        Returns:
            Computed LDAM loss
        """
        batch_size = inputs.size(0)

        # Apply margins
        batch_m = self.m_list[targets]
        batch_m = batch_m.view(batch_size, 1)

        inputs_m = inputs.clone()
        inputs_m.scatter_(
            1, targets.view(-1, 1), inputs.gather(1, targets.view(-1, 1)) - batch_m
        )

        # Scale and compute cross entropy
        inputs_m = inputs_m * self.s
        loss = F.cross_entropy(inputs_m, targets, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary classification.

    The Dice loss is based on the Dice coefficient (F1 score) and is
    particularly useful for imbalanced datasets as it focuses on the
    overlap between predicted and true positive regions.
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method
        """
        super().__init__()
        self.smooth: float = smooth
        self.reduction: str = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute Dice loss.

        Args:
            inputs: Predictions tensor [N, C]
            targets: Ground truth labels [N]

        Returns:
            Computed Dice loss
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Calculate Dice coefficient for each class
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that mixes multiple loss types.

    This allows combining the benefits of different loss functions,
    such as cross-entropy for overall accuracy and focal loss for
    handling class imbalance.
    """

    def __init__(self, losses: dict, weights: Optional[dict] = None):
        """
        Args:
            losses: Dictionary of loss functions {name: loss_fn}
            weights: Dictionary of loss weights {name: weight}
        """
        super().__init__()

        self.losses: nn.ModuleDict = nn.ModuleDict(losses)
        self.weights: dict = weights or {name: 1.0 for name in losses}

    def forward(self, inputs: Tensor, targets: Tensor) -> dict:
        """
        Compute combined loss.

        Args:
            inputs: Predictions tensor [N, C]
            targets: Ground truth labels [N]

        Returns:
            Dictionary containing individual losses and total loss
        """
        loss_dict = {}
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(inputs, targets)
            weight = self.weights[name]

            loss_dict[name] = loss_value
            total_loss += weight * loss_value

        loss_dict["total"] = total_loss
        return loss_dict


def create_loss_function(
    loss_type: str, class_counts: Optional[Tensor] = None, **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions based on type and class statistics.

    Args:
        loss_type: Type of loss function ('cross_entropy', 'focal', 'weighted_focal',
                  'class_balanced', 'ldam', 'dice', 'combined')
        class_counts: Number of samples per class for computing weights
        **kwargs: Additional arguments for specific loss functions

    Returns:
        Configured loss function
    """
    if loss_type == "cross_entropy":
        if class_counts is not None:
            weights = 1.0 / class_counts.float()
            weights = weights / weights.sum() * len(weights)
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        return FocalLoss(**kwargs)

    elif loss_type == "weighted_focal":
        if class_counts is not None:
            weights = 1.0 / class_counts.float()
            weights = weights / weights.sum() * len(weights)
            # Remove alpha from kwargs to avoid conflict
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != "alpha"}
            return WeightedFocalLoss(alpha=weights, **filtered_kwargs)
        else:
            return WeightedFocalLoss(**kwargs)

    elif loss_type == "class_balanced":
        if class_counts is None:
            raise ValueError("class_counts required for ClassBalancedLoss")
        return ClassBalancedLoss(samples_per_class=class_counts, **kwargs)

    elif loss_type == "ldam":
        if class_counts is None:
            raise ValueError("class_counts required for LDAMLoss")
        return LDAMLoss(cls_num_list=class_counts.tolist(), **kwargs)

    elif loss_type == "dice":
        return DiceLoss(**kwargs)

    elif loss_type == "combined":
        # Example combined loss with CE and Focal
        losses = {"cross_entropy": nn.CrossEntropyLoss(), "focal": FocalLoss(gamma=2.0)}
        weights = {"cross_entropy": 0.5, "focal": 0.5}
        return CombinedLoss(losses, weights)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
