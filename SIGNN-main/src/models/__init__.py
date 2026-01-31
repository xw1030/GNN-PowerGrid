"""
Neural network models and layers for SIGNN.
Author: Charlotte Cambier van Nooten
"""

# Import layers
from .layers import (
    MLPLayer,
    GCNLayer,
    GraphAttentionLayer,
    TransformerLayer,
    GINLayer,
    SIGNNLayer,
    ResidualBlock,
)

# Import complete models
from .models import (
    MLP,
    GCN,
    GAT,
    GraphTransformer,
    GIN,
    SIGNN,
    ResidualGNN,
)

# Import loss functions
from .losses import (
    FocalLoss,
    WeightedFocalLoss,
    ClassBalancedLoss,
)

__all__ = [
    # Layers
    "MLPLayer",
    "GCNLayer",
    "GraphAttentionLayer",
    "TransformerLayer",
    "GINLayer",
    "SIGNNLayer",
    "ResidualBlock",
    # Complete Models
    "MLP",
    "GCN",
    "GAT",
    "GraphTransformer",
    "GIN",
    "SIGNN",
    "ResidualGNN",
    # Loss functions
    "FocalLoss",
    "WeightedFocalLoss",
    "ClassBalancedLoss",
]
