"""
Complete graph neural network models for edge classification.
Author: Charlotte Cambier van Nooten
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

from .layers import (
    MLPLayer,
    GCNLayer,
    GraphAttentionLayer,
    TransformerLayer,
    GINLayer,
    SIGNNLayer,
    ResidualBlock,
)


class MLP(nn.Module):
    """Multi-Layer Perceptron for edge classification."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            node_features: Number of node feature dimensions
            edge_features: Number of edge feature dimensions
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features

        # Input dimension: 2 * node_features (source + target) + edge_features
        input_dim = 2 * node_features + edge_features

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(
                MLPLayer(
                    input_dim=prev_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation="relu",
                )
            )
            prev_dim = hidden_dim

        # Output layer for binary classification
        layers.append(
            MLPLayer(
                input_dim=prev_dim,
                output_dim=2,
                dropout=0.0,
                use_batch_norm=False,
                activation="none",
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for edge classification.

        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges] - source and target nodes for each edge
            edge_attr: [num_edges, edge_feature_dim]

        Returns:
            logits: [num_edges, 2] - binary classification logits
        """
        # Get source and target node features for each edge
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        source_features = node_features[source_nodes]
        target_features = node_features[target_nodes]

        # Concatenate source, target, and edge features
        edge_representations = torch.cat(
            [source_features, target_features, edge_attr], dim=1
        )

        # Predict edge labels
        logits = self.network(edge_representations)
        return logits


class GCN(nn.Module):
    """Graph Convolutional Network for edge classification."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
        classifier_hidden_dims: List[int] = [128, 64],
    ):
        """
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension for graph convolutions
            num_layers: Number of graph convolution layers
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            classifier_hidden_dims: Hidden dimensions for edge classifier
        """
        super().__init__()

        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # Graph convolution layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim if i > 0 else hidden_dim
            self.gcn_layers.append(
                GCNLayer(
                    input_dim=layer_input_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )

        # Edge classifier
        classifier_layers = []
        classifier_input_dim = 2 * hidden_dim + edge_features
        prev_dim = classifier_input_dim

        for hidden_dim_cls in classifier_hidden_dims:
            classifier_layers.append(
                MLPLayer(
                    input_dim=prev_dim,
                    output_dim=hidden_dim_cls,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation="relu",
                )
            )
            prev_dim = hidden_dim_cls

        classifier_layers.append(
            MLPLayer(
                input_dim=prev_dim,
                output_dim=2,
                dropout=0.0,
                use_batch_norm=False,
                activation="none",
            )
        )

        self.edge_classifier = nn.Sequential(*classifier_layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with graph convolutions followed by edge classification.

        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feature_dim]

        Returns:
            logits: [num_edges, 2] - binary classification logits
        """
        # Encode initial node features
        x = F.relu(self.node_encoder(node_features))

        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)

        # Edge classification
        if edge_index.shape[1] > 0:
            source_features = x[edge_index[0]]
            target_features = x[edge_index[1]]
            edge_input = torch.cat([source_features, target_features, edge_attr], dim=1)
            logits = self.edge_classifier(edge_input)
        else:
            logits = torch.zeros(0, 2, device=node_features.device)

        return logits


class GAT(nn.Module):
    """Graph Attention Network for edge classification."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        classifier_hidden_dims: List[int] = [128, 64],
    ):
        """
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension for attention layers
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout probability
            classifier_hidden_dims: Hidden dimensions for edge classifier
        """
        super().__init__()

        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            concat_heads = i < num_layers - 1  # Don't concat on last layer
            self.gat_layers.append(
                GraphAttentionLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat_heads=concat_heads,
                )
            )

        # Edge classifier
        classifier_layers = []
        classifier_input_dim = 2 * hidden_dim + edge_features
        prev_dim = classifier_input_dim

        for hidden_dim_cls in classifier_hidden_dims:
            classifier_layers.append(
                MLPLayer(
                    input_dim=prev_dim,
                    output_dim=hidden_dim_cls,
                    dropout=dropout,
                    use_batch_norm=True,
                    activation="relu",
                )
            )
            prev_dim = hidden_dim_cls

        classifier_layers.append(
            MLPLayer(
                input_dim=prev_dim,
                output_dim=2,
                dropout=0.0,
                use_batch_norm=False,
                activation="none",
            )
        )

        self.edge_classifier = nn.Sequential(*classifier_layers)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with graph attention followed by edge classification.

        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feature_dim]

        Returns:
            logits: [num_edges, 2] - binary classification logits
        """
        # Encode initial node features
        x = F.relu(self.node_encoder(node_features))

        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)

        # Edge classification
        if edge_index.shape[1] > 0:
            source_features = x[edge_index[0]]
            target_features = x[edge_index[1]]
            edge_input = torch.cat([source_features, target_features, edge_attr], dim=1)
            logits = self.edge_classifier(edge_input)
        else:
            logits = torch.zeros(0, 2, device=node_features.device)

        return logits


class GraphTransformer(nn.Module):
    """Graph Transformer for edge classification."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_nodes: int = 1000,
        classifier_hidden_dims: List[int] = [256, 128],
    ):
        """
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            max_nodes: Maximum number of nodes for positional encoding
            classifier_hidden_dims: Hidden dimensions for edge classifier
        """
        super().__init__()

        self.d_model = d_model
        self.max_nodes = max_nodes

        # Input projections
        self.node_encoder = nn.Linear(node_features, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(max_nodes, d_model))

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Edge classifier
        classifier_layers = []
        classifier_input_dim = 2 * d_model + edge_features
        prev_dim = classifier_input_dim

        for hidden_dim in classifier_hidden_dims:
            classifier_layers.append(
                MLPLayer(
                    input_dim=prev_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=True,
                    activation="relu",
                )
            )
            prev_dim = hidden_dim

        classifier_layers.append(
            MLPLayer(
                input_dim=prev_dim,
                output_dim=2,
                dropout=0.0,
                use_batch_norm=False,
                activation="none",
            )
        )

        self.edge_classifier = nn.Sequential(*classifier_layers)

    def create_attention_mask(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> Optional[torch.Tensor]:
        """Create attention mask based on graph connectivity."""
        if edge_index.shape[1] == 0:
            return None

        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Undirected

        # Add self-connections
        adj.fill_diagonal_(1)

        # Create attention mask (0 for attention, -inf for no attention)
        mask = torch.zeros_like(adj)
        mask[adj == 0] = float("-inf")

        return mask

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through transformer layers followed by edge classification.

        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feature_dim]

        Returns:
            logits: [num_edges, 2] - binary classification logits
        """
        num_nodes = node_features.size(0)

        # Encode node features and add positional encoding
        x = self.node_encoder(node_features)
        if num_nodes <= self.max_nodes:
            x = x + self.positional_encoding[:num_nodes]

        # Add batch dimension for transformer
        x = x.unsqueeze(0)  # [1, num_nodes, d_model]

        # Create attention mask
        attention_mask = self.create_attention_mask(edge_index, num_nodes)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask)

        # Remove batch dimension
        x = x.squeeze(0)  # [num_nodes, d_model]

        # Edge classification
        if edge_index.shape[1] > 0:
            source_features = x[edge_index[0]]
            target_features = x[edge_index[1]]
            edge_input = torch.cat([source_features, target_features, edge_attr], dim=1)
            logits = self.edge_classifier(edge_input)
        else:
            logits = torch.zeros(0, 2, device=node_features.device)

        return logits


class GIN(nn.Module):
    """Graph Isomorphism Network for edge classification."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        eps: float = 0.0,
        learn_eps: bool = True,
        dropout: float = 0.2,
        classifier_hidden_dims: List[int] = [128, 64],
    ):
        """
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension for GIN layers
            num_layers: Number of GIN layers
            eps: Initial epsilon value
            learn_eps: Whether to learn epsilon
            dropout: Dropout probability
            classifier_hidden_dims: Hidden dimensions for edge classifier
        """
        super().__init__()

        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # GIN layers
        self.gin_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            self.gin_layers.append(
                GINLayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim * 2,
                    output_dim=hidden_dim,
                    eps=eps,
                    learn_eps=learn_eps,
                )
            )

        # Edge classifier
        classifier_layers = []
        classifier_input_dim = 2 * hidden_dim + edge_features
        prev_dim = classifier_input_dim

        for hidden_dim_cls in classifier_hidden_dims:
            classifier_layers.append(
                MLPLayer(
                    input_dim=prev_dim,
                    output_dim=hidden_dim_cls,
                    dropout=dropout,
                    use_batch_norm=True,
                    activation="relu",
                )
            )
            prev_dim = hidden_dim_cls

        classifier_layers.append(
            MLPLayer(
                input_dim=prev_dim,
                output_dim=2,
                dropout=0.0,
                use_batch_norm=False,
                activation="none",
            )
        )

        self.edge_classifier = nn.Sequential(*classifier_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with GIN layers followed by edge classification.

        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feature_dim]

        Returns:
            logits: [num_edges, 2] - binary classification logits
        """
        # Encode initial node features
        x = F.relu(self.node_encoder(node_features))

        # Apply GIN layers
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # Edge classification
        if edge_index.shape[1] > 0:
            source_features = x[edge_index[0]]
            target_features = x[edge_index[1]]
            edge_input = torch.cat([source_features, target_features, edge_attr], dim=1)
            logits = self.edge_classifier(edge_input)
        else:
            logits = torch.zeros(0, 2, device=node_features.device)

        return logits


class SIGNN(nn.Module):
    """
    Symmetry Isomorphism Graph Neural Network for edge classification.

    This model uses SIGNN layers with message passing between nodes and edges.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        node_types: int = 6,
        dropout: float = 0.2,
        update_edges: bool = True,
        classifier_hidden_dims: List[int] = [128, 64],
    ):
        """
        Args:
            node_features: Number of input node features
            edge_features: Number of input edge features
            hidden_dim: Hidden dimension for SIGNN layers
            num_layers: Number of SIGNN layers
            node_types: Number of different node types in power grids
            dropout: Dropout probability
            update_edges: Whether to update edge features
            classifier_hidden_dims: Hidden dimensions for edge classifier
        """
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.update_edges = update_edges

        # Input encoders
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)

        # Node type embeddings
        self.node_type_embedding = nn.Embedding(node_types, hidden_dim // 4)

        # SIGNN layers
        self.signn_layers = nn.ModuleList(
            [
                SIGNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    hidden_dim=hidden_dim * 2,
                    dropout=dropout,
                    update_edges=update_edges,
                )
                for _ in range(num_layers)
            ]
        )

        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through SIGNN layers followed by edge classification.

        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feature_dim]
            node_types: [num_nodes] - optional node type indices

        Returns:
            logits: [num_edges, 2] - binary classification logits
        """
        # Encode initial features
        node_h = self.node_encoder(node_features)

        # Add node type embeddings if available
        if node_types is not None:
            type_embed = self.node_type_embedding(node_types)
            node_h = torch.cat([node_h, type_embed], dim=1)
            # Adjust dimensions if needed
            if node_h.size(1) != self.hidden_dim:
                adjustment_layer = nn.Linear(node_h.size(1), self.hidden_dim).to(
                    node_h.device
                )
                node_h = adjustment_layer(node_h)

        if edge_index.shape[1] > 0:
            edge_h = self.edge_encoder(edge_attr)
        else:
            edge_h = torch.zeros(0, self.hidden_dim, device=node_features.device)

        # Apply SIGNN layers
        for signn_layer in self.signn_layers:
            node_h, edge_h = signn_layer(node_h, edge_index, edge_h)

        # Edge classification
        if edge_index.shape[1] > 0:
            source_features = node_h[edge_index[0]]
            target_features = node_h[edge_index[1]]
            edge_input = torch.cat([source_features, target_features, edge_attr], dim=1)
            logits = self.edge_classifier(edge_input)
        else:
            logits = torch.zeros(0, 2, device=node_features.device)

        return logits


class ResidualGNN(nn.Module):
    """Graph Neural Network with residual connections for edge classification."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
        classifier_hidden_dims: List[int] = [128, 64],
    ):
        """
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension for residual blocks
            num_blocks: Number of residual blocks
            gnn_type: Type of GNN layer ("gcn", "gat", "gin")
            dropout: Dropout probability
            classifier_hidden_dims: Hidden dimensions for edge classifier
        """
        super().__init__()

        self.node_encoder = nn.Linear(node_features, hidden_dim)

        # Create GNN layers with residual connections
        self.gnn_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if gnn_type.lower() == "gcn":
                gnn_layer = GCNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
            elif gnn_type.lower() == "gat":
                gnn_layer = GraphAttentionLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=4,
                    dropout=dropout,
                    concat_heads=False,
                )
            elif gnn_type.lower() == "gin":
                gnn_layer = GINLayer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim * 2,
                    output_dim=hidden_dim,
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            # Wrap in residual block
            block = ResidualGNNBlock(gnn_layer, hidden_dim, dropout)
            self.gnn_blocks.append(block)

        # Edge classifier with residual blocks
        classifier_layers = []
        classifier_input_dim = 2 * hidden_dim + edge_features

        # Input projection
        self.input_projection = nn.Linear(classifier_input_dim, hidden_dim)

        # Residual blocks for classifier
        self.classifier_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(2)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through residual GNN."""
        # Encode node features
        x = F.relu(self.node_encoder(node_features))

        # Apply residual GNN blocks
        for block in self.gnn_blocks:
            x = block(x, edge_index)

        # Edge classification
        if edge_index.shape[1] > 0:
            source_features = x[edge_index[0]]
            target_features = x[edge_index[1]]
            edge_input = torch.cat([source_features, target_features, edge_attr], dim=1)

            # Project to hidden dimension
            x_cls = F.relu(self.input_projection(edge_input))

            # Apply residual blocks
            for block in self.classifier_blocks:
                x_cls = block(x_cls)

            # Output classification
            logits = self.output_layer(x_cls)
        else:
            logits = torch.zeros(0, 2, device=node_features.device)

        return logits


class ResidualGNNBlock(nn.Module):
    """Residual block for GNN layers."""

    def __init__(self, gnn_layer: nn.Module, hidden_dim: int, dropout: float):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.gnn_layer(x, edge_index)
        x = self.norm(x + residual)
        x = self.dropout(x)
        return x
