"""
Neural network layers for graph neural networks.
Author: Charlotte Cambier van Nooten
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MLPLayer(nn.Module):
    """Multi-layer perceptron layer with optional batch normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)

        if self.use_batch_norm:
            x = self.batch_norm(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer with message passing."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        use_batch_norm: bool = True,
        add_self_loops: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_self_loops = add_self_loops

        self.linear = nn.Linear(input_dim, output_dim)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GCN layer.

        Args:
            x: [num_nodes, input_dim] - node features
            edge_index: [2, num_edges] - edge connectivity
            edge_weight: [num_edges] - optional edge weights

        Returns:
            out: [num_nodes, output_dim] - updated node features
        """
        if edge_index.shape[1] == 0:
            return self.linear(x)

        num_nodes = x.size(0)

        # Add self loops if requested
        if self.add_self_loops:
            self_loop_index = (
                torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            )
            edge_index = torch.cat([edge_index, self_loop_index], dim=1)

            if edge_weight is not None:
                self_loop_weight = torch.ones(num_nodes, device=x.device)
                edge_weight = torch.cat([edge_weight, self_loop_weight])

        # Compute degree normalization
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, row, torch.ones(edge_index.shape[1], device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Apply normalization
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if edge_weight is not None:
            norm = norm * edge_weight

        # Message passing
        x = self.linear(x)

        # Aggregate messages
        out = torch.zeros_like(x)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += norm[i] * x[src]

        if self.batch_norm is not None:
            out = self.batch_norm(out)

        out = F.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with multi-head attention."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
        use_bias: bool = True,
        concat_heads: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads

        if concat_heads:
            assert output_dim % num_heads == 0
            self.head_dim = output_dim // num_heads
        else:
            self.head_dim = output_dim

        # Linear transformations for each head
        self.W = nn.Linear(input_dim, self.head_dim * num_heads, bias=use_bias)

        # Attention mechanism - single attention vector for all heads
        self.attention = nn.Parameter(torch.randn(2 * self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.attention.unsqueeze(0), gain=gain)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through GAT layer.

        Args:
            x: [num_nodes, input_dim] - node features
            edge_index: [2, num_edges] - edge connectivity

        Returns:
            out: [num_nodes, output_dim] - updated node features
        """
        if edge_index.shape[1] == 0:
            return torch.zeros(x.size(0), self.output_dim, device=x.device)

        N = x.size(0)

        # Linear transformation
        h = self.W(x)  # [N, head_dim * num_heads]
        h = h.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]

        # Get source and target node features
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # For each head, compute attention
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)

        for head in range(self.num_heads):
            h_head = h[:, head, :]  # [N, head_dim]

            h_i = h_head[source_nodes]  # [num_edges, head_dim]
            h_j = h_head[target_nodes]  # [num_edges, head_dim]

            # Concatenate for attention computation
            h_cat = torch.cat([h_i, h_j], dim=-1)  # [num_edges, 2*head_dim]

            # Compute attention scores
            e = torch.matmul(h_cat, self.attention)  # [num_edges]
            e = self.leaky_relu(e)

            # Apply softmax per target node
            attention_weights = torch.zeros(edge_index.shape[1], device=x.device)
            for i in range(N):
                mask = target_nodes == i
                if mask.any():
                    attention_weights[mask] = F.softmax(e[mask], dim=0)

            # Apply dropout to attention weights
            attention_weights = self.dropout(attention_weights)

            # Apply attention to source node features
            weighted_h = h_i * attention_weights.unsqueeze(-1)  # [num_edges, head_dim]

            # Aggregate messages for this head
            out[:, head, :].index_add_(0, target_nodes, weighted_h)

        if self.concat_heads:
            out = out.view(N, -1)  # [N, output_dim]
        else:
            out = out.mean(dim=1)  # [N, head_dim]

        return out


class TransformerLayer(nn.Module):
    """Transformer layer for graph data with positional encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Feed forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer layer.

        Args:
            x: [batch_size, seq_len, d_model] - input features
            attention_mask: [batch_size, seq_len, seq_len] - attention mask

        Returns:
            out: [batch_size, seq_len, d_model] - updated features
        """
        # Self attention
        x2, _ = self.self_attn(x, x, x, attn_mask=attention_mask)
        x = self.norm1(x + self.dropout1(x2))

        # Feed forward
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x))))
        x = self.norm2(x + x2)

        return x


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        eps: float = 0.0,
        learn_eps: bool = True,
    ):
        super().__init__()

        self.eps = nn.Parameter(torch.tensor(eps)) if learn_eps else eps

        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through GIN layer.

        Args:
            x: [num_nodes, input_dim] - node features
            edge_index: [2, num_edges] - edge connectivity

        Returns:
            out: [num_nodes, output_dim] - updated node features
        """
        if edge_index.shape[1] == 0:
            return self.mlp((1 + self.eps) * x)

        num_nodes = x.size(0)

        # Aggregate neighbor features
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, target_nodes, x[source_nodes])

        # Apply MLP to aggregated features
        out = self.mlp((1 + self.eps) * x + aggregated)

        return out


class SIGNNLayer(nn.Module):
    """SIGNN layer with message passing and edge updates."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        update_edges: bool = True,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.update_edges = update_edges

        # Message functions
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Node update functions
        self.node_update = nn.GRUCell(node_dim, node_dim)

        # Edge update functions
        if update_edges:
            self.edge_update = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, edge_dim),
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SIGNN layer.

        Args:
            node_h: [num_nodes, node_dim] - node features
            edge_index: [2, num_edges] - edge connectivity
            edge_h: [num_edges, edge_dim] - edge features

        Returns:
            updated_node_h: [num_nodes, node_dim]
            updated_edge_h: [num_edges, edge_dim]
        """
        if edge_index.shape[1] == 0:
            return node_h, edge_h

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # Compute messages
        source_h = node_h[source_nodes]
        target_h = node_h[target_nodes]

        message_input = torch.cat([source_h, target_h, edge_h], dim=1)
        messages = self.message_net(message_input)

        # Aggregate messages for each node
        num_nodes = node_h.size(0)
        aggregated = torch.zeros_like(node_h)

        # Sum messages for each target node
        aggregated.index_add_(0, target_nodes, messages)

        # Update node features using GRU
        updated_node_h = self.node_update(aggregated, node_h)
        updated_node_h = self.norm(updated_node_h)
        updated_node_h = self.dropout(updated_node_h)

        # Update edge features if enabled
        if self.update_edges:
            edge_input = torch.cat(
                [updated_node_h[source_nodes], updated_node_h[target_nodes], edge_h],
                dim=1,
            )
            updated_edge_h = self.edge_update(edge_input)
            updated_edge_h = F.relu(updated_edge_h + edge_h)  # Residual connection
        else:
            updated_edge_h = edge_h

        return updated_node_h, updated_edge_h


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x

        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.batch_norm2(x)

        # Add residual connection
        x = x + residual
        x = F.relu(x)

        return x
