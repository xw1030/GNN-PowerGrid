"""
Subgraph transforms for SIGNN (ESAN-style ego_nets_plus).
Converts a single graph into a list of subgraphs for subgraph-based training.
"""

import torch
import random
from torch_geometric.utils import k_hop_subgraph
from typing import List, Tuple, Optional

from .multi_grid_dataloader import PowerGridGraphData


def ego_nets_plus_subgraphs(
    data: PowerGridGraphData,
    num_hops: int = 2,
    max_centers: Optional[int] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    ESAN-style ego_nets_plus: one subgraph per node (k-hop neighborhood),
    with center node marked by prepending [1,0] vs [0,1] to node features.

    Args:
        data: PowerGridGraphData (x, edge_index, edge_attr, y)
        num_hops: k for k-hop neighborhood
        max_centers: if set, only use this many random center nodes per sample (reduces OOM).

    Returns:
        List of (x_plus, edge_index, edge_attr, y) per subgraph.
        x_plus: [num_nodes, node_features+2], edge_index/edge_attr/y: subset for this subgraph.
    """
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
    num_nodes = data.num_nodes
    device = x.device
    dtype = x.dtype

    if edge_index.shape[1] == 0:
        return []

    center_indices = list(range(num_nodes))
    if max_centers is not None and num_nodes > max_centers:
        center_indices = random.sample(center_indices, max_centers)

    out = []
    for i in center_indices:
        _, _, _, edge_mask = k_hop_subgraph(
            i, num_hops, edge_index, relabel_nodes=False, num_nodes=num_nodes
        )
        ei = edge_index[:, edge_mask]
        ea = edge_attr[edge_mask] if edge_attr is not None else None
        y_sub = y[edge_mask]

        ids = torch.zeros(num_nodes, 2, dtype=dtype, device=device)
        ids[:, 1] = 1.0
        ids[i] = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        x_plus = torch.hstack([ids, x])

        out.append((x_plus, ei, ea, y_sub))
    return out


def pad_center_marker(x: torch.Tensor, device=None) -> torch.Tensor:
    """Prepend [0,0] for all nodes (no center). Use when not in subgraph mode."""
    device = device or x.device
    x = x.to(device)
    z = torch.zeros(x.size(0), 2, dtype=x.dtype, device=device)
    return torch.hstack([z, x])
