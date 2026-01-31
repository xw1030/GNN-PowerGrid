"""
Multi-Grid DataLoader
Author: Charlotte Cambier van Nooten
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerGridGraphData:
    """
    Data container for a single power grid graph with scenario-specific labels.
    """

    def __init__(
        self,
        x,
        edge_index,
        edge_attr,
        y,
        edge_ids=None,
        grid_id=None,
        scenario=None,
        num_nodes=None,
    ):
        self.x = x  # Node features [num_nodes, num_features]
        self.edge_index = edge_index  # Edge connectivity [2, num_edges]
        self.edge_attr = edge_attr  # Edge features [num_edges, num_edge_features]
        self.y = y  # Edge labels [num_edges]
        self.edge_ids = edge_ids  # Original edge IDs [num_edges]
        self.grid_id = grid_id  # Grid identifier
        self.scenario = scenario  # Scenario name
        self.num_nodes = num_nodes if num_nodes is not None else x.shape[0]

    def __repr__(self):
        return (
            f"PowerGridGraphData(grid_id={self.grid_id}, scenario={self.scenario}, "
            f"num_nodes={self.num_nodes}, num_edges={self.edge_index.shape[1]})"
        )


class MultiGridPowerGridDataset:
    """
    Multi-Grid PyTorch dataset for power grid n-1 classification.

    The dataset loads multiple power grids from CSV files, where each grid_id
    represents a separate graph. Each graph has multiple scenarios with different
    n-1 classification labels.

    Total samples = num_grids × num_scenarios
    """

    def __init__(
        self,
        root_dir: str,
        scenario_columns: Optional[List[str]] = None,
        grid_ids: Optional[List[int]] = None,
        normalize_features: bool = True,
        include_node_features: bool = True,
        include_transformers: bool = True,
    ):
        """
        Args:
            root_dir: Path to directory containing CSV files
            scenario_columns: List of scenario column names to use. If None, uses all status columns
            grid_ids: List of specific grid IDs to load. If None, loads all grids
            normalize_features: Whether to normalize node and edge features
            include_node_features: Whether to include node features in the graph
            include_transformers: Whether to include transformers in the dataset
        """
        self.root_dir = Path(root_dir)
        self.normalize_features = normalize_features
        self.include_node_features = include_node_features
        self.include_transformers = include_transformers

        # Load data
        logger.info("Loading CSV files...")
        self.nodes_df = pd.read_csv(self.root_dir / "nodes.csv")
        self.lines_df = pd.read_csv(self.root_dir / "lines.csv")
        if self.include_transformers:
            self.transformers_df = pd.read_csv(self.root_dir / "transformers.csv")
        else:
            # Create empty dataframe with expected columns for compatibility
            self.transformers_df = pd.DataFrame(
                columns=["grid_id", "from_node", "to_node"]
            )

        # Get available grid IDs
        available_grid_ids = sorted(self.nodes_df["grid_id"].unique())
        if grid_ids is None:
            self.grid_ids = available_grid_ids
        else:
            self.grid_ids = [gid for gid in grid_ids if gid in available_grid_ids]

        logger.info(
            f"Found {len(available_grid_ids)} grids, using {len(self.grid_ids)} grids"
        )

        # Get scenario columns if not provided
        if scenario_columns is None:
            self.scenario_columns = [
                col for col in self.lines_df.columns if col.startswith("status_")
            ]
        else:
            self.scenario_columns = scenario_columns

        logger.info(
            f"Using {len(self.scenario_columns)} scenarios: {self.scenario_columns}"
        )

        # Create sample index: (grid_id, scenario) pairs
        self.samples = []
        for grid_id in self.grid_ids:
            for scenario in self.scenario_columns:
                self.samples.append((grid_id, scenario))

        logger.info(
            f"Total samples: {len(self.samples)} ({len(self.grid_ids)} grids × {len(self.scenario_columns)} scenarios)"
        )

        # Prepare per-grid data structures
        self._prepare_grid_data()

    def _prepare_grid_data(self):
        """Prepare node features and edge information for each grid."""

        self.grid_data = {}

        for grid_id in self.grid_ids:
            # logger.info(f"Processing grid {grid_id}...")

            # Filter data for this grid
            grid_nodes = self.nodes_df[self.nodes_df["grid_id"] == grid_id].copy()
            grid_lines = self.lines_df[self.lines_df["grid_id"] == grid_id].copy()
            grid_transformers = self.transformers_df[
                self.transformers_df["grid_id"] == grid_id
            ].copy()

            if len(grid_nodes) == 0:
                logger.warning(f"No nodes found for grid {grid_id}, skipping...")
                continue

            # Create node mapping for this grid
            node_ids = sorted(grid_nodes["id"].unique())
            node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

            # Prepare node features for this grid
            if self.include_node_features:
                node_feature_cols = [
                    "x",
                    "y",
                    "node_type",
                    "u_rated",
                    "netvlak",
                    "voltage",
                    "voltage_angle",
                ]

                # Add power features for each scenario
                power_cols = []
                for scenario in self.scenario_columns:
                    scenario_num = scenario.split("_")[1]
                    power_cols.extend(
                        [f"real_power_{scenario_num}", f"reactive_power_{scenario_num}"]
                    )

                all_feature_cols = node_feature_cols + power_cols

                # Sort nodes by ID for consistent ordering
                nodes_sorted = grid_nodes.set_index("id").loc[node_ids].copy()

                # Fill NaN values
                for col in all_feature_cols:
                    if col in nodes_sorted.columns:
                        nodes_sorted[col] = nodes_sorted[col].fillna(0)
                    else:
                        nodes_sorted[col] = 0

                node_features = torch.tensor(
                    nodes_sorted[all_feature_cols].values, dtype=torch.float32
                )
            else:
                # Use node embeddings/one-hot
                node_features = torch.eye(len(node_ids), dtype=torch.float32)

            # Prepare edges for this grid
            edges_data = []

            # Process lines
            for _, line in grid_lines.iterrows():
                from_node = line["from_node_id"]
                to_node = line["to_node_id"]

                if from_node in node_id_to_idx and to_node in node_id_to_idx:
                    edge_features = [
                        line["resistance"],
                        line["reactance"],
                        line["capacitance"],
                        line["loss_factor"] if pd.notna(line["loss_factor"]) else 0.0,
                        line["max_current"],
                        1.0,  # edge_type: 1 for line
                    ]

                    # Add status for each scenario
                    scenario_labels = {}
                    for scenario in self.scenario_columns:
                        status = line[scenario]
                        if status == "n-1":
                            scenario_labels[scenario] = 1
                        elif status == "not-n-1":
                            scenario_labels[scenario] = 0
                        else:  # 'unrecoverable'
                            scenario_labels[scenario] = -1  # Will be filtered out

                    edges_data.append(
                        {
                            "from_node": from_node,
                            "to_node": to_node,
                            "features": edge_features,
                            "labels": scenario_labels,
                            "edge_id": line["id"],
                            "edge_type": "line",
                        }
                    )

            # Process transformers (if enabled)
            if self.include_transformers:
                for _, transformer in grid_transformers.iterrows():
                    from_node = transformer["from_node"]
                    to_node = transformer["to_node"]

                    if from_node in node_id_to_idx and to_node in node_id_to_idx:
                        edge_features = [
                            transformer["u1"] / 1000,  # Normalize voltage
                            transformer["u2"] / 1000,  # Normalize voltage
                            transformer["sn"] / 1000000,  # Normalize power (MVA)
                            transformer["uk"],
                            transformer["pk"] / 1000,  # Normalize power (kW)
                            0.0,  # edge_type: 0 for transformer
                        ]

                        # Add status for each scenario
                        scenario_labels = {}
                        for scenario in self.scenario_columns:
                            status = transformer[scenario]
                            if status == "n-1":
                                scenario_labels[scenario] = 1
                            elif status == "not-n-1":
                                scenario_labels[scenario] = 0
                            else:  # 'unrecoverable'
                                scenario_labels[scenario] = -1  # Will be filtered out

                        edges_data.append(
                            {
                                "from_node": from_node,
                                "to_node": to_node,
                                "features": edge_features,
                                "labels": scenario_labels,
                                "edge_id": transformer["id"],
                                "edge_type": "transformer",
                            }
                        )

            # Store grid data
            self.grid_data[grid_id] = {
                "node_features": node_features,
                "node_id_to_idx": node_id_to_idx,
                "edges_data": edges_data,
                "num_nodes": len(node_ids),
                "num_lines": len(grid_lines),
                "num_transformers": len(grid_transformers),
            }

            # logger.info(
            #     f"Grid {grid_id}: {len(node_ids)} nodes, {len(edges_data)} edges"
            # )

        # Normalize edge features globally if requested
        if self.normalize_features and len(self.grid_data) > 0:
            self._normalize_edge_features()

        logger.info(f"Prepared data for {len(self.grid_data)} grids")

    def _normalize_edge_features(self):
        """Normalize edge features across all grids."""
        all_edge_features = []

        # Collect all edge features
        for grid_data in self.grid_data.values():
            for edge in grid_data["edges_data"]:
                all_edge_features.append(edge["features"])

        if not all_edge_features:
            return

        # Calculate global statistics
        all_features_array = np.array(all_edge_features)
        feature_mean = np.mean(all_features_array, axis=0)
        feature_std = np.std(all_features_array, axis=0)
        feature_std = np.where(
            feature_std == 0, 1, feature_std
        )  # Avoid division by zero

        # Normalize all edge features
        for grid_data in self.grid_data.values():
            for edge in grid_data["edges_data"]:
                normalized = (np.array(edge["features"]) - feature_mean) / feature_std
                edge["features"] = normalized.tolist()

    def __len__(self):
        """Return total number of samples (grids × scenarios)."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get graph data for a specific (grid_id, scenario) pair."""
        grid_id, scenario = self.samples[idx]

        if grid_id not in self.grid_data:
            raise ValueError(f"Grid {grid_id} not found in prepared data")

        grid_info = self.grid_data[grid_id]

        # Build edge index and edge attributes for this scenario
        edge_indices = []
        edge_features = []
        edge_labels = []
        edge_ids = []

        for edge_data in grid_info["edges_data"]:
            if scenario not in edge_data["labels"]:
                continue

            label = edge_data["labels"][scenario]

            # Skip unrecoverable edges
            if label == -1:
                continue

            from_idx = grid_info["node_id_to_idx"][edge_data["from_node"]]
            to_idx = grid_info["node_id_to_idx"][edge_data["to_node"]]

            # Add both directions (undirected graph)
            edge_indices.extend([[from_idx, to_idx], [to_idx, from_idx]])
            edge_features.extend([edge_data["features"], edge_data["features"]])
            edge_labels.extend([label, label])
            edge_ids.extend([edge_data["edge_id"], edge_data["edge_id"]])

        # Convert to tensors
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
            y = torch.tensor(edge_labels, dtype=torch.long)
            edge_ids_tensor = torch.tensor(edge_ids, dtype=torch.long)
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float32)  # 6 edge features
            y = torch.zeros(0, dtype=torch.long)
            edge_ids_tensor = torch.zeros(0, dtype=torch.long)

        # Create data object
        data = PowerGridGraphData(
            x=grid_info["node_features"],
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            edge_ids=edge_ids_tensor,
            grid_id=grid_id,
            scenario=scenario,
            num_nodes=grid_info["num_nodes"],
        )

        return data

    def get_sample_info(self, idx):
        """Get (grid_id, scenario) for a given sample index."""
        return self.samples[idx]

    def get_grid_stats(self):
        """Get statistics for each grid."""
        stats = {}
        for grid_id, grid_info in self.grid_data.items():
            stats[grid_id] = {
                "num_nodes": grid_info["num_nodes"],
                "num_lines": grid_info["num_lines"],
                "num_transformers": grid_info["num_transformers"],
                "total_edges": len(grid_info["edges_data"]),
            }
        return stats

    def get_scenario_stats(self):
        """Get statistics for each scenario across all grids."""
        stats = {}

        for scenario in self.scenario_columns:
            total_n1 = 0
            total_not_n1 = 0
            total_edges = 0

            for grid_id in self.grid_ids:
                if grid_id not in self.grid_data:
                    continue

                for edge_data in self.grid_data[grid_id]["edges_data"]:
                    if scenario in edge_data["labels"]:
                        label = edge_data["labels"][scenario]
                        if label == 1:
                            total_n1 += 1
                        elif label == 0:
                            total_not_n1 += 1
                        # Skip unrecoverable (-1)

            total_edges = total_n1 + total_not_n1
            stats[scenario] = {
                "n_1": total_n1,
                "not_n_1": total_not_n1,
                "total": total_edges,
                "n_1_ratio": total_n1 / total_edges if total_edges > 0 else 0,
            }

        return stats


class MultiGridDataLoader:
    """
    Simple DataLoader for multi-grid dataset.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch_data = []

            for idx in batch_indices:
                batch_data.append(self.dataset[idx])

            if len(batch_data) == 1:
                yield batch_data[0]
            else:
                yield batch_data


def create_multi_grid_dataloader(
    dataset_path: str,
    batch_size: int = 1,
    shuffle: bool = True,
    scenarios: Optional[List[str]] = None,
    grid_ids: Optional[List[int]] = None,
    normalize_features: bool = True,
) -> MultiGridDataLoader:
    """
    Create a multi-grid DataLoader for the power grid dataset.

    Args:
        dataset_path: Path to directory containing CSV files
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        scenarios: List of specific scenarios to load. If None, loads all scenarios
        grid_ids: List of specific grid IDs to load. If None, loads all grids
        normalize_features: Whether to normalize features

    Returns:
        MultiGridDataLoader instance
    """
    dataset = MultiGridPowerGridDataset(
        dataset_path,
        scenario_columns=scenarios,
        grid_ids=grid_ids,
        normalize_features=normalize_features,
    )

    return MultiGridDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def analyze_multi_grid_dataset(dataset_path: str):
    """
    Analyze the multi-grid dataset and print statistics.

    Args:
        dataset_path: Path to directory containing CSV files
    """
    dataset = MultiGridPowerGridDataset(dataset_path)

    print(f"Multi-Grid Dataset Analysis:")
    print(f"Number of grids: {len(dataset.grid_ids)}")
    print(f"Number of scenarios: {len(dataset.scenario_columns)}")
    print(
        f"Total samples: {len(dataset)} ({len(dataset.grid_ids)} × {len(dataset.scenario_columns)})"
    )

    # Grid statistics
    print(f"\nGrid Statistics:")
    grid_stats = dataset.get_grid_stats()

    nodes_per_grid = [stats["num_nodes"] for stats in grid_stats.values()]
    edges_per_grid = [stats["total_edges"] for stats in grid_stats.values()]

    print(
        f"Nodes per grid: min={min(nodes_per_grid)}, max={max(nodes_per_grid)}, avg={np.mean(nodes_per_grid):.1f}"
    )
    print(
        f"Edges per grid: min={min(edges_per_grid)}, max={max(edges_per_grid)}, avg={np.mean(edges_per_grid):.1f}"
    )

    # Scenario statistics
    print(f"\nScenario Statistics (across all grids):")
    scenario_stats = dataset.get_scenario_stats()
    for scenario, stat in scenario_stats.items():
        print(
            f"{scenario}: n-1={stat['n_1']}, not-n-1={stat['not_n_1']}, "
            f"total={stat['total']}, n-1 ratio={stat['n_1_ratio']:.3f}"
        )

    # Sample data
    print(f"\nSample data (first sample):")
    sample_data = dataset[0]
    print(f"Grid ID: {sample_data.grid_id}")
    print(f"Scenario: {sample_data.scenario}")
    print(f"Nodes: {sample_data.num_nodes}")
    print(f"Edges: {sample_data.edge_index.shape[1]}")
    print(f"Node features shape: {sample_data.x.shape}")
    print(f"Edge features shape: {sample_data.edge_attr.shape}")
    print(f"Labels shape: {sample_data.y.shape}")
    if len(sample_data.y) > 0:
        print(
            f"Label distribution: n-1={torch.sum(sample_data.y == 1).item()}, "
            f"not-n-1={torch.sum(sample_data.y == 0).item()}"
        )

    return dataset


if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset"

    # Analyze dataset
    dataset = analyze_multi_grid_dataset(dataset_path)

    # Create dataloader
    dataloader = create_multi_grid_dataloader(dataset_path, batch_size=4, shuffle=True)

    print(f"\nDataLoader created with {len(dataloader)} batches")

    # Test iteration
    for i, batch in enumerate(dataloader):
        if isinstance(batch, list):
            print(f"Batch {i}: {len(batch)} graphs")
            for j, graph in enumerate(batch):
                print(
                    f"  Graph {j}: Grid {graph.grid_id}, Scenario {graph.scenario}, "
                    f"{graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges"
                )
        else:
            print(
                f"Batch {i}: Grid {batch.grid_id}, Scenario {batch.scenario}, "
                f"{batch.num_nodes} nodes, {batch.edge_index.shape[1]} edges"
            )

        if i >= 2:  # Just show first few batches
            break
