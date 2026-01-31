# SIGNN Multi-Grid PowerGrid n-1 Classification DataLoader

A dataloader for SIGNN (Symmetry Isomorphism Graph Neural Networks) power grid n-1 contingency classification using CSV data containing stations (nodes), lines (edges), and transformers across multiple power grids.

## Overview

This dataloader processes power grid data from CSV files and creates graph structures suitable for training SIGNN models to predict n-1 contingencies. The key insight is that **`grid_id` determines individual power grids**, resulting in **159 different power grids** with **11 scenarios each**, giving us **159 × 11 = 1,749 total samples**.

## Dataset Structure

### Multi-Grid Architecture
- **159 individual power grids** (identified by `grid_id`)
- **11 scenarios per grid** (different load conditions)
- **Total samples: 1,749** (159 grids × 11 scenarios)
- **Variable graph sizes**: 4 to 1,802 nodes per grid

### CSV Files Structure

The dataloader expects three CSV files in a `dataset/` directory:

#### `nodes.csv`
Contains power grid nodes with features:
- `id`: Unique node identifier
- `grid_id`: **Grid identifier (determines which power grid)**
- `x`, `y`: Node coordinates
- `node_type`: Type of node (1=OS, 2=SS, 3=RS, 4=MSR, 5=DSR, 0=UNKNOWN)
- `u_rated`: Rated voltage
- `netvlak`: Network level
- `voltage`, `voltage_angle`: Electrical measurements
- `real_power_X`, `reactive_power_X`: Power measurements for scenario X

#### `lines.csv`
Contains power lines with features:
- `id`: Unique line identifier
- `grid_id`: **Grid identifier (determines which power grid)**
- `from_node_id`, `to_node_id`: Connected nodes
- `resistance`, `reactance`, `capacitance`: Electrical properties
- `loss_factor`, `max_current`: Line characteristics
- `status_X`: n-1 classification label for scenario X

#### `transformers.csv`
Contains transformers with features:
- `id`: Unique transformer identifier
- `grid_id`: **Grid identifier (determines which power grid)**
- `from_node`, `to_node`: Connected nodes
- `u1`, `u2`: Voltage levels
- `sn`: Apparent power rating
- `uk`, `pk`: Transformer parameters
- `status_X`: n-1 classification label for scenario X

## Classification Labels

Each edge (line or transformer) has status labels for different scenarios:
- `"n-1"`: Removing this edge results in n-1 contingency (label = 1)
- `"not-n-1"`: Removing this edge does not result in n-1 contingency (label = 0)
- `"unrecoverable"`: Edge is excluded from analysis (filtered out)

# Dataset Statistics

```
Multi-Grid Dataset Analysis:
Number of grids: 159
Number of scenarios: 11
Total samples: 1,749 (159 × 11)

Grid Statistics:
Nodes per grid: min=4, max=1,802, avg=506.2
Edges per grid: min=2, max=1,862, avg=528.1

Scenario Statistics (across all grids):
- ~98.8% n-1 edges (class imbalance)
- ~1.2% not-n-1 edges
- Consistent across all scenarios
```

### Individual Grid Examples
- **Grid 0**: 1,237 nodes, 1,288 edges
- **Grid 61**: 6 nodes, 3 edges (smallest)
- **Grid 165**: 1,802 nodes, 1,862 edges (largest)

## Data Structure Per Sample

### Node Features (29 dimensions)
Each sample contains a graph with:
1. **Spatial/Electrical**: coordinates, node type, voltage ratings
2. **Scenario-specific**: power measurements for the specific scenario
3. **Normalized**: optional feature normalization across all grids

### Edge Features (6 dimensions)
Each edge has features:
1. **Lines**: resistance, reactance, capacitance, loss_factor, max_current, edge_type=1
2. **Transformers**: u1/1000, u2/1000, sn/1000000, uk, pk/1000, edge_type=0

### Graph Properties
- **Undirected**: Each edge appears in both directions
- **Variable size**: 4 to 1,802 nodes depending on grid
