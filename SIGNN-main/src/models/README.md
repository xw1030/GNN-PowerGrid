# Models and Layers Documentation

Graph Neural Network models and layers for the (SI)GNN framework for edge classification tasks.

## Structure

- `layers.py`: Contains individual neural network layer implementations
- `models.py`: Contains complete model architectures that combine layers for edge classification
- `losses.py`: Contains loss functions for training

## Available Models

1. **`MLP`**: Simple multi-layer perceptron baseline (*simple baseline*)
2. **`GCN`**: Graph Convolutional Network (*general model*)
3. **`GAT`**: Graph Attention Network with multi-head attention (*attention mechanisms and interpretability*)
4. **`GraphTransformer`**: Transformer architecture for graphs (*large graphs and global attention*)
5. **`GIN`**: Graph Isomorphism Network (*guarantees about graph isomorphism*)
6. **`SIGNN`**: Symmetry Isomorphism Graph Neural Network (*the main model*)
7. **`ResidualGNN`**: Graph neural network with residual connections (*deeper networks*)
