import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)

print(">>> run_train.py file loaded")

from src.data.multi_grid_dataloader import MultiGridPowerGridDataset
from src.models.models import MLP, SIGNN
from src.training.trainer import TrainingConfig, MultiGridTrainer

import torch


def main():
    print("Training starting...")

    # -------------------------
    # 1. Load dataset
    # -------------------------
    dataset_path = "dataset"   # SIGNN-main/dataset folder
    dataset = MultiGridPowerGridDataset(dataset_path)
    
    # Use first sample to determine input dims
    sample = dataset[0]

    # -------------------------
    # 2. Config (set before model so we know use_subgraph for input dim)
    # -------------------------
    config = TrainingConfig()
    config.num_epochs = 50
    config.learning_rate = 0.001
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.use_subgraph = False   # True = ESAN-style ego_nets_plus subgraph training
    config.subgraph_num_hops = 2
    config.subgraph_max_centers_per_sample = None  # None = 全图所有节点作中心；设整数则限制中心数
    config.subgraph_chunk_size = 100  # 每批 backward 的子图数，越大显存越高、越接近拉满（OOM 可改小）
    config.subgraph_num_workers = 4  # 子图生成并行进程数，0=不并行；若 subgraph 占比高可设为 4~8

    # -------------------------
    # 3. Build SIGNN model (node_features + 2 when use_subgraph for center marker)
    # -------------------------
    node_in = sample.x.shape[1] + 2  # +2 for subgraph center marker (pad zeros when not use_subgraph)
    model = SIGNN(
        node_features=node_in,
        edge_features=sample.edge_attr.shape[1],
        hidden_dim=128,
        num_layers=3,
        node_types=6,  # Default for power grid node types
        dropout=0.2,
        update_edges=True,
        classifier_hidden_dims=[128, 64]
    )

#     trainer = Trainer(
#         model=model,
#         dataloader=dataloader,
#         lr=0.001,
#         epochs=50,
#     )

#     trainer.train()


    # -------------------------
    # 4. Trainer
    # -------------------------
    trainer = MultiGridTrainer(
        model=model,
        dataset=dataset,
        config=config,
        logger=logging.getLogger("SIGNN_MultiGrid")
    )

    # -------------------------
    # 5. Train
    # -------------------------
    trainer.fit_multi_grid()

    # -------------------------
    # 6. Evaluate
    # -------------------------
    results = trainer.evaluate_test_set()
    # 只打印汇总，避免 results 里的长列表（如 scenarios）刷屏
    if results:
        s = results.get("summary", {})
        print("Final Test Results (summary):")
        print(f"  overall_accuracy: {s.get('overall_accuracy', results.get('accuracy'))}")
        print(f"  overall_f1: {s.get('overall_f1', results.get('f1'))}")
        print(f"  num_test_samples: {results.get('num_test_samples')}")
        print(f"  num_test_grids: {results.get('num_test_grids')}")
        if "grid_accuracy_mean" in s:
            print(f"  grid_accuracy: mean={s['grid_accuracy_mean']:.4f}, min={s['grid_accuracy_min']:.4f}, max={s['grid_accuracy_max']:.4f}")
    else:
        print("Final Test Results: (empty)")

if __name__ == "__main__":
    main()
    