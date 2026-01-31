import logging
logging.basicConfig(level=logging.INFO)

print(">>> run_train.py file loaded")

from src.data.multi_grid_dataloader import MultiGridPowerGridDataset
from src.models.models import MLP
from src.training.trainer import TrainingConfig, MultiGridTrainer

import torch


def main():
    print("Training starting...")

    # TODO: you MUST fill these in based on your dataset and model
#     dataloader = MultiGridDataLoader(...)
#     model = SIGNN(
#         in_channels=...,
#         hidden_channels=...,
#         out_channels=...,
#         num_layers=...
#     )

#     trainer = Trainer(
#         model=model,
#         dataloader=dataloader,
#         lr=0.001,
#         epochs=50,
#     )

#     trainer.train()

# if __name__ == "__main__":
#     main()

    # -------------------------
    # 1. Load dataset
    # -------------------------
    dataset_path = "dataset"   # SIGNN-main/dataset folder
    dataset = MultiGridPowerGridDataset(dataset_path)

    # -------------------------
    # 2. Build model
    # -------------------------
    # use first sample to determine input dims
    sample = dataset[0]
    model = MLP(
        node_features=sample.x.shape[1],
        edge_features=sample.edge_attr.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        use_batch_norm=True
    )

    # -------------------------
    # 3. Config
    # -------------------------
    config = TrainingConfig()
    config.num_epochs = 50
    config.learning_rate = 0.001
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 4. Trainer
    # -------------------------
    trainer = MultiGridTrainer(
        model=model,
        dataset=dataset,
        config=config,
        logger=logging.getLogger("MLP_MultiGrid")
    )

    # -------------------------
    # 5. Train
    # -------------------------
    trainer.fit_multi_grid()

    # -------------------------
    # 6. Evaluate
    # -------------------------
    results = trainer.evaluate_test_set()
    print("Final Test Results:", results)

if __name__ == "__main__":
    main()
    