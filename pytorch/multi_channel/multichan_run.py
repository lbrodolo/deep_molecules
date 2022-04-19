import math
import os
import random
from datetime import datetime
from pathlib import Path
from sched import scheduler

import numpy as np
import torch
from model import CNN
from dataset import MultichannelDataset
from torchmetrics import MeanAbsoluteError, R2Score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm, trange

SEED = 42
DATA_DIR = "/home/lbrodoloni/data"
DISABLE_BAR = True
SCHEDULER = True
EPOCHS = 80
PHYSIC_LEN = 12
N_POINTS = 32
BATCH_SIZE = 16
GAMMA = 0.375
ALPHA = 0.7
NAME = (
    f"10kval-BATCH{BATCH_SIZE}-{EPOCHS}epochs-{N_POINTS}grid-{PHYSIC_LEN}A-Gamma{GAMMA}"
)


def run():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    Path(f"checkpoints/{NAME}-{date_time}").mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import model and move to device (if available)
    model = CNN().to(device)
    model.initialize_weights
    print(model)
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    # Import loss function and optimizer
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.MSELoss()
    opt = torch.optim.NAdam(model.parameters(), lr=0.001)

    # Set up learning rate scheduler
    if SCHEDULER:
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10)
    print("Optimizer setted up")

    # Load dataset
    train_paths = [
        f"{DATA_DIR}/train/{name}" for name in os.listdir(f"{DATA_DIR}/train")
    ]
    test_paths = [f"{DATA_DIR}/test/{name}" for name in os.listdir(f"{DATA_DIR}/test")][
        :10000
    ]  # Uso solo i primi diecimila dati di test come validation, lascio gli ultimi 10000 come vero e proprio test set
    train_dataset = MultichannelDataset(
        train_paths, physic_len=PHYSIC_LEN, gamma=GAMMA, n_points=N_POINTS, alpha=ALPHA
    )
    print("Train Dataset generato")
    test_data = MultichannelDataset(
        test_paths, physic_len=PHYSIC_LEN, gamma=GAMMA, n_points=N_POINTS, alpha=ALPHA
    )  # Faccio previsioni direttamente sul test set, non uso il validation set
    print(f"Test Dataset generato \nDuration: {datetime.now() - now}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=False, num_workers=1
    )

    # Train and Test metrics
    train_mae = MeanAbsoluteError().to(device)
    test_mae = MeanAbsoluteError().to(device)
    r2_test = R2Score().to(device)
    best_mae = float("inf")
    best_r2 = 0.0

    pbar = tqdm(range(EPOCHS), leave=True, disable=DISABLE_BAR)

    # Train model
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")

        train_mae.reset()
        model.train()

        train_pbar = tqdm(train_loader, leave=True, disable=DISABLE_BAR)
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            pred = pred.squeeze()
            loss = criterion(pred, target)
            train_mae.update(pred, target)
            train_pbar.set_postfix({"mae": train_mae.compute().item()})
            # Backpropagation
            opt.zero_grad()
            loss.backward()
            # Optimizer step
            opt.step()
        test_mae.reset()
        r2_test.reset()
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(test_loader, leave=True, disable=DISABLE_BAR):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                pred = pred.squeeze()

                test_mae.update(pred, target)
                r2_test.update(pred, target)
        pbar.set_postfix(
            {
                "mae": train_mae.compute().item(),
                "test_mae": test_mae.compute().item(),
                "test_r2": r2_test.compute().item(),
            }
        )

        if SCHEDULER:
            scheduler.step()
        if DISABLE_BAR:
            print(
                f"EPOCH: {epoch}: \tMAE: {train_mae.compute().item()} - Test MAE: {test_mae.compute().item()} - Test R2: {r2_test.compute().item()}"
            )

        if test_mae < best_mae:
            torch.save(
                model.state_dict(), f"checkpoints/{NAME}-{date_time}/best_model.pt"
            )
            best_mae = test_mae
        if r2_test > best_r2:
            torch.save(
                model.state_dict(), f"checkpoints/{NAME}-{date_time}/best_model_r2.pt"
            )
            best_r2 = r2_test
    print(f"Best R2 on test: {best_r2.compute().item()}")


if __name__ == "__main__":
    run()
