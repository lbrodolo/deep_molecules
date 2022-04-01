import os
import random
import numpy as np
import torch
from torchmetrics import R2Score, MeanAbsoluteError
from model import CNN
from torch_dataset import MoleculesDataset
import math
from datetime import datetime
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm import trange, tqdm

SEED = 42
DATA_DIR = "/home/lbrodoloni/data/train"
DISABLE_BAR = True
SCHEDULER = True


def run():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    Path(f"checkpoints/{date_time}").mkdir(parents=True, exist_ok=True)

    # reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import model and move it to GPU (if available)
    model = CNN().to(device)
    model.initialize_weights
    print(model)
    print(
        f"TOTAL PARAMETERS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())

    # scheduler option
    if SCHEDULER:
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10)

    print("Ottimizzatore caricato")
    # start = datetime.now()

    # xyz_paths = [f"{DATA_DIR}/{name}" for name in os.listdir(DATA_DIR)]

    # print(len(xyz_paths))
    # val_idx = np.zeros(len(xyz_paths), dtype=bool)
    # true_idx = np.random.choice(
    #     range(len(xyz_paths)), size=math.ceil(0.1 * len(xyz_paths)), replace=False
    # )
    # val_idx[true_idx] = True
    # val_paths = np.asarray(xyz_paths)[val_idx].tolist()
    # train_paths = np.asarray(xyz_paths)[np.logical_not(val_idx)].tolist()

    # train_dataset = MoleculesDataset(train_paths, use_numba=False)
    # val_dataset = MoleculesDataset(val_paths, use_numba=False)
    # print(f"Dataset caricato, durata: {datetime.now() - start}")

    paths = [
        f"/home/lbrodoloni/Larger_Dataset/32grid_pot/all_dataset_train_32/instances/{name}"
        for name in os.listdir(DATA_DIR)
    ]
    test_paths = paths[math.ceil(0.9 * len(paths)) :]
    train_paths = paths[: math.ceil(0.9 * len(paths))]

    train_dataset = MoleculesDataset(train_paths)
    val_dataset = MoleculesDataset(test_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=1
    )

    # train metrics
    train_mae = MeanAbsoluteError().to(device)
    # validation metrics
    val_mae = MeanAbsoluteError().to(device)
    r2_val = R2Score().to(device)
    best_mae = 9999.0
    best_r2 = 0.0
    # print(train_dataset.__getitem__(4))
    pbar = tqdm(range(80), leave=True, disable=DISABLE_BAR)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")

        train_mae.reset()
        model.train()
        train_pbar = tqdm(train_loader, leave=False, disable=DISABLE_BAR)
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            pred = pred.squeeze()
            loss = criterion(pred, target)
            train_mae.update(pred, target)
            train_pbar.set_postfix({"mae": train_mae.compute().item()})
            # backpropagation
            opt.zero_grad()
            loss.backward()
            # optimizer step
            opt.step()
            # pbar.set_postfix({"train_mae": train_mae.compute()})

        val_mae.reset()
        r2_val.reset()
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(val_loader, leave=False, disable=DISABLE_BAR):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                pred = pred.squeeze()

                val_mae.update(pred, target)
                r2_val.update(pred, target)

        pbar.set_postfix(
            {
                "mae": train_mae.compute().item(),
                "val_mae": val_mae.compute().item(),
                "val_r2": r2_val.compute().item(),
            }
        )
        if SCHEDULER:
            scheduler.step()

        if DISABLE_BAR:
            print(
                f"MAE: {train_mae.compute().item()} val_mae: {val_mae.compute().item()} val_r2 {r2_val.compute().item()}"
            )

        if val_mae < best_mae:
            torch.save(model.state_dict(), f"checkpoints/{date_time}/best_mae.ckpt")
            best_mae = val_mae
        if r2_val > best_r2:
            torch.save(model.state_dict(), f"checkpoints/{date_time}/best_r2.ckpt")


if __name__ == "__main__":
    run()
