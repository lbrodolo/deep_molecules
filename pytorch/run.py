import os
import random
import numpy as np
import torch
from torchmetrics import R2Score, MeanAbsoluteError
from sklearn.metrics import mean_absolute_error
from torchmetrics.functional import r2_score
from model import CNN
from tqdm import trange, tqdm
from torch_dataset import MoleculesDataset
import math
from datetime import datetime

SEED = 42
DATA_DIR = "/home/lbrodoloni/data/train"

def run():
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

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.NAdam(model.parameters(), lr=1e-4)

    print("Ottimizzatore caricato")
    start = datetime.now()

    xyz_paths = [f"{DATA_DIR}/{name}" for name in os.listdir(DATA_DIR)]

    # print(len(xyz_paths))
    val_idx = np.zeros(len(xyz_paths), dtype=bool)
    true_idx = np.random.choice(
        range(len(xyz_paths)), size=math.ceil(0.1 * len(xyz_paths)), replace=False
    )
    val_idx[true_idx] = True
    val_paths = np.asarray(xyz_paths)[val_idx].tolist()
    train_paths = np.asarray(xyz_paths)[np.logical_not(val_idx)].tolist()

    train_dataset = MoleculesDataset(train_paths)
    val_dataset = MoleculesDataset(val_paths)
    print(f"Dataset caricato, durata: {datetime.now() - start}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=1
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=1
    )
    val_mae = MeanAbsoluteError().to(device)
    r2_val = R2Score().to(device)
    # print(train_dataset.__getitem__(4))
    for epoch in trange(80):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            pred = pred.squeeze()
            loss = criterion(pred, target)
            # backpropagation
            opt.zero_grad()
            loss.backward()

            # optimizer step
            opt.step()

        val_loss = []
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                pred = pred.squeeze()

                val_loss.append(criterion(pred, target))
                val_mae.update(pred, target)
                r2_val.update(pred, target)
            print(f"epoch {epoch} val: {torch.mean(torch.tensor(val_loss))} val_mae: {val_mae.compute()} R2_val: {r2_val.compute()}")
        val_mae.reset()
        r2_val.reset()

if __name__ == "__main__":
    run()
