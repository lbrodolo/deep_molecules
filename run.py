import random

import numpy as np
import torch

# from ignite.contrib.metrics.regression import R2Score
from model import CNN
from tqdm import trange


SEED = 42


def run():
    # reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import model and move it to GPU (if available)
    model = CNN().to(device)
    print(model)

    # import loss and optimizer
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())

    train_set = torch.utils.data.TensorDataset(
        torch.ones((80, 1, 50, 50, 50), dtype=torch.float)
        + torch.arange(80, dtype=torch.float)[..., None, None, None, None],
        torch.arange(80, dtype=torch.float),
    )
    val_set = torch.utils.data.TensorDataset(
        torch.ones((20, 1, 50, 50, 50), dtype=torch.float)
        + torch.arange(20, dtype=torch.float)[..., None, None, None, None],
        torch.arange(20, dtype=torch.float),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=4, shuffle=False, num_workers=4
    )

    for epoch in range(20):
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
            print(f"epoch {epoch} val: {torch.mean(torch.tensor(val_loss))}")


if __name__ == "__main__":
    run()
