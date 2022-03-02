import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    def __init__(
        self,
    ):
        super(CNN, self).__init__()

        self.conv = nn.ModuleList()
        # Nx1x50x50x50
        self.conv.append(nn.Conv3d(1, 8, kernel_size=3))
        self.conv.append(nn.ELU())
        # Nx8x48x48x48
        self.conv.append(nn.Conv3d(8, 16, kernel_size=3, padding=1))
        self.conv.append(nn.ELU())
        # Nx16x48x48x48
        self.conv.append(nn.AvgPool3d(kernel_size=2))
        # Nx16x24x24x24
        self.conv.append(nn.Conv3d(16, 32, kernel_size=3, padding=1))
        self.conv.append(nn.ELU())
        # Nx32x24x24x24
        self.conv.append(nn.AvgPool3d(kernel_size=2))

        self.conv = nn.Sequential(*self.conv)

        # Nx32x12x12x12
        self.dense = nn.ModuleList()
        self.dense.append(nn.Linear(55296, 512))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(512, 256))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(256, 128))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(128, 1))

        self.dense = nn.Sequential(*self.dense)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 55296)
        x = self.dense(x)
        return x
