import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    def __init__(
        self,
    ):
        super(CNN, self).__init__()

        self.conv = nn.ModuleList()
        # Nx32x32x32
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3, padding="valid"))
        self.conv.append(nn.ELU())
        # Nx32x30x30
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3, padding="valid"))
        self.conv.append(nn.ELU())
        # Nx32x28x28
        self.conv.append(nn.AvgPool2d(kernel_size=2))
        # Nx32x14x14
        self.conv.append(nn.Conv2d(32, 32, kernel_size=3, padding="valid"))
        self.conv.append(nn.ELU())
        # Nx32x12x12
        self.conv.append(nn.AvgPool2d(kernel_size=2))
        # Nx32x6x6
        self.conv = nn.Sequential(*self.conv)

        # Nx1152
        self.dense = nn.ModuleList()
        self.dense.append(nn.Linear(1152, 128))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(128, 64))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(64, 1))

        self.dense = nn.Sequential(*self.dense)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 1152)
        x = self.dense(x)
        return x

    def initialize_weights(m):
        if isinstance(m, nn.Conv3d):
            # nn.init.xavier_uniform_(m.weight.data, nonlinearity='elu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
