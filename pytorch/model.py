import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    def __init__(
        self,
    ):
        super(CNN, self).__init__()

        self.conv = nn.ModuleList()
        # Nx1x32x32x32
        self.conv.append(nn.Conv3d(1, 8, kernel_size=3, padding='valid'))
        self.conv.append(nn.ELU())
        # Nx8x30x30x30
        self.conv.append(nn.Conv3d(8, 16, kernel_size=3, padding='valid'))
        self.conv.append(nn.ELU())
        # Nx16x28x28x28
        self.conv.append(nn.AvgPool3d(kernel_size=2))
        # Nx16x14x14x14
        self.conv.append(nn.Conv3d(16, 32, kernel_size=3, padding='valid'))
        self.conv.append(nn.ELU())
        # Nx32x12x12x12
        self.conv.append(nn.AvgPool3d(kernel_size=2))
        # Nx32x6x6x6
        self.conv = nn.Sequential(*self.conv)

        # Nx8x6x6x6
        self.dense = nn.ModuleList()
        self.dense.append(nn.Linear(6912, 16))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(16, 8))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(8, 8))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(8, 4))
        self.dense.append(nn.ELU())
        self.dense.append(nn.Linear(4, 1))

        self.dense = nn.Sequential(*self.dense)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 6912)
        x = self.dense(x)
        return x

    def initialize_weights(m):
        if isinstance(m, nn.Conv3d):
            #nn.init.xavier_uniform_(m.weight.data, nonlinearity='elu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            #nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)