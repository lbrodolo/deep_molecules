import torch.nn as nn
from torch import Tensor


class MultichannelCNN(nn.Module):
    def __init__(
        self,
    ):
        super(MultichannelCNN, self).__init__()
        
        # Convolutional layers
        self.conv = nn.ModuleList()
        # 5x32x32x32
        self.conv.append(nn.Conv3d(5, 8, kernel_size=3, padding="valid"))
        self.conv.append(nn.ELU())
        # 8x30x30x30
        self.conv.append(nn.Conv3d(8, 16, kernel_size=3, padding="valid"))
        self.conv.append(nn.ELU())
        # 16x28x28x28
        self.conv.append(nn.AvgPool3d(kernel_size=2))
        # 16x14x14x14
        self.conv.append(nn.Conv3d(16, 32, kernel_size=3, padding="valid"))
        self.conv.append(nn.ELU())
        # 32x12x12x12
        self.conv.append(nn.AvgPool3d(kernel_size=2))
        # 32x6x6x6
        self.conv = nn.Sequential(*self.conv)
        
        # Dense layers
        self.dense = nn.ModuleList()
        self.dense.append(nn.Linear(32*6*6*6, 128))
        self.dense.append(nn.ELU())
        self.dense.append(16, 8)
        self.dense.append(nn.ELU())
        self.dense.append(8, 8)
        self.dense.append(nn.ELU())
        self.dense.append(8, 4)
        self.dense.append(nn.ELU())
        self.dense.append(4, 1)
        
        self.dense = nn.Sequential(*self.dense)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(-1, 32*6*6*6)
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
        