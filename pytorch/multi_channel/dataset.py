from datetime import datetime
from typing import List
import numpy as np
import torch
from tqdm import tqdm
from multi_chann_gen import channel_potential_generator

SEED = 42


class MultichannelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xyz_paths: List[str],
        n_points: int = 32,
        physic_length: float = 12.0,
        gamma: float = 0.375,
        alpha: float = 6.0,
    ):
        start_time = datetime.now()
        self.xyz_paths = xyz_paths
        dataset = []
        print(f"Start loading dataset...")
        for path in self.xyz_paths:
            dataset.append(
                channel_potential_generator(path, gamma, alpha, n_points, physic_length)
            )
        self.dataset = dataset
        print(f"Multi-channel dataset loaded. \nTime elapsed: {datetime.now() - start_time}")
    
    def __len__(self):
        return len(self.xyz_paths)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
        