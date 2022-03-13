import torch
import numpy as np
import os
from data_gen_numba import read_data  # get_data, gaussian_pot
from multiprocessing import Pool
from typing import List, Tuple


SEED = 42


class MoleculesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
    ):
        # root = "/Users/lucabrodoloni/Desktop/Stage/Vettore_download/data"
        self.xyz_paths = [f"{data_dir}/{name}" for name in os.listdir(data_dir)]
        # print(self.xyz_paths)
        pool = Pool(6)
        dataset = pool.map(read_data, self.xyz_paths)
        train_size = self.__len__ * 0.8
        val_size = self.__len__ - self.__len__ * 0.8
        self.train_data, self.val_data = torch.utils.data.random_split(
            dataset, lengths=[train_size, val_size]
        )

    def __len__(self) -> int:
        return len(self.xyz_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[index]
