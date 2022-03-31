import os
from datetime import datetime
from multiprocessing import Pool
from typing import List

import numpy as np
import torch
from data_gen import read_data  # get_data, gaussian_pot

SEED = 42
RESULTS = []


class MoleculesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        xyz_paths: List[str],
        n_points: int = 32,
        gamma: float = 0.36,
        use_numba: bool = True,
        # data_dir: str,
    ):
        global RESULTS
        # root = "/Users/lucabrodoloni/Desktop/Stage/Vettore_download/data"
        self.xyz_paths = xyz_paths
        # = [f"{data_dir}/{name}" for name in os.listdir(data_dir)]

        # pool = Pool(3)
        dataset = []
        print("Pool selezionato... \nIn attesa di 'map'...")
        start_time = datetime.now()
        for path in self.xyz_paths:
            dataset.append(
                read_data(
                    path,
                    angles=None,
                    n_points=n_points,
                    gamma=gamma,
                    use_numba=use_numba,
                )
            )

        # for path in self.xyz_paths:
        #     pool.apply_async(read_data, args=(path,))  # , callback=self.async_callback)
        # pool.close()
        # pool.join()

        # dataset = RESULTS
        RESULTS = []

        self.dataset = dataset
        print("Pool map eseguito")
        print(f"Durata: {datetime.now() - start_time}")

    def __len__(self) -> int:
        return len(self.xyz_paths)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def async_callback(self, res):
        # print("stocazzo")
        RESULTS.append(res)
