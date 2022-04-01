from datetime import datetime
from typing import List
import os

import numpy as np
import torch
from tqdm import tqdm
from data_gen import read_data, new_read_data  # get_data, gaussian_pot

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

        self.xyz_paths = xyz_paths

        # for path in tqdm(self.xyz_paths, leave=False):
        #     dataset.append(
        #         read_data(
        #             path,
        #             angles=None,
        #             n_points=n_points,
        #             gamma=gamma,
        #             use_numba=use_numba,
        #         )

        #     )

        # dataset = new_read_data(
        #     self.xyz_paths,
        #     angles=None,
        #     n_points=n_points,
        #     gamma=gamma,
        #     use_numba=use_numba,
        # )

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
        el = np.load(
            f"/home/lbrodoloni/Larger_Dataset/32grid_pot/all_dataset_train_32/instances/mol_{index}.npz"
        )
        pot, diff_eng = torch.from_numpy(el["potential"]), torch.from_numpy(
            el["difference_energy"]
        )
        return pot.float(), diff_eng.float()
        # return self.dataset[index]

    def async_callback(self, res):
        # print("stocazzo")
        RESULTS.append(res)
