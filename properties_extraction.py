import os
import numpy as np


def properties_extraction(
    train_path: str, test_path: str, property: str, save_dict: bool
) -> None:
    """Extracts molecular properties from .npz dictionaries

    Args:
        train_path (str): Train Path
        test_path (str): Test Path
        property (str): Property to extract: ('size', 'energy', 'ids')
        save_dict (bool): Save .npz (.txt for 'ids') file within the selected property
    """
    train_molecules = os.listdir(train_path)
    test_molecules = os.listdir(test_path)

    if property == "size":
        sizes_x_train = []
        sizes_y_train = []
        sizes_z_train = []
        sizes_train = []
        sizes = []

        for molecules in train_molecules:
            mol = np.load(f"{train_path}{molecules}")
            sizes_x_train.append(mol["sizex"])
            sizes_y_train.append(mol["sizey"])
            sizes_z_train.append(mol["sizez"])
            sizes.append(mol["size"])
        for molecules in test_molecules:
            mol = np.load(f"{test_path}{molecules}")
            sizes_x_train.append(mol["sizex"])
            sizes_y_train.append(mol["sizey"])
            sizes_z_train.append(mol["sizez"])
            sizes.append(mol["size"])

        train_sizes_dict = {
            "sizes_x": sizes_x_train,
            "sizes_y": sizes_y_train,
            "sizes_z": sizes_z_train,
            "size": sizes,
        }
        if save_dict == True:
            np.savez("train_size_dict.npz", **train_sizes_dict)

    elif property == "energy":

        energy_0k = []
        diff_energy = []
        for molecules in train_molecules:
            mol = np.load(f"{train_path}{molecules}")
            energy_0k.append(mol["internal_energy0k"])
            diff_energy.append(mol["difference_energy"])
        for molecules in test_molecules:
            mol = np.load(f"{test_path}{molecules}")
            energy_0k.append(mol["internal_energy0k"])
            diff_energy.append(mol["difference_energy"])

        energies_dict = {
            "internal_energy0k": energy_0k,
            "difference_energy": diff_energy,
        }
        if save_dict == True:
            np.savez("energies_dict.npz", **energies_dict)

    elif property == "ids":

        ids = []
        for molecules in train_molecules:
            mol = np.load(f"{train_path}{molecules}")
            ids.append["id"]
        for molecules in test_molecules:
            mol = np.load(f"{test_path}{molecules}")
            ids.append["id"]

        if save_dict == True:
            p.savetxt("ids.txt", ids, fmt="%i")


if __name__ == "__main__":
    properties_extraction("energy", save_dict=True)
