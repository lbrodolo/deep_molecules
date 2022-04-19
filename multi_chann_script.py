import os
from typing import Tuple

import numpy as np
import torch

from data_gen import get_data


def multi_potential(
    atomic_info: np.ndarray,
    gamma: float = 0.36,
    alpha: float = 6.0,
    n_points: int = 32,
    physic_length: float = 12.0,
) -> np.ndarray:

    """Create a gaussian potential for a given atomic configuration.
        height and variance of the gaussian are fixed parameters.

    Args:
        atomic_info (np.ndarray): atomic coordinates and atomic number.
        gamma (float, optional): variance of the gaussian. Defaults to 0.36.
        alpha (float, optional): constant (height). Defaults to 6.0.
        n_points (int, optional): number of points for each side of the cube. Defaults to 32.
        physic_length (float, optional): physical length of each side of the cube. Defaults to 12.0 Å.

    Returns:
        np.ndarray: 3D array with the potential.
    """

    V_pot = np.zeros((n_points, n_points, n_points))
    grid_space = physic_length / n_points
    n_atoms = atomic_info.shape[0]
    offset = physic_length / 2
    offset_x = (abs(max(atomic_info[:, 0])) - abs(min(atomic_info[:, 0]))) / 2
    offset_y = (abs(max(atomic_info[:, 1])) - abs(min(atomic_info[:, 1]))) / 2
    offset_z = (abs(max(atomic_info[:, 2])) - abs(min(atomic_info[:, 2]))) / 2
    for i in range(n_points):
        for j in range(n_points):
            for k in range(n_points):
                V_term_space = np.zeros(n_atoms)
                for l in range(n_atoms):
                    V_term_space[l] = alpha * np.exp(
                        (-1.0 / (2 * (gamma) ** 2))
                        * (
                            (((i * grid_space) - offset + offset_x) - atomic_info[l, 0])
                            ** 2
                            + (
                                ((j * grid_space) - offset + offset_y)
                                - atomic_info[l, 1]
                            )
                            ** 2
                            + (
                                ((k * grid_space) - offset + offset_z)
                                - atomic_info[l, 2]
                            )
                            ** 2
                        )
                    )
                # sum the terms in the function and make it single precision.
                V_pot[i, j, k] = np.single(np.sum(V_term_space))
    return V_pot


def channel_potential_generator(path: str) -> Tuple[torch.Tensor, torch.Tensor]:

    """Create a multichannel potential and return Difference Energy:
    CHANNEL 0: H potential
    CHANNEL 1: C potential
    CHANNEL 2: N potential
    CHANNEL 3: O potential
    CHANNEL 4: F potential

    Returns:
        Tuple(torh.Tensor, torh.Tensor): Multichannel potential, Difference Energy.
    """

    mol = get_data(path)
    n_atoms = mol[0]
    difference_energy = mol[5]
    difference_energy = torch.tensor(difference_energy)
    frame = mol[2]
    # multichannel potential initialization
    multichannel_potential = np.zeros((5, 32, 32, 32))
    H_coord = []  # canale 0
    C_coord = []  # canale 1
    N_coord = []  # canale 2
    O_coord = []  # canale 3
    F_coord = []  # canale 4
    for i in range(0, len(frame)):
        if frame[i][0] == 1.0:
            H_coord.append(frame[i][1:4])
        elif frame[i][0] == 6.0:
            C_coord.append(frame[i][1:4])
        elif frame[i][0] == 7.0:
            N_coord.append(frame[i][1:4])
        elif frame[i][0] == 8.0:
            O_coord.append(frame[i][1:4])
        elif frame[i][0] == 9.0:
            F_coord.append(frame[i][1:4])
    if len(H_coord) != 0:
        # list -> array
        H_coord = np.array(H_coord).reshape(len(H_coord), 3)
        # compute the potential
        H_pot = multi_potential(atomic_info=H_coord)
        # add the potential to the multichannel potential
        multichannel_potential[0] = H_pot
    else:
        # if there are no H atoms, add a zero potential
        multichannel_potential[0] = np.zeros((32, 32, 32))
    if len(C_coord) != 0:
        C_coord = np.array(C_coord).reshape(len(C_coord), 3)
        C_pot = multi_potential(atomic_info=C_coord)
        multichannel_potential[1] = C_pot
    else:
        multichannel_potential[1] = np.zeros((32, 32, 32))
    if len(N_coord) != 0:
        N_coord = np.array(N_coord).reshape(len(N_coord), 3)
        N_pot = multi_potential(atomic_info=N_coord)
        multichannel_potential[2] = N_pot
    else:
        multichannel_potential[2] = np.zeros((32, 32, 32))
    if len(O_coord) != 0:
        O_coord = np.array(O_coord).reshape(len(O_coord), 3)
        O_pot = multi_potential(atomic_info=O_coord)
        multichannel_potential[3] = O_pot
    else:
        multichannel_potential[3] = np.zeros((32, 32, 32))
    if len(F_coord) != 0:
        F_coord = np.array(F_coord).reshape(len(F_coord), 3)
        F_pot = multi_potential(atomic_info=F_coord)
        multichannel_potential[4] = F_pot
    else:
        multichannel_potential[4] = np.zeros((32, 32, 32))
    # print(f"Hydrogen coordinates: \n{len(H_coord)}")
    # H_coord = np.array(H_coord).reshape(len(H_coord), 3)
    # print(H_coord)
    # print(f"Carbon coordinates: \n{C_coord}")
    # print(f"Nitrogen coordinates: \n{N_coord}")
    # print(f"Oxigen coordinates: \n{O_coord}")
    # print(f"Fluorine coordinates: \n{F_coord}")
    multichannel_potential = torch.from_numpy(multichannel_potential)
    return multichannel_potential, difference_energy


if __name__ == "__main__":
    multichannel_potential = main(
        "/Users/lucabrodoloni/Desktop/Stage/Vettore_download/dati_puliti/train/dsgdb9nsd_000017.xyz"
    )
    np.save(
        "/Users/lucabrodoloni/Desktop/multichannel_potential.npy",
        multichannel_potential,
    )

    print(f"Potential has shape: {multichannel_potential.shape}")
    print(f"Potential has max value: {np.max(multichannel_potential)}")
    print(f"Potential has min value: {np.min(multichannel_potential)}")
    print(f"Potential has mean value: {np.mean(multichannel_potential)}")
