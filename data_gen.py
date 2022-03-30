from datetime import datetime
from multiprocessing import Pool
from typing import Tuple, Optional

import numpy as np
import torch
from data_gen_vec import np_gaussian_pot
from numba import njit
from scipy.spatial.transform import Rotation as R
from torch.functional import Tensor

Nx = 50  # Number of points in x-direction
Ny = 50  # Number of points in y-direction
Nz = 50  # Number of points in z-direction
gamma = 0.36  # Angstrom
old_grid_space = 0.155  # Angstrom
int_energy_data = []
int_energy_per_atom_data = []
sum_ind_atom_energies = []
difference_energy = []
id_numbers = []

Lx = Nx * old_grid_space  # Angstrom
Ly = Ny * old_grid_space  # Angstrom
Lz = Nz * old_grid_space  # Angstrom

old_offset = Lx / 2  # The offset for "COM" coordinates.


def get_data(file_name):
    fin = open(file_name)
    first = fin.readline().split()
    no_atoms = int(first[0])
    frame = np.zeros((no_atoms, 4))
    info = fin.readline().split()
    internal_energy_0K = float(info[12])
    id_number = int(info[1])
    temp_dict = []
    ind_energy = 0.0

    for i in range(no_atoms):
        temp_dict = []
        temp_dict = fin.readline().split()
        # Here we check the atomic species and assign the atomic number.
        if temp_dict[0] == "H":
            frame[i, 0] = 1
            ind_energy += -0.5
        elif temp_dict[0] == "C":
            frame[i, 0] = 6
            ind_energy += -37.8450
        elif temp_dict[0] == "N":
            frame[i, 0] = 7
            ind_energy += -54.5892
        elif temp_dict[0] == "O":
            frame[i, 0] = 8
            ind_energy += -75.0673
        elif temp_dict[0] == "F":
            frame[i, 0] = 9
            ind_energy += -99.7339

        frame[i, 1] = np.single(temp_dict[1])  # Insert x coordinate
        frame[i, 2] = np.single(temp_dict[2])  # Insert y coordinate
        frame[i, 3] = np.single(temp_dict[3])  # Insert z coordinate
        diff_energy = internal_energy_0K - ind_energy

    return no_atoms, internal_energy_0K, frame, id_number, ind_energy, diff_energy


@njit
def numba_gaussian_pot(
    atomic_info,
    gamma: float = 0.36,
    n_points: int = 50,
    physic_len: float = 7.75,
) -> np.ndarray:
    # Initalise the grid for Gaussian Potential
    V_pot = np.zeros((n_points, n_points, n_points))
    # initialize n_atoms
    n_atoms = atomic_info.shape[0]
    # get grid space
    grid_space = physic_len / n_points
    # compute offset
    offset = physic_len / 2

    for i in range(n_points):
        for j in range(n_points):
            for k in range(n_points):
                # create a space for the summation
                #  according to the number of atoms.
                V_term_space = np.zeros(n_atoms)

                for l in range(n_atoms):

                    V_term_space[l] = atomic_info[l, 0] * np.exp(
                        (-1.0 / (2 * (gamma) ** 2))
                        * (
                            (((i * grid_space) - offset) - atomic_info[l, 1]) ** 2
                            + (((j * grid_space) - offset) - atomic_info[l, 2]) ** 2
                            + (((k * grid_space) - offset) - atomic_info[l, 3]) ** 2
                        )
                    )
                # sum the terms in the function and make it single precision.
                V_pot[i, j, k] = np.single(np.sum(V_term_space))
    return V_pot


def np_gaussian_pot(
    atomic_info: np.ndarray,
    gamma: float = 0.36,
    physic_len: float = 7.75,
    n_points: int = 50,
) -> np.ndarray:

    x = np.linspace(0, physic_len, n_points, endpoint=False)
    y = np.linspace(0, physic_len, n_points, endpoint=False)
    z = np.linspace(0, physic_len, n_points, endpoint=False)
    # atom_info N x 3, only coordinates
    atom_info = atomic_info[:, 1:]
    # zeta N
    zeta = atomic_info[:, 0]

    xxx, yyy, zzz = np.meshgrid(x, y, z)
    xxx, yyy, zzz = xxx - x.max() / 2, yyy - y.max() / 2, zzz - z.max() / 2

    square_norm = (
        (xxx[None, ...] - atom_info[:, 0][..., None, None, None]) ** 2
        + (yyy[None, ...] - atom_info[:, 1][..., None, None, None]) ** 2
        + (zzz[None, ...] - atom_info[:, 2][..., None, None, None]) ** 2
    )
    pot = np.sum(
        zeta[..., None, None, None] * np.exp(-square_norm / (2 * gamma ** 2)), axis=0
    )

    return pot


def read_data(
    path: str,
    n_points: int = 50,
    gamma: float = 0.36,
    angles: Optional[Tuple[float, float, float]] = None,
    use_numba: bool = False,
) -> Tuple[Tensor, Tensor]:
    my_pot = None
    diff_energy = None

    mol = get_data(path)
    atomic_info = mol[2]
    diff_energy = torch.tensor(mol[5])

    if angles is not None:
        rot = R.from_euler("zyx", angles, degrees=False)
        atomic_info[..., 1:] = rot.apply(atomic_info[..., 1:])

    if use_numba:
        my_pot = numba_gaussian_pot(atomic_info, n_points=n_points, gamma=gamma)
    else:
        my_pot = np_gaussian_pot(atomic_info, n_points=n_points, gamma=gamma)

    my_pot = torch.from_numpy(my_pot)
    my_pot = my_pot.unsqueeze(0).float()

    return my_pot, diff_energy


if __name__ == "__main__":

    list_id = [f"{i}".zfill(6) for i in range(1000)]
    # for i in range(1, 133886):
    start_time = datetime.now()
    pool = Pool(6)

    # read_data(i)
    #     pool.apply_async(read_data, args=(i,))
    results = pool.map(read_data, list_id)

    print(len(results), type(results))
    print(f"Duration {datetime.now() - start_time}")
