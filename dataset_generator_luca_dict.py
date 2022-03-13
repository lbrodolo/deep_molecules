import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import rotate
import math as m
from numba import jit
from typing import AnyStr

"""Set the number of grid points in the x,y,z direction"""
Nx = 50  # Number of points in x-direction
Ny = 50  # Number of points in y-direction
Nz = 50  # Number of points in z-direction
"""Hyper parameter for psuedo potential"""
gamma = 0.36  # Angstrom
"""Set the grid spacing - smaller provides a finer grid"""
grid_space = 0.155  # Angstrom
"""Physical length if the grid in Angstrom"""
Lx = Nx * grid_space  # Angstrom
Ly = Ny * grid_space  # Angstrom
Lz = Nz * grid_space  # Angstrom
"""To offset the 3D grid 0,0,0 to the center of cube"""
offset = Lx / 2  # The offset for "COM" coordinates.

int_energy_data = []
int_energy_per_atom_data = []
sum_ind_atom_energies = []
difference_energy = []
id_numbers = []


def get_data(file_name: AnyStr) -> np.array:

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


@jit(nopython=True)
def gaussian_pot(
    Nx,
    Ny,
    Nz,
    no_atoms,
    atomic_info,
    gamma,
    grid_space,
    offset_x,
    offset_y,
    offset_z,
    id_number,
):

    V_pot = np.zeros((Nx, Ny, Nz))  # Initalise the grid for Gaussian Potential

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # create a space for the summation term of length according to number of atoms.
                V_term_space = np.zeros(no_atoms)

                for l in range(no_atoms):

                    V_term_space[l] = atomic_info[l, 0] * np.exp(
                        (-1.0 / (2 * (gamma) ** 2))
                        * (
                            (((i * grid_space) - offset + offset_x) - atomic_info[l, 1])
                            ** 2
                            + (
                                ((j * grid_space) - offset + offset_y)
                                - atomic_info[l, 2]
                            )
                            ** 2
                            + (
                                ((k * grid_space) - offset + offset_z)
                                - atomic_info[l, 3]
                            )
                            ** 2
                        )
                    )

                # sum the terms in the function and make it single precision.
                V_pot[i, j, k] = np.single(np.sum(V_term_space))

                # np.savetxt('Gaussian_potential'+str(id_number)'.txt',V_pot)
    return V_pot


def Rx(phi):
    return np.matrix(
        [[1, 0, 0], [0, m.cos(phi), -m.sin(phi)], [0, m.sin(phi), m.cos(phi)]]
    )


def Ry(theta):
    return np.matrix(
        [[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]]
    )


def Rz(psi):
    return np.matrix(
        [[m.cos(psi), -m.sin(psi), 0], [m.sin(psi), m.cos(psi), 0], [0, 0, 1]]
    )


def rotate_matrix(atomic_info, phi, theta, psi):

    R = Rx(phi) * Ry(theta) * Rz(psi)

    for l in range(len(atomic_info[:, 0])):
        atomic_info[l, 1:4] = atomic_info[l, 1:4] * R
    return atomic_info


list_id = []

for i in range(1, 133886):
    list_id.append(f"{i}".zfill(6))

k = -1
for i in list_id:
    try:
        mol_1 = get_data(
            f"/Users/lucabrodoloni/Desktop/Stage/Dataset/dsgdb9nsd_{i}.xyz"
        )
        no_atoms = mol_1[0]
        internal_energy = mol_1[1]
        atomic_info = mol_1[2]
        id_number = mol_1[3]
        ind_energy = mol_1[4]
        diff_energy = mol_1[5]
        offset_x = np.sum(atomic_info[:, 1]) / len(atomic_info[:, 1])
        offset_y = np.sum(atomic_info[:, 2]) / len(atomic_info[:, 2])
        offset_z = np.sum(atomic_info[:, 3]) / len(atomic_info[:, 3])

        sizex = np.amax(atomic_info[:, 1]) - np.amin(atomic_info[:, 1])
        sizey = np.amax(atomic_info[:, 2]) - np.amin(atomic_info[:, 2])
        sizez = np.amax(atomic_info[:, 3]) - np.amin(atomic_info[:, 3])

        size = np.sqrt(sizex**2.0 + sizey**2.0 + sizez**2.0)

        if sizex <= 5.5 and sizey <= 5.5 and sizez <= 5.5:
            k += 1
            my_pot = gaussian_pot(
                Nx,
                Ny,
                Nz,
                no_atoms,
                atomic_info,
                gamma,
                grid_space,
                offset_x,
                offset_y,
                offset_z,
                id_number,
            )

            mol_dict = {
                "id": id_number,
                "potential": my_pot,
                "difference_energy": diff_energy,
                "coordinates": atomic_info[:, 1:],
                "size": size,
                "internal_energy0k": internal_energy,
                "atomic_numbers": atomic_info[:, 0],
            }
            np.savez(f"/Users/lucabrodoloni/Desktop/split_temp/mol_{k}.npz", **mol_dict)

            del my_pot
            del offset_x
            del offset_y
            del offset_z
            int_energy_data.append(internal_energy)
            int_energy_per_atom_data.append(internal_energy / no_atoms)
            sum_ind_atom_energies.append(ind_energy)
            difference_energy.append(diff_energy)
            id_numbers.append(id_number)

            for j in range(1):
                k += 1

                phi = (m.pi * np.random.randint(11)) / (np.random.randint(10) + 1)
                theta = (m.pi * np.random.randint(11)) / (np.random.randint(10) + 1)
                psi = (m.pi * np.random.randint(11)) / (np.random.randint(10) + 1)

                atomic_info = rotate_matrix(atomic_info, phi, theta, psi)

                offset_x = np.sum(atomic_info[:, 1]) / len(atomic_info[:, 1])
                offset_y = np.sum(atomic_info[:, 2]) / len(atomic_info[:, 2])
                offset_z = np.sum(atomic_info[:, 3]) / len(atomic_info[:, 3])

                my_pot = gaussian_pot(
                    Nx,
                    Ny,
                    Nz,
                    no_atoms,
                    atomic_info,
                    gamma,
                    grid_space,
                    offset_x,
                    offset_y,
                    offset_z,
                    id_number,
                )
                np.save(
                    f"/Volumes/T7/Quantum_molecules_stuff/Datasets/Potentials_5_5A_1rot/pot_{str(k)}.npy",
                    my_pot,
                )

                del my_pot
                del offset_x
                del offset_y
                del offset_z
                int_energy_data.append(internal_energy)
                int_energy_per_atom_data.append(internal_energy / no_atoms)
                sum_ind_atom_energies.append(ind_energy)
                difference_energy.append(diff_energy)
                id_numbers.append(id_number)
    except:
        pass


np.savetxt("internal_energies_medium_atoms.txt", int_energy_data, fmt="%1.7f")
np.savetxt(
    "internal_energies_per_atom_medium_atoms.txt", int_energy_per_atom_data, fmt="%1.7f"
)
np.savetxt("individual_atoms_energy_summed.txt", sum_ind_atom_energies, fmt="%1.7f")
np.savetxt("difference_gs_summed_energies.txt", difference_energy, fmt="%1.7f")
np.savetxt("id_numbers.txt", id_numbers, fmt="%i")
