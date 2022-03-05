import numpy as np


def Rx(alpha):
    return np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def Ry(beta):
    return np.matrix(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )


def Rz(gamma):
    return np.matrix(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )


def Rotation(alpha: float, beta: float, gamma: float) -> np.ndarray:
    return Rz(gamma) * Ry(beta) * Rx(alpha)


# da implementare rotazioni random
def main():

    x = np.random.rand(3, 3)
    gamma = np.pi / 2
    beta = np.pi
    alpha = np.pi / 4
    print(Rotation(alpha, beta, gamma) * x)


if __name__ == "__main__":
    main()
