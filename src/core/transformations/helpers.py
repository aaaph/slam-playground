import numpy as np


def omega(w: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 skew-symmetric matrix from a 3D vector w.

    # [ -|w x| w ]
    # [  -w    0 ]
    """
    omega = np.zeros((4, 4))
    omega[:3, :3] = -skew(w)
    omega[3, :3] = -w
    omega[:3, 3] = w
    return omega


def skew(vector: np.ndarray) -> np.ndarray:
    """
    Create a 3x3 skew-symmetric matrix from a 3D vector.

    # [ 0 -z y ]
    # [ z  0 -x ]
    # [ -y x  0 ]
    """
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
