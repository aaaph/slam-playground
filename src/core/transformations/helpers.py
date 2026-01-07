import numpy as np

from core.transformations.special_euclidian_3_dim import SE3


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


def ate(estimated_pose: SE3, ground_truth_pose: SE3) -> float:
    """Calculate the absolute position error between two poses."""
    result = np.linalg.norm(estimated_pose.translation() - ground_truth_pose.translation())
    return float(result)


def are(estimated_pose: SE3, ground_truth_pose: SE3) -> float:
    """
    Calculate the absolute rotation error between two poses.

    Args:
        estimated_pose: The estimated pose.
        ground_truth_pose: The ground truth pose.

    Returns:
        The absolute rotation error in degrees.

    """
    rel_pose = ground_truth_pose.inverse() * estimated_pose
    result = np.degrees(np.linalg.norm(rel_pose.rotation().as_rotvec()))
    return float(result)


def calculate_ape(estimated_pose: SE3, ground_truth_pose: SE3) -> tuple[float, float]:
    """Calculate the absolute position and rotation error between two poses."""
    return ate(estimated_pose, ground_truth_pose), are(estimated_pose, ground_truth_pose)


def rte(estimated_delta: SE3, ground_truth_delta: SE3) -> float:
    """
    Calculate the relative translation error between two poses.

    Args:
        estimated_delta: The estimated delta.
        ground_truth_delta: The ground truth delta.

    Returns:
        The relative translation error.

    """
    error_matrix = ground_truth_delta.inverse() * estimated_delta
    result = np.linalg.norm(error_matrix.translation())
    return float(result)


def rre(estimated_delta: SE3, ground_truth_delta: SE3) -> float:
    """
    Calculate the relative rotation error between two poses.

    Args:
        estimated_delta: The estimated delta.
        ground_truth_delta: The ground truth delta.

    Returns:
        The relative rotation error.

    """
    error_matrix = ground_truth_delta.inverse() * estimated_delta
    angle_rad = np.linalg.norm(error_matrix.rotation().as_rotvec())
    return float(np.degrees(angle_rad))


def calculate_rpe(estimated_delta: SE3, ground_truth_delta: SE3) -> tuple[float, float]:
    """Calculate the relative position and rotation error between two poses."""
    return rte(estimated_delta, ground_truth_delta), rre(estimated_delta, ground_truth_delta)
