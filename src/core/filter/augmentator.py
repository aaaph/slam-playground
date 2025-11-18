import numpy as np
from jax.scipy.spatial.transform import Rotation

from core.filter.state import CameraClone, State
from core.transformations.helpers import skew

t_imu_cam0 = np.array(
    [
        [0.014865542981794, 0.999557249008346, -0.025774436697440, 0.065222909535531],
        [-0.999880929698575, 0.014967213324719, 0.003756188357967, -0.020706385492719],
        [0.004140296794224, 0.025715529947966, 0.999660727177902, -0.008054602460030],
        [0, 0, 0, 1.000000000000000],
    ]
)
t_cam0_imu = np.linalg.inv(t_imu_cam0)
rotation_imu_cam0 = t_cam0_imu[:3, :3].T
translation_cam0_imu = t_cam0_imu[:3, 3]


class Augmentator:
    """Augmentator of the Multi-State Constraint Kalman Filter."""

    def __init__(self) -> None:
        """Initialize the augmentator."""

    def augment_clone(self, state: State) -> tuple[CameraClone, State]:
        """Augment the state."""
        pose = np.array(state.inertial_state.get_pose())
        timestamp = state.ts
        exist = state.sliding_window.get_by_timestamp(timestamp)
        if exist is not None:
            msg = f"Timestamp {timestamp} already exists in the sliding window"
            raise ValueError(msg)
        clone = state.sliding_window.add(timestamp, pose)

        rot_i_to_c0 = Rotation.from_matrix(rotation_imu_cam0).as_matrix()
        rot_w_to_i = Rotation.from_quat(state.inertial_state.q).as_matrix()
        p_c_in_i = translation_cam0_imu
        """
        J = [
            [∂p_c/∂p_i, ∂p_c/∂q_i, ∂p_c/∂v_i, ∂p_c/∂b_a, ∂p_c/∂b_g]
            [∂q_c/∂p_i, ∂q_c/∂q_i, ∂q_c/∂v_i, ∂q_c/∂b_a, ∂q_c/∂b_g]
        ]
        """
        j = np.zeros((6, 15))
        j[:3, 3:6] = rot_i_to_c0  # ∂p_c/∂q_i
        j[3:6, 3:6] = rot_w_to_i @ skew(p_c_in_i)  # ∂q_c/∂q_i
        j[3:6, 0:3] = np.eye(3)  # ∂q_c/∂p_i

        prev_row_count = state.covariance.sigma.shape[0]
        new_sigma = np.zeros((prev_row_count + 6, prev_row_count + 6))
        new_sigma[:prev_row_count, :prev_row_count] = state.covariance.sigma
        j_full = np.zeros((6, prev_row_count))
        j_full[:, :15] = j

        pjt = state.covariance.sigma @ j_full.T
        new_sigma[:prev_row_count, prev_row_count:] = pjt
        new_sigma[prev_row_count:, :prev_row_count] = pjt.T
        new_sigma[prev_row_count:, prev_row_count:] = j_full @ state.covariance.sigma @ j_full.T

        new_sigma = (new_sigma + new_sigma.T) / 2
        state.apply_covariance(new_sigma)

        return clone, state
