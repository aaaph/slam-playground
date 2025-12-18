from dataclasses import dataclass

import numpy as np

import gtsam
from core.transformations.special_euclidian_3_dim import SE3


@dataclass(frozen=True)
class StereoContext:
    """Stereo context."""

    stereo_k: np.ndarray
    cam0_k: np.ndarray
    cam1_k: np.ndarray

    baseline: float

    cam0_in_body: SE3
    cam1_in_body: SE3

    @property
    def stereo_k_gtsam(self) -> gtsam.Cal3_S2Stereo:
        """Get the camera matrix in GTSAM format."""
        k_matrix = self.stereo_k
        baseline = self.baseline
        fx = k_matrix[0, 0]
        fy = k_matrix[1, 1]
        skew = k_matrix[0, 1]
        cx = k_matrix[0, 2]
        cy = k_matrix[1, 2]
        return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)

    @property
    def cam0_k_gtsam(self) -> gtsam.Cal3_S2:
        """Get the camera matrix in GTSAM format."""
        k_matrix = self.cam0_k
        fx = k_matrix[0, 0]
        fy = k_matrix[1, 1]
        skew = k_matrix[0, 1]
        cx = k_matrix[0, 2]
        cy = k_matrix[1, 2]
        return gtsam.Cal3_S2(fx, fy, skew, cx, cy)
