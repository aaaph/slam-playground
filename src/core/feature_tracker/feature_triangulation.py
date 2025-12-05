import numpy as np
from numpy.typing import NDArray

from core.feature_tracker.feature import Feature


class FeatureTriangulation:
    """Feature triangulation module."""

    def __init__(self, k_matrix: np.ndarray, baseline: float) -> None:
        """Initialize the feature triangulation module."""
        self.k_matrix = k_matrix
        self.baseline = baseline

    def make_initial_guess(self, feature: Feature) -> NDArray[np.float64]:  # shape: (3,)
        """
        Make an initial guess for a feature over last stereo pair.

        Args:
            feature: The feature to make an initial guess for.

        Returns:
            np.ndarray: The initial guess for the feature in camera frame.
            Shape: (3,)

        """
        active_stereo_pair = feature.get_active_stereo_pair()
        if active_stereo_pair is None:
            raise ValueError("Feature has no active stereo pair")
        _, left_uv, right_uv = active_stereo_pair
        if right_uv is None:
            raise ValueError("Feature has no active right point")
        left_u, v = left_uv
        right_u = right_uv[0]
        fx = self.k_matrix[0, 0]
        fy = self.k_matrix[1, 1]
        cx = self.k_matrix[0, 2]
        cy = self.k_matrix[1, 2]
        baseline = self.baseline
        disp = left_u - right_u
        if disp <= 0:
            raise ValueError("Disparity is non-positive")
        z = fx * baseline / disp
        x = (left_u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])
