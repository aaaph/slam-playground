import numpy as np
import pytest
from numpy.typing import NDArray

from core.pose_tracker.feature_triangulation import FeatureTriangulation
from core.transformations.special_euclidian_3_dim import SE3


class TestFeatureTriangulation:
    """Test feature triangulation module."""

    @staticmethod
    def project_point(feat_in_world_vec: np.ndarray, world_in_camera_se3: SE3, k_matrix: np.ndarray) -> np.ndarray:
        """Project a point from the feature to the camera."""
        feat_in_camera_vec = world_in_camera_se3 @ feat_in_world_vec
        p_norm = feat_in_camera_vec[:2] / feat_in_camera_vec[2]
        uv_homog = k_matrix @ np.array([*p_norm, 1.0])
        return uv_homog[:2]

    @pytest.fixture
    def triangulator(self) -> FeatureTriangulation:
        """Create a feature triangulation module."""
        left_k_matrix = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        right_k_matrix = np.array([[75, 0, 37.5], [0, 75, 37.5], [0, 0, 1]], dtype=float)
        k_matricies = (
            left_k_matrix,
            left_k_matrix,
            right_k_matrix,
        )
        baseline = 0.1
        body_sensor_transforms = (
            SE3.identity() * SE3(t=np.array([0.1, 0.1, 0.1])),
            SE3.identity() * SE3(t=np.array([0.2, 0.2, 0.2])),
        )
        return FeatureTriangulation(k_matricies, baseline, body_sensor_transforms)

    def test_should_be_possible_to_create(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can be created."""
        assert triangulator is not None
        assert triangulator.k_stereo is not None
        assert triangulator.k_left is not None
        assert triangulator.k_right is not None
        assert triangulator.k_stereo_inv is not None
        assert triangulator.k_left_inv is not None
        assert triangulator.k_right_inv is not None
        assert triangulator.body_in_cam0 is not None
        assert triangulator.body_in_cam1 is not None

    def test_make_initial_guess_by_stereo_batch(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can make an initial guess by stereo batch."""
        stereo_tensor = np.array([[1, 100, 115, 95, 110], [2, 100, 115, 95, 110]])
        result = triangulator.make_initial_guess_by_stereo_batch(stereo_tensor)
        assert result is not None
        assert result.shape == (2, 5)
        assert np.allclose(result[:, 4], np.array([1, 1]), atol=1e-5)

    def test_make_initial_guess_by_stereo_batch_invalid(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can make an initial guess by stereo batch."""
        stereo_tensor = np.array([[1, 99, 115, 100, 110], [2, 100, 115, 101, 110]])
        result = triangulator.make_initial_guess_by_stereo_batch(stereo_tensor)
        assert result is not None
        assert result.shape == (2, 5)
        assert np.allclose(result[:, 4], np.array([0, 0]), atol=1e-5)

    def test_stereo_batch_status(self, triangulator: FeatureTriangulation, stereo_batch: NDArray[np.float32]):
        """Test that the feature triangulation module can make an initial guess by stereo batch."""
        result = triangulator.make_initial_guess_by_stereo_batch(stereo_batch)
        status_column = result[:, 4]
        nan_mask = np.isnan(stereo_batch[:, 3])
        # test that rows with nan has 0 flag
        np.testing.assert_array_equal(status_column[nan_mask], np.array([0, 0, 0, 0]))
