from unittest.mock import MagicMock

import numpy as np
import pytest

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_triangulation import FeatureTriangulation
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
        assert hasattr(triangulator, "compute_feature_linear_system_update")
        assert callable(triangulator.compute_feature_linear_system_update)
        assert hasattr(triangulator, "make_initial_guess_by_stereo_pair")
        assert callable(triangulator.make_initial_guess_by_stereo_pair)

    def test_make_initial_guess(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can make an initial guess."""
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (95, 110))
        _, initial_guess = triangulator.make_initial_guess_by_stereo_pair(feature)
        assert initial_guess is not None
        assert initial_guess.shape == (3,)

        # should throw an error if the feature has no active stereo pair
        feature = Feature(1)
        with pytest.raises(ValueError, match="Feature has no active left point"):
            triangulator.make_initial_guess_by_stereo_pair(feature)

        feature.apply_left_only(2, (0, 0))
        with pytest.raises(ValueError, match="Feature has no active right point"):
            triangulator.make_initial_guess_by_stereo_pair(feature)

        # should throw an error if the disparity is non-positive
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (100, 115))
        with pytest.raises(ValueError, match="Disparity is non-positive"):
            triangulator.make_initial_guess_by_stereo_pair(feature)

    def test_compute_feature_linear_system_update(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can update the feature linear system."""
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (95, 110))
        pose = SE3.identity()
        delta_a, delta_b = triangulator.compute_feature_linear_system_update(feature, 1, pose)
        assert delta_a is not None
        assert delta_b is not None
        assert delta_a.shape == (3, 3)
        assert delta_b.shape == (3,)

    def test_linear_system_match(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can match the linear system."""
        k_matrix = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)

        feat_in_world_vec_true = np.array([10.0, 2.0, 5.0], dtype=float)
        body_in_world_se3 = SE3.from_rpy_xyz(np.array([0.1, 0.2, 0.0]), np.array([1.0, 1.0, 0.0]))

        camera_in_body_se3 = SE3.identity() * SE3(t=np.array([0.1, 0.1, 0.1]))

        camera_in_world_se3 = body_in_world_se3 * camera_in_body_se3
        world_in_camera_se3 = camera_in_world_se3.inverse()

        u, v = TestFeatureTriangulation.project_point(feat_in_world_vec_true, world_in_camera_se3, k_matrix)
        mock_feature = MagicMock()
        mock_feature.get_uv_by_timestamp.return_value = [(0, u, v)]
        k_matricies = (k_matrix, k_matrix, k_matrix)
        baseline = 0.1
        body_sensor_transforms = (camera_in_body_se3, camera_in_body_se3)
        triangulator = FeatureTriangulation(k_matricies, baseline, body_sensor_transforms)

        delta_a, delta_b = triangulator.compute_feature_linear_system_update(mock_feature, 0, body_in_world_se3)

        assert np.allclose(delta_a @ feat_in_world_vec_true, delta_b, atol=1e-5)

    def test_linear_system_match_stereo(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can no match the linear system."""
        left_k_matrix = triangulator.k_left
        right_k_matrix = triangulator.k_right
        left_camera_in_body_se3 = triangulator.cam0_in_body
        right_camera_in_body_se3 = triangulator.cam1_in_body

        feat_in_world_vec_true = np.array([10.0, 2.0, 5.0], dtype=float)
        body_in_world_se3 = SE3.from_rpy_xyz(np.array([0.1, 0.2, 0.0]), np.array([1.0, 1.0, 0.0]))

        left_camera_in_world_se3 = body_in_world_se3 * left_camera_in_body_se3
        right_camera_in_world_se3 = body_in_world_se3 * right_camera_in_body_se3
        world_in_left_camera_se3 = left_camera_in_world_se3.inverse()
        world_in_right_camera_se3 = right_camera_in_world_se3.inverse()

        left_u, left_v = TestFeatureTriangulation.project_point(
            feat_in_world_vec_true, world_in_left_camera_se3, left_k_matrix
        )
        right_u, right_v = TestFeatureTriangulation.project_point(
            feat_in_world_vec_true, world_in_right_camera_se3, right_k_matrix
        )

        feat = Feature.spawn_from_left_and_right(1, 0.0, (left_u, left_v), (right_u, right_v))

        delta_a, delta_b = triangulator.compute_feature_linear_system_update(feat, 0, body_in_world_se3)
        assert np.allclose(delta_a @ feat_in_world_vec_true, delta_b, atol=1e-5)

        solved_feat_in_world_vec = np.linalg.solve(delta_a, delta_b)
        assert np.allclose(solved_feat_in_world_vec, feat_in_world_vec_true, atol=1e-5)

    def test_make_linear_triangulation_guess(self, triangulator: FeatureTriangulation):
        """Test that the feature triangulation module can make a linear triangulation guess."""
        feature = MagicMock()
        feature.ready_to_triangulate = True

        # Simple system: x=10, y=2, z=5
        # Use 4 equations to simulate a redefined system (stereo)
        feat_in_world_vec_true = np.array([10.0, 2.0, 5.0])
        feature.A = np.vstack([np.eye(3), [1, 1, 1]])  # 4x3 matrix
        feature.b = feature.A @ feat_in_world_vec_true  # Ideal b

        camera_in_world_se3 = SE3.identity()  # Camera at 0,0,0

        success, vec = triangulator.make_linear_triangulation_guess(feature, camera_in_world_se3)

        assert success is True
        np.testing.assert_allclose(vec, feat_in_world_vec_true, atol=1e-5)
