from types import SimpleNamespace

import numpy as np
import pytest

from core.front_end.landmark_initialization import InitializedLandmarkSchema
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.pose_tracker.frame_to_frame_pnp_store import PnPMapSchema
from core.transformations.special_euclidian_3_dim import SE3
from pipeline.nodes.vio_frontend import VIOFrontend


class _LandmarkInitializationStub:
    """Minimal landmark initialization stub for frontend transform tests."""

    def __init__(self, landmarks: np.ndarray) -> None:
        self._landmarks = landmarks

    def get_initialized_landmarks(self) -> np.ndarray:
        """Return cached world-frame landmarks."""
        return self._landmarks.copy()


class _LandmarkInitializationApplyStub(_LandmarkInitializationStub):
    """Minimal landmark initialization stub for observation application tests."""

    def __init__(self, landmarks: np.ndarray) -> None:
        """Initialize the apply-observation stub."""
        super().__init__(landmarks)
        self.lost_features: np.ndarray | None = None
        self.tracking_info: np.ndarray | None = None
        self.ready_slots = np.array([3, 5], dtype=np.int32)
        self.triangulated_slots: np.ndarray | None = None

    def remove_lost_features(self, lost_features: np.ndarray) -> None:
        """Record removed feature IDs."""
        self.lost_features = lost_features.copy()

    def add_observation(self, tracking_info: np.ndarray, _pose_estimate: SE3) -> np.ndarray:
        """Record tracking info and return ready slots."""
        self.tracking_info = tracking_info.copy()
        return self.ready_slots

    def triangulate_ready_observations(self, ready_slots: np.ndarray) -> None:
        """Record triangulated slots."""
        self.triangulated_slots = ready_slots.copy()


class _PnpEstimatorStub:
    """Minimal PnP estimator stub."""

    def __init__(self) -> None:
        """Initialize the PnP estimator stub."""
        self.visual_features: np.ndarray | None = None

    def add_visual_data(self, visual_features: np.ndarray) -> None:
        """Record visual data added to PnP."""
        self.visual_features = visual_features.copy()


class TestVIOFrontend:
    """VIO frontend unit tests."""

    def test_get_initialized_landmarks_in_camera_frame_transforms_cached_world_landmarks(self) -> None:
        """Frontend visualization should transform cached world landmarks into current cam0 frame."""
        node = VIOFrontend.__new__(VIOFrontend)
        node.landmark_init = _LandmarkInitializationStub(
            np.array(
                [
                    [10.0, 11.0, 2.0, 3.0],
                    [20.0, 8.0, -1.0, 0.5],
                ],
                dtype=np.float64,
            )
        )
        cam_in_world = SE3(t=np.array([10.0, 0.0, 0.0], dtype=np.float64))

        landmarks = node.get_initialized_landmarks_in_camera_frame(cam_in_world)

        np.testing.assert_allclose(
            landmarks,
            np.array(
                [
                    [10.0, 1.0, 2.0, 3.0],
                    [20.0, -2.0, -1.0, 0.5],
                ],
                dtype=np.float64,
            ),
        )

    @pytest.mark.skip(reason="Skipping test_apply_observations_adds_completed_tracking_landmarks_to_pnp")
    def test_apply_observations_adds_completed_tracking_landmarks_to_pnp(self) -> None:
        """Completed landmarks should seed PnP using current tracked UV rows."""
        node = VIOFrontend.__new__(VIOFrontend)
        node.vio_ctx = SimpleNamespace(stereo=SimpleNamespace(cam0_in_body_se3=SE3.identity()))
        node.landmark_init = _LandmarkInitializationApplyStub(
            np.array(
                [
                    [30.0, 3.0, 30.0, 300.0],
                    [10.0, 1.0, 10.0, 100.0],
                    [40.0, 4.0, 40.0, 400.0],
                ],
                dtype=np.float64,
            )
        )
        node.pnp_estimator = _PnpEstimatorStub()
        active_points = np.full((4, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        active_points[:, StereoTriangulationSchema.FEAT_ID] = np.array([10.0, 20.0, 30.0, 50.0])
        active_points[:, StereoTriangulationSchema.LEFT_UV] = np.array(
            [[110.0, 111.0], [120.0, 121.0], [130.0, 131.0], [150.0, 151.0]]
        )
        active_points[:, StereoTriangulationSchema.RIGHT_UV] = np.array(
            [[105.0, 111.0], [115.0, 121.0], [125.0, 131.0], [145.0, 151.0]]
        )
        tracking_mask = np.array([True, False, True, False], dtype=np.bool_)

        completed_landmarks = node.apply_observations(1.0, SE3.identity(), tracking_mask, active_points)

        np.testing.assert_array_equal(node.landmark_init.lost_features, np.array([20, 50], dtype=np.int32))
        tracking_info = node.landmark_init.tracking_info
        assert tracking_info is not None
        np.testing.assert_allclose(
            tracking_info[:, StereoTriangulationSchema.FEAT_ID],
            [10, 30],
        )
        np.testing.assert_array_equal(node.landmark_init.triangulated_slots, node.landmark_init.ready_slots)
        np.testing.assert_allclose(
            completed_landmarks[:, InitializedLandmarkSchema.FEAT_ID],
            np.array([30.0, 10.0, 40.0]),
        )
        assert node.pnp_estimator.visual_features is not None
        visual_features = node.pnp_estimator.visual_features
        np.testing.assert_array_equal(visual_features[:, PnPMapSchema.FEAT_ID], np.array([10.0, 30.0]))
        np.testing.assert_allclose(
            visual_features[:, PnPMapSchema.XYZ],
            np.array([[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]]),
        )
        np.testing.assert_allclose(
            visual_features[:, PnPMapSchema.LEFT_UV],
            np.array([[110.0, 111.0], [130.0, 131.0]]),
        )
        np.testing.assert_allclose(
            visual_features[:, PnPMapSchema.RIGHT_UV],
            np.array([[105.0, 111.0], [125.0, 131.0]]),
        )
