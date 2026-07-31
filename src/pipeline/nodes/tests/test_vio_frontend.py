from types import SimpleNamespace

import numpy as np
import pytest

from core.front_end.landmark_initialization import InitializedLandmarkSchema
from core.front_end.observation_store import ObservationSchema
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
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
        self.observations: np.ndarray | None = None
        self.ready_slots = np.array([3, 5], dtype=np.int32)
        self.triangulated_slots: np.ndarray | None = None

    def remove_lost_features(self, lost_features: np.ndarray) -> None:
        """Record removed feature IDs."""
        self.lost_features = lost_features.copy()

    def add_observation(self, observations: np.ndarray) -> np.ndarray:
        """Record observation rows and return ready slots."""
        self.observations = observations.copy()
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


def _initialized_landmark_row(feat_id: float, xyz: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Build one initialized-landmark row with covariance."""
    row = np.full((InitializedLandmarkSchema.count(),), np.nan, dtype=np.float64)
    row[InitializedLandmarkSchema.FEAT_ID] = feat_id
    row[InitializedLandmarkSchema.XYZ] = xyz
    row[InitializedLandmarkSchema.COV] = covariance.reshape(9)
    row[InitializedLandmarkSchema.DEPTH_SIGMA] = np.sqrt(max(covariance[2, 2], 0.0))
    return row


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

    def test_get_initialized_landmarks_in_camera_frame_rotates_covariance(self) -> None:
        """Frontend should rotate cached world-frame covariance into the current cam0 frame."""
        node = VIOFrontend.__new__(VIOFrontend)
        covariance_world = np.diag(np.array([1.0, 4.0, 9.0], dtype=np.float64))
        node.landmark_init = _LandmarkInitializationStub(
            np.array([_initialized_landmark_row(10.0, np.array([1.0, 0.0, 2.0]), covariance_world)])
        )
        cam_in_world = SE3.from_rpy_xyz(np.array([0.0, 0.0, np.pi / 2], dtype=np.float64), np.zeros(3))

        landmarks = node.get_initialized_landmarks_in_camera_frame(cam_in_world)

        np.testing.assert_allclose(
            landmarks[0, InitializedLandmarkSchema.XYZ],
            np.array([0.0, -1.0, 2.0]),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            landmarks[0, InitializedLandmarkSchema.COV].reshape(3, 3),
            np.diag(np.array([4.0, 1.0, 9.0], dtype=np.float64)),
            atol=1e-7,
        )
        assert landmarks[0, InitializedLandmarkSchema.DEPTH_SIGMA] == pytest.approx(3.0)

    def test_build_landmark_observations_drops_right_uv_for_bad_stereo_status(self) -> None:
        """Frontend should convert one-shot BAD_STEREO rows into left-only landmark observations."""
        node = VIOFrontend.__new__(VIOFrontend)
        active_points = np.full((3, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        active_points[:, StereoTriangulationSchema.FEAT_ID] = np.array([10.0, 20.0, 30.0])
        active_points[:, StereoTriangulationSchema.LEFT_UV] = np.array(
            [[110.0, 111.0], [120.0, 121.0], [130.0, 131.0]]
        )
        active_points[:, StereoTriangulationSchema.RIGHT_UV] = np.array(
            [[105.0, 111.0], [115.0, 121.0], [125.0, 131.0]]
        )
        active_points[:, StereoTriangulationSchema.STATUS] = np.array(
            [
                StereoTriangulationStatus.TRIANGULATED.value,
                StereoTriangulationStatus.BAD_STEREO.value,
                StereoTriangulationStatus.TRIANGULATED.value,
            ],
            dtype=np.float32,
        )
        tracking_mask = np.array([True, True, False], dtype=np.bool_)
        cam0_in_world = SE3(t=np.array([1.0, 2.0, 3.0], dtype=np.float64))

        observations = node.build_landmark_observations(42, active_points, tracking_mask, cam0_in_world)

        assert observations.shape == (2, ObservationSchema.size())
        np.testing.assert_allclose(observations[:, ObservationSchema.FEAT_ID], np.array([10.0, 20.0]))
        np.testing.assert_allclose(observations[:, ObservationSchema.FRAME_ID], np.array([42.0, 42.0]))
        np.testing.assert_allclose(
            observations[:, ObservationSchema.LEFT_UV],
            np.array([[110.0, 111.0], [120.0, 121.0]]),
        )
        np.testing.assert_allclose(observations[0, ObservationSchema.RIGHT_UV], np.array([105.0, 111.0]))
        assert np.all(np.isnan(observations[1, ObservationSchema.RIGHT_UV]))
        np.testing.assert_allclose(
            observations[:, ObservationSchema.CAM0_MATRIX],
            np.tile(cam0_in_world.as_matrix().reshape(-1), (2, 1)),
        )

    def test_apply_observations_passes_prebuilt_observation_rows_to_landmark_init(self) -> None:
        """Frontend should own active-point to landmark-observation conversion."""
        node = VIOFrontend.__new__(VIOFrontend)
        node.vio_ctx = SimpleNamespace(stereo=SimpleNamespace(cam0_in_body_se3=SE3.identity()))
        node.landmark_init = _LandmarkInitializationApplyStub(np.empty((0, InitializedLandmarkSchema.count())))
        active_points = np.full((3, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        active_points[:, StereoTriangulationSchema.FEAT_ID] = np.array([10.0, 20.0, 30.0])
        active_points[:, StereoTriangulationSchema.LEFT_UV] = np.array(
            [[110.0, 111.0], [120.0, 121.0], [130.0, 131.0]]
        )
        active_points[:, StereoTriangulationSchema.RIGHT_UV] = np.array(
            [[105.0, 111.0], [115.0, 121.0], [125.0, 131.0]]
        )
        active_points[:, StereoTriangulationSchema.STATUS] = np.array(
            [
                StereoTriangulationStatus.TRIANGULATED.value,
                StereoTriangulationStatus.BAD_STEREO.value,
                StereoTriangulationStatus.TRIANGULATED.value,
            ],
            dtype=np.float32,
        )
        tracking_mask = np.array([True, True, False], dtype=np.bool_)

        node.apply_observations(42, SE3.identity(), tracking_mask, active_points)

        np.testing.assert_array_equal(node.landmark_init.lost_features, np.array([30], dtype=np.int32))
        observations = node.landmark_init.observations
        assert observations is not None
        np.testing.assert_array_equal(node.landmark_init.triangulated_slots, node.landmark_init.ready_slots)
        np.testing.assert_allclose(observations[:, ObservationSchema.FEAT_ID], np.array([10.0, 20.0]))
        np.testing.assert_allclose(observations[:, ObservationSchema.FRAME_ID], np.array([42.0, 42.0]))
        np.testing.assert_allclose(observations[0, ObservationSchema.RIGHT_UV], np.array([105.0, 111.0]))
        assert np.all(np.isnan(observations[1, ObservationSchema.RIGHT_UV]))

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

        completed_landmarks = node.apply_observations(42, SE3.identity(), tracking_mask, active_points)

        np.testing.assert_array_equal(node.landmark_init.lost_features, np.array([20, 50], dtype=np.int32))
        tracking_info = node.landmark_init.observations
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
