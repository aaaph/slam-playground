import numpy as np

from core.front_end.landmark_initialization import InitializedLandmarkSchema, LandmarkInitialization
from core.front_end.observation_store import ObservationSchema, ObservationStore, ReadyObservationCriteria
from core.front_end.ray_triangulation import TriangulationStatus
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3


class _TriangulatorSpy:
    """Spy triangulator that records one triangulation call."""

    def __init__(self) -> None:
        self.left_uv: np.ndarray | None = None
        self.right_uv: np.ndarray | None = None
        self.cam0_matrix: np.ndarray | None = None

    def triangulate_feature_observations(
        self, left_uv: np.ndarray, right_uv: np.ndarray, cam0_poses: np.ndarray
    ) -> tuple[TriangulationStatus, np.ndarray]:
        """Record triangulation inputs."""
        self.left_uv = left_uv.copy()
        self.right_uv = right_uv.copy()
        self.cam0_matrix = cam0_poses.copy()
        return TriangulationStatus.SUCCESS, np.array([1.0, 2.0, 3.0], dtype=np.float64)


class _TriangulatorSequence:
    """Spy triangulator that returns a configured status sequence."""

    def __init__(self, results: list[tuple[TriangulationStatus, np.ndarray]]) -> None:
        self.results = results
        self.calls = 0

    def triangulate_feature_observations(
        self, left_uv: np.ndarray, right_uv: np.ndarray, cam0_poses: np.ndarray
    ) -> tuple[TriangulationStatus, np.ndarray]:
        """Return the next configured triangulation result."""
        _ = (left_uv, right_uv, cam0_poses)
        result = self.results[self.calls]
        self.calls += 1
        return result


def _tracking_info(feat_ids: np.ndarray, left_u: float) -> np.ndarray:
    """Build tracking rows for landmark initialization tests."""
    rows = np.full((feat_ids.shape[0], StereoTriangulationSchema.count()), np.nan, dtype=np.float64)
    rows[:, StereoTriangulationSchema.FEAT_ID] = feat_ids
    rows[:, StereoTriangulationSchema.LEFT_U] = left_u
    rows[:, StereoTriangulationSchema.LEFT_V] = 20.0
    rows[:, StereoTriangulationSchema.RIGHT_U] = left_u - 1.0
    rows[:, StereoTriangulationSchema.RIGHT_V] = 20.0
    return rows


def test_try_to_triangulate_observation_uses_per_feature_history_mask() -> None:
    """Landmark initialization should select valid rows for one ready feature."""
    triangulator = _TriangulatorSpy()
    initializer = LandmarkInitialization(
        ObservationStore(capacity=1, history_size=5),
        triangulator,
        ReadyObservationCriteria(min_history_size=3, min_pixel_displacement=1.0),
    )
    ready_observations = np.full((1, 5, ObservationSchema.size()), np.nan, dtype=np.float64)
    history_mask = np.array([[True, True, True, False, False]], dtype=np.bool_)

    for i in range(3):
        ready_observations[0, i, ObservationSchema.FEAT_ID] = 42
        ready_observations[0, i, ObservationSchema.LEFT_UV] = np.array([10.0 + i, 20.0 + i])
        ready_observations[0, i, ObservationSchema.RIGHT_UV] = np.array([9.0 + i, 20.0 + i])
        ready_observations[0, i, ObservationSchema.CAM0_MATRIX] = (
            SE3(t=np.array([float(i), float(i + 1), float(i + 2)])).as_matrix().reshape(-1)
        )

    current_world_from_cam0 = SE3(t=np.array([2.0, 3.0, 4.0])).as_matrix()
    initialized = initializer._try_to_triangulate_observation(  # noqa: SLF001
        np.array([42], dtype=np.int32), history_mask, ready_observations, current_world_from_cam0
    )

    assert triangulator.left_uv is not None
    assert triangulator.right_uv is not None
    assert triangulator.cam0_matrix is not None
    np.testing.assert_allclose(triangulator.left_uv, np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]))
    np.testing.assert_allclose(triangulator.right_uv, np.array([[9.0, 20.0], [10.0, 21.0], [11.0, 22.0]]))
    assert triangulator.cam0_matrix.shape == (3, 4, 4)
    np.testing.assert_allclose(
        triangulator.cam0_matrix[:, :3, 3],
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    np.testing.assert_allclose(initialized[:, InitializedLandmarkSchema.XYZ], np.array([[-1.0, 0.0, 1.0]]))


def test_add_observation_returns_initialized_current_cam0_landmarks_without_removing_successes() -> None:
    """Successful triangulations should be emitted in current cam0 while histories are kept."""
    store = ObservationStore(capacity=2, history_size=3)
    triangulator = _TriangulatorSequence(
        [
            (TriangulationStatus.SUCCESS, np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            (TriangulationStatus.ILL_CONDITIONED, np.array([9.0, 9.0, 9.0], dtype=np.float64)),
            (TriangulationStatus.SUCCESS, np.array([1.0, 2.0, 3.0], dtype=np.float64)),
            (TriangulationStatus.ILL_CONDITIONED, np.array([9.0, 9.0, 9.0], dtype=np.float64)),
        ]
    )
    initializer = LandmarkInitialization(
        store,
        triangulator,
        ReadyObservationCriteria(min_history_size=3, min_pixel_displacement=1.0, min_displacement_observations=2),
    )
    feat_ids = np.array([10, 20], dtype=np.int32)

    first = initializer.add_observation(_tracking_info(feat_ids, 0.0), SE3(t=np.array([10.0, 0.0, 0.0])))
    second = initializer.add_observation(_tracking_info(feat_ids, 2.0), SE3(t=np.array([11.0, 0.0, 0.0])))
    third = initializer.add_observation(_tracking_info(feat_ids, 3.0), SE3(t=np.array([12.0, 0.0, 0.0])))

    assert first.shape == (0, InitializedLandmarkSchema.count())
    assert second.shape == (0, InitializedLandmarkSchema.count())
    assert third.shape == (1, InitializedLandmarkSchema.count())
    assert third[0, InitializedLandmarkSchema.FEAT_ID] == 10
    np.testing.assert_allclose(third[0, InitializedLandmarkSchema.XYZ], np.array([-1.0, 2.0, 3.0]))
    assert store.get_feat_history(10).shape == (3, ObservationSchema.size())
    assert store.get_feat_history(20).shape == (3, ObservationSchema.size())

    initializer.remove_lost_features(np.array([20], dtype=np.int32))
    repeated = initializer.add_observation(
        _tracking_info(np.array([10], dtype=np.int32), 4.0),
        SE3(t=np.array([13.0, 0.0, 0.0])),
    )

    assert repeated.shape == (1, InitializedLandmarkSchema.count())
    assert store.get_feat_history(10).shape == (3, ObservationSchema.size())
    assert store.get_feat_history(20).shape == (0, ObservationSchema.size())
    assert triangulator.calls == 2
