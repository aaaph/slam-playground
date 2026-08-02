import numpy as np
import pytest

from core.front_end.landmark_cache import LandmarkCache, LandmarkCacheSchema, LandmarkCacheStatus
from core.front_end.landmark_initialization import InitializedLandmarkSchema, LandmarkInitialization
from core.front_end.landmark_triangulation import TriangulationStatus
from core.front_end.observation_store import ObservationSchema, ObservationStore, ReadyObservationCriteria
from core.transformations.special_euclidian_3_dim import SE3

K_INV = np.eye(3, dtype=np.float64)
REFINER_COVARIANCE = np.diag(np.array([0.1, 0.2, 0.3], dtype=np.float64))
ZERO_COVARIANCE = np.zeros((3, 3), dtype=np.float64)


class _MixedTriangulatorSpy:
    """Spy triangulator that records mixed triangulation calls."""

    def __init__(self, results: list[tuple[TriangulationStatus, np.ndarray]] | None = None) -> None:
        self.results = results or [(TriangulationStatus.SUCCESS, np.array([1.0, 3.0, 5.0], dtype=np.float64))]
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def triangulate_mixed(
        self, left_uvs: np.ndarray, right_uvs: np.ndarray, left_poses: np.ndarray
    ) -> tuple[TriangulationStatus, np.ndarray]:
        """Record triangulation inputs and return the next configured point."""
        self.calls.append((left_uvs.copy(), right_uvs.copy(), left_poses.copy()))
        status, point = self.results[len(self.calls) - 1]
        return status, point.copy()


def _observations(feat_ids: np.ndarray, left_u: float, pose_estimate: SE3) -> np.ndarray:
    """Build observation rows for landmark initialization tests."""
    rows = np.full((feat_ids.shape[0], ObservationSchema.size()), np.nan, dtype=np.float64)
    rows[:, ObservationSchema.FEAT_ID] = feat_ids
    rows[:, ObservationSchema.CAM0_MATRIX] = pose_estimate.as_matrix().reshape(-1)
    rows[:, ObservationSchema.LEFT_U] = left_u
    rows[:, ObservationSchema.LEFT_V] = 20.0
    rows[:, ObservationSchema.RIGHT_U] = left_u - 1.0
    rows[:, ObservationSchema.RIGHT_V] = 20.0
    rows[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = 0.0
    return rows


def _initialized_row(feat_id: float, xyz: np.ndarray, covariance: np.ndarray = REFINER_COVARIANCE) -> np.ndarray:
    """Build one initialized landmark row with covariance."""
    row = np.full((InitializedLandmarkSchema.count(),), np.nan, dtype=np.float64)
    row[InitializedLandmarkSchema.FEAT_ID] = feat_id
    row[InitializedLandmarkSchema.XYZ] = xyz
    row[InitializedLandmarkSchema.COV] = covariance.reshape(9)
    row[InitializedLandmarkSchema.DEPTH_SIGMA] = np.sqrt(max(covariance[2, 2], 0.0))
    return row


def test_ready_slots_returns_store_slots_history_versions_and_feature_ids() -> None:
    """Ready slot selection should expose slot-space data for cache indexing."""
    store = ObservationStore(
        k_inv=K_INV,
        capacity=2,
        history_size=3,
        ready_criteria=ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=0.05,
            min_parallax_observations=2,
        ),
    )
    feat_ids = np.array([10, 20], dtype=np.int32)

    for pose_x, ready_u, pending_u in [(10.0, 0.0, 0.0), (11.0, 2.0, 0.2), (12.0, 3.0, 0.4)]:
        pose_matrix = SE3(t=np.array([pose_x, 0.0, 0.0])).as_matrix()
        observations = np.full((2, ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[:, ObservationSchema.FEAT_ID] = feat_ids
        observations[:, ObservationSchema.CAM0_MATRIX] = pose_matrix.reshape(-1)
        observations[:, ObservationSchema.LEFT_U] = np.array([ready_u, pending_u], dtype=np.float64)
        observations[:, ObservationSchema.LEFT_V] = 20.0
        observations[:, ObservationSchema.RIGHT_U] = np.array([ready_u - 1.0, pending_u - 1.0], dtype=np.float64)
        observations[:, ObservationSchema.RIGHT_V] = 20.0
        store.add_observations(observations)

    ready_slots, ready_history, ready_feat_ids = store.ready_slots(np.array([0, 1], dtype=np.int32))

    np.testing.assert_array_equal(ready_slots, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(ready_history, np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(ready_feat_ids, np.array([10], dtype=np.int32))


def test_triangulate_ready_observations_uses_per_feature_history_mask_and_writes_cache() -> None:
    """Ready observation triangulation should select valid rows and write results to cache."""
    store = ObservationStore(k_inv=K_INV, capacity=1, history_size=5)
    triangulator = _MixedTriangulatorSpy()
    cache = LandmarkCache.default_factory(capacity=1)
    initializer = LandmarkInitialization(
        store,
        cache,
        triangulator,
    )

    for i in range(3):
        observations = np.full((1, ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[0, ObservationSchema.FEAT_ID] = 42
        observations[0, ObservationSchema.LEFT_UV] = np.array([10.0 + i, 20.0 + i])
        observations[0, ObservationSchema.RIGHT_UV] = np.array([9.0 + i, 20.0 + i])
        if i == 1:
            observations[0, ObservationSchema.RIGHT_UV] = np.nan
        observations[0, ObservationSchema.CAM0_MATRIX] = (
            SE3(t=np.array([float(i), float(i + 1), float(i + 2)])).as_matrix().reshape(-1)
        )
        store.add_observations(observations)

    initializer.triangulate_ready_observations(np.array([0], dtype=np.int32))

    assert len(triangulator.calls) == 1
    left_uvs, right_uvs, left_poses = triangulator.calls[0]
    np.testing.assert_allclose(
        left_uvs,
        np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]),
    )
    np.testing.assert_allclose(
        right_uvs,
        np.array(
            [
                [9.0, 20.0],
                [np.nan, np.nan],
                [11.0, 22.0],
            ]
        ),
    )
    assert left_poses.shape == (3, 4, 4)
    np.testing.assert_allclose(left_poses[:, :3, 3], np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.COMPLETED.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.FEAT_ID] == 42.0  # noqa: SLF001
    np.testing.assert_allclose(cache._data[0, LandmarkCacheSchema.XYZ], np.array([1.0, 3.0, 5.0]))  # noqa: SLF001
    np.testing.assert_allclose(cache._data[0, LandmarkCacheSchema.COV], ZERO_COVARIANCE.reshape(9))  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.DEPTH_SIGMA] == 0.0  # noqa: SLF001
    np.testing.assert_allclose(
        initializer.get_initialized_landmarks(),
        np.array([_initialized_row(42.0, np.array([1.0, 3.0, 5.0]), ZERO_COVARIANCE)]),
    )


@pytest.mark.parametrize(
    "status",
    [
        TriangulationStatus.NOT_VALID,
        TriangulationStatus.BIG_REPROJECTION_ERROR,
        TriangulationStatus.BIG_DEPTH_VARIANCE,
        TriangulationStatus.COVARIANCE_NOT_VALID,
        TriangulationStatus.INVALID_POINT_DEPTH,
    ],
)
def test_triangulate_ready_observations_marks_failed_triangulation_status_as_failed(
    status: TriangulationStatus,
) -> None:
    """Non-success triangulation statuses should keep the landmark out of the completed cache."""
    store = ObservationStore(k_inv=K_INV, capacity=1, history_size=5)
    triangulator = _MixedTriangulatorSpy([(status, np.full((3,), np.nan, dtype=np.float64))])
    cache = LandmarkCache.default_factory(capacity=1)
    initializer = LandmarkInitialization(
        store,
        cache,
        triangulator,
    )

    for i in range(3):
        observations = np.full((1, ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[0, ObservationSchema.FEAT_ID] = 42
        observations[0, ObservationSchema.LEFT_UV] = np.array([10.0 + i, 20.0 + i])
        observations[0, ObservationSchema.RIGHT_UV] = np.array([9.0 + i, 20.0 + i])
        observations[0, ObservationSchema.CAM0_MATRIX] = (
            SE3(t=np.array([float(i), float(i + 1), float(i + 2)])).as_matrix().reshape(-1)
        )
        store.add_observations(observations)

    initializer.triangulate_ready_observations(np.array([0], dtype=np.int32))

    assert len(triangulator.calls) == 1
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_SOFT.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 4.0  # noqa: SLF001
    assert np.all(np.isnan(cache._data[0, LandmarkCacheSchema.XYZ]))  # noqa: SLF001


def test_landmark_cache_get_completed_landmarks_returns_visualization_rows() -> None:
    """Completed cache entries should be exposed as feat_id + xyz rows."""
    cache = LandmarkCache(capacity=3)
    cache.apply_completed(
        np.array([20], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        REFINER_COVARIANCE.reshape(1, 3, 3),
    )
    cache.apply_failed(np.array([2], dtype=np.int32), np.array([1], dtype=np.int32))

    landmarks = cache.get_completed_landmarks()

    np.testing.assert_allclose(
        landmarks,
        np.array([_initialized_row(20.0, np.array([1.0, 2.0, 3.0]))]),
    )


def test_add_observation_writes_histories_and_does_not_triangulate_in_ingest_step() -> None:
    """Landmark observation ingest should store histories without running heavy initialization."""
    store = ObservationStore(
        k_inv=K_INV,
        capacity=2,
        history_size=3,
        ready_criteria=ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=0.05,
            min_parallax_observations=2,
        ),
    )
    triangulator = _MixedTriangulatorSpy()
    cache = LandmarkCache.default_factory(capacity=2)
    initializer = LandmarkInitialization(
        store,
        cache,
        triangulator,
    )
    feat_ids = np.array([10, 20], dtype=np.int32)

    first_slots = initializer.add_observation(_observations(feat_ids, 0.0, SE3(t=np.array([10.0, 0.0, 0.0]))))
    np.testing.assert_array_equal(first_slots, np.empty((0,), dtype=np.int32))
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.FEAT_ID],  # noqa: SLF001
        np.array([10.0, 20.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.STATUS],  # noqa: SLF001
        np.array([LandmarkCacheStatus.OBSERVING.value, LandmarkCacheStatus.OBSERVING.value], dtype=np.float64),
    )

    second_slots = initializer.add_observation(_observations(feat_ids, 2.0, SE3(t=np.array([11.0, 0.0, 0.0]))))
    np.testing.assert_array_equal(second_slots, np.empty((0,), dtype=np.int32))
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.STATUS],  # noqa: SLF001
        np.array([LandmarkCacheStatus.OBSERVING.value, LandmarkCacheStatus.OBSERVING.value], dtype=np.float64),
    )

    third_slots = initializer.add_observation(_observations(feat_ids, 3.0, SE3(t=np.array([12.0, 0.0, 0.0]))))

    np.testing.assert_array_equal(third_slots, np.array([0, 1], dtype=np.int32))
    assert store.get_feat_history(10).shape == (3, ObservationSchema.size())
    assert store.get_feat_history(20).shape == (3, ObservationSchema.size())
    assert len(triangulator.calls) == 0
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.FEAT_ID],  # noqa: SLF001
        np.array([10.0, 20.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.STATUS],  # noqa: SLF001
        np.array([LandmarkCacheStatus.READY.value, LandmarkCacheStatus.READY.value], dtype=np.float64),
    )

    initializer.remove_lost_features(np.array([20], dtype=np.int32))
    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001
    assert np.isnan(cache._data[1, LandmarkCacheSchema.FEAT_ID])  # noqa: SLF001
    repeated_slots = initializer.add_observation(
        _observations(np.array([10], dtype=np.int32), 4.0, SE3(t=np.array([13.0, 0.0, 0.0]))),
    )

    np.testing.assert_array_equal(repeated_slots, np.array([0], dtype=np.int32))
    assert store.get_feat_history(10).shape == (3, ObservationSchema.size())
    assert store.get_feat_history(20).shape == (0, ObservationSchema.size())
    assert len(triangulator.calls) == 0


def test_add_observation_retries_soft_failed_slots_only_after_retry_history_version() -> None:
    """Soft-failed ready slots should stay banned until the observation history advances."""
    store = ObservationStore(
        k_inv=K_INV,
        capacity=1,
        history_size=4,
        ready_criteria=ReadyObservationCriteria(
            min_history_size=3,
            min_parallax_rad=0.05,
            min_parallax_observations=2,
        ),
    )
    triangulator = _MixedTriangulatorSpy([(TriangulationStatus.NOT_VALID, np.full(3, np.nan, dtype=np.float64))])
    cache = LandmarkCache.default_factory(capacity=1)
    initializer = LandmarkInitialization(
        store,
        cache,
        triangulator,
    )
    feat_ids = np.array([10], dtype=np.int32)

    initializer.add_observation(_observations(feat_ids, 0.0, SE3(t=np.array([10.0, 0.0, 0.0]))))
    initializer.add_observation(_observations(feat_ids, 2.0, SE3(t=np.array([11.0, 0.0, 0.0]))))
    ready_slots = initializer.add_observation(_observations(feat_ids, 3.0, SE3(t=np.array([12.0, 0.0, 0.0]))))
    initializer.triangulate_ready_observations(ready_slots)

    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_SOFT.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 4.0  # noqa: SLF001
    store_ready_slots, store_history_versions, store_feat_ids = store.ready_slots(np.array([0], dtype=np.int32))
    still_banned_slots = cache.apply_ready(store_feat_ids, store_ready_slots, store_history_versions)

    np.testing.assert_array_equal(still_banned_slots, np.empty((0,), dtype=np.int32))

    retried_slots = initializer.add_observation(_observations(feat_ids, 4.0, SE3(t=np.array([13.0, 0.0, 0.0]))))

    np.testing.assert_array_equal(retried_slots, np.array([0], dtype=np.int32))
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.READY.value  # noqa: SLF001


def test_landmark_cache_apply_observing_updates_only_empty_slots() -> None:
    """Observing status should not downgrade ready or failed cache entries."""
    cache = LandmarkCache(capacity=2)
    cache.apply_ready(np.array([10], dtype=np.int32), np.array([0], dtype=np.int32), np.array([1], dtype=np.int32))

    cache.apply_observing(np.array([30, 40], dtype=np.int32), np.array([0, 1], dtype=np.int32))

    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.READY.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.FEAT_ID] == 10.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.OBSERVING.value  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.FEAT_ID] == 40.0  # noqa: SLF001


def test_landmark_cache_apply_failed_bans_after_soft_attempts() -> None:
    """Failed cache slots should become hard-failed only after the soft attempt budget."""
    cache = LandmarkCache(capacity=3)
    cache._data[0, LandmarkCacheSchema.ATTEMPTS] = 4.0  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.ATTEMPTS] = 5.0  # noqa: SLF001
    cache.apply_completed(
        np.array([30], dtype=np.int32),
        np.array([2], dtype=np.int32),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        REFINER_COVARIANCE.reshape(1, 3, 3),
    )

    cache.apply_failed(np.array([0, 1], dtype=np.int32), np.array([3, 3], dtype=np.int32))

    assert cache._data[0, LandmarkCacheSchema.ATTEMPTS] == 5.0  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_SOFT.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 4.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.ATTEMPTS] == 6.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_HARD.value  # noqa: SLF001

    banned_slots = cache.apply_ready(
        np.array([10, 20, 30], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([3, 3, 3], dtype=np.int32),
    )
    ready_slots = cache.apply_ready(
        np.array([10, 20, 30], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([4, 4, 4], dtype=np.int32),
    )

    np.testing.assert_array_equal(banned_slots, np.empty((0,), dtype=np.int32))
    np.testing.assert_array_equal(ready_slots, np.array([0], dtype=np.int32))
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.READY.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 0.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_HARD.value  # noqa: SLF001
    assert cache._data[2, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.COMPLETED.value  # noqa: SLF001


def test_landmark_cache_clear_slots_resets_payload_and_status() -> None:
    """Clearing cache slots should remove stale ready state before slot reuse."""
    cache = LandmarkCache(capacity=2)
    cache.apply_ready(np.array([10], dtype=np.int32), np.array([1], dtype=np.int32), np.array([1], dtype=np.int32))
    cache._data[1, LandmarkCacheSchema.XYZ] = np.array([1.0, 2.0, 3.0])  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.ATTEMPTS] = 4.0  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 7.0  # noqa: SLF001

    cache.clear_slots(np.array([1], dtype=np.int32))

    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.ATTEMPTS] == 0.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 0.0  # noqa: SLF001
    assert np.isnan(cache._data[1, LandmarkCacheSchema.FEAT_ID])  # noqa: SLF001
    assert np.all(np.isnan(cache._data[1, LandmarkCacheSchema.XYZ]))  # noqa: SLF001
