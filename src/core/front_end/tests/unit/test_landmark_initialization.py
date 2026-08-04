import numpy as np
import pytest

from core.front_end.landmark_cache import LandmarkCache, LandmarkCacheSchema, LandmarkCacheStatus
from core.front_end.landmark_initialization import LandmarkInitialization, LandmarkInitializationFrameSchema
from core.front_end.landmark_triangulation import TriangulationStatus
from core.front_end.observation_store import ObservationSchema, ObservationStore, ReadyObservationCriteria
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
from core.transformations.special_euclidian_3_dim import SE3

K_INV = np.eye(3, dtype=np.float64)


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


def _stereo_frame(
    feat_ids: np.ndarray,
    left_u: float | np.ndarray,
    stereo_status: StereoTriangulationStatus | np.ndarray = StereoTriangulationStatus.TRIANGULATED,
) -> np.ndarray:
    """Build a stereo-triangulation frame for landmark initialization tests."""
    feat_ids = np.asarray(feat_ids, dtype=np.float32)
    left_u_arr = np.broadcast_to(np.asarray(left_u, dtype=np.float32), feat_ids.shape).astype(
        np.float32,
        copy=True,
    )
    left_v_arr = np.full(feat_ids.shape, 20.0, dtype=np.float32)
    status = stereo_status.value if isinstance(stereo_status, StereoTriangulationStatus) else stereo_status
    status_arr = np.broadcast_to(np.asarray(status, dtype=np.float32), feat_ids.shape).astype(
        np.float32,
        copy=True,
    )

    frame = np.full((feat_ids.shape[0], StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
    frame[:, StereoTriangulationSchema.FEAT_ID] = feat_ids
    frame[:, StereoTriangulationSchema.LEFT_U] = left_u_arr
    frame[:, StereoTriangulationSchema.LEFT_V] = left_v_arr
    frame[:, StereoTriangulationSchema.RIGHT_U] = left_u_arr - 1.0
    frame[:, StereoTriangulationSchema.RIGHT_V] = left_v_arr
    frame[:, StereoTriangulationSchema.STEREO_STATUS] = status_arr
    return frame


def _observation_row(feat_id: int, left_u: float, pose_x: float, *, right_valid: bool = True) -> np.ndarray:
    """Build one store observation row."""
    observation = np.full((1, ObservationSchema.size()), np.nan, dtype=np.float64)
    observation[0, ObservationSchema.FEAT_ID] = feat_id
    observation[0, ObservationSchema.LEFT_U] = left_u
    observation[0, ObservationSchema.LEFT_V] = 20.0
    if right_valid:
        observation[0, ObservationSchema.RIGHT_U] = left_u - 1.0
        observation[0, ObservationSchema.RIGHT_V] = 20.0
    observation[0, ObservationSchema.CAM0_MATRIX] = SE3(t=np.array([pose_x, 0.0, 0.0])).as_matrix().reshape(-1)
    return observation


def test_apply_observation_frame_populates_observations_and_observing_cache() -> None:
    """Tracked rows should be observed while lost rows are removed."""
    store = ObservationStore(k_inv=K_INV, capacity=3, history_size=3)
    cache = LandmarkCache.default_factory(capacity=3)
    initializer = LandmarkInitialization(store, cache, _MixedTriangulatorSpy())
    stereo_frame = _stereo_frame(
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        np.array([110.0, 120.0, 130.0], dtype=np.float32),
        np.array(
            [
                StereoTriangulationStatus.TRIANGULATED.value,
                StereoTriangulationStatus.BAD_STEREO.value,
                StereoTriangulationStatus.TRIANGULATED.value,
            ],
            dtype=np.float32,
        ),
    )

    success_mask, landmark_frame = initializer.apply_observation_frame(
        SE3(t=np.array([1.0, 2.0, 3.0], dtype=np.float64)).as_matrix(),
        np.array([True, True, False], dtype=np.bool_),
        stereo_frame,
    )

    np.testing.assert_array_equal(success_mask, np.array([False, False, False], dtype=np.bool_))
    np.testing.assert_allclose(landmark_frame[:, LandmarkInitializationFrameSchema.STEREO], stereo_frame)
    np.testing.assert_array_equal(
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS],
        np.array(
            [
                LandmarkCacheStatus.OBSERVING.value,
                LandmarkCacheStatus.OBSERVING.value,
                LandmarkCacheStatus.EMPTY.value,
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.STATUS],  # noqa: SLF001
        np.array([LandmarkCacheStatus.OBSERVING.value, LandmarkCacheStatus.OBSERVING.value], dtype=np.float64),
    )
    assert cache._data[2, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001
    assert store.get_feat_history(30).shape == (0, ObservationSchema.size())
    assert np.all(np.isnan(store.get_feat_history(20)[0, ObservationSchema.RIGHT_UV]))


def test_apply_observation_frame_triangulates_ready_rows_and_commits_cache() -> None:
    """Ready rows should be triangulated and committed from the frame state."""
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
    triangulator = _MixedTriangulatorSpy(
        [
            (TriangulationStatus.SUCCESS, np.array([1.0, 3.0, 5.0], dtype=np.float64)),
            (TriangulationStatus.SUCCESS, np.array([2.0, 4.0, 6.0], dtype=np.float64)),
        ]
    )
    cache = LandmarkCache.default_factory(capacity=2)
    initializer = LandmarkInitialization(store, cache, triangulator)
    feat_ids = np.array([10.0, 20.0], dtype=np.float32)

    for pose_x, left_u in [(10.0, 0.0), (11.0, 2.0)]:
        success_mask, _landmark_frame = initializer.apply_observation_frame(
            SE3(t=np.array([pose_x, 0.0, 0.0])).as_matrix(),
            np.array([True, True], dtype=np.bool_),
            _stereo_frame(feat_ids, left_u),
        )
        np.testing.assert_array_equal(success_mask, np.array([False, False], dtype=np.bool_))

    success_mask, landmark_frame = initializer.apply_observation_frame(
        SE3(t=np.array([12.0, 0.0, 0.0])).as_matrix(),
        np.array([True, True], dtype=np.bool_),
        _stereo_frame(feat_ids, 3.0),
    )

    np.testing.assert_array_equal(success_mask, np.array([True, True], dtype=np.bool_))
    np.testing.assert_array_equal(
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS],
        np.full((2,), LandmarkCacheStatus.COMPLETED.value, dtype=np.float64),
    )
    np.testing.assert_allclose(
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_XYZ],
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        cache._data[:2, LandmarkCacheSchema.STATUS],  # noqa: SLF001
        np.full((2,), LandmarkCacheStatus.COMPLETED.value, dtype=np.float64),
    )
    assert len(triangulator.calls) == 2


def test_apply_observation_frame_limits_triangulation_batch_and_keeps_extra_ready() -> None:
    """At most six ready rows should be triangulated per frame."""
    store = ObservationStore(
        k_inv=K_INV,
        capacity=8,
        history_size=3,
        ready_criteria=ReadyObservationCriteria(min_history_size=1, min_parallax_rad=0.0),
    )
    triangulator = _MixedTriangulatorSpy(
        [
            (TriangulationStatus.SUCCESS, np.array([float(i), float(i + 1), float(i + 2)], dtype=np.float64))
            for i in range(6)
        ]
    )
    cache = LandmarkCache.default_factory(capacity=8)
    initializer = LandmarkInitialization(store, cache, triangulator)

    success_mask, landmark_frame = initializer.apply_observation_frame(
        np.eye(4, dtype=np.float64),
        np.ones((8,), dtype=np.bool_),
        _stereo_frame(np.arange(100.0, 108.0, dtype=np.float32), np.arange(8.0, dtype=np.float32)),
    )

    np.testing.assert_array_equal(
        success_mask,
        np.array([True, True, True, True, True, True, False, False], dtype=np.bool_),
    )
    assert len(triangulator.calls) == 6
    np.testing.assert_array_equal(
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS],
        np.array(
            [
                LandmarkCacheStatus.COMPLETED.value,
                LandmarkCacheStatus.COMPLETED.value,
                LandmarkCacheStatus.COMPLETED.value,
                LandmarkCacheStatus.COMPLETED.value,
                LandmarkCacheStatus.COMPLETED.value,
                LandmarkCacheStatus.COMPLETED.value,
                LandmarkCacheStatus.READY.value,
                LandmarkCacheStatus.READY.value,
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        cache._data[:, LandmarkCacheSchema.STATUS],  # noqa: SLF001
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS],
    )


def test_apply_observation_frame_uses_cached_completed_and_hard_failed_rows() -> None:
    """Completed and hard-failed cached rows should skip observation ingest."""
    store = ObservationStore(k_inv=K_INV, capacity=2, history_size=3)
    cache = LandmarkCache.default_factory(capacity=2)
    cache.commit(
        np.array([10.0, 20.0], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([LandmarkCacheStatus.COMPLETED.value, LandmarkCacheStatus.FAILED_HARD.value], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]], dtype=np.float64),
    )
    initializer = LandmarkInitialization(store, cache, _MixedTriangulatorSpy())

    success_mask, landmark_frame = initializer.apply_observation_frame(
        np.eye(4, dtype=np.float64),
        np.array([True, True], dtype=np.bool_),
        _stereo_frame(np.array([10.0, 20.0], dtype=np.float32), 0.0),
    )

    np.testing.assert_array_equal(success_mask, np.array([True, False], dtype=np.bool_))
    np.testing.assert_array_equal(
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS],
        np.array([LandmarkCacheStatus.COMPLETED.value, LandmarkCacheStatus.FAILED_HARD.value], dtype=np.float64),
    )
    np.testing.assert_allclose(
        landmark_frame[0, LandmarkInitializationFrameSchema.LANDMARK_XYZ],
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    assert store.get_feat_history(10).shape == (0, ObservationSchema.size())
    assert store.get_feat_history(20).shape == (0, ObservationSchema.size())


def test_apply_observation_frame_uses_cache_failure_policy_for_hard_escalation() -> None:
    """Frame status for failed triangulation should come from cache lifecycle policy."""
    store = ObservationStore(
        k_inv=K_INV,
        capacity=1,
        history_size=3,
        ready_criteria=ReadyObservationCriteria(min_history_size=1, min_parallax_rad=0.0),
    )
    cache = LandmarkCache.default_factory(capacity=1)
    cache._data[0, LandmarkCacheSchema.FEAT_ID] = 10.0  # noqa: SLF001
    cache._data[0, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.FAILED_SOFT.value  # noqa: SLF001
    cache._data[0, LandmarkCacheSchema.ATTEMPTS] = 5.0  # noqa: SLF001
    cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0.0  # noqa: SLF001
    triangulator = _MixedTriangulatorSpy(
        [(TriangulationStatus.NOT_VALID, np.full((3,), np.nan, dtype=np.float64))]
    )
    initializer = LandmarkInitialization(store, cache, triangulator)

    success_mask, landmark_frame = initializer.apply_observation_frame(
        np.eye(4, dtype=np.float64),
        np.array([True], dtype=np.bool_),
        _stereo_frame(np.array([10.0], dtype=np.float32), 0.0),
    )

    np.testing.assert_array_equal(success_mask, np.array([False], dtype=np.bool_))
    assert (
        landmark_frame[0, LandmarkInitializationFrameSchema.LANDMARK_STATUS]
        == LandmarkCacheStatus.FAILED_HARD.value
    )
    assert cache._data[0, LandmarkCacheSchema.ATTEMPTS] == 6.0  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_HARD.value  # noqa: SLF001


def test_remove_lost_features_clears_store_and_cache_slots() -> None:
    """Lost rows should release observation-store slots and clear matching cache rows."""
    store = ObservationStore(k_inv=K_INV, capacity=2, history_size=3)
    cache = LandmarkCache.default_factory(capacity=2)
    initializer = LandmarkInitialization(store, cache, _MixedTriangulatorSpy())
    slots = store._get_feature_slots(np.array([10, 20], dtype=np.int32))  # noqa: SLF001
    cache.commit(
        np.array([10.0, 20.0], dtype=np.float64),
        slots.astype(np.float64),
        np.array([LandmarkCacheStatus.OBSERVING.value, LandmarkCacheStatus.COMPLETED.value], dtype=np.float64),
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]], dtype=np.float64),
    )
    landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
    landmark_frame[:, LandmarkInitializationFrameSchema.FEAT_ID] = np.array([10.0, 20.0], dtype=np.float64)
    landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_SLOT] = slots

    result = initializer.remove_lost_features(landmark_frame, np.array([False, True], dtype=np.bool_))

    assert result is None
    assert store.get_feat_history(10).shape == (0, ObservationSchema.size())
    assert store.get_feat_history(20).shape == (0, ObservationSchema.size())
    assert cache._data[slots[0], LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.OBSERVING.value  # noqa: SLF001
    assert cache._data[slots[1], LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001


def test_triangulate_ready_observations_returns_masks_and_compact_xyz_without_cache_mutation() -> None:
    """Triangulation should return data to the caller instead of mutating the cache."""
    store = ObservationStore(k_inv=K_INV, capacity=1, history_size=5)
    for i in range(3):
        store.add_observations(_observation_row(42, 10.0 + i, float(i), right_valid=i != 1))
    cache = LandmarkCache.default_factory(capacity=1)
    triangulator = _MixedTriangulatorSpy()
    initializer = LandmarkInitialization(store, cache, triangulator)
    landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
    landmark_frame[0, LandmarkInitializationFrameSchema.LANDMARK_SLOT] = 0.0

    failed_mask, success_mask, xyz = initializer.triangulate_ready_observations(
        landmark_frame,
        np.array([True, False], dtype=np.bool_),
    )

    np.testing.assert_array_equal(failed_mask, np.array([False, False], dtype=np.bool_))
    np.testing.assert_array_equal(success_mask, np.array([True, False], dtype=np.bool_))
    np.testing.assert_allclose(xyz, np.array([[1.0, 3.0, 5.0]], dtype=np.float64))
    assert len(triangulator.calls) == 1
    _left_uvs, right_uvs, left_poses = triangulator.calls[0]
    np.testing.assert_allclose(
        right_uvs,
        np.array([[9.0, 20.0], [np.nan, np.nan], [11.0, 20.0]], dtype=np.float64),
    )
    assert left_poses.shape == (3, 4, 4)
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001


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
def test_triangulate_ready_observations_returns_failed_mask_for_non_success_status(
    status: TriangulationStatus,
) -> None:
    """Non-success triangulation statuses should be returned as failed frame rows."""
    store = ObservationStore(k_inv=K_INV, capacity=1, history_size=5)
    for i in range(3):
        store.add_observations(_observation_row(42, 10.0 + i, float(i)))
    cache = LandmarkCache.default_factory(capacity=1)
    triangulator = _MixedTriangulatorSpy([(status, np.full((3,), np.nan, dtype=np.float64))])
    initializer = LandmarkInitialization(store, cache, triangulator)
    landmark_frame = np.full((1, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
    landmark_frame[0, LandmarkInitializationFrameSchema.LANDMARK_SLOT] = 0.0

    failed_mask, success_mask, xyz = initializer.triangulate_ready_observations(
        landmark_frame,
        np.array([True], dtype=np.bool_),
    )

    np.testing.assert_array_equal(failed_mask, np.array([True], dtype=np.bool_))
    np.testing.assert_array_equal(success_mask, np.array([False], dtype=np.bool_))
    assert xyz.shape == (0, 3)
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001


def test_landmark_cache_lookup_returns_empty_for_misaligned_feature_rows() -> None:
    """Lookup should align by slot and feature ID, returning EMPTY for mismatches."""
    cache = LandmarkCache(capacity=1)
    cache.commit(
        np.array([10.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([LandmarkCacheStatus.COMPLETED.value], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
    )

    lookup = cache.lookup(np.array([10, 99], dtype=np.int32), np.array([0, 0], dtype=np.int32))

    assert lookup[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.COMPLETED.value
    np.testing.assert_allclose(lookup[0, LandmarkCacheSchema.XYZ], np.array([1.0, 2.0, 3.0]))
    assert lookup[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value
    assert np.all(np.isnan(lookup[1, LandmarkCacheSchema.XYZ]))


def test_landmark_cache_commit_tracks_soft_failures_hard_failures_and_retry_gate() -> None:
    """Commit should own cache mutation and retry gating for failed rows."""
    cache = LandmarkCache(capacity=3)
    cache._data[0, LandmarkCacheSchema.ATTEMPTS] = 4.0  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.ATTEMPTS] = 5.0  # noqa: SLF001

    cache.commit(
        np.array([10.0, 20.0, 30.0], dtype=np.float64),
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        np.array(
            [
                LandmarkCacheStatus.FAILED_SOFT.value,
                LandmarkCacheStatus.FAILED_SOFT.value,
                LandmarkCacheStatus.COMPLETED.value,
            ],
            dtype=np.float64,
        ),
        np.array([3.0, 3.0, 3.0], dtype=np.float64),
        np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]], dtype=np.float64),
    )

    assert cache._data[0, LandmarkCacheSchema.ATTEMPTS] == 5.0  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_SOFT.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 4.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.ATTEMPTS] == 6.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_HARD.value  # noqa: SLF001
    assert cache._data[2, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.COMPLETED.value  # noqa: SLF001

    cache.commit(
        np.array([10.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([LandmarkCacheStatus.READY.value], dtype=np.float64),
        np.array([3.0], dtype=np.float64),
        np.array([[np.nan, np.nan, np.nan]], dtype=np.float64),
    )
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_SOFT.value  # noqa: SLF001

    cache.commit(
        np.array([10.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([LandmarkCacheStatus.READY.value], dtype=np.float64),
        np.array([4.0], dtype=np.float64),
        np.array([[np.nan, np.nan, np.nan]], dtype=np.float64),
    )
    assert cache._data[0, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.READY.value  # noqa: SLF001
    assert cache._data[0, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 0.0  # noqa: SLF001


def test_landmark_cache_get_completed_landmarks_returns_visualization_rows() -> None:
    """Completed cache entries should be exposed as feat_id + xyz rows."""
    cache = LandmarkCache(capacity=3)
    cache.commit(
        np.array([20.0, 30.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([LandmarkCacheStatus.COMPLETED.value, LandmarkCacheStatus.FAILED_SOFT.value], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]], dtype=np.float64),
    )

    landmarks = cache.get_completed_landmarks()

    np.testing.assert_allclose(
        landmarks,
        np.array([[20.0, 1.0, 2.0, 3.0]], dtype=np.float64),
    )


def test_landmark_cache_clear_slots_resets_payload_and_status() -> None:
    """Clearing cache slots should remove stale ready state before slot reuse."""
    cache = LandmarkCache(capacity=2)
    cache._data[1, LandmarkCacheSchema.FEAT_ID] = 10.0  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.READY.value  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.XYZ] = np.array([1.0, 2.0, 3.0])  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.ATTEMPTS] = 4.0  # noqa: SLF001
    cache._data[1, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 7.0  # noqa: SLF001

    cache.clear_slots(np.array([1], dtype=np.int32))

    assert cache._data[1, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.ATTEMPTS] == 0.0  # noqa: SLF001
    assert cache._data[1, LandmarkCacheSchema.RETRY_AFTER_VERSION] == 0.0  # noqa: SLF001
    assert np.isnan(cache._data[1, LandmarkCacheSchema.FEAT_ID])  # noqa: SLF001
    assert np.all(np.isnan(cache._data[1, LandmarkCacheSchema.XYZ]))  # noqa: SLF001
