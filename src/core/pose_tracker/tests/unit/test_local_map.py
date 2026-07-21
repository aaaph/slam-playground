import numpy as np
import pytest

from core.pose_tracker.local_map import LocalMap, LocalMapPointSource, LocalMapSchema


def make_local_map_row(
    feat_id: int,
    xyz: list[float],
    covariance: list[float] | None = None,
    depth_sigma: float | None = None,
) -> np.ndarray:
    """Create a local-map row for tests."""
    row = np.full(LocalMapSchema.count(), np.nan, dtype=np.float64)
    row[LocalMapSchema.FEAT_ID] = feat_id
    row[LocalMapSchema.XYZ] = xyz
    if covariance is not None:
        row[LocalMapSchema.COV] = covariance
    if depth_sigma is not None:
        row[LocalMapSchema.DEPTH_SIGMA] = depth_sigma
    return row


class TestLocalMap:
    """Unit test for local map."""

    @pytest.fixture
    def local_map(self) -> LocalMap:
        """Create a local map."""
        return LocalMap(10)

    def test_local_map_keep_points_within_capacity(self, local_map: LocalMap):
        """Test that the local map keeps points within capacity."""
        for i in range(10):
            local_map.add_points({i: np.array([i, i, i])})
        assert len(local_map.landmarks) == 10
        local_map.add_points({10: np.array([10, 10, 10])})
        assert len(local_map.landmarks) == 10
        assert 0 not in local_map.landmarks

    def test_local_map_should_be_lru(self, local_map: LocalMap):
        """Test that the local map is LRU."""
        for i in range(10):
            local_map.add_points({i: np.array([i, i, i])})
        assert local_map.get_point(0) is not None
        local_map.add_points({20: np.array([20, 20, 20])})
        assert local_map.get_point(0) is not None, "Element 0 should have survived because it was accessed"
        assert local_map.get_point(1) is None, "Element 1 should have been evicted"

    def test_local_map_order_integrity(self, local_map: LocalMap):
        """Test that the local map maintains order integrity."""
        for i in range(10):
            local_map.add_points({i: np.array([i, i, i])})

        keys = list(local_map.landmarks.keys())
        assert keys == list(range(10))
        local_map.get_point(0)
        keys = list(local_map.landmarks.keys())
        assert keys == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    def test_local_map_exists_method(self, local_map: LocalMap):
        """Test that the local map exists method works."""
        for i in range(10):
            local_map.add_points({i: np.array([i, i, i])})
        assert local_map.exists(0)
        assert not local_map.exists(100)

    def test_local_map_add_ndarray(self, local_map: LocalMap):
        """Test that the local map add ndarray method works."""
        tensor = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

        assert hasattr(local_map, "add_ndarray")
        local_map.add_ndarray(tensor)
        assert local_map.exists(0)
        assert local_map.exists(1)
        assert local_map.exists(2)
        assert not local_map.exists(3)
        assert local_map.get_point(0) is not None
        assert local_map.get_point(1) is not None
        assert local_map.get_point(2) is not None
        assert local_map.get_point(3) is None

    def test_frontend_observation_stores_covariance_and_metadata(self, local_map: LocalMap) -> None:
        """Test that frontend observations populate covariance and ownership metadata."""
        covariance = [1.0, 0.1, 0.2, 0.1, 2.0, 0.3, 0.2, 0.3, 3.0]
        row = make_local_map_row(7, [1.0, 2.0, 3.0], covariance, depth_sigma=0.5)

        local_map.add_frontend_observations(np.array([row]), timestamp_ns=100.0)

        mask, points = local_map.get_stable_batch(np.array([7], dtype=np.int32))
        assert mask[0]
        np.testing.assert_allclose(points[0, LocalMapSchema.XYZ], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(points[0, LocalMapSchema.COV], np.array(covariance))
        assert points[0, LocalMapSchema.DEPTH_SIGMA] == pytest.approx(0.5)
        assert points[0, LocalMapSchema.SOURCE] == LocalMapPointSource.FRONTEND_CANDIDATE.value
        assert points[0, LocalMapSchema.OBS_COUNT] == pytest.approx(1.0)
        assert points[0, LocalMapSchema.FIRST_SEEN_TS] == pytest.approx(100.0)
        assert points[0, LocalMapSchema.LAST_UPDATED_TS] == pytest.approx(100.0)

    def test_frontend_candidate_updates_use_covariance_fusion(self, local_map: LocalMap) -> None:
        """Test that frontend candidate updates fuse repeated observations using covariance."""
        covariance = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        first_row = make_local_map_row(7, [0.0, 0.0, 0.0], covariance, depth_sigma=1.0)
        second_row = make_local_map_row(7, [2.0, 0.0, 0.0], covariance, depth_sigma=1.0)

        local_map.add_frontend_observations(np.array([first_row]), timestamp_ns=100.0)
        local_map.add_frontend_observations(np.array([second_row]), timestamp_ns=200.0)

        mask, points = local_map.get_stable_batch(np.array([7], dtype=np.int32))
        assert mask[0]
        np.testing.assert_allclose(points[0, LocalMapSchema.XYZ], np.array([1.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(
            points[0, LocalMapSchema.COV],
            np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]),
            atol=1e-8,
        )
        assert points[0, LocalMapSchema.OBS_COUNT] == pytest.approx(2.0)
        assert points[0, LocalMapSchema.LAST_UPDATED_TS] == pytest.approx(200.0)

    def test_backend_landmark_owns_point_after_feedback(self, local_map: LocalMap) -> None:
        """Test that backend optimized landmarks block later frontend xyz/cov overwrites."""
        frontend_covariance = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        frontend_row = make_local_map_row(7, [1.0, 2.0, 3.0], frontend_covariance, depth_sigma=1.0)
        backend_row = np.array([[7.0, 10.0, 20.0, 30.0, 1.0]], dtype=np.float64)
        later_frontend_row = make_local_map_row(
            7,
            [100.0, 200.0, 300.0],
            [9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0],
            depth_sigma=9.0,
        )

        local_map.add_frontend_observations(np.array([frontend_row]), timestamp_ns=100.0)
        local_map.apply_backend_landmarks(backend_row, timestamp_ns=200.0)
        local_map.add_frontend_observations(np.array([later_frontend_row]), timestamp_ns=300.0)

        mask, points = local_map.get_stable_batch(np.array([7], dtype=np.int32))
        assert mask[0]
        np.testing.assert_allclose(points[0, LocalMapSchema.XYZ], np.array([10.0, 20.0, 30.0]))
        np.testing.assert_allclose(points[0, LocalMapSchema.COV], np.array(frontend_covariance))
        assert points[0, LocalMapSchema.DEPTH_SIGMA] == pytest.approx(1.0)
        assert points[0, LocalMapSchema.SOURCE] == LocalMapPointSource.BACKEND_OPTIMIZED.value
        assert points[0, LocalMapSchema.OBS_COUNT] == pytest.approx(2.0)
        assert points[0, LocalMapSchema.BACKEND_VERSION] == pytest.approx(1.0)
        assert points[0, LocalMapSchema.LAST_OBSERVED_TS] == pytest.approx(300.0)
        assert points[0, LocalMapSchema.LAST_UPDATED_TS] == pytest.approx(200.0)

    def test_local_map_get_batch(self, local_map: LocalMap):
        """Test that the local map get batch method works."""
        for i in range(10):
            local_map.add_points({i: np.array([i, i, i])})
        feat_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 50])
        mask, points = local_map.get_batch(feat_ids)
        assert points.shape[0] == 10
        assert points.shape[1] == 4
        assert points[0, LocalMapSchema.FEAT_ID] == 0
        assert points[0, LocalMapSchema.X] == 0
        assert points[0, LocalMapSchema.Y] == 0
        assert points[0, LocalMapSchema.Z] == 0

        assert points[9, LocalMapSchema.FEAT_ID] == 50
        assert np.isnan(points[9, LocalMapSchema.X])
        assert np.isnan(points[9, LocalMapSchema.Y])
        assert np.isnan(points[9, LocalMapSchema.Z])

        assert not mask[9]

    def test_local_map_health_updates(self, local_map: LocalMap):
        """Test that the local map health can be updated by feature id."""
        local_map.add_points({7: np.array([7, 7, 7])})
        feat_ids = np.array([7])

        np.testing.assert_array_equal(local_map.increase_health(feat_ids), np.array([2], dtype=np.int32))
        np.testing.assert_array_equal(local_map.decrease_health(feat_ids), np.array([1], dtype=np.int32))

        idx = local_map._find_slots(feat_ids)  # noqa: SLF001
        np.testing.assert_array_equal(local_map._data[idx, LocalMapSchema.HEALTH], np.array([1], dtype=np.float32))  # noqa: SLF001

    def test_local_map_get_stable_batch(self, local_map: LocalMap):
        """Test that stable batch returns only landmarks above the health threshold."""
        local_map.add_points(
            {
                1: np.array([1, 1, 1], dtype=np.float32),
                2: np.array([2, 2, 2], dtype=np.float32),
                3: np.array([3, 3, 3], dtype=np.float32),
            }
        )
        for _ in range(5):
            local_map.decrease_health(np.array([2], dtype=np.int32))

        mask, points = local_map.get_stable_batch(np.array([1, 2, 3, 50], dtype=np.int32))

        np.testing.assert_array_equal(mask, np.array([True, False, True, False]))
        assert points.shape == (4, LocalMapSchema.count())
        assert points[0, LocalMapSchema.FEAT_ID] == 1
        assert points[0, LocalMapSchema.X] == 1
        assert points[0, LocalMapSchema.HEALTH] == 1
        assert points[1, LocalMapSchema.FEAT_ID] == 2
        assert np.isnan(points[1, LocalMapSchema.X])
        assert np.isnan(points[1, LocalMapSchema.HEALTH])
        assert points[2, LocalMapSchema.FEAT_ID] == 3
        assert points[2, LocalMapSchema.X] == 3
        assert points[2, LocalMapSchema.HEALTH] == 1
        assert points[3, LocalMapSchema.FEAT_ID] == 50
        assert np.isnan(points[3, LocalMapSchema.X])

    def test_local_map_get_points_with_covariance(self, local_map: LocalMap) -> None:
        """Test that local-map covariance snapshots include only rows with finite covariance."""
        covariance = [1.0, 0.1, 0.2, 0.1, 2.0, 0.3, 0.2, 0.3, 3.0]
        local_map.add_frontend_observations(
            np.array([make_local_map_row(7, [1.0, 2.0, 3.0], covariance, depth_sigma=0.5)]),
            timestamp_ns=100.0,
        )
        local_map.add_point(8, np.array([4.0, 5.0, 6.0], dtype=np.float64))

        points = local_map.get_points_with_covariance()
        points[0, LocalMapSchema.X] = 999.0

        fresh_points = local_map.get_points_with_covariance()

        assert fresh_points.shape == (1, LocalMapSchema.count())
        assert fresh_points[0, LocalMapSchema.FEAT_ID] == 7.0
        np.testing.assert_allclose(fresh_points[0, LocalMapSchema.XYZ], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(fresh_points[0, LocalMapSchema.COV], np.array(covariance))
