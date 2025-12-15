import numpy as np
import pytest

from core.pose_tracker.local_map import LocalMap


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
