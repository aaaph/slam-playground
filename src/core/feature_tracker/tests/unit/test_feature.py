import numpy as np
import pytest

from core.feature_tracker.feature import Feature, FeatureStatus


class TestFeature:
    """Unit test for feature."""

    @pytest.fixture
    def feature(self) -> Feature:
        """Create a feature."""
        return Feature(1)

    def test_feature_methods_and_properties_accessability(self, feature: Feature):
        """Test that the feature can be created."""
        assert feature is not None
        assert hasattr(feature, "_add")

    def test_feature_spawn_method(self):
        """Test that the feature has a from_observations method."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None
        assert feature.feat_id == 1
        assert feature.size > 0
        assert feature.head > 0
        assert feature.iteration_life > 0
        assert feature.u[0] == 0
        assert feature.v[0] == 0
        assert feature.u[1] == 1
        assert feature.v[1] == 1

    def test_feature_column_structure(self):
        """Test that the feature has a column structure."""
        feature = Feature(1, capacity=2)
        feature.apply_left_only(1, (0, 0))
        feature.apply_left_only(2, (1, 1))
        assert np.array_equal(feature.ts, np.array([1, 2], dtype=np.float32))
        assert np.array_equal(feature.cam_id, np.array([0, 0], dtype=np.int32))
        assert np.array_equal(feature.u, np.array([0, 1], dtype=np.float32))
        assert np.array_equal(feature.v, np.array([0, 1], dtype=np.float32))
        assert feature.size == 2

    def test_feature_ring_buffer(self):
        """Test that the feature is ring buffer."""
        feature = Feature(1, capacity=2)
        feature.apply_left_only(1, (0, 0))
        feature.apply_left_only(2, (1, 1))
        feature.apply_left_only(3, (2, 2))
        assert feature.size == 2
        assert feature.head == 1
        assert np.array_equal(feature.ts, np.array([3, 2], dtype=np.float32))
        feature.apply_left_only(4, (3, 3))
        assert feature.size == 2
        assert feature.head == 0
        assert np.array_equal(feature.ts, np.array([3, 4], dtype=np.float32))
        assert feature.u[0] == 2

    def test_feature_stereo_pair_idx(self):
        """Test that the feature stereo pair idx works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None

        _, left_uv, right_uv = feature.get_active_stereo_pair()
        assert left_uv == (0, 0)
        assert right_uv == (1, 1)

        feature.apply_left_only(2, (2, 2))
        _, left_uv, right_uv = feature.get_active_stereo_pair()
        assert left_uv == (2, 2)
        assert right_uv is None

        feature.apply_stereo_pair(4, (4, 4), (5, 5))
        _, left_uv, right_uv = feature.get_active_stereo_pair()
        assert left_uv == (4, 4)
        assert right_uv == (5, 5)

    def test_feature_get_tail(self):
        """Test that the feature get tail works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1), feat_capacity=10)
        feature.apply_stereo_pair(2, (2, 2), (3, 3))
        feature.apply_stereo_pair(3, (3, 3), (4, 4))
        feature.apply_stereo_pair(4, (4, 4), (5, 5))
        feature.apply_stereo_pair(5, (5, 5), (6, 6))

        assert hasattr(feature, "get_tail")

        _, active_left, _ = feature.get_active_stereo_pair()
        tail = feature.get_tail(0)
        assert active_left == (5, 5)
        # assert for array includes all of the following:
        assert all(item in tail for item in [(4, 4), (3, 3), (2, 2), (0, 0)])

    def test_feature_get_uv_by_timestamp(self):
        """Test that the feature get uv by timestamp works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1), feat_capacity=10)
        feature.apply_stereo_pair(2, (2, 2), (3, 3))
        feature.apply_stereo_pair(3, (3, 3), (4, 4))
        feature.apply_stereo_pair(4, (4, 4), (5, 5))
        feature.apply_stereo_pair(5, (5, 5), (6, 6))
        assert hasattr(feature, "get_uv_by_timestamp")
        assert callable(feature.get_uv_by_timestamp)
        uv = feature.get_uv_by_timestamp(2)
        assert uv == [(0, 2, 2), (1, 3, 3)]
        uv = feature.get_uv_by_timestamp(3)
        assert uv == [(0, 3, 3), (1, 4, 4)]
        uv = feature.get_uv_by_timestamp(4)
        assert uv == [(0, 4, 4), (1, 5, 5)]
        uv = feature.get_uv_by_timestamp(5)
        assert uv == [(0, 5, 5), (1, 6, 6)]

    def test_feature_iterate(self):
        """Test that the feature iterate works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None
        assert hasattr(feature, "iterate")
        assert callable(feature.iterate)
        iterator = feature.iterate()
        assert iterator is not None
        assert hasattr(iterator, "__iter__")
        assert callable(iterator.__iter__)
        list_uvs = []
        for item in iterator:
            assert item is not None
            assert isinstance(item, tuple)
            u = item[2]
            v = item[3]
            list_uvs.append((u, v))
        assert len(list_uvs) == feature.size
        assert all(item in list_uvs for item in [(0, 0), (1, 1)])

    def test_feature_get_active_measurement(self):
        """Test that the feature get current feature measurement works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None
        assert hasattr(feature, "get_active_measurement")
        assert callable(feature.get_active_measurement)
        measurement = feature.get_active_measurement()
        assert measurement is not None

        feature.apply_left_only(2, (2, 2))
        measurement = feature.get_active_measurement()
        assert measurement.timestamp == 2
        assert measurement.left == (2, 2)
        assert measurement.right is None
        assert measurement.is_left_only()

        feature.apply_stereo_pair(3, (3, 3), (4, 4))
        measurement = feature.get_active_measurement()
        assert measurement.timestamp == 3
        assert measurement.left == (3, 3)
        assert measurement.right == (4, 4)
        assert measurement.is_stereo()
        assert measurement.as_tuple() == (3, (3, 3), (4, 4))

    def test_feature_spawn_from_ndarray(self):
        """Test that the feature spawn from ndarray works."""
        ndarray = np.array([1, 1, 0, 0, 1, 1, 1], dtype=np.float32)
        feature = Feature.spawn_from_ndarray(ndarray)
        assert feature is not None
        assert feature.feat_id == 1
        assert feature.size == 2
        assert feature.head == 2
        assert feature.ts[0] == 1
        assert feature.cam_id[0] == 0
        assert feature.u[0] == 0
        assert feature.v[0] == 0
        assert feature.u[1] == 1
        assert feature.v[1] == 1
        assert feature.state == FeatureStatus.TRACKED
