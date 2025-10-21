import numpy as np

from core.feature_tracker.feature import Feature


class TestUnitFeature:
    """Unit test for feature."""

    def test_should_be_possible_to_create(self):
        """Test that the feature can be created."""
        feature = Feature(1)
        assert feature is not None

    def test_should_have_add_method(self):
        """Test that the feature has an _add method."""
        feature = Feature(1)
        assert hasattr(feature, "_add")

    def test_should_have_from_observations_method(self):
        """Test that the feature has a from_observations method."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None

    def test_should_have_obs_count_method(self):
        """Test that the feature has an obs_count method."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None
        assert feature.obs_count() == 2

    def test_should_have_last_observation_method(self):
        """Test that the feature has a last_observation method."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature.get_last_left() == (1, 1, 0, 0)

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

    def test_feature_should_be_ring_buffer(self):
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

    def test_feature_method_spawn_from_left_and_right(self):
        """Test that the feature method spawn from left and right works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert feature is not None
        assert feature.feat_id == 1
        assert feature.size == 2
        assert feature.head > 0
        assert feature.u[0] == 0
        assert feature.v[0] == 0
        assert feature.u[1] == 1
        assert feature.v[1] == 1

    def test_feature_select_method(self):
        """Test that the feature select method works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        assert hasattr(feature, "select")
        assert callable(feature.select)

        uv = feature.select(1, 0).at[0].get()
        u, v = uv
        assert u == 0
        assert v == 0
        u, v = feature.select(1, 1).at[0].get()
        assert u == 1
        assert v == 1

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
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
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
