import numpy as np
import pytest

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

    def test_feature_get_uv_by_timestamp(self):
        """Test that the feature get uv by timestamp works."""
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
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

    def test_feature_stereo_initial_guess(self):
        """Test that the feature stereo initial guess works."""
        k_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        baseline = 0.1
        feature = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        feature.apply_stereo_pair(2, (2, 2), (3, 3))
        assert feature is not None
        assert hasattr(feature, "make_initial_guess")
        assert callable(feature.make_initial_guess)
        # should throw an error if there are no 2 observations
        with pytest.raises(ValueError, match="Feature has no active stereo pair"):
            feature.make_initial_guess(k_matrix, baseline)

        # should throw an error if the disparity is non-positive
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (100, 115))
        with pytest.raises(ValueError, match="Disparity is non-positive"):
            feature.make_initial_guess(k_matrix, baseline)

        # should return the correct initial guess with shape (3,)
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (95, 110))
        initial_guess = feature.make_initial_guess(k_matrix, baseline)
        assert initial_guess is not None
        assert isinstance(initial_guess, np.ndarray)
        assert initial_guess.shape == (3,)
