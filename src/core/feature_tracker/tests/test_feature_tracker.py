import jax.numpy as jnp

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_pool import FeaturePool
from core.feature_tracker.feature_tracker import FeatureTracker
from core.feature_tracker.image_preprocess import StereoImagePreprocess


class TestUnitFeatureTracker:
    """Unit test for feature tracker."""

    def test_should_be_possible_to_create(self, mocker):
        """Test that the feature tracker can be created."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        feature_tracker = FeatureTracker(mock_preprocessor)
        assert feature_tracker is not None

    def test_feature_tracker_initialization(self, mocker):
        """Test that the feature tracker initialization works."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        mock_preprocessor.preprocess_stereo.return_value = (
            jnp.array([[1, 2], [3, 4]]),
            jnp.array([[5, 6], [7, 8]]),
        )
        feature_tracker = FeatureTracker(mock_preprocessor)
        assert feature_tracker.grid is not None
        assert feature_tracker.fast is not None

        for region in feature_tracker.grid:
            assert region is not None

    def test_feed_first_method(self, mocker):
        """Test that the feature tracker has a feed_first method."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        feature_tracker = FeatureTracker(mock_preprocessor)
        assert hasattr(feature_tracker, "feed_first")
        assert callable(feature_tracker.feed_first)

    def test_feed_method(self, mocker):
        """Test that the feature tracker has a feed_second method."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        feature_tracker = FeatureTracker(mock_preprocessor)
        assert hasattr(feature_tracker, "feed")

    def test_get_features_by_timestamp(self, mocker):
        """Test that the feature tracker has a get_features_by_timestamp method."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        ft = FeatureTracker(mock_preprocessor)
        assert hasattr(ft, "get_features_spawned_in_timestamp")
        assert callable(ft.get_features_spawned_in_timestamp)

        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        ft.pool = FeaturePool()

        ft.pool.add_feature(feat)

        features = ft.get_features_spawned_in_timestamp(1)
        assert len(features) == 1
        assert features[0] == feat

    def test_drop_features_method(self, mocker):
        """Test that the feature tracker has a drop_features method."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        ft = FeatureTracker(mock_preprocessor)
        assert hasattr(ft, "drop_features")
        assert callable(ft.drop_features)

        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        ft.pool = FeaturePool()
        ft.pool.add_feature(feat)

        features = ft.get_features_spawned_in_timestamp(1)
        assert len(features) == 1
        assert features[0] == feat

        ft.drop_features(features)
        features = ft.get_features_spawned_in_timestamp(1)
        assert len(features) == 0
