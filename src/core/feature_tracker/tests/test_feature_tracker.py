import jax.numpy as jnp

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
