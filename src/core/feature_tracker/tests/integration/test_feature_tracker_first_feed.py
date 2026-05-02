import numpy as np

from core.feature_tracker.feature_frame import FeatureFrame
from core.feature_tracker.feature_schema import FeatureSchema
from core.feature_tracker.feature_tracker import FeatureTracker


class TestFeatureTrackerFirstFeed:
    """Test for feature tracker first feed."""

    def test_feature_tracker_return_active_features(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker return active features."""
        iteration_before = feature_tracker.iterator_count
        left, right = stereo_frame
        active_features = feature_tracker.feed_first(1, (left, right))
        assert feature_tracker.iterator_count == iteration_before + 1
        assert isinstance(active_features, FeatureFrame)

    def test_feature_tracker_init_only_stereo(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """
        Test that the feature tracker init using only stereo points.

        Should return ActiveFeatures without nan right points.
        """
        left, right = stereo_frame
        active_features = feature_tracker.feed_first(1, (left, right))

        assert active_features.ndarray.shape[0] != 0
        assert active_features.ndarray.shape[1] == FeatureSchema.count()
        assert not np.isnan(active_features.ndarray[:, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1]).any()
        np.testing.assert_array_equal(
            active_features.ndarray[:, FeatureSchema.STEREO_SCORE],
            np.zeros(active_features.ndarray.shape[0], dtype=np.float32),
        )

    def test_feature_tracker_init_by_black_image(self, feature_tracker: FeatureTracker):
        """Test that the feature tracker init by black image."""
        black_image = np.zeros((480, 752), dtype=np.uint8)
        active_features = feature_tracker.feed_first(1, (black_image, black_image))
        assert active_features.ndarray.shape[0] == 0
        assert active_features.ndarray.shape[1] == FeatureSchema.count()
        assert np.isnan(active_features.ndarray[:, 1:5]).all()

    def test_feature_tracker_stereo_same(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker init by stereo points that are the same."""
        left, _ = stereo_frame
        left_shifted = np.roll(left, 1, axis=0)
        active_features = feature_tracker.feed_first(1, (left, left_shifted))
        assert active_features.ndarray.shape[0] != 0
        assert active_features.ndarray.shape[1] == FeatureSchema.count()
