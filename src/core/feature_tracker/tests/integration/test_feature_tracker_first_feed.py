import numpy as np

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
        tracking_mask, active_features = feature_tracker.feed_first(1, (left, right))
        assert feature_tracker.iterator_count == iteration_before + 1
        assert tracking_mask.shape == (active_features.shape[0],)
        assert active_features.shape[1] == FeatureSchema.count()

    def test_feature_tracker_init_only_stereo(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """
        Test that the feature tracker init using only stereo points.

        Should return ActiveFeatures without nan right points.
        """
        left, right = stereo_frame
        tracking_mask, active_features = feature_tracker.feed_first(1, (left, right))

        assert active_features.shape[0] != 0
        assert active_features.shape[1] == FeatureSchema.count()
        assert not np.isnan(active_features[:, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1]).any()
        np.testing.assert_array_equal(tracking_mask, np.ones((active_features.shape[0],), dtype=np.bool_))
        np.testing.assert_array_equal(
            active_features[:, FeatureSchema.STEREO_SCORE],
            np.zeros(active_features.shape[0], dtype=np.float32),
        )

    def test_feature_tracker_init_by_black_image(self, feature_tracker: FeatureTracker):
        """Test that the feature tracker init by black image."""
        black_image = np.zeros((480, 752), dtype=np.uint8)
        tracking_mask, active_features = feature_tracker.feed_first(1, (black_image, black_image))
        assert tracking_mask.shape == (0,)
        assert active_features.shape[0] == 0
        assert active_features.shape[1] == FeatureSchema.count()
        assert np.isnan(active_features[:, 1:5]).all()

    def test_feature_tracker_stereo_same(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker init by stereo points that are the same."""
        left, _ = stereo_frame
        left_shifted = np.roll(left, 1, axis=0)
        _tracking_mask, active_features = feature_tracker.feed_first(1, (left, left_shifted))
        assert active_features.shape[0] != 0
        assert active_features.shape[1] == FeatureSchema.count()
