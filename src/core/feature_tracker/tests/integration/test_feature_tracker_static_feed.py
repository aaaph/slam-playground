import numpy as np

from core.feature_tracker.feature_schema import FeatureSchema
from core.feature_tracker.feature_tracker import FeatureTracker


class TestFeatureTrackerStaticFeed:
    """Test for feature tracker static feed."""

    def test_feature_tracker_static_frame_keeping_features(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker output is successful for a static frame and keeps features."""
        left, right = stereo_frame

        tracking_mask, features = feature_tracker.feed(1, (left, right))
        assert features is not None
        feature_size_first = int(np.count_nonzero(tracking_mask))

        tracking_mask, features = feature_tracker.feed(2, (left, right))
        feature_size_second = int(np.count_nonzero(tracking_mask))
        assert feature_size_second >= feature_size_first

        tracking_mask, features = feature_tracker.feed(3, (left, right))

        feature_size_third = int(np.count_nonzero(tracking_mask))
        assert feature_size_third >= feature_size_second

    def test_feature_tracker_features_keep_in_same_position_but_could_have_noise(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the features keep in same position with minimal noise."""
        left, right = stereo_frame
        frames = []
        n_frames = 20
        for ts in range(1, n_frames + 1):
            _tracking_mask, frame = feature_tracker.feed(ts, (left, right))
            frames.append(frame.copy())
        feature_id = 1
        feature_slice = np.full((20, FeatureSchema.count()), np.nan, dtype=np.float32)

        for i, frame in enumerate(frames):
            feat_array = frame[frame[:, FeatureSchema.FEAT_ID].astype(np.int32) == feature_id]
            feature_slice[i, :] = feat_array

        np.testing.assert_array_equal(
            feature_slice[:, 1], np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        )

        u_values = feature_slice[:, 2]
        v_values = feature_slice[:, 3]
        u_sigma = np.std(u_values)
        v_sigma = np.std(v_values)
        noise_threshold = 0.1
        drift_threshold = 0.1
        assert u_sigma < noise_threshold
        assert v_sigma < noise_threshold
        drift_u = abs(u_values[0] - u_values[-1])
        assert drift_u < drift_threshold

    def test_feature_tracker_calculate_age_of_features(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker calculates the age of features correctly."""
        left, right = stereo_frame
        last_age = 0
        for ts in range(1, 11):
            _tracking_mask, frame = feature_tracker.feed(ts, (left, right))
            my_feature_age = frame[1, :]
            last_age = int(my_feature_age[FeatureSchema.AGE])
            assert last_age == ts - 1
            assert int(my_feature_age[FeatureSchema.STEREO_SCORE]) == ts - 1

        assert last_age == 9

    def test_feature_tracker_stereo_close_vertical_points(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker do rectification correctly by validating v values of features."""
        left, right = stereo_frame

        _tracking_mask, feat_frame = feature_tracker.feed(1, (left, right))
        has_stereo = ~np.isnan(feat_frame[:, 4])
        stereo_data = feat_frame[has_stereo]
        left_points = stereo_data[:, 2:4]
        right_points = stereo_data[:, 4:6]
        diff = left_points[:, 1] - right_points[:, 1]

        assert abs(diff).mean() < 1.0
