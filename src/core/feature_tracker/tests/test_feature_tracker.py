from pathlib import Path

import cv2
import jax.numpy as jnp
import numpy as np
import pytest

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_pool import FeaturePool
from core.feature_tracker.feature_tracker import FeatureTracker
from core.feature_tracker.image_preprocess import StereoImagePreprocess
from dataset.dataset_config import CameraConfig, StereoConfig


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

    def test_sliding_window(self, mocker):
        """Test that the feature tracker manipulates the oldest and newest timestamps."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        ft = FeatureTracker(mock_preprocessor)
        assert hasattr(ft, "oldest_and_newest_timestamps")
        assert callable(ft.oldest_and_newest_timestamps)

        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        ft.pool = FeaturePool()
        ft.pool.add_feature(feat)

        oldest_ts, newest_ts = ft.oldest_and_newest_timestamps()
        assert oldest_ts == 1
        assert newest_ts == 1

        feat = Feature.spawn_from_left_and_right(2, 100, (0, 0), (1, 1))
        ft.pool.add_feature(feat)
        feat = Feature.spawn_from_left_and_right(3, 200, (0, 0), (1, 1))
        ft.pool.add_feature(feat)

        oldest_ts, newest_ts = ft.oldest_and_newest_timestamps()
        assert oldest_ts == 1
        assert newest_ts == 200

    def test_feature_tracker_get_grouped_features(self, mocker):
        """Test that the feature tracker has a get_grouped_features method."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        ft = FeatureTracker(mock_preprocessor)
        assert hasattr(ft, "get_features_grouped_by_status")
        assert callable(ft.get_features_grouped_by_status)

        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        ft.pool = FeaturePool()
        ft.pool.add_feature(feat)

        features = ft.get_features_grouped_by_status()
        assert len(features["new"]) == 1
        assert len(features["tracked"]) == 0
        assert len(features["lost"]) == 0
        ft.pool.apply_left_point(2, 1, 1, 1)
        features = ft.get_features_grouped_by_status()
        assert len(features["new"]) == 0
        assert len(features["tracked"]) == 1
        assert len(features["lost"]) == 0
        ft.pool.mark_features_as_lost(np.array([[1, 1, 1]]).reshape(-1, 3))
        features = ft.get_features_grouped_by_status()
        assert len(features["new"]) == 0
        assert len(features["tracked"]) == 0
        assert len(features["lost"]) == 1

    def test_feature_tracker_get_feature_by_id(self, mocker):
        """Test that the feature tracker has a get_feature_by_id method."""
        mock_preprocessor = mocker.Mock(spec=StereoImagePreprocess)
        ft = FeatureTracker(mock_preprocessor)
        assert hasattr(ft, "get_feature_by_id")
        assert callable(ft.get_feature_by_id)

        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        ft.pool = FeaturePool()
        ft.pool.add_feature(feat)
        feat_from_method = ft.get_feature_by_id(1)
        assert feat_from_method is not None
        assert feat_from_method.feat_id == 1
        with pytest.raises(ValueError, match="Feature with ID 2 not found"):
            ft.get_feature_by_id(2)


class TestIntegrationFeatureTracker:
    """Integration test for feature tracker."""

    CAMERA_CONFIG_0 = CameraConfig(
        {
            "resolution": (752, 480),
            "camera_model": "pinhole",
            "intrinsics": (458.654, 457.296, 367.215, 248.375),
            "distortion_model": "radial-tangential",
            "distortion_coefficients": (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05),
            "T_BS": {
                "cols": 4,
                "rows": 4,
                "data": [
                    0.0148655429818,
                    -0.999880929698,
                    0.00414029679422,
                    -0.0216401454975,
                    0.999557249008,
                    0.0149672133247,
                    0.025715529948,
                    -0.064676986768,
                    -0.0257744366974,
                    0.00375618835797,
                    0.999660727178,
                    0.00981073058949,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        }
    )
    CAMERA_CONFIG_1 = CameraConfig(
        {
            "resolution": (752, 480),
            "camera_model": "pinhole",
            "intrinsics": (457.587, 456.134, 379.999, 255.238),
            "distortion_model": "radial-tangential",
            "distortion_coefficients": (-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05),
            "T_BS": {
                "cols": 4,
                "rows": 4,
                "data": [
                    0.0125552670891,
                    -0.999755099723,
                    0.0182237714554,
                    -0.0198435579556,
                    0.999598781151,
                    0.0130119051815,
                    0.0251588363115,
                    0.0453689425024,
                    -0.0253898008918,
                    0.0179005838253,
                    0.999517347078,
                    0.00786212447038,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        }
    )

    def test_feature_tracker_integration(self, mocker):
        """Test that the feature tracker integration works."""
        current_dir = Path(__file__).resolve().parent
        testing_image_left = cv2.imread(str(current_dir / "testing_image_left.png"))
        testing_image_left = cv2.cvtColor(testing_image_left, cv2.COLOR_BGR2GRAY)
        testing_image_right = cv2.imread(str(current_dir / "testing_image_right.png"))
        testing_image_right = cv2.cvtColor(testing_image_right, cv2.COLOR_BGR2GRAY)
        left, right = np.array(testing_image_left), np.array(testing_image_right)

        config = StereoConfig(self.CAMERA_CONFIG_0, self.CAMERA_CONFIG_1)
        ft = FeatureTracker(config)

        left_out, right_out = ft.feed(1, (left, right))
        assert left_out is not None
        assert right_out is not None
        left_out, right_out = ft.feed(2, (left, right))
        assert left_out is not None
        assert right_out is not None

        old_ts, new_ts = ft.oldest_and_newest_timestamps()
        assert old_ts == 1
        assert new_ts == 2

        feat_size = ft.feat_count()
        assert feat_size > 0
