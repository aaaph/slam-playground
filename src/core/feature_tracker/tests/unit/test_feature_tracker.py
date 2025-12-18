import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_pool import FeaturePool
from core.feature_tracker.feature_tracker import FeatureTracker
from core.transformations.special_euclidian_3_dim import SE3


class TestFeatureTracker:
    """Unit test for feature tracker."""

    @pytest.fixture
    def stereo_ctx(self) -> StereoContext:
        """Create a stereo camera DTO."""
        return StereoContext(
            stereo_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam0_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam1_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            baseline=1.0,
            cam0_in_body=SE3.identity(),
            cam1_in_body=SE3.identity(),
        )

    @pytest.fixture
    def feature_tracker(self, stereo_ctx: StereoContext) -> FeatureTracker:
        """Create a feature tracker."""
        return FeatureTracker.default_factory(stereo_ctx)

    @pytest.fixture
    def feature_tracker_with_pool(self, feature_tracker: FeatureTracker) -> FeatureTracker:
        """Create a feature tracker with a pool."""
        feature_tracker.pool = FeaturePool()
        return feature_tracker

    def test_feature_tracker_creation_and_properties(self, feature_tracker: FeatureTracker):
        """Test that the feature tracker can be created and has the correct properties."""
        assert feature_tracker is not None
        assert feature_tracker.grid is not None
        assert feature_tracker.fast is not None

        for region in feature_tracker.grid:
            assert region is not None

        assert hasattr(feature_tracker, "feed_first")
        assert callable(feature_tracker.feed_first)
        assert hasattr(feature_tracker, "feed")
        assert callable(feature_tracker.feed)
        assert hasattr(feature_tracker, "get_features_spawned_in_timestamp")
        assert callable(feature_tracker.get_features_spawned_in_timestamp)
        assert hasattr(feature_tracker, "drop_features")
        assert callable(feature_tracker.drop_features)
        assert hasattr(feature_tracker, "get_features_grouped_by_status")
        assert callable(feature_tracker.get_features_grouped_by_status)

    def test_get_features_by_timestamp(self, feature_tracker_with_pool: FeatureTracker):
        """Test that the feature tracker has a get_features_by_timestamp method."""
        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        feature_tracker_with_pool.pool.add_feature(feat)

        features = feature_tracker_with_pool.get_features_spawned_in_timestamp(1)
        assert len(features) == 1
        assert features[0] == feat

    def test_drop_features_method(self, feature_tracker_with_pool: FeatureTracker):
        """Test that the feature tracker has a drop_features method."""
        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        feature_tracker_with_pool.pool.add_feature(feat)

        features = feature_tracker_with_pool.get_features_spawned_in_timestamp(1)
        assert len(features) == 1
        assert features[0] == feat

        feature_tracker_with_pool.drop_features(features)
        features = feature_tracker_with_pool.get_features_spawned_in_timestamp(1)
        assert len(features) == 0

    def test_feature_tracker_get_grouped_features(self, feature_tracker_with_pool: FeatureTracker):
        """Test that the feature tracker has a get_grouped_features method."""
        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        feature_tracker_with_pool.pool.add_feature(feat)

        features = feature_tracker_with_pool.get_features_grouped_by_status()
        assert len(features["new"]) == 1
        assert len(features["tracked"]) == 0
        assert len(features["lost"]) == 0
        feature_tracker_with_pool.pool.apply_left_point(2, 1, 1, 1)
        features = feature_tracker_with_pool.get_features_grouped_by_status()
        assert len(features["new"]) == 0
        assert len(features["tracked"]) == 1
        assert len(features["lost"]) == 0
        feature_tracker_with_pool.pool.mark_features_as_lost(np.array([[1, 1, 1]]).reshape(-1, 3))
        features = feature_tracker_with_pool.get_features_grouped_by_status()
        assert len(features["new"]) == 0
        assert len(features["tracked"]) == 0
        assert len(features["lost"]) == 1

    def test_feature_tracker_get_feature_by_id(self, feature_tracker_with_pool: FeatureTracker):
        """Test that the feature tracker has a get_feature_by_id method."""
        feat = Feature.spawn_from_left_and_right(1, 1, (0, 0), (1, 1))
        feature_tracker_with_pool.pool.add_feature(feat)
        feat_from_method = feature_tracker_with_pool.get_feature_by_id(1)
        assert feat_from_method is not None
        assert feat_from_method.feat_id == 1
        with pytest.raises(ValueError, match="Feature with ID 2 not found"):
            feature_tracker_with_pool.get_feature_by_id(2)
