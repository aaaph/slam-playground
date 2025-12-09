import numpy as np
import pytest

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_pool import FeaturePool


class TestFeaturePool:
    """Test the feature pool."""

    @pytest.fixture
    def feature_pool(self) -> FeaturePool:
        """Create a feature pool."""
        return FeaturePool()

    def test_feature_pool_methods_properites(self, feature_pool: FeaturePool):
        """Test that the feature pool methods and properties are callable and have the correct type."""
        assert hasattr(feature_pool, "add_feature")
        assert callable(feature_pool.add_feature)
        assert isinstance(feature_pool.features, dict)
        assert hasattr(feature_pool, "get_active_points_ready_for_klt")
        assert callable(feature_pool.get_active_points_ready_for_klt)

    def test_feature_pool_add_empty_feature(self, feature_pool: FeaturePool):
        """Test that the feature pool throws an exception if add feature with no observations."""
        with pytest.raises(ValueError, match="Feature has no observations"):
            feature_pool.add_feature(Feature(1))

    def test_feature_pool_increment_feat_id(self) -> None:
        """Test fp should increment feat id counter on add feature."""
        fp = FeaturePool(100)
        feat = Feature.spawn_from_left_and_right(fp.feat_id_counter, 1, (0, 0), (1, 1))
        fp.add_feature(feat)
        assert fp.feat_id_counter == 101

    def test_feature_pool_dictionary_after_add_feature(self, feature_pool: FeaturePool):
        """Test fp after add feature should have feature in features map."""
        feat = Feature.spawn_from_left_and_right(feature_pool.feat_id_counter, 1, (0, 0), (1, 1))
        feature_pool.add_feature(feat)
        assert feature_pool.features[feat.feat_id] == feat

    def test_feature_pool_get_active_points_ready_for_klt(self, feature_pool: FeaturePool) -> None:
        """Test that the feature pool can get active points ready for KLT."""
        feat = Feature.spawn_from_left_and_right(10, 1, (0, 0), (1, 1))
        feat.apply_left_only(1, (0, 0))
        feature_pool.add_feature(feat)
        assert np.array_equal(
            feature_pool.get_active_points_ready_for_klt(), np.array([[10, 0, 0]]).reshape(-1, 3)
        )

        new_feat = Feature.spawn_from_left_and_right(10, 2, (2, 2), (3, 3))
        new_feat.apply_stereo_pair(2, (2, 2), (3, 3))
        feature_pool.add_feature(new_feat)
        assert np.array_equal(
            feature_pool.get_active_points_ready_for_klt(), np.array([[10, 2, 2]]).reshape(-1, 3)
        )

        third_feat = Feature.spawn_from_left_and_right(11, 2, (4, 4), (5, 5))
        feature_pool.add_feature(third_feat)
        assert np.array_equal(
            feature_pool.get_active_points_ready_for_klt(), np.array([[10, 2, 2], [11, 4, 4]]).reshape(-1, 3)
        )
