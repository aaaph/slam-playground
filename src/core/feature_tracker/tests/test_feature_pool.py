import numpy as np
import pytest

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_pool import FeaturePool


class TestFeaturePool:
    """Test the feature pool."""

    def test_feature_pool(self) -> None:
        """Test the feature pool."""
        feature_pool = FeaturePool()
        assert feature_pool is not None

    def test_should_have_method_add_feature(self) -> None:
        """Test class should have method add_feature."""
        fp = FeaturePool()
        assert hasattr(fp, "add_feature")

    def test_fp_should_have_features_field_with_map_type(self) -> None:
        """Test fp should have features field with map type."""
        fp = FeaturePool()
        assert isinstance(fp.features, dict)

    def test_fp_should_throw_exception_if_add_feature_with_no_observations(self) -> None:
        """Test fp should throw exception if add feature with no observations."""
        fp = FeaturePool()
        with pytest.raises(ValueError, match="Feature has no observations"):
            fp.add_feature(Feature(1))

    def test_fp_should_increment_feat_id_counter_on_add_feature(self) -> None:
        """Test fp should increment feat id counter on add feature."""
        fp = FeaturePool(100)
        feat = Feature.spawn_from_left_and_right(fp.feat_id_counter, 1, (0, 0), (1, 1))
        fp.add_feature(feat)
        assert fp.feat_id_counter == 101

    def test_fp_after_add_feature_should_have_feature_in_features_map(self) -> None:
        """Test fp after add feature should have feature in features map."""
        fp = FeaturePool(100)
        feat = Feature.spawn_from_left_and_right(fp.feat_id_counter, 1, (0, 0), (1, 1))
        fp.add_feature(feat)
        assert fp.features[feat.feat_id] == feat

    def test_fp_should_have_method_get_active_points_ready_for_klt(self) -> None:
        """Test fp should have method get active points ready for KLT."""
        fp = FeaturePool()
        assert hasattr(fp, "get_active_points_ready_for_klt")
        assert callable(fp.get_active_points_ready_for_klt)

        feat = Feature.spawn_from_left_and_right(10, 1, (0, 0), (1, 1))
        feat.apply_left_only(1, (0, 0))
        fp.add_feature(feat)
        assert np.array_equal(fp.get_active_points_ready_for_klt(), np.array([[10, 0, 0]]).reshape(-1, 3))

        new_feat = Feature.spawn_from_left_and_right(10, 2, (2, 2), (3, 3))
        new_feat.apply_stereo_pair(2, (2, 2), (3, 3))
        fp.add_feature(new_feat)
        assert np.array_equal(fp.get_active_points_ready_for_klt(), np.array([[10, 2, 2]]).reshape(-1, 3))

        third_feat = Feature.spawn_from_left_and_right(11, 2, (4, 4), (5, 5))
        fp.add_feature(third_feat)
        assert np.array_equal(
            fp.get_active_points_ready_for_klt(), np.array([[10, 2, 2], [11, 4, 4]]).reshape(-1, 3)
        )
