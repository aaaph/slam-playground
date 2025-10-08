import jax.numpy as jnp
import pytest

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_tracker import FeatureTracker
from core.feature_tracker.observation import Observation


class TestUnitFeatureTracker:
    """Unit test for feature tracker."""

    def test_should_be_possible_to_create(self):
        """Test that the feature tracker can be created."""
        feature_tracker = FeatureTracker()
        assert feature_tracker is not None

    def test_should_have_feed_first_method(self):
        """Test that the feature tracker has a feed_first method."""
        feature_tracker = FeatureTracker()
        assert hasattr(feature_tracker, "feed_first")

    def test_should_have_feed_method(self):
        """Test that the feature tracker has a feed_second method."""
        feature_tracker = FeatureTracker()
        assert hasattr(feature_tracker, "feed")

    def test_should_have_add_feature_method(self):
        """Test that the feature tracker has an add_feature method."""
        feature_tracker = FeatureTracker()
        assert hasattr(feature_tracker, "add_feature")
        feat = Feature(1)
        feat.append(Observation(1, 1, jnp.array([0, 0])))  # pyright: ignore[reportUndefinedVariable]
        feature_tracker.add_feature(feat)
        assert feature_tracker.features[1] is not None

    def test_should_raise_error_if_feature_has_no_observations(self):
        """Test that the feature tracker raises an error if a feature has no observations."""
        feature_tracker = FeatureTracker()
        feat = Feature(1)
        with pytest.raises(ValueError, match="Feature has no observations"):
            feature_tracker.add_feature(feat)
