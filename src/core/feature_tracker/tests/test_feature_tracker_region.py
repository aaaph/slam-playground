import jax.numpy as jnp
import pytest

from core.feature_tracker.feature_tracker_region import FeatureTrackerRegion


class TestUnitFeatureTrackerRegion:
    """Unit test for feature tracker region."""

    def test_should_be_possible_to_create(self):
        """Test that the feature tracker region can be created."""
        region = FeatureTrackerRegion((1, 1), jnp.array([[1, 1], [1, 1]]))
        assert region is not None

    def test_should_have_box_property(self):
        """Test that the feature tracker region has a box property."""
        region = FeatureTrackerRegion((1, 1), jnp.array([[1, 1], [1, 1]]))
        assert region.box is not None

    def test_should_raise_error_if_region_has_no_pixels(self):
        """Test that the feature tracker region raises an error if the region has no pixels."""
        with pytest.raises(ValueError, match="Region has no pixels"):
            FeatureTrackerRegion((1, 1), jnp.array([[0, 0], [0, 0]]))

    def test_should_have_region_id_to_string_method(self):
        """Test that the feature tracker region has a region_id_to_string method."""
        region = FeatureTrackerRegion((1, 1), jnp.array([[1, 1], [1, 1]]))
        assert region.region_id_to_string() is not None
        assert region.region_id_to_string() == "[1,1]"
