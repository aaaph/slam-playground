import numpy as np
import pytest

from core.feature_tracker.feature_tracker_region import FeatureTrackerRegion


class TestFeatureTrackerRegion:
    """Unit test for feature tracker region."""

    @pytest.fixture
    def feature_tracker_region(self) -> FeatureTrackerRegion:
        """Create a feature tracker region."""
        return FeatureTrackerRegion(1, np.array([[1, 1], [1, 1]]))

    def test_should_be_possible_to_create(self, feature_tracker_region: FeatureTrackerRegion):
        """Test that the feature tracker region can be created."""
        assert feature_tracker_region is not None

    def test_should_have_box_property(self, feature_tracker_region: FeatureTrackerRegion):
        """Test that the feature tracker region has a box property."""
        assert feature_tracker_region.box is not None

    def test_should_raise_error_if_region_has_no_pixels(self):
        """Test that the feature tracker region raises an error if the region has no pixels."""
        with pytest.raises(ValueError, match="Region has no pixels"):
            FeatureTrackerRegion(1, np.array([[0, 0], [0, 0]]))

    def test_should_have_region_id_to_string_method(self):
        """Test that the feature tracker region has a region_id_to_string method."""
        region = FeatureTrackerRegion(1, np.array([[1, 1], [1, 1]]))
        assert region.region_id_to_string() is not None
        assert region.region_id_to_string() == "1"
