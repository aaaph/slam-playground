import numpy as np
import pytest

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_triangulation import FeatureTriangulation


class TestFeatureTriangulation:
    """Test feature triangulation module."""

    def test_should_be_possible_to_create(self):
        """Test that the feature triangulation module can be created."""
        feat_triang = FeatureTriangulation(np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]), 0.1)
        assert feat_triang is not None

    def test_make_initial_guess(self):
        """Test that the feature triangulation module can make an initial guess."""
        k_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
        baseline = 0.1
        feat_triang = FeatureTriangulation(k_matrix, baseline)
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (95, 110))
        initial_guess = feat_triang.make_initial_guess(feature)
        assert initial_guess is not None
        assert initial_guess.shape == (3,)

        # should throw an error if the feature has no active stereo pair
        feature = Feature(1)
        with pytest.raises(ValueError, match="Feature has no active stereo pair"):
            feat_triang.make_initial_guess(feature)

        feature.apply_left_only(2, (0, 0))
        with pytest.raises(ValueError, match="Feature has no active right point"):
            feat_triang.make_initial_guess(feature)

        # should throw an error if the disparity is non-positive
        feature = Feature.spawn_from_left_and_right(1, 1, (100, 115), (100, 115))
        with pytest.raises(ValueError, match="Disparity is non-positive"):
            feat_triang.make_initial_guess(feature)
