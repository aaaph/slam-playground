from core.feature_tracker.helper import grid_factor
from core.feature_tracker.my_collections import ResettableDict


class TestFeatureTrackerHelpers:
    """Unit test for feature tracker helper."""

    def test_grid_factor(self):
        """Test that the grid_factor method returns the correct values."""
        assert grid_factor(8) == (2, 4)
        assert grid_factor(4) == (2, 2)
        assert grid_factor(2) == (1, 2)
        assert grid_factor(1) == (1, 1)
        assert grid_factor(0) == (1, 1)
        assert grid_factor(16) == (4, 4)
        assert grid_factor(32) == (4, 8)
        assert grid_factor(160) == (10, 16)

    def test_my_collections(self):
        """Test that the my_collections method returns the correct values."""
        my_default_dict = {0: 0}
        my_resettable_dict = ResettableDict(my_default_dict)
        my_resettable_dict[0] = 1
        assert my_resettable_dict[0] == 1
        my_resettable_dict.clear()
        assert my_resettable_dict[0] == 0

        my_dict_with_set = {0: set()}
        my_resettable_dict_with_set = ResettableDict(my_dict_with_set)
        my_resettable_dict_with_set[0].add(1)
        assert my_resettable_dict_with_set[0] == {1}
        my_resettable_dict_with_set.clear()
        assert my_resettable_dict_with_set[0] == set()
