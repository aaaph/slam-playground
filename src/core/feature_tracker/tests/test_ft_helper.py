from core.feature_tracker.helper import grid_factor


class TestUnitFTHelper:
    """Unit test for feature tracker helper."""

    def test_grid_factor_should_return_correct_values(self):
        """Test that the grid_factor method returns the correct values."""
        assert grid_factor(8) == (2, 4)
        assert grid_factor(4) == (2, 2)
        assert grid_factor(2) == (1, 2)
        assert grid_factor(1) == (1, 1)
        assert grid_factor(0) == (1, 1)
        assert grid_factor(16) == (4, 4)
        assert grid_factor(32) == (4, 8)
