import pytest

from core.front_end.zero_velocity_tracker import ZeroVelocityTracker, ZeroVelocityTrackerState


class TestZeroVelocityTracker:
    """Unit test for zero velocity tracker."""

    @pytest.fixture
    def zero_velocity_tracker(self) -> ZeroVelocityTracker:
        """Create a zero velocity tracker."""
        return ZeroVelocityTracker()

    def test_zero_velocity_tracker_feed(self):
        """Test that the zero velocity tracker can be fed."""
        tracker = ZeroVelocityTracker(initial_state=ZeroVelocityTrackerState.NON_ZERO_VELOCITY)

        for _ in range(10):
            tracker.feed(0.0)

        assert tracker.state == ZeroVelocityTrackerState.ZERO_VELOCITY

        for _ in range(10):
            tracker.feed(2.0)

        assert tracker.state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
