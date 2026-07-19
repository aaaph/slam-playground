import pytest

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTracker, ZeroVelocityTrackerState


class TestZeroVelocityTracker:
    """Unit test for zero velocity tracker."""

    @pytest.fixture
    def zero_velocity_tracker(self) -> ZeroVelocityTracker:
        """Create a zero velocity tracker."""
        return ZeroVelocityTracker()

    def test_zero_velocity_tracker_returns_unknown_until_window_is_full(self) -> None:
        """Test that the zero velocity tracker returns UNKNOWN when the window is not full."""
        tracker = ZeroVelocityTracker(window_size=4)

        assert tracker.state == ZeroVelocityTrackerState.UNKNOWN
        assert tracker.feed(0.0) == ZeroVelocityTrackerState.UNKNOWN
        assert tracker.feed(0.0) == ZeroVelocityTrackerState.UNKNOWN
        assert tracker.feed(0.0) == ZeroVelocityTrackerState.UNKNOWN
        assert tracker.state == ZeroVelocityTrackerState.UNKNOWN

    def test_zero_velocity_tracker_selects_zero_velocity_on_static_consensus(self) -> None:
        """Test that mostly static frames switch the tracker to ZERO_VELOCITY."""
        tracker = ZeroVelocityTracker(window_size=4, consensus_ratio=0.75)

        for disparity in [0.0, 0.2, 0.5, 2.0]:
            state = tracker.feed(disparity)

        assert state == ZeroVelocityTrackerState.ZERO_VELOCITY
        assert tracker.state == ZeroVelocityTrackerState.ZERO_VELOCITY

    def test_zero_velocity_tracker_selects_non_zero_velocity_on_dynamic_consensus(self) -> None:
        """Test that mostly dynamic frames switch the tracker to NON_ZERO_VELOCITY."""
        tracker = ZeroVelocityTracker(window_size=4, consensus_ratio=0.75)

        for disparity in [2.0, 3.0, 4.0, 0.0]:
            state = tracker.feed(disparity)

        assert state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        assert tracker.state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY

    def test_zero_velocity_tracker_keeps_unknown_for_first_mixed_window(self) -> None:
        """Test that a mixed first full window remains UNKNOWN."""
        tracker = ZeroVelocityTracker(window_size=4, consensus_ratio=0.75)

        for disparity in [0.0, 2.0, 0.0, 2.0]:
            state = tracker.feed(disparity)

        assert state == ZeroVelocityTrackerState.UNKNOWN
        assert tracker.state == ZeroVelocityTrackerState.UNKNOWN

    def test_zero_velocity_tracker_keeps_previous_known_state_for_mixed_window(self) -> None:
        """Test that mixed windows keep the last known state."""
        tracker = ZeroVelocityTracker(window_size=4, consensus_ratio=0.75)

        for disparity in [0.0, 0.2, 0.5, 0.0]:
            tracker.feed(disparity)

        assert tracker.state == ZeroVelocityTrackerState.ZERO_VELOCITY

        for disparity in [2.0, 0.0, 2.0, 0.0]:
            state = tracker.feed(disparity)

        assert state == ZeroVelocityTrackerState.ZERO_VELOCITY
        assert tracker.state == ZeroVelocityTrackerState.ZERO_VELOCITY
