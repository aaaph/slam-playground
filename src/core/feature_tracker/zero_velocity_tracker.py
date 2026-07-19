from collections import deque
from enum import IntEnum

import numpy as np


class ZeroVelocityTrackerState(IntEnum):
    """Zero velocity tracker state."""

    ZERO_VELOCITY = 1
    NON_ZERO_VELOCITY = 0
    UNKNOWN = -1


class ZeroVelocityTracker:
    """Zero velocity tracker based on feature-tracker motion metrics."""

    def __init__(
        self,
        window_size: int = 4,
        disparity_threshold: float = 1.0,
        consensus_ratio: float = 0.75,
        initial_state: ZeroVelocityTrackerState = ZeroVelocityTrackerState.UNKNOWN,
    ) -> None:
        """Initialize the zero velocity tracker."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 < consensus_ratio <= 1.0:
            raise ValueError("consensus_ratio must be in (0.0, 1.0]")
        self.state = initial_state
        self.window_size = window_size
        self.disparity_threshold = disparity_threshold
        self.consensus_ratio = consensus_ratio
        self.debouncer = deque(maxlen=self.window_size)

    def feed(self, current_disparity: float) -> ZeroVelocityTrackerState:
        """Feed the zero velocity tracker with the current temporal disparity."""
        zero_velocity_in_frame = current_disparity < self.disparity_threshold
        self.debouncer.append(zero_velocity_in_frame)
        if len(self.debouncer) < self.window_size:
            self.state = ZeroVelocityTrackerState.UNKNOWN
            return self.state

        window_sum = np.sum(self.debouncer)
        window_size = len(self.debouncer)
        zero_velocity_ratio = window_sum / window_size
        non_zero_velocity_ratio = 1.0 - zero_velocity_ratio
        if zero_velocity_ratio >= self.consensus_ratio:
            self.state = ZeroVelocityTrackerState.ZERO_VELOCITY
        elif non_zero_velocity_ratio >= self.consensus_ratio:
            self.state = ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        return self.state
