from collections import deque
from enum import IntEnum

import numpy as np


class ZeroVelocityTrackerState(IntEnum):
    """Zero velocity tracker state."""

    ZERO_VELOCITY = 1
    NON_ZERO_VELOCITY = 0


class ZeroVelocityTracker:
    """Zero velocity tracker based on the VIO streams."""

    def __init__(
        self,
        window_size: int = 4,
        disparity_threshold: float = 1.0,
        initial_state: ZeroVelocityTrackerState = ZeroVelocityTrackerState.ZERO_VELOCITY,
    ) -> None:
        """Initialize the zero velocity tracker."""
        self.state = initial_state
        self.window_size = window_size
        self.disparity_threshold = disparity_threshold
        self.debouncer = deque(maxlen=self.window_size)

    def feed(self, current_disparity: float) -> ZeroVelocityTrackerState:
        """Feed the zero velocity tracker with the current disparity."""
        zero_velocity_in_frame = current_disparity < self.disparity_threshold
        self.debouncer.append(zero_velocity_in_frame)
        if len(self.debouncer) < self.window_size:
            return self.state

        # check that all values in the queue are the same
        window_sum = np.sum(self.debouncer)
        window_size = len(self.debouncer)
        if window_sum == window_size:
            self.state = ZeroVelocityTrackerState.ZERO_VELOCITY
        elif window_sum == 0:
            self.state = ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        return self.state
