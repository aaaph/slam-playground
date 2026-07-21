from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState


class FeatureMetricsSchema(IntEnum):
    """Feature metrics schema."""

    ACTIVE_COUNT = 0
    GOOD_COUNT = 1
    LOST_COUNT = 2
    TRACKED_COUNT = 3
    STEREO_OK_COUNT = 4
    STEREO_OK_RATIO = 5
    TEMPORAL_PIXEL_DISPLACEMENT = 6
    TEMPORAL_PIXEL_DISPLACEMENT_P90 = 7
    ZERO_VELOCITY_STATE = 8

    @classmethod
    def count(cls) -> int:
        """Return the number of metrics."""
        return len(cls)


class FeatureTrackerMetrics:
    """Feature tracker metrics view over a reusable numpy row."""

    __slots__ = ("_data",)

    def __init__(self, data: NDArray[np.float32]) -> None:
        """Initialize the metrics view."""
        self._data = data

    @property
    def ndarray(self) -> NDArray[np.float32]:
        """Return the raw metrics row."""
        return self._data

    @property
    def active_count(self) -> int:
        """Return the active feature count."""
        return int(self._data[FeatureMetricsSchema.ACTIVE_COUNT])

    @property
    def good_count(self) -> int:
        """Return the non-lost feature count."""
        return int(self._data[FeatureMetricsSchema.GOOD_COUNT])

    @property
    def lost_count(self) -> int:
        """Return the lost feature count."""
        return int(self._data[FeatureMetricsSchema.LOST_COUNT])

    @property
    def tracked_count(self) -> int:
        """Return the count of features tracked from a previous frame."""
        return int(self._data[FeatureMetricsSchema.TRACKED_COUNT])

    @property
    def stereo_ok_count(self) -> int:
        """Return the count of good features with a stereo match."""
        return int(self._data[FeatureMetricsSchema.STEREO_OK_COUNT])

    @property
    def stereo_ok_ratio(self) -> float:
        """Return the ratio of good features with a stereo match."""
        return float(self._data[FeatureMetricsSchema.STEREO_OK_RATIO])

    @property
    def temporal_pixel_displacement(self) -> float:
        """Return the median temporal pixel displacement."""
        return float(self._data[FeatureMetricsSchema.TEMPORAL_PIXEL_DISPLACEMENT])

    @property
    def temporal_pixel_displacement_p90(self) -> float:
        """Return the p90 temporal pixel displacement."""
        return float(self._data[FeatureMetricsSchema.TEMPORAL_PIXEL_DISPLACEMENT_P90])

    @property
    def zero_velocity_state(self) -> ZeroVelocityTrackerState:
        """Return the debounced zero-velocity state."""
        return ZeroVelocityTrackerState(int(self._data[FeatureMetricsSchema.ZERO_VELOCITY_STATE]))

    def copy(self) -> NDArray[np.float32]:
        """Return a copy of the current metrics row."""
        return self._data.copy()

    def __str__(self) -> str:
        """Return a string representation of the metrics."""
        return (
            f"FeatureTrackerMetrics(active_count={self.active_count}, good_count={self.good_count}, "
            f"lost_count={self.lost_count}, tracked_count={self.tracked_count}, "
            f"stereo_ok_count={self.stereo_ok_count}, stereo_ok_ratio={self.stereo_ok_ratio}, "
            f"temporal_pixel_displacement={self.temporal_pixel_displacement}, "
            f"temporal_pixel_displacement_p90={self.temporal_pixel_displacement_p90}, "
            f"zero_velocity_state={self.zero_velocity_state})"
        )
