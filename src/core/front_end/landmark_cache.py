from enum import IntEnum, auto
from typing import Self

import numpy as np
from numpy.typing import NDArray


class LandmarkCacheStatus(IntEnum):
    """Status of the landmark cache."""

    EMPTY = 0
    OBSERVING = auto()
    READY = auto()
    FAILED_SOFT = auto()
    FAILED_HARD = auto()
    COMPLETED = auto()


class LandmarkCacheSchema:
    """Schema for the landmark cache."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3
    STATUS = 4
    ATTEMPTS = 5
    RETRY_AFTER_VERSION = 6

    XYZ = slice(X, Z + 1)

    @classmethod
    def size(cls) -> int:
        """Return the size of the landmark cache schema."""
        return 7


class LandmarkCache:
    """Cache-Queue that caches landmarks after triangulation/refinement."""

    MAX_SOFT_FAILURE_ATTEMPTS = 5.0
    SOFT_FAILURE_RETRY_VERSION_STEP = 1.0

    def __init__(self, capacity: int) -> None:
        """Initialize the landmark cache."""
        self._capacity = capacity
        self._data = np.full((capacity, LandmarkCacheSchema.size()), np.nan, dtype=np.float64)
        self._data[:, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.EMPTY.value
        self._data[:, LandmarkCacheSchema.ATTEMPTS] = 0.0
        self._data[:, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0

    @classmethod
    def default_factory(cls, capacity: int) -> Self:
        """Create a default landmark cache."""
        return cls(capacity)

    def apply_ready(
        self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        """Apply the ready status to the landmark cache."""
        status = self._data[slots, LandmarkCacheSchema.STATUS]
        retry_after_versions = self._data[slots, LandmarkCacheSchema.RETRY_AFTER_VERSION]
        retry_ready_mask = (status != LandmarkCacheStatus.FAILED_SOFT.value) | (
            history_versions >= retry_after_versions
        )
        mutable_mask = (
            (status != LandmarkCacheStatus.FAILED_HARD.value)
            & (status != LandmarkCacheStatus.COMPLETED.value)
            & retry_ready_mask
        )
        ready_slots = slots[mutable_mask]
        self._data[ready_slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.READY.value
        self._data[ready_slots, LandmarkCacheSchema.FEAT_ID] = feat_ids[mutable_mask]
        self._data[ready_slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0.0
        return ready_slots

    def apply_completed(
        self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32], xyz: NDArray[np.float64]
    ) -> None:
        """Apply completed triangulation results to the landmark cache."""
        self._data[slots, LandmarkCacheSchema.FEAT_ID] = feat_ids
        self._data[slots, LandmarkCacheSchema.XYZ] = xyz
        self._data[slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.COMPLETED.value

    def get_completed_landmarks(self) -> NDArray[np.float64]:
        """Get completed landmarks as visualization rows."""
        completed_mask = self._data[:, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.COMPLETED.value
        completed = self._data[completed_mask]
        landmarks = np.full((completed.shape[0], 4), np.nan, dtype=np.float64)
        landmarks[:, 0] = completed[:, LandmarkCacheSchema.FEAT_ID]
        landmarks[:, 1:4] = completed[:, LandmarkCacheSchema.XYZ]
        return landmarks

    def apply_failed(self, slots: NDArray[np.int32], history_versions: NDArray[np.int32]) -> None:
        """Apply a triangulation failure to landmark cache slots."""
        self._data[slots, LandmarkCacheSchema.ATTEMPTS] += 1.0
        attempts = self._data[slots, LandmarkCacheSchema.ATTEMPTS]
        failed_hard_mask = attempts > self.MAX_SOFT_FAILURE_ATTEMPTS
        self._data[slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.FAILED_SOFT.value
        self._data[slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = (
            history_versions + self.SOFT_FAILURE_RETRY_VERSION_STEP
        )
        self._data[slots[failed_hard_mask], LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.FAILED_HARD.value

    def apply_observing(self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32]) -> None:
        """Apply the observing status to empty landmark cache slots."""
        empty_mask = self._data[slots, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value
        observing_slots = slots[empty_mask]
        self._data[observing_slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.OBSERVING.value
        self._data[observing_slots, LandmarkCacheSchema.FEAT_ID] = feat_ids[empty_mask]

    def clear_slots(self, slots: NDArray[np.int32]) -> None:
        """Clear cache slots."""
        self._data[slots, :] = np.nan
        self._data[slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.EMPTY.value
        self._data[slots, LandmarkCacheSchema.ATTEMPTS] = 0.0
        self._data[slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0.0
