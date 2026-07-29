from enum import IntEnum, auto

import numpy as np


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

    def __init__(self, capacity: int) -> None:
        """Initialize the landmark cache."""
        self._capacity = capacity
        self._data = np.full((capacity, LandmarkCacheSchema.size()), np.nan, dtype=np.float64)
        self._data[:, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.EMPTY.value
        self._data[:, LandmarkCacheSchema.ATTEMPTS] = 0.0
        self._data[:, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0
