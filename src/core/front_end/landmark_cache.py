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
    COMPLETED_ROW = slice(FEAT_ID, Z + 1)

    @classmethod
    def size(cls) -> int:
        """Return the size of the landmark cache schema."""
        return cls.RETRY_AFTER_VERSION + 1


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

    def get_history_version(self, slots: NDArray[np.int32]) -> NDArray[np.int32]:
        """Get the history version of the landmark cache."""
        return self._data[slots, LandmarkCacheSchema.RETRY_AFTER_VERSION]

    def lookup(self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32]) -> NDArray[np.float64]:
        """Return feature-matched cache rows aligned with input rows."""
        cache_rows = self._data[slots]
        feature_match_mask = cache_rows[:, LandmarkCacheSchema.FEAT_ID] == feat_ids
        lookup = np.full((slots.shape[0], LandmarkCacheSchema.size()), np.nan, dtype=np.float64)
        lookup[:, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.EMPTY.value
        lookup[feature_match_mask] = cache_rows[feature_match_mask]
        return lookup

    def resolve_failed_attempt_statuses(
        self, slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> NDArray[np.float64]:
        """Return statuses after failed triangulation attempts without mutating cache."""
        resolved_statuses = self._data[slots, LandmarkCacheSchema.STATUS].copy()
        failed_attempt_mask = self._failed_attempt_mask(slots, history_versions)
        next_attempts = self._data[slots, LandmarkCacheSchema.ATTEMPTS] + 1.0
        resolved_statuses[failed_attempt_mask] = LandmarkCacheStatus.FAILED_SOFT.value
        resolved_statuses[failed_attempt_mask & (next_attempts > self.MAX_SOFT_FAILURE_ATTEMPTS)] = (
            LandmarkCacheStatus.FAILED_HARD.value
        )
        return resolved_statuses

    def commit(
        self,
        feat_ids: NDArray[np.float64],
        slots: NDArray[np.float64],
        statuses: NDArray[np.float64],
        history_versions: NDArray[np.float64],
        xyz: NDArray[np.float64],
    ) -> None:
        """Commit frame-aligned landmark lifecycle state into the cache."""
        if feat_ids.shape[0] == 0:
            return

        valid_slot_mask = np.isfinite(slots)
        if not np.any(valid_slot_mask):
            return

        valid_slots = slots[valid_slot_mask].astype(np.int32, copy=False)
        valid_feat_ids = feat_ids[valid_slot_mask].astype(np.int32, copy=False)
        valid_statuses = statuses[valid_slot_mask]
        valid_history_versions = history_versions[valid_slot_mask].astype(np.int32, copy=False)
        valid_xyz = xyz[valid_slot_mask]

        empty_mask = valid_statuses == LandmarkCacheStatus.EMPTY.value
        if np.any(empty_mask):
            self.clear_slots(valid_slots[empty_mask])

        observing_mask = valid_statuses == LandmarkCacheStatus.OBSERVING.value
        if np.any(observing_mask):
            self._commit_observing(valid_feat_ids[observing_mask], valid_slots[observing_mask])

        ready_mask = valid_statuses == LandmarkCacheStatus.READY.value
        if np.any(ready_mask):
            self._commit_ready(
                valid_feat_ids[ready_mask],
                valid_slots[ready_mask],
                valid_history_versions[ready_mask],
            )

        failed_soft_mask = valid_statuses == LandmarkCacheStatus.FAILED_SOFT.value
        if np.any(failed_soft_mask):
            self._commit_failed_attempts(
                valid_feat_ids[failed_soft_mask],
                valid_slots[failed_soft_mask],
                valid_history_versions[failed_soft_mask],
            )

        failed_hard_mask = valid_statuses == LandmarkCacheStatus.FAILED_HARD.value
        if np.any(failed_hard_mask):
            self._commit_failed_hard(
                valid_feat_ids[failed_hard_mask],
                valid_slots[failed_hard_mask],
                valid_history_versions[failed_hard_mask],
            )

        completed_mask = valid_statuses == LandmarkCacheStatus.COMPLETED.value
        if np.any(completed_mask):
            self._commit_completed(
                valid_feat_ids[completed_mask],
                valid_slots[completed_mask],
                valid_xyz[completed_mask],
            )

    def _commit_observing(self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32]) -> None:
        """Commit observing rows only into empty cache slots."""
        empty_mask = self._data[slots, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value
        observing_slots = slots[empty_mask]
        self._data[observing_slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.OBSERVING.value
        self._data[observing_slots, LandmarkCacheSchema.FEAT_ID] = feat_ids[empty_mask]

    def _commit_ready(
        self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> None:
        """Commit already-gated ready rows without selecting a return subset."""
        status = self._data[slots, LandmarkCacheSchema.STATUS]
        retry_ready_mask = self._retry_ready_mask(slots, history_versions)
        mutable_mask = (
            (status != LandmarkCacheStatus.FAILED_HARD.value)
            & (status != LandmarkCacheStatus.COMPLETED.value)
            & retry_ready_mask
        )
        ready_slots = slots[mutable_mask]
        self._data[ready_slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.READY.value
        self._data[ready_slots, LandmarkCacheSchema.FEAT_ID] = feat_ids[mutable_mask]
        self._data[ready_slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0.0

    def _commit_failed_attempts(
        self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> None:
        """Commit failed triangulation attempts and update retry state."""
        failed_attempt_mask = self._failed_attempt_mask(slots, history_versions)
        failed_slots = slots[failed_attempt_mask]
        resolved_statuses = self.resolve_failed_attempt_statuses(slots, history_versions)
        self._data[failed_slots, LandmarkCacheSchema.FEAT_ID] = feat_ids[failed_attempt_mask]
        self._data[failed_slots, LandmarkCacheSchema.ATTEMPTS] += 1.0
        self._data[failed_slots, LandmarkCacheSchema.STATUS] = resolved_statuses[failed_attempt_mask]
        self._data[failed_slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = (
            history_versions[failed_attempt_mask] + self.SOFT_FAILURE_RETRY_VERSION_STEP
        )

    def _failed_attempt_mask(
        self, slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> NDArray[np.bool_]:
        """Return rows that can accept a failed triangulation attempt."""
        status = self._data[slots, LandmarkCacheSchema.STATUS]
        return (
            (status != LandmarkCacheStatus.FAILED_HARD.value)
            & (status != LandmarkCacheStatus.COMPLETED.value)
            & self._retry_ready_mask(slots, history_versions)
        )

    def _retry_ready_mask(
        self, slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> NDArray[np.bool_]:
        """Return rows whose retry version allows another triangulation attempt."""
        status = self._data[slots, LandmarkCacheSchema.STATUS]
        retry_after_versions = self._data[slots, LandmarkCacheSchema.RETRY_AFTER_VERSION]
        return (status != LandmarkCacheStatus.FAILED_SOFT.value) | (history_versions >= retry_after_versions)

    def _commit_failed_hard(
        self, feat_ids: NDArray[np.int32], slots: NDArray[np.int32], history_versions: NDArray[np.int32]
    ) -> None:
        """Commit hard-failed triangulation attempts without re-counting already hard-failed rows."""
        already_hard_mask = self._data[slots, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.FAILED_HARD.value
        new_hard_slots = slots[~already_hard_mask]
        self._data[slots, LandmarkCacheSchema.FEAT_ID] = feat_ids
        self._data[new_hard_slots, LandmarkCacheSchema.ATTEMPTS] += 1.0
        self._data[new_hard_slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = (
            history_versions[~already_hard_mask] + self.SOFT_FAILURE_RETRY_VERSION_STEP
        )
        self._data[slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.FAILED_HARD.value

    def _commit_completed(
        self,
        feat_ids: NDArray[np.int32],
        slots: NDArray[np.int32],
        xyz: NDArray[np.float64],
    ) -> None:
        """Commit completed triangulation results."""
        self._data[slots, LandmarkCacheSchema.FEAT_ID] = feat_ids
        self._data[slots, LandmarkCacheSchema.XYZ] = xyz
        self._data[slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.COMPLETED.value

    def get_completed_landmarks(self) -> NDArray[np.float64]:
        """Get completed landmarks as visualization rows."""
        completed_mask = self._data[:, LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.COMPLETED.value
        completed = self._data[completed_mask]
        return completed[:, LandmarkCacheSchema.COMPLETED_ROW].copy()

    def clear_slots(self, slots: NDArray[np.int32]) -> None:
        """Clear cache slots."""
        self._data[slots, :] = np.nan
        self._data[slots, LandmarkCacheSchema.STATUS] = LandmarkCacheStatus.EMPTY.value
        self._data[slots, LandmarkCacheSchema.ATTEMPTS] = 0.0
        self._data[slots, LandmarkCacheSchema.RETRY_AFTER_VERSION] = 0.0
