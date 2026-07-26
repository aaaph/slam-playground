from typing import Self

import numpy as np
from numpy.typing import NDArray


class FeatureDuplicateError(Exception):
    """Exception raised when a feature is duplicated."""

    def __init__(self, message: str = "Feature is duplicated.") -> None:
        """Initialize the FeatureDuplicateError exception."""
        self.message = message
        super().__init__(self.message)


class NotEnoughSlotsError(Exception):
    """Exception raised when there are not enough slots available."""

    def __init__(self, message: str = "Not enough slots available.") -> None:
        """Initialize the NotEnoughSlotsError exception."""
        self.message = message
        super().__init__(self.message)


class PnPMapSchema:
    """Schema for PnP map features."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3
    LEFT_U = 4
    LEFT_V = 5
    RIGHT_U = 6
    RIGHT_V = 7

    XYZ = slice(X, Z + 1)
    LEFT_UV = slice(LEFT_U, LEFT_V + 1)
    RIGHT_UV = slice(RIGHT_U, RIGHT_V + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of features."""
        return cls.RIGHT_V + 1


class FrameToFramePnpStore:
    """Frame-to-frame PnP store."""

    def __init__(self, map_capacity: int = 400, max_outlier_streak: int = 2) -> None:
        """Initialize the PnP store."""
        self.map_capacity = map_capacity
        self.max_outlier_streak = max_outlier_streak
        self._head = 0
        self.depth_capacity = 2
        self._map = np.empty((self.depth_capacity, self.map_capacity, PnPMapSchema.count()), dtype=np.float64)
        self._observed = np.zeros((self.depth_capacity, self.map_capacity), dtype=np.bool_)
        self._outlier_streak = np.zeros(self.map_capacity, dtype=np.int8)
        self._free_slots = []
        self._feat_to_slot = {}
        self._next_slot = 0

    @classmethod
    def default_factory(cls, map_capacity: int = 400, max_outlier_streak: int = 2) -> Self:
        """Return a default factory for the FrameToFramePnpStore."""
        return cls(map_capacity, max_outlier_streak)

    def finish_frame_and_advance(self) -> None:
        """Finish the current frame and advance to the next frame."""
        used_slots = np.fromiter(self._feat_to_slot.values(), dtype=np.int32)
        if used_slots.size > 0:
            missing_mask = ~self._observed[self._head, used_slots]
            slots_to_clear = used_slots[missing_mask]
            if slots_to_clear.size > 0:
                slots_to_clear_list = slots_to_clear.tolist()
                slots_to_clear_set = set(slots_to_clear_list)
                feat_ids_to_clear = [
                    feat_id for feat_id, slot in self._feat_to_slot.items() if slot in slots_to_clear_set
                ]
                for feat_id in feat_ids_to_clear:
                    del self._feat_to_slot[feat_id]
                self._outlier_streak[slots_to_clear] = 0
                self._free_slots.extend(slots_to_clear_list)

        self._head = (self._head + 1) % self.depth_capacity
        self._map[self._head].fill(np.nan)
        self._observed[self._head].fill(False)  # noqa: FBT003

    def _pop_free_slots(self, size: int) -> NDArray[np.int32]:
        """Pop free slots."""
        if size == 0:
            return np.empty(0, dtype=np.int32)

        slots = np.empty(size, dtype=np.int32)
        filled = 0

        reused_count = min(size, len(self._free_slots))
        if reused_count > 0:
            slots[:reused_count] = self._free_slots[-reused_count:]
            del self._free_slots[-reused_count:]
            filled += reused_count

        remaining = size - filled
        if remaining > 0:
            if self._next_slot + remaining > self.map_capacity:
                raise NotEnoughSlotsError
            slots[filled:] = np.arange(self._next_slot, self._next_slot + remaining, dtype=np.int32)
            self._next_slot += remaining

        return slots

    def _get_feature_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Get the slots for the features."""
        feat_ids = np.asarray(feat_ids, dtype=np.int32)
        if feat_ids.ndim != 1:
            raise ValueError("feat_ids must be a 1D array")
        if feat_ids.shape[0] != np.unique(feat_ids).shape[0]:
            raise FeatureDuplicateError

        output = np.full_like(feat_ids, -1, dtype=np.int32)
        new_feat_mask = np.array([fid.item() not in self._feat_to_slot for fid in feat_ids])
        for i in np.where(~new_feat_mask)[0]:
            output[i] = self._feat_to_slot[feat_ids[i].item()]
        new_size = np.sum(new_feat_mask)
        if new_size > 0:
            new_slots = self._pop_free_slots(new_size)
            output[new_feat_mask] = new_slots

            for feat_id, slot in zip(feat_ids[new_feat_mask], new_slots, strict=True):
                self._feat_to_slot[int(feat_id)] = int(slot)

        return output

    def _lookup_feature_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Lookup existing slots for the features."""
        return np.fromiter(
            (self._feat_to_slot[int(feat_id)] for feat_id in feat_ids),
            dtype=np.int32,
            count=feat_ids.shape[0],
        )

    def add_features(self, ndarray: NDArray[np.float64]) -> None:
        """Add features to the store."""
        feat_ids = ndarray[:, PnPMapSchema.FEAT_ID].astype(np.int32, copy=False)
        slots = self._get_feature_slots(feat_ids)
        self._map[self._head, slots, :] = ndarray
        self._observed[self._head, slots] = True

    def update_outlier_streak(self, feat_ids: NDArray[np.int32], inlier_mask: NDArray[np.bool_]) -> None:
        """Update the outlier streak for the features."""
        slots = self._lookup_feature_slots(feat_ids)

        self._outlier_streak[slots[~inlier_mask]] += 1
        self._outlier_streak[slots[inlier_mask]] = 0

    def get_previous_xyz(self, feat_ids: NDArray[np.int32]) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Get the previous xyz for the features."""
        previous_idx = (self._head - 1) % self.depth_capacity
        found_mask = np.zeros(feat_ids.shape[0], dtype=np.bool_)
        xyz = np.full((feat_ids.shape[0], 3), np.nan, dtype=np.float64)

        for i, feat_id in enumerate(feat_ids):
            slot = self._feat_to_slot.get(int(feat_id))
            if slot is None:
                continue
            if not self._observed[previous_idx, slot]:
                continue
            if self._outlier_streak[slot] > self.max_outlier_streak:
                continue

            xyz[i, :] = self._map[previous_idx, slot, PnPMapSchema.XYZ]
            found_mask[i] = True

        return found_mask, xyz

    def get_outlier_streak(self) -> dict[int, int]:
        """Get the outlier streak for the features."""
        streaks = {}
        for feat_id, slot in self._feat_to_slot.items():
            if self._outlier_streak[slot] > 0:
                streaks[int(feat_id)] = int(self._outlier_streak[slot])
        return streaks
