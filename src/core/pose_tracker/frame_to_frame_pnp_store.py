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

    @classmethod
    def count(cls) -> int:
        """Return the number of features."""
        return cls.RIGHT_V + 1


class FrameToFramePnpStore:
    """Frame-to-frame PnP store."""

    def __init__(self, map_capacity: int = 400) -> None:
        """Initialize the PnP store."""
        self.map_capacity = map_capacity
        self._head = 0
        self.depth_capacity = 2
        self._map = np.empty((self.depth_capacity, self.map_capacity, PnPMapSchema.count()), dtype=np.float64)
        self._observed = np.zeros((self.depth_capacity, self.map_capacity), dtype=np.bool_)
        self._free_slots = []
        self._feat_to_slot = {}
        self._next_slot = 0
        self._timestamp_row = np.empty(2, dtype=np.float64)

    @classmethod
    def default_factory(cls, map_capacity: int = 400) -> Self:
        """Return a default factory for the FrameToFramePnpStore."""
        return cls(map_capacity)

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

    def add_features(self, ndarray: NDArray[np.float64]) -> None:
        """Add features to the store."""
        feat_ids = ndarray[:, PnPMapSchema.FEAT_ID].astype(np.int32, copy=False)
        slots = self._get_feature_slots(feat_ids)
        self._map[self._head, slots, :] = ndarray
        self._observed[self._head, slots] = True
