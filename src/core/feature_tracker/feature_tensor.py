from collections import deque

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from core.feature_tracker.feature_frame import FeatureFrame
from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema, active_feat_arrow_schema
from logger import spawn_logger

Point2 = tuple[float, float]

_DEFAULT_STATUS_VALUES = np.array(
    [
        FeatureLifecycle.ACTIVE.value,
        FeatureLifecycle.LOST.value,
    ],
    dtype=np.float32,
)
_FEATURE_COLORS: dict[FeatureLifecycle, tuple[int, int, int]] = {
    FeatureLifecycle.ACTIVE: (0, 255, 0),
    FeatureLifecycle.LOST: (128, 128, 128),
}
_MAX_STATUS_VALUE = max(status.value for status in FeatureLifecycle)
_FEATURE_COLORS_LUT = np.zeros((_MAX_STATUS_VALUE + 1, 3), dtype=np.uint8)
for status, color in _FEATURE_COLORS.items():
    _FEATURE_COLORS_LUT[status.value] = color


class FeatureTensor:
    """Feature tensor(Feature Pool)."""

    schema = active_feat_arrow_schema

    def __init__(self, feat_capacity: int = 200) -> None:
        """Initialize the feature tensor."""
        self.feat_capacity = feat_capacity
        self.history_capacity = 2

        self._ts_head = 0
        self._last_timestamp = -float("inf")
        self._prev_timestamp = -float("inf")

        # Rows follow FeatureSchema.
        self._data = np.full(
            (self.history_capacity, self.feat_capacity, FeatureSchema.count()), np.nan, dtype=np.float32
        )
        self._id_to_idx: dict[int, int] = {}
        self.free_slots = list(range(feat_capacity - 1, -1, -1))
        self.ts_deque = deque(maxlen=self.history_capacity)
        self.timestamps = np.full(self.history_capacity, -1, dtype=np.int64)
        self.logger = spawn_logger(app="feature_tensor")

    def step(self, new_timestamp: float) -> None:
        """Step the feature tensor."""
        occupated_slots = self.active_indeces
        if occupated_slots.size > 0:
            current_data = self._data[self._ts_head, occupated_slots]
            is_nan = np.isnan(current_data[:, 1])
            is_lost = current_data[:, 6] == FeatureLifecycle.LOST.value
            to_remove = is_nan | is_lost
            slots_to_free = occupated_slots[to_remove]

            if slots_to_free.size > 0:
                ids_to_remove = [fid for fid, s in self._id_to_idx.items() if s in slots_to_free]
                for fid in ids_to_remove:
                    del self._id_to_idx[fid]
                self.free_slots.extend(slots_to_free.tolist())

        self._ts_head = (self._ts_head + 1) % self.history_capacity
        self._data[self._ts_head].fill(np.nan)
        self._prev_timestamp = self._last_timestamp
        self._last_timestamp = new_timestamp
        self.timestamps[self._ts_head] = new_timestamp

    @property
    def initiated(self) -> bool:
        """Check if the feature tensor is initiated."""
        return self._last_timestamp > 0

    @property
    def active_indeces(self) -> NDArray[np.int32]:
        """Get the active indeces of the feature tensor."""
        return np.array(list(self._id_to_idx.values()), dtype=np.int32)

    @property
    def current_data(self) -> NDArray[np.float32]:
        """Get the current data of the feature tensor."""
        return self._data[self._ts_head]

    @property
    def active_data(self) -> NDArray[np.float32]:
        """Get the active data of the feature tensor."""
        return self.current_data[self.active_indeces]

    @property
    def active_frame(self) -> FeatureFrame:
        """Get the active features of the feature tensor."""
        return FeatureFrame(
            data=self.current_data,
            active_indeces=self.active_indeces,
            active_mask=~np.isnan(self.current_data[:, 1]),
            timestamp=self._last_timestamp,
        )

    @property
    def prev_data(self) -> NDArray[np.float32]:
        """Get the previous data of the feature tensor."""
        prev_idx = (self._ts_head - 1) % self.history_capacity
        return self._data[prev_idx]

    @property
    def pair_exists(self) -> bool:
        """Check if the previous and current timestamp are valid."""
        return self._prev_timestamp > 0 and self._last_timestamp > 0

    def get_frame_by_timestamp(self, timestamp: float) -> FeatureFrame:
        """Get the frame by timestamp."""
        ts_index = self.timestamp_index(timestamp)
        frame_data = self._data[ts_index]
        active_mask = ~np.isnan(frame_data[:, FeatureSchema.TIMESTAMP])
        active_indeces = np.flatnonzero(active_mask).astype(np.int32)
        return FeatureFrame(
            data=frame_data,
            active_indeces=active_indeces,
            active_mask=active_mask,
            timestamp=timestamp,
        )

    def get_active_features(self, states: list[FeatureLifecycle] | None = None) -> NDArray[np.float32]:
        """Get the active features of the feature tensor."""
        if states is None:
            mask = np.isin(self.current_data[:, 6], _DEFAULT_STATUS_VALUES)
        else:
            values = np.array([state.value for state in states], dtype=np.float32)
            mask = np.isin(self.current_data[:, 6], values)
        mask &= ~np.isnan(self.current_data[:, 1])
        return self.current_data[mask]

    @classmethod
    def default_factory(cls, capacity: int = 1000) -> "FeatureTensor":
        """Create a feature tensor from a capacity. Capacity means the history of features."""
        return cls(capacity)

    def __repr__(self) -> str:
        """Return the representation of the feature tensor."""
        return f"FeatureTensor(capacity={self.feat_capacity}, free_slots={len(self.free_slots)})"

    def get_free_indexes(self, size: int) -> NDArray[np.int32]:
        """Get the indexes of the feature tensor."""
        size = min(size, len(self.free_slots))

        indexes = [self.free_slots.pop() for _ in range(size)]
        return np.array(indexes, dtype=np.int32)

    def allocate_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Allocate slots for the features."""
        if len(feat_ids) != len(np.unique(feat_ids)):
            raise ValueError("Duplicate feature IDs")
        output = np.full_like(feat_ids, -1, dtype=np.int32)
        new_mask = np.array([fid.item() not in self._id_to_idx for fid in feat_ids])
        for i in np.where(~new_mask)[0]:
            output[i] = self._id_to_idx[feat_ids[i].item()]
        num_new = np.sum(new_mask)
        if num_new > 0:
            new_slots = self.get_free_indexes(num_new)
            if len(new_slots) != num_new:
                raise ValueError("No free slots available")
            output[new_mask] = new_slots
        return output

    def find_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Find the slots of the features in the feature tensor."""
        if len(feat_ids) != len(np.unique(feat_ids)):
            raise ValueError("Duplicate feature IDs")
        output = np.zeros_like(feat_ids, dtype=np.int32)
        for i, feat_id in enumerate(feat_ids):
            feat_id_val = feat_id.item()
            idx = self._id_to_idx.get(feat_id_val, None)
            if idx is None:
                msg = f"Feature with ID {feat_id_val} not found"
                raise ValueError(msg)
            output[i] = idx
        return output

    def exists(self, feat_id: int) -> bool:
        """Check if the feature exists in the feature tensor."""
        return feat_id in self._id_to_idx

    def timestamp_index(self, timestamp: float) -> int:
        """Find the index of the timestamp in the feature tensor."""
        matches = np.where(self.timestamps == timestamp)[0]
        if matches.size == 0:
            msg = f"Timestamp {timestamp} not found"
            raise ValueError(msg)
        return int(matches[0])

    def add(
        self, feat_id: int, timestamp: float, left_uv: Point2, right_uv: Point2 | None, state: FeatureLifecycle
    ) -> None:
        """Add a feature to the feature tensor."""
        if timestamp < self._last_timestamp:
            raise ValueError("Old timestamp")
        if timestamp > self._last_timestamp:
            self.step(timestamp)
        indexes = self.allocate_slots(np.array([feat_id]))
        if len(indexes) == 0:
            raise ValueError("No free slots available")
        index = indexes[0].item()
        t = self._ts_head
        data = np.full(FeatureSchema.count(), np.nan, dtype=np.float32)
        data[FeatureSchema.FEAT_ID] = feat_id
        data[FeatureSchema.TIMESTAMP] = timestamp
        data[FeatureSchema.LEFT_U] = left_uv[0]
        data[FeatureSchema.LEFT_V] = left_uv[1]
        data[FeatureSchema.RIGHT_U] = right_uv[0] if right_uv is not None else np.nan
        data[FeatureSchema.RIGHT_V] = right_uv[1] if right_uv is not None else np.nan
        data[FeatureSchema.LIFECYCLE] = state.value
        data[FeatureSchema.AGE] = 0
        data[FeatureSchema.FRAME_PIXEL_DISPLACEMENT] = 0.0
        self._data[t, index] = data
        self._id_to_idx[feat_id] = index

    def add_batch(self, timestamp: float, batch: NDArray[np.float32]) -> None:
        """Add a batch of features to the feature tensor."""
        if batch.shape[0] == 0:
            return
        if timestamp < self._last_timestamp:
            raise ValueError("Old timestamp")
        if timestamp > self._last_timestamp:
            self.step(timestamp)
        feat_ids = batch[:, 0].astype(np.int32)
        indexes = self.allocate_slots(feat_ids)
        t = self._ts_head
        if batch.shape[1] > FeatureSchema.count():
            msg = f"Feature batch has {batch.shape[1]} columns, schema has {FeatureSchema.count()}"
            raise ValueError(msg)
        self._data[t, indexes, : batch.shape[1]] = batch
        self._id_to_idx.update(dict(zip(feat_ids.tolist(), indexes.tolist(), strict=True)))

    def update_state(self, feat_ids: NDArray[np.int32], state: FeatureLifecycle) -> None:
        """Update the state of the features in the feature tensor."""
        t = self._ts_head
        slots = self.find_slots(feat_ids)
        self._data[t, slots, FeatureSchema.LIFECYCLE] = state.value

    def get_slots_by_status(self, status: FeatureLifecycle) -> NDArray[np.int32]:
        """Get the features by status."""
        mask = self.current_data[:, 6] == status.value
        return np.where(mask)[0].astype(np.int32)

    def as_arrow(self) -> pa.RecordBatch:
        """Convert the feature tensor to a struct array."""
        active_data = self.active_data
        left_points = pa.StructArray.from_arrays(
            [active_data[:, FeatureSchema.LEFT_U], active_data[:, FeatureSchema.LEFT_V]],
            names=["u", "v"],
        )
        right_points = pa.StructArray.from_arrays(
            [active_data[:, FeatureSchema.RIGHT_U], active_data[:, FeatureSchema.RIGHT_V]],
            names=["u", "v"],
        )
        stereo_struct = pa.StructArray.from_arrays(
            [left_points, right_points],
            names=["left", "right"],
        )
        return pa.RecordBatch.from_arrays(
            [
                active_data[:, FeatureSchema.FEAT_ID],
                active_data[:, FeatureSchema.TIMESTAMP],
                stereo_struct,
                active_data[:, FeatureSchema.LIFECYCLE],
                active_data[:, FeatureSchema.AGE],
            ],
            schema=active_feat_arrow_schema,
        )

    @classmethod
    def from_arrow(cls, arrow: pa.RecordBatch) -> "FeatureTensor":
        """Create a feature tensor from a record batch."""
        num_features = arrow.num_rows
        capacity = num_features
        tensor = cls(capacity)
        if num_features == 0:
            return tensor

        batch = np.full((num_features, FeatureSchema.count()), np.nan, dtype=np.float32)
        batch[:, FeatureSchema.FEAT_ID] = arrow.column(0).to_numpy()
        batch[:, FeatureSchema.TIMESTAMP] = arrow.column(1).to_numpy()
        stereo_struct = arrow.column(2)
        left_points = stereo_struct.field("left")
        right_points = stereo_struct.field("right")
        batch[:, FeatureSchema.LEFT_U] = left_points.field("u").to_numpy()
        batch[:, FeatureSchema.LEFT_V] = left_points.field("v").to_numpy()
        batch[:, FeatureSchema.RIGHT_U] = right_points.field("u").to_numpy()
        batch[:, FeatureSchema.RIGHT_V] = right_points.field("v").to_numpy()
        batch[:, FeatureSchema.LIFECYCLE] = arrow.column(3).to_numpy()
        batch[:, FeatureSchema.AGE] = arrow.column(4).to_numpy()
        batch[:, FeatureSchema.FRAME_PIXEL_DISPLACEMENT] = 0.0
        timestamp = batch[:, FeatureSchema.TIMESTAMP].max().item()
        tensor.add_batch(timestamp, batch)
        return tensor

    @staticmethod
    def to_color_array(data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Convert the data to a color array."""
        if data.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        feature_statuses = data[:, FeatureSchema.LIFECYCLE].astype(np.int32)
        return _FEATURE_COLORS_LUT[feature_statuses]
