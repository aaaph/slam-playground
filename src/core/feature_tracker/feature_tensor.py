import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from core.feature_tracker.feature import FeatureStatus

Point2 = tuple[float, float]

point2_schema = pa.struct(
    [
        pa.field("u", pa.float32(), nullable=True),
        pa.field("v", pa.float32(), nullable=True),
    ]
)

active_feat_schema = pa.schema(
    [
        pa.field("feat_id", pa.int32()),
        pa.field("timestamp", pa.float32()),
        pa.field(
            "stereo",
            pa.struct(
                [
                    pa.field("left", point2_schema),
                    pa.field("right", point2_schema),
                ]
            ),
        ),
        pa.field("state", pa.int32()),
        pa.field("age", pa.int32()),
    ]
)
_DEFAULT_STATUS_VALUES = np.array(
    [
        FeatureStatus.NEW.value,
        FeatureStatus.TRACKED.value,
        FeatureStatus.STABLE.value,
        FeatureStatus.UNSTABLE.value,
    ],
    dtype=np.float32,
)
_FEATURE_COLORS: dict[FeatureStatus, tuple[int, int, int]] = {
    FeatureStatus.NEW: (0, 255, 0),
    FeatureStatus.TRACKED: (255, 0, 0),
    FeatureStatus.STABLE: (255, 0, 255),
    FeatureStatus.UNSTABLE: (0, 0, 255),
    FeatureStatus.LOST: (128, 128, 128),
}
_MAX_STATUS_VALUE = max(status.value for status in FeatureStatus)
_FEATURE_COLORS_LUT = np.zeros((_MAX_STATUS_VALUE + 1, 3), dtype=np.uint8)
for status, color in _FEATURE_COLORS.items():
    _FEATURE_COLORS_LUT[status.value] = color


class FeatureTensor:
    """Feature tensor."""

    schema = active_feat_schema

    def __init__(self, feat_capacity: int = 200, history_capacity: int = 2) -> None:
        """Initialize the feature tensor."""
        self.feat_capacity = feat_capacity
        self.history_capacity = history_capacity

        self._ts_head = 0
        self._last_timestamp = -float("inf")
        self._prev_timestamp = -float("inf")

        # (feat_id, timestamp, ul, vl, ur, vr, state, age)
        self._data = np.full((history_capacity, feat_capacity, 8), np.nan, dtype=np.float32)
        self._id_to_idx: dict[int, int] = {}
        self.free_slots = list(range(feat_capacity - 1, -1, -1))

    def step(self, new_timestamp: float) -> None:
        """Step the feature tensor."""
        occupated_slots = self.active_indeces
        if occupated_slots.size > 0:
            current_data = self._data[self._ts_head, occupated_slots]
            is_nan = np.isnan(current_data[:, 1])
            is_lost = current_data[:, 6] == FeatureStatus.LOST.value
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
    def prev_data(self) -> NDArray[np.float32]:
        """Get the previous data of the feature tensor."""
        prev_idx = (self._ts_head - 1) % self.history_capacity
        return self._data[prev_idx]

    def active_features(self, states: None | list[FeatureStatus] = None) -> NDArray[np.float32]:
        """Get the active features of the feature tensor."""
        if states is None:
            mask = np.isin(self.current_data[:, 6], _DEFAULT_STATUS_VALUES)
        else:
            values = np.array([state.value for state in states], dtype=np.float32)
            mask = np.isin(self.current_data[:, 6], values)
        mask &= ~np.isnan(self.current_data[:, 1])
        return self.current_data[mask]

    @classmethod
    def from_capacity(cls, capacity: int = 5) -> "FeatureTensor":
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

    def add(
        self, feat_id: int, timestamp: float, left_uv: Point2, right_uv: Point2 | None, state: FeatureStatus
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
        data = np.array(
            [
                feat_id,
                timestamp,
                left_uv[0],
                left_uv[1],
                right_uv[0] if right_uv is not None else np.nan,
                right_uv[1] if right_uv is not None else np.nan,
                state.value,
                0,
            ],
            dtype=np.float32,
        )
        self._data[t, index] = data
        self._id_to_idx[feat_id] = index

    def add_batch(self, timestamp: float, batch: NDArray[np.float32]) -> None:
        """Add a batch of features to the feature tensor."""
        if timestamp < self._last_timestamp:
            raise ValueError("Old timestamp")
        if timestamp > self._last_timestamp:
            self.step(timestamp)
        feat_ids = batch[:, 0].astype(np.int32)
        indexes = self.allocate_slots(feat_ids)
        t = self._ts_head
        self._data[t, indexes, 0] = feat_ids
        self._data[t, indexes, 1] = batch[:, 1]
        self._data[t, indexes, 2] = batch[:, 2]
        self._data[t, indexes, 3] = batch[:, 3]
        self._data[t, indexes, 4] = batch[:, 4]
        self._data[t, indexes, 5] = batch[:, 5]
        self._data[t, indexes, 6] = batch[:, 6]
        self._data[t, indexes, 7] = batch[:, 7]
        self._id_to_idx.update(dict(zip(feat_ids.tolist(), indexes.tolist(), strict=True)))

    def update_state(self, feat_ids: NDArray[np.int32], state: FeatureStatus) -> None:
        """Update the state of the features in the feature tensor."""
        t = self._ts_head
        slots = self.find_slots(feat_ids)
        self._data[t, slots, 6] = state.value

    def get_slots_by_status(self, status: FeatureStatus) -> NDArray[np.int32]:
        """Get the features by status."""
        mask = self.current_data[:, 6] == status.value
        return np.where(mask)[0].astype(np.int32)

    def as_arrow(self) -> pa.RecordBatch:
        """Convert the feature tensor to a struct array."""
        active_data = self.active_data
        left_points = pa.StructArray.from_arrays(
            [active_data[:, 2], active_data[:, 3]],
            names=["u", "v"],
        )
        right_points = pa.StructArray.from_arrays(
            [active_data[:, 4], active_data[:, 5]],
            names=["u", "v"],
        )
        stereo_struct = pa.StructArray.from_arrays(
            [left_points, right_points],
            names=["left", "right"],
        )
        return pa.RecordBatch.from_arrays(
            [
                active_data[:, 0],
                active_data[:, 1],
                stereo_struct,
                active_data[:, 6],
                active_data[:, 7],
            ],
            schema=active_feat_schema,
        )

    @classmethod
    def from_arrow(cls, arrow: pa.RecordBatch, history_capacity: int = 1) -> "FeatureTensor":
        """Create a feature tensor from a record batch."""
        num_features = arrow.num_rows
        capacity = num_features
        tensor = cls(capacity, history_capacity)
        if num_features == 0:
            return tensor

        batch = np.full((num_features, 8), np.nan, dtype=np.float32)
        batch[:, 0] = arrow.column(0).to_numpy()
        batch[:, 1] = arrow.column(1).to_numpy()
        stereo_struct = arrow.column(2)
        left_points = stereo_struct.field("left")
        right_points = stereo_struct.field("right")
        batch[:, 2] = left_points.field("u").to_numpy()
        batch[:, 3] = left_points.field("v").to_numpy()
        batch[:, 4] = right_points.field("u").to_numpy()
        batch[:, 5] = right_points.field("v").to_numpy()
        batch[:, 6] = arrow.column(3).to_numpy()
        batch[:, 7] = arrow.column(4).to_numpy()
        timestamp = batch[:, 1].max().item()
        tensor.add_batch(timestamp, batch)
        return tensor

    @staticmethod
    def to_color_array(data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Convert the data to a color array."""
        if data.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        feature_statuses = data[:, 6].astype(np.int32)
        return _FEATURE_COLORS_LUT[feature_statuses]
