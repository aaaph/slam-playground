from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

type Observations = NDArray[np.float64]
type ObservationSlots = NDArray[np.int32]
type ObservationHistories = NDArray[np.float64]
type ObservationHistorySlots = NDArray[np.int32]
type ObservationHistoryMask = NDArray[np.bool_]
type ObservationHistoryVersions = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class ReadyObservationCriteria:
    """Criteria for ready observation histories."""

    min_parallax_rad: float = 0.02
    min_history_size: int = 5
    min_parallax_observations: int = 3
    min_pixel_displacement: float = 1.0


DEFAULT_READY_OBSERVATION_CRITERIA = ReadyObservationCriteria()


class ObservationSchema:
    """Schema for the observation of landmarks."""

    FEAT_ID = 0
    LEFT_U = 1
    LEFT_V = 2
    RIGHT_U = 3
    RIGHT_V = 4
    ANCHOR_PIXEL_DISPLACEMENT = 5

    CAM0_MATRIX_00 = 6
    CAM0_MATRIX_01 = 7
    CAM0_MATRIX_02 = 8
    CAM0_MATRIX_03 = 9
    CAM0_MATRIX_10 = 10
    CAM0_MATRIX_11 = 11
    CAM0_MATRIX_12 = 12
    CAM0_MATRIX_13 = 13
    CAM0_MATRIX_20 = 14
    CAM0_MATRIX_21 = 15
    CAM0_MATRIX_22 = 16
    CAM0_MATRIX_23 = 17
    CAM0_MATRIX_30 = 18
    CAM0_MATRIX_31 = 19
    CAM0_MATRIX_32 = 20
    CAM0_MATRIX_33 = 21

    LEFT_BEARING_0 = 22
    LEFT_BEARING_1 = 23
    LEFT_BEARING_2 = 24

    RIGHT_BEARING_0 = 25
    RIGHT_BEARING_1 = 26
    RIGHT_BEARING_2 = 27
    FRAME_ID = 28

    LEFT_UV = slice(LEFT_U, LEFT_V + 1)
    RIGHT_UV = slice(RIGHT_U, RIGHT_V + 1)
    CAM0_MATRIX = slice(CAM0_MATRIX_00, CAM0_MATRIX_33 + 1)
    LEFT_BEARING = slice(LEFT_BEARING_0, LEFT_BEARING_2 + 1)
    RIGHT_BEARING = slice(RIGHT_BEARING_0, RIGHT_BEARING_2 + 1)

    @classmethod
    def size(cls) -> int:
        """Return the size of the observation schema."""
        return cls.FRAME_ID + 1

    @classmethod
    def pose_matrix(cls, flat_array: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the pose matrix from a flat array."""
        return flat_array[cls.CAM0_MATRIX].reshape(4, 4)


class CompressPolicy(Enum):
    """Policy for compressing the observations."""

    UNIFORM_RECENT = auto()
    TOP_DISPLACEMENT = auto()


class SelectPolicy(Enum):
    """Policy for selecting the observations."""

    P90_PARALLAX = auto()
    ANCHOR_TO_LATEST_PARALLAX = auto()
    PIXEL_DISPLACEMENT = auto()


class ObservationStore:
    """Store for the observations of landmarks."""

    def __init__(  # noqa: PLR0913
        self,
        k_inv: NDArray[np.float64],
        capacity: int = 1000,
        history_size: int = 20,
        compressed_history_size: int = 5,
        compress_policy: CompressPolicy = CompressPolicy.TOP_DISPLACEMENT,
        select_policy: SelectPolicy = SelectPolicy.ANCHOR_TO_LATEST_PARALLAX,
        ready_criteria: ReadyObservationCriteria = DEFAULT_READY_OBSERVATION_CRITERIA,
    ) -> None:
        """Initialize the observation store."""
        self._compress_policy = compress_policy
        self._select_policy = select_policy
        self._ready_criteria = ready_criteria
        self._capacity = capacity
        self._history_size = history_size
        self._compressed_history_size = compressed_history_size
        self._k_inv_T = k_inv.T
        self._observations = np.full((capacity, history_size, ObservationSchema.size()), np.nan)
        self._index = 0
        self._feat_ids_to_slot: dict[int, int] = {}
        self._slot_to_feat: NDArray[np.int32] = np.full(self._capacity, -1, np.int32)
        self._history_sizes: NDArray[np.int32] = np.zeros(self._capacity, dtype=np.int32)
        self._history_versions: NDArray[np.int32] = np.zeros(self._capacity, dtype=np.int32)

        self._free_slots: list[int] = []
        self._next_slot = 0

    @classmethod
    def default_factory(  # noqa: PLR0913
        cls,
        k_inv: NDArray[np.float64],
        capacity: int = 1000,
        history_size: int = 20,
        compressed_history_size: int = 5,
        compress_policy: CompressPolicy = CompressPolicy.TOP_DISPLACEMENT,
        select_policy: SelectPolicy = SelectPolicy.ANCHOR_TO_LATEST_PARALLAX,
        ready_criteria: ReadyObservationCriteria = DEFAULT_READY_OBSERVATION_CRITERIA,
    ) -> Self:
        """Create a default observation store."""
        return cls(
            k_inv,
            capacity,
            history_size,
            compressed_history_size,
            compress_policy,
            select_policy,
            ready_criteria,
        )

    def _get_feature_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        feat_ids_arr = np.asarray(feat_ids, dtype=np.int32)
        if feat_ids_arr.ndim != 1:
            raise ValueError("feat_ids must be a 1D array")
        if feat_ids_arr.shape[0] != np.unique(feat_ids_arr).shape[0]:
            raise ValueError("feat_ids must be unique")

        n = feat_ids_arr.shape[0]
        output = np.empty(n, dtype=np.int32)
        new_indices = []
        new_feats = []

        for i, fid in enumerate(feat_ids_arr):
            fid_int = int(fid)
            slot = self._feat_ids_to_slot.get(fid_int, -1)
            if slot == -1:
                new_indices.append(i)
                new_feats.append(fid_int)
            else:
                output[i] = slot

        if new_feats:
            new_slots = self._pop_free_slots(len(new_feats))
            for idx, fid_int, slot in zip(new_indices, new_feats, new_slots, strict=True):
                output[idx] = slot
                self._feat_ids_to_slot[fid_int] = int(slot)
                self._slot_to_feat[slot] = fid_int
                self._history_sizes[slot] = 0
                self._history_versions[slot] = 0

        return output

    def _pop_free_slots(self, size: int) -> NDArray[np.int32]:
        """Pop free slots from the observation store."""
        if size == 0:
            return np.empty(0, dtype=np.int32)
        slots = np.empty(size, dtype=np.int32)
        filled = 0

        reused_count = min(size, len(self._free_slots))
        if reused_count > 0:
            for i in range(reused_count):
                slots[i] = self._free_slots.pop()
            filled += reused_count

        remaining = size - filled
        if remaining > 0:
            if self._next_slot + remaining > self._capacity:
                raise ValueError("Not enough free slots")
            slots[filled:] = np.arange(self._next_slot, self._next_slot + remaining, dtype=np.int32)
            self._next_slot += remaining
            filled += remaining

        return slots[:filled]

    def store_info(self) -> dict[str, Any]:
        """Get the store info."""
        return {
            "capacity": self._capacity,
            "history_size": self._history_size,
            "compressed_history_size": self._compressed_history_size,
            "compress_policy": self._compress_policy,
            "select_policy": self._select_policy,
            "ready_criteria": self._ready_criteria,
            "feats_to_slots": (self._feat_ids_to_slot),
            "history_sizes": self._history_sizes,
            "history_versions": self._history_versions,
        }

    def get_feat_history(self, feat_id: int) -> NDArray[np.float64]:
        """Get the history of a feature."""
        slot = self._feat_ids_to_slot.get(feat_id, -1)
        history_size = 0 if slot == -1 else int(self._history_sizes[slot])
        if slot == -1 or history_size == 0:
            return np.empty((history_size, ObservationSchema.size()), dtype=np.float64)
        return self._observations[slot, :history_size, :]

    def ready_slots(
        self, candidate_slots: ObservationSlots
    ) -> tuple[ObservationSlots, ObservationHistoryVersions, NDArray[np.int32]]:
        """Get ready slots, history versions, and feature IDs by readiness criteria."""
        criteria = self._ready_criteria
        used_slots = np.asarray(candidate_slots, dtype=np.int32)
        if used_slots.size == 0:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty, empty

        history_sizes = self._history_sizes[used_slots]

        history_ready_mask = history_sizes >= criteria.min_history_size
        candidate_slots = used_slots[history_ready_mask]
        candidate_history_sizes = history_sizes[history_ready_mask]
        if candidate_slots.size == 0:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty, empty

        history_mask = np.arange(self._history_size)[None, :] < candidate_history_sizes[:, None]

        match self._select_policy:
            case SelectPolicy.PIXEL_DISPLACEMENT:
                anchor_pixel_displacement = self._observations[
                    candidate_slots, :, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT
                ]
                anchor_pixel_displacement = np.where(history_mask, anchor_pixel_displacement, np.nan)
                anchor_pixel_displacement[:, 0] = np.nan
                p90_pixel_displacement = np.nanpercentile(anchor_pixel_displacement, 90, axis=1)
                ready_observations = np.sum(anchor_pixel_displacement >= criteria.min_pixel_displacement, axis=1)
                ready = (p90_pixel_displacement >= criteria.min_pixel_displacement) & (
                    ready_observations >= criteria.min_parallax_observations
                )
            case SelectPolicy.ANCHOR_TO_LATEST_PARALLAX:
                latest_history_indices = candidate_history_sizes - 1
                anchor_bearing = self._observations[candidate_slots, 0, ObservationSchema.LEFT_BEARING]
                latest_bearing = self._observations[
                    candidate_slots, latest_history_indices, ObservationSchema.LEFT_BEARING
                ]
                world_from_anchor = self._observations[candidate_slots, 0, ObservationSchema.CAM0_MATRIX].reshape(
                    candidate_slots.shape[0], 4, 4
                )
                world_from_latest = self._observations[
                    candidate_slots, latest_history_indices, ObservationSchema.CAM0_MATRIX
                ].reshape(candidate_slots.shape[0], 4, 4)
                anchor_from_latest_rot = np.einsum(
                    "nij,njk->nik", np.swapaxes(world_from_anchor[:, :3, :3], 1, 2), world_from_latest[:, :3, :3]
                )
                latest_bearing_anchor = np.einsum("nij,nj->ni", anchor_from_latest_rot, latest_bearing)
                latest_bearing_anchor /= np.linalg.norm(latest_bearing_anchor, axis=1, keepdims=True)
                cos_angles = np.einsum("nj,nj->n", latest_bearing_anchor, anchor_bearing)
                latest_parallax = np.arccos(np.clip(cos_angles, -1.0, 1.0))
                ready = latest_parallax >= criteria.min_parallax_rad
            case SelectPolicy.P90_PARALLAX:
                bearings_camera = self._observations[candidate_slots, :, ObservationSchema.LEFT_BEARING]

                world_from_cam0 = self._observations[candidate_slots, :, ObservationSchema.CAM0_MATRIX].reshape(
                    candidate_slots.shape[0], self._history_size, 4, 4
                )
                anchor_from_world_rot = np.swapaxes(world_from_cam0[:, 0, :3, :3], 1, 2)
                anchor_from_cam0_rot = np.einsum(
                    "nij,nhjk->nhik", anchor_from_world_rot, world_from_cam0[:, :, :3, :3]
                )
                bearings_anchor: Any = np.einsum("nhij,nhj->nhi", anchor_from_cam0_rot, bearings_camera)
                bearings_anchor /= np.linalg.norm(bearings_anchor, axis=2, keepdims=True)

                anchor_bearing = bearings_anchor[:, 0, :]
                cos_angles = np.einsum("nhj,nj->nh", bearings_anchor, anchor_bearing)
                angular_parallax = np.arccos(np.clip(cos_angles, -1.0, 1.0))
                angular_parallax = np.where(history_mask, angular_parallax, np.nan)
                angular_parallax[:, 0] = np.nan
                p90_parallax = np.nanpercentile(angular_parallax, 90, axis=1)
                ready_observations = np.sum(angular_parallax >= criteria.min_parallax_rad, axis=1)
                ready = (p90_parallax >= criteria.min_parallax_rad) & (
                    ready_observations >= criteria.min_parallax_observations
                )
        slots = candidate_slots[ready]
        return slots, self._history_versions[slots], self._slot_to_feat[slots]

    def get_slots_by_criteria(self, candidate_slots: ObservationSlots) -> NDArray[np.int32]:
        """Get feature slots by readiness criteria."""
        slots, _history_versions, _feat_ids = self.ready_slots(candidate_slots)
        return slots

    def get_feat_by_criteria(self, candidate_slots: ObservationSlots) -> NDArray[np.int32]:
        """Get feature IDs by readiness criteria."""
        _slots, _history_versions, feat_ids = self.ready_slots(candidate_slots)
        return feat_ids

    def get_feat_ids_by_slots(self, slots: NDArray[np.int32]) -> NDArray[np.int32]:
        """Get feature IDs by store slots."""
        return self._slot_to_feat[slots]

    def get_history_versions_by_slots(self, slots: NDArray[np.int32]) -> ObservationHistoryVersions:
        """Get history versions by store slots."""
        return self._history_versions[slots]

    def get_feature_slice_by_slots(
        self, slots: NDArray[np.int32]
    ) -> tuple[ObservationHistories, ObservationHistoryMask]:
        """Get fixed-depth histories and valid masks by store slots."""
        history_sizes = self._history_sizes[slots]
        history_mask = np.arange(self._history_size)[None, :] < history_sizes[:, None]
        return self._observations[slots, :, :], history_mask

    def get_ready_feature_slice(
        self, candidate_slots: ObservationSlots
    ) -> tuple[NDArray[np.int32], ObservationHistories, ObservationHistoryMask]:
        """Get ready feature IDs, fixed-depth histories, and valid history mask."""
        slots, _history_versions, feat_ids = self.ready_slots(candidate_slots)
        histories, history_mask = self.get_feature_slice_by_slots(slots)
        return feat_ids, histories, history_mask

    def remove_features(self, feat_ids: NDArray[np.int32]) -> ObservationSlots:
        """Remove features from the observation store."""
        feat_ids_arr = np.asarray(feat_ids, dtype=np.int32)
        if feat_ids_arr.ndim != 1:
            raise ValueError("feat_ids must be a 1D array")
        if feat_ids_arr.shape[0] != np.unique(feat_ids_arr).shape[0]:
            raise ValueError("feat_ids must be unique")

        removed_slots = np.full(feat_ids_arr.shape[0], -1, dtype=np.int32)
        removed_count = 0
        for fid in feat_ids_arr:
            fid_int = int(fid)
            slot = self._feat_ids_to_slot.pop(fid_int, -1)
            if slot == -1:
                continue
            self._observations[slot].fill(np.nan)
            self._slot_to_feat[slot] = -1
            self._history_sizes[slot] = 0
            self._history_versions[slot] = 0
            self._free_slots.append(slot)
            removed_slots[removed_count] = slot
            removed_count += 1

        return removed_slots[:removed_count]

    def add_observations(self, observations: Observations) -> tuple[ObservationSlots, ObservationHistorySlots]:  # noqa: PLR0915
        """Add observations to the observation store."""
        feat_ids = observations[:, ObservationSchema.FEAT_ID].astype(np.int32, copy=False)

        history_slots = np.full(feat_ids.shape[0], -1, dtype=np.int32)
        index_slots = self._get_feature_slots(feat_ids)
        next_history_sizes = np.empty(feat_ids.shape[0], dtype=np.int32)
        match self._compress_policy:
            case CompressPolicy.UNIFORM_RECENT:
                for i, slot in enumerate(index_slots):
                    feat_history = int(self._history_sizes[slot])
                    if feat_history >= self._history_size:
                        compressed_size = min(self._compressed_history_size, self._history_size - 1)
                        keep_indices = np.rint(np.linspace(0, feat_history - 1, compressed_size)).astype(np.int32)
                        compressed_history = self._observations[slot, keep_indices, :].copy()
                        self._observations[slot].fill(np.nan)
                        self._observations[slot, :compressed_size, :] = compressed_history
                        feat_history = compressed_size

                    history_slots[i] = feat_history
                    next_history_sizes[i] = min(feat_history + 1, self._history_size)

            case CompressPolicy.TOP_DISPLACEMENT:
                for i, slot in enumerate(index_slots):
                    feat_history = int(self._history_sizes[slot])
                    if feat_history >= self._history_size:
                        compressed_size = min(self._compressed_history_size, self._history_size - 1)
                        latest_idx = feat_history - 1
                        middle_indices = np.arange(1, latest_idx, dtype=np.int32)
                        middle_count = min(max(compressed_size - 2, 0), middle_indices.shape[0])
                        if middle_count > 0:
                            middle_displacement = self._observations[
                                slot, middle_indices, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT
                            ]
                            top_indices = middle_indices[
                                np.argpartition(middle_displacement, -middle_count)[-middle_count:]
                            ]
                            keep_indices = np.sort(np.concatenate((np.array([0, latest_idx]), top_indices)))
                        else:
                            keep_indices = np.array([0, latest_idx], dtype=np.int32)

                        compressed_history = self._observations[slot, keep_indices, :].copy()
                        self._observations[slot].fill(np.nan)
                        self._observations[slot, : keep_indices.shape[0], :] = compressed_history
                        feat_history = keep_indices.shape[0]

                    history_slots[i] = feat_history
                    next_history_sizes[i] = min(feat_history + 1, self._history_size)

        anchor_uv = self._observations[index_slots, 0, ObservationSchema.LEFT_UV]
        current_uv = observations[:, ObservationSchema.LEFT_UV]
        anchor_pixel_displacement = np.linalg.norm(current_uv - anchor_uv, axis=1)
        is_anchor = history_slots == 0
        anchor_pixel_displacement[is_anchor] = 0.0

        left_pixel_homogeneous = np.ones((observations.shape[0], 3), dtype=np.float64)
        left_pixel_homogeneous[:, :2] = observations[:, ObservationSchema.LEFT_UV]
        left_bearings = left_pixel_homogeneous @ self._k_inv_T
        left_bearings /= np.linalg.norm(left_bearings, axis=1, keepdims=True)

        right_uv = observations[:, ObservationSchema.RIGHT_UV]
        right_valid_mask = np.all(np.isfinite(right_uv), axis=1)
        right_bearings = np.full((observations.shape[0], 3), np.nan, dtype=np.float64)
        right_pixel_homogeneous = np.ones((np.count_nonzero(right_valid_mask), 3), dtype=np.float64)
        right_pixel_homogeneous[:, :2] = right_uv[right_valid_mask]
        right_bearings_valid = right_pixel_homogeneous @ self._k_inv_T
        right_bearings_valid /= np.linalg.norm(right_bearings_valid, axis=1, keepdims=True)
        right_bearings[right_valid_mask] = right_bearings_valid

        observations[:, ObservationSchema.LEFT_BEARING] = left_bearings
        observations[:, ObservationSchema.RIGHT_BEARING] = right_bearings

        self._observations[index_slots, history_slots, :] = observations
        self._observations[index_slots, history_slots, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = (
            anchor_pixel_displacement
        )
        self._history_sizes[index_slots] = next_history_sizes
        self._history_versions[index_slots] += 1
        return index_slots, history_slots
