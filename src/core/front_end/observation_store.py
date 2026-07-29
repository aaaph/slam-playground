from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

type Observations = NDArray[np.float64]
type ObservationHistories = NDArray[np.float64]
type ObservationHistoryMask = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ReadyObservationCriteria:
    """Criteria for ready observation histories."""

    min_history_size: int
    min_parallax_rad: float
    min_parallax_observations: int = 3


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

    LEFT_UV = slice(LEFT_U, LEFT_V + 1)
    RIGHT_UV = slice(RIGHT_U, RIGHT_V + 1)
    CAM0_MATRIX = slice(CAM0_MATRIX_00, CAM0_MATRIX_33 + 1)

    @classmethod
    def size(cls) -> int:
        """Return the size of the observation schema."""
        return cls.CAM0_MATRIX_33 + 1

    @classmethod
    def pose_matrix(cls, flat_array: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the pose matrix from a flat array."""
        return flat_array[cls.CAM0_MATRIX].reshape(4, 4)


class CompressPolicy(Enum):
    """Policy for compressing the observations."""

    UNIFORM_RECENT = auto()
    TOP_DISPLACEMENT = auto()


class ObservationStore:
    """Store for the observations of landmarks."""

    def __init__(
        self,
        capacity: int = 1000,
        history_size: int = 20,
        compressed_history_size: int = 5,
        compress_policy: CompressPolicy = CompressPolicy.TOP_DISPLACEMENT,
        k_inv: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the observation store."""
        self._compress_policy = compress_policy
        self._capacity = capacity
        self._history_size = history_size
        self._compressed_history_size = compressed_history_size
        self._k_inv = np.eye(3, dtype=np.float64) if k_inv is None else k_inv
        self._observations = np.full((capacity, history_size, ObservationSchema.size()), np.nan)
        self._index = 0
        self._feat_ids_to_slot: dict[int, int] = {}
        self._slot_to_feat: NDArray[np.int32] = np.full(self._capacity, -1, np.int32)
        self._feat_ids_to_history_size: dict[int, int] = {}

        self._free_slots: list[int] = []
        self._next_slot = 0

    @classmethod
    def default_factory(
        cls,
        capacity: int = 1000,
        history_size: int = 20,
        compressed_history_size: int = 5,
        compress_policy: CompressPolicy = CompressPolicy.TOP_DISPLACEMENT,
        k_inv: NDArray[np.float64] | None = None,
    ) -> Self:
        """Create a default observation store."""
        return cls(capacity, history_size, compressed_history_size, compress_policy, k_inv)

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
                self._feat_ids_to_history_size[fid_int] = 0

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
            "feats_to_slots": (self._feat_ids_to_slot),
            "feats_history": (self._feat_ids_to_history_size),
        }

    def get_feat_history(self, feat_id: int) -> NDArray[np.float64]:
        """Get the history of a feature."""
        slot = self._feat_ids_to_slot.get(feat_id, -1)
        history_size = self._feat_ids_to_history_size.get(feat_id, 0)
        if slot == -1 or history_size == 0:
            return np.empty((history_size, ObservationSchema.size()), dtype=np.float64)
        return self._observations[slot, :history_size, :]

    def get_slots_by_criteria(self, criteria: ReadyObservationCriteria) -> NDArray[np.int32]:
        """Get feature slots by readiness criteria."""
        used_slots = np.fromiter(self._feat_ids_to_slot.values(), dtype=np.int32)
        if used_slots.size == 0:
            return np.empty((0,), dtype=np.int32)

        feat_ids = self._slot_to_feat[used_slots]
        history_sizes = np.array(
            [self._feat_ids_to_history_size[int(fid)] for fid in feat_ids],
            dtype=np.int32,
        )

        history_ready_mask = history_sizes >= max(criteria.min_history_size, 2)
        candidate_slots = used_slots[history_ready_mask]
        candidate_history_sizes = history_sizes[history_ready_mask]
        if candidate_slots.size == 0:
            return np.empty((0,), dtype=np.int32)

        history_mask = np.arange(self._history_size)[None, :] < candidate_history_sizes[:, None]
        uv = self._observations[candidate_slots, :, ObservationSchema.LEFT_UV]
        pixel_homogeneous = np.ones((candidate_slots.shape[0], self._history_size, 3), dtype=np.float64)
        pixel_homogeneous[:, :, :2] = uv
        bearings_camera = pixel_homogeneous @ self._k_inv.T
        bearings_camera /= np.linalg.norm(bearings_camera, axis=2, keepdims=True)

        world_from_cam0 = self._observations[candidate_slots, :, ObservationSchema.CAM0_MATRIX].reshape(
            candidate_slots.shape[0], self._history_size, 4, 4
        )
        anchor_from_world_rot = np.swapaxes(world_from_cam0[:, 0, :3, :3], 1, 2)
        anchor_from_cam0_rot = np.einsum("nij,nhjk->nhik", anchor_from_world_rot, world_from_cam0[:, :, :3, :3])
        bearings_anchor = np.einsum("nhij,nhj->nhi", anchor_from_cam0_rot, bearings_camera)
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
        return candidate_slots[ready]

    def get_feat_by_criteria(self, criteria: ReadyObservationCriteria) -> NDArray[np.int32]:
        """Get feature IDs by readiness criteria."""
        slots = self.get_slots_by_criteria(criteria)
        return self._slot_to_feat[slots]

    def get_ready_feature_slice(
        self, criteria: ReadyObservationCriteria
    ) -> tuple[NDArray[np.int32], ObservationHistories, ObservationHistoryMask]:
        """Get ready feature IDs, fixed-depth histories, and valid history mask."""
        slots = self.get_slots_by_criteria(criteria)
        feat_ids = self._slot_to_feat[slots]
        history_sizes = np.array(
            [self._feat_ids_to_history_size[int(fid)] for fid in feat_ids],
            dtype=np.int32,
        )
        history_mask = np.arange(self._history_size)[None, :] < history_sizes[:, None]
        return feat_ids, self._observations[slots, :, :], history_mask

    def remove_features(self, feat_ids: NDArray[np.int32]) -> None:
        """Remove features from the observation store."""
        feat_ids_arr = np.asarray(feat_ids, dtype=np.int32)
        if feat_ids_arr.ndim != 1:
            raise ValueError("feat_ids must be a 1D array")
        if feat_ids_arr.shape[0] != np.unique(feat_ids_arr).shape[0]:
            raise ValueError("feat_ids must be unique")

        for fid in feat_ids_arr:
            fid_int = int(fid)
            slot = self._feat_ids_to_slot.pop(fid_int, -1)
            if slot == -1:
                continue
            self._observations[slot].fill(np.nan)
            self._slot_to_feat[slot] = -1
            self._free_slots.append(slot)
            self._feat_ids_to_history_size.pop(fid_int, None)

    def add_observations(self, observations: Observations) -> None:  # noqa: PLR0915
        """Add observations to the observation store."""
        feat_ids = observations[:, ObservationSchema.FEAT_ID].astype(np.int32, copy=False)

        history_slots = np.full(feat_ids.shape[0], -1, dtype=np.int32)
        index_slots: NDArray[np.int32] | None = None
        next_history_sizes: dict[int, int] = {}
        match self._compress_policy:
            case CompressPolicy.UNIFORM_RECENT:
                index_slots = self._get_feature_slots(feat_ids)
                for i, fid in enumerate(feat_ids):
                    fid_int = int(fid)
                    feat_history = self._feat_ids_to_history_size.get(fid_int, 0)
                    if feat_history >= self._history_size:
                        compressed_size = min(self._compressed_history_size, self._history_size - 1)
                        keep_indices = np.rint(np.linspace(0, feat_history - 1, compressed_size)).astype(np.int32)
                        slot = int(index_slots[i])
                        compressed_history = self._observations[slot, keep_indices, :].copy()
                        self._observations[slot].fill(np.nan)
                        self._observations[slot, :compressed_size, :] = compressed_history
                        feat_history = compressed_size

                    history_slots[i] = feat_history
                    next_history_sizes[fid_int] = min(feat_history + 1, self._history_size)

            case CompressPolicy.TOP_DISPLACEMENT:
                index_slots = self._get_feature_slots(feat_ids)
                for i, fid in enumerate(feat_ids):
                    fid_int = int(fid)
                    feat_history = self._feat_ids_to_history_size.get(fid_int, 0)
                    if feat_history >= self._history_size:
                        compressed_size = min(self._compressed_history_size, self._history_size - 1)
                        slot = int(index_slots[i])
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
                    next_history_sizes[fid_int] = min(feat_history + 1, self._history_size)

        if index_slots is None:
            index_slots = self._get_feature_slots(feat_ids)

        anchor_uv = self._observations[index_slots, 0, ObservationSchema.LEFT_UV]
        current_uv = observations[:, ObservationSchema.LEFT_UV]
        anchor_pixel_displacement = np.linalg.norm(current_uv - anchor_uv, axis=1)
        is_anchor = history_slots == 0
        anchor_pixel_displacement[is_anchor] = 0.0
        self._observations[index_slots, history_slots, :] = observations
        self._observations[index_slots, history_slots, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = (
            anchor_pixel_displacement
        )
        self._feat_ids_to_history_size.update(next_history_sizes)
