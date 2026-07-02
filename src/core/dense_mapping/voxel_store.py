from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from core.dense_mapping.voxel_schema import VoxelSchema

type VoxelKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class UpdateStoreResult:
    """Result of updating the voxel store."""

    new_voxel_indices: list[int]
    updated_voxel_indices: list[int]

    @property
    def new_voxels_count(self) -> int:
        """Get the number of new voxels."""
        return len(self.new_voxel_indices)

    @property
    def updated_voxels_count(self) -> int:
        """Get the number of updated voxels."""
        return len(self.updated_voxel_indices)


class VoxelStatus(Enum):
    """Voxel status."""

    TENTATIVE = 1
    CONFIRMED = auto()


class StoreIsFullError(Exception):
    """Exception raised when the store is full."""

    def __init__(self, message: str = "Voxel store is full.") -> None:
        """Initialize the StoreFullError exception."""
        self.message = message
        super().__init__(self.message)


class VoxelStore:
    """Voxel store."""

    def __init__(self, store_size: int = 150_000) -> None:
        """Initialize the voxel store."""
        self.store_size = store_size
        self.voxels = {}
        self.data = np.full((self.store_size, VoxelSchema.count()), np.nan, dtype=np.float32)
        self.voxel_key_to_idx: dict[VoxelKey, int] = {}
        self.free_indices = []
        self.size = 0

    @classmethod
    def default_factory(cls, store_size: int = 100_000) -> "VoxelStore":
        """Create a default voxel store."""
        return cls(store_size=store_size)

    def allocate_index(self) -> int:
        """Allocate an index for a voxel."""
        if self.free_indices:
            return self.free_indices.pop()
        if self.size < self.store_size:
            slot = self.size
            self.size += 1
            return slot

        raise StoreIsFullError

    def add_voxels(self, voxels: NDArray[np.float32]) -> NDArray[np.int32]:
        """Add voxels to the store."""
        if voxels.size == 0:
            return np.empty(0, dtype=np.int32)

        indices = np.empty(voxels.shape[0], dtype=np.int32)
        indices_idx = 0

        for row in voxels:
            voxel_key = (
                int(row[VoxelSchema.VOXEL_KEY][0]),
                int(row[VoxelSchema.VOXEL_KEY][1]),
                int(row[VoxelSchema.VOXEL_KEY][2]),
            )
            slot = self.voxel_key_to_idx.get(voxel_key)
            if slot is None:
                index = self.allocate_index()
                self.voxel_key_to_idx[voxel_key] = index
                self.data[index, VoxelSchema.VOXEL_KEY] = row[VoxelSchema.VOXEL_KEY]
                self.data[index, VoxelSchema.VOXEL_COLOR] = row[VoxelSchema.VOXEL_COLOR]
                self.data[index, VoxelSchema.VOXEL_CENTER] = row[VoxelSchema.VOXEL_CENTER]
                self.data[index, VoxelSchema.VOXEL_HITS] = row[VoxelSchema.VOXEL_HITS]
                self.data[index, VoxelSchema.VOXEL_OBSERVATIONS] = 1.0
                self.data[index, VoxelSchema.VOXEL_STATUS] = VoxelStatus.TENTATIVE.value
                indices[indices_idx] = index
                indices_idx += 1
            else:
                old_hits = self.data[slot, VoxelSchema.VOXEL_HITS]
                new_hits = row[VoxelSchema.VOXEL_HITS]
                old_colors = self.data[slot, VoxelSchema.VOXEL_COLOR]
                new_colors = row[VoxelSchema.VOXEL_COLOR]
                self.data[slot, VoxelSchema.VOXEL_COLOR] = (old_colors * old_hits + new_colors * new_hits) / (
                    old_hits + new_hits
                )

                self.data[slot, VoxelSchema.VOXEL_HITS] += row[VoxelSchema.VOXEL_HITS]
                self.data[slot, VoxelSchema.VOXEL_OBSERVATIONS] += 1.0

                indices[indices_idx] = slot
                indices_idx += 1
        return indices

    def active_voxel_view(self) -> NDArray[np.float32]:
        """Get all voxels."""
        return self.data[: self.size, :]

    def get_voxels_by_indices(self, indices: NDArray[np.int32]) -> NDArray[np.float32]:
        """Get voxels by indices."""
        return self.data[indices, :]

    def voxel_stats(self, indices: NDArray[np.int32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get voxel stats by indices."""
        hits = self.data[indices, VoxelSchema.VOXEL_HITS]
        observations = self.data[indices, VoxelSchema.VOXEL_OBSERVATIONS]
        return hits, observations

    def update_voxel_status(self, indices: NDArray[np.int32], statuses: NDArray[np.int32]) -> None:
        """Update voxel status by indices."""
        self.data[indices, VoxelSchema.VOXEL_STATUS] = statuses
