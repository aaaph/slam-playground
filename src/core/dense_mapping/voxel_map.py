from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.dense_mapping.voxel_builder import VoxelBuilder
from core.dense_mapping.voxel_store import VoxelStatus, VoxelStore


@dataclass(frozen=True)
class VoxelConfig:
    """Voxel configuration."""

    voxel_size_m: float = 0.1
    store_size: int = 150_000
    min_confirmed_hits: int = 8
    min_confirmed_observations: int = 2


class VoxelMap:
    """Voxel map."""

    def __init__(self, config: VoxelConfig, builder: VoxelBuilder, store: VoxelStore) -> None:
        """Initialize the voxel map."""
        self.config = config
        self.builder = builder
        self.store = store

    @classmethod
    def default_factory(cls, config: VoxelConfig) -> Self:
        """Create a default voxel map."""
        return cls(
            config=config,
            builder=VoxelBuilder.default_factory(config.voxel_size_m),
            store=VoxelStore.default_factory(config.store_size),
        )

    def integrate_voxels(self, voxels: NDArray[np.float32]) -> int:
        """Integrate voxels into the voxel map."""
        indices = self.store.add_voxels(voxels)

        hits, observations = self.store.voxel_stats(indices)
        is_confirmed = (hits >= self.config.min_confirmed_hits) & (
            observations >= self.config.min_confirmed_observations
        )
        self.store.update_voxel_status(
            indices, np.where(is_confirmed, VoxelStatus.CONFIRMED.value, VoxelStatus.TENTATIVE.value)
        )
        return indices.shape[0]

    def map_size(self) -> int:
        """Get the size of the voxel map."""
        return self.store.size
