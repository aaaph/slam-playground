import numpy as np
from numpy.typing import NDArray

from core.dense_mapping.voxel_schema import VoxelSchema


class VoxelBuilder:
    """Voxel builder."""

    def __init__(self, voxel_size_m: float = 0.1) -> None:
        """Initialize the voxel builder."""
        self.voxel_size_m = voxel_size_m

    @classmethod
    def default_factory(cls, voxel_size_m: float = 0.1) -> "VoxelBuilder":
        """Create a default voxel builder."""
        return cls(voxel_size_m=voxel_size_m)

    def build_from_point_cloud(self, point_cloud: NDArray[np.float32]) -> NDArray[np.float32]:
        """Build the voxel map from the point cloud."""
        if point_cloud.size == 0:
            return np.empty((0, VoxelSchema.count()), dtype=np.float32)
        voxel_keys = np.floor(point_cloud[:, :3] / self.voxel_size_m).astype(np.int32)
        unique_keys, inverse, counts = np.unique(
            voxel_keys,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        voxel_count = unique_keys.shape[0]
        voxel_batch = np.full((voxel_count, VoxelSchema.count()), fill_value=0.0, dtype=np.float32)
        voxel_batch[:, VoxelSchema.VOXEL_KEY] = unique_keys
        voxel_batch[:, VoxelSchema.VOXEL_CENTER] = (
            unique_keys.astype(np.float32, copy=False) + 0.5
        ) * self.voxel_size_m
        voxel_batch[:, VoxelSchema.VOXEL_HITS] = counts.astype(np.float32, copy=False)
        np.add.at(
            voxel_batch[:, VoxelSchema.VOXEL_COLOR], inverse, point_cloud[:, 3:].astype(np.float32, copy=False)
        )
        voxel_batch[:, VoxelSchema.VOXEL_COLOR] = (
            voxel_batch[:, VoxelSchema.VOXEL_COLOR] / voxel_batch[:, VoxelSchema.VOXEL_HITS][:, None]
        )
        return voxel_batch
