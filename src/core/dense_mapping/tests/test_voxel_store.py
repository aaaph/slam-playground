import numpy as np

from core.dense_mapping.voxel_schema import VoxelSchema
from core.dense_mapping.voxel_store import VoxelStatus, VoxelStore


def test_add_voxels_preserves_float_centers_for_arrow_visualization() -> None:
    """Voxel rows stored for Arrow/Rerun should preserve metric centers."""
    store = VoxelStore(store_size=4)
    voxels = np.zeros((1, VoxelSchema.count()), dtype=np.float32)
    voxels[0, VoxelSchema.VOXEL_KEY] = np.array([0, 1, 2], dtype=np.float32)
    voxels[0, VoxelSchema.VOXEL_HITS] = 3.0
    voxels[0, VoxelSchema.VOXEL_COLOR] = np.array([10, 20, 30], dtype=np.float32)
    voxels[0, VoxelSchema.VOXEL_CENTER] = np.array([0.05, 0.15, 0.25], dtype=np.float32)

    indices = store.add_voxels(voxels)
    stored_voxels = store.active_voxel_view()

    assert np.array_equal(indices, np.array([0], dtype=np.int32))
    assert stored_voxels.dtype == np.float32
    assert np.allclose(stored_voxels[0, VoxelSchema.VOXEL_CENTER], np.array([0.05, 0.15, 0.25]))
    assert stored_voxels[0, VoxelSchema.VOXEL_STATUS] == VoxelStatus.TENTATIVE.value
