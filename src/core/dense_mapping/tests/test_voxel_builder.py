import numpy as np

from core.dense_mapping.voxel_builder import VoxelBuilder
from core.dense_mapping.voxel_schema import VoxelSchema


def test_build_from_point_cloud_groups_points_by_grid_key() -> None:
    """VoxelBuilder should aggregate sampled points into grid-cell voxel rows."""
    builder = VoxelBuilder.default_factory(voxel_size_m=0.5)
    point_cloud = np.array(
        [
            [0.10, 0.10, 0.10, 10.0, 20.0, 30.0],
            [0.49, 0.20, 0.20, 30.0, 40.0, 50.0],
            [0.50, -0.10, 1.00, 100.0, 0.0, 50.0],
        ],
        dtype=np.float32,
    )

    voxels = builder.build_from_point_cloud(point_cloud)

    assert voxels.shape == (2, VoxelSchema.count())
    assert voxels.dtype == np.float32
    assert np.array_equal(
        voxels[:, VoxelSchema.VOXEL_KEY],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(voxels[:, VoxelSchema.VOXEL_HITS], np.array([2.0, 1.0], dtype=np.float32))
    assert np.allclose(
        voxels[:, VoxelSchema.VOXEL_CENTER],
        np.array(
            [
                [0.25, 0.25, 0.25],
                [0.75, -0.25, 1.25],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(
        voxels[:, VoxelSchema.VOXEL_COLOR],
        np.array(
            [
                [20.0, 30.0, 40.0],
                [100.0, 0.0, 50.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_from_point_cloud_returns_empty_voxel_batch_for_empty_input() -> None:
    """VoxelBuilder should preserve the VoxelSchema column contract for empty input."""
    builder = VoxelBuilder.default_factory(voxel_size_m=0.5)

    voxels = builder.build_from_point_cloud(np.empty((0, 6), dtype=np.float32))

    assert voxels.shape == (0, VoxelSchema.count())
    assert voxels.dtype == np.float32
