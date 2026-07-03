import numpy as np

from core.dense_mapping.voxel_builder import VoxelBuilder
from core.dense_mapping.voxel_map import VoxelConfig, VoxelMap
from core.dense_mapping.voxel_schema import VoxelSchema
from core.dense_mapping.voxel_store import VoxelStatus, VoxelStore


def make_voxel(
    key: tuple[int, int, int],
    *,
    hits: float,
    color: tuple[float, float, float] = (10.0, 20.0, 30.0),
    center: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> np.ndarray:
    """Create one VoxelSchema row for map/store tests."""
    voxel = np.zeros((1, VoxelSchema.count()), dtype=np.float32)
    voxel[0, VoxelSchema.VOXEL_KEY] = np.array(key, dtype=np.float32)
    voxel[0, VoxelSchema.VOXEL_HITS] = hits
    voxel[0, VoxelSchema.VOXEL_COLOR] = np.array(color, dtype=np.float32)
    voxel[0, VoxelSchema.VOXEL_CENTER] = np.array(center, dtype=np.float32)
    return voxel


def test_integrate_voxels_confirms_voxel_only_after_hit_and_observation_thresholds() -> None:
    """VoxelMap should apply confirmation policy on top of store stats."""
    config = VoxelConfig(
        voxel_size_m=0.1,
        store_size=4,
        min_confirmed_hits=3,
        min_confirmed_observations=2,
    )
    voxel_map = VoxelMap(
        config=config,
        builder=VoxelBuilder.default_factory(config.voxel_size_m),
        store=VoxelStore(store_size=config.store_size),
    )

    first_count = voxel_map.integrate_voxels(make_voxel((0, 0, 0), hits=3.0))
    first_view = voxel_map.store.active_voxel_view()

    assert first_count == 1
    assert voxel_map.map_size() == 1
    assert first_view[0, VoxelSchema.VOXEL_STATUS] == VoxelStatus.TENTATIVE.value

    second_count = voxel_map.integrate_voxels(make_voxel((0, 0, 0), hits=1.0))
    second_view = voxel_map.store.active_voxel_view()

    assert second_count == 1
    assert voxel_map.map_size() == 1
    assert second_view[0, VoxelSchema.VOXEL_HITS] == 4.0
    assert second_view[0, VoxelSchema.VOXEL_OBSERVATIONS] == 2.0
    assert second_view[0, VoxelSchema.VOXEL_STATUS] == VoxelStatus.CONFIRMED.value


def test_integrate_voxels_handles_empty_batch_without_changing_store() -> None:
    """VoxelMap should keep empty updates as a no-op."""
    config = VoxelConfig(store_size=4)
    voxel_map = VoxelMap(
        config=config,
        builder=VoxelBuilder.default_factory(config.voxel_size_m),
        store=VoxelStore(store_size=config.store_size),
    )

    integrated_count = voxel_map.integrate_voxels(np.empty((0, VoxelSchema.count()), dtype=np.float32))

    assert integrated_count == 0
    assert voxel_map.map_size() == 0
