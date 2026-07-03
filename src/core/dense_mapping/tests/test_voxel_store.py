import numpy as np
import pytest

from core.dense_mapping.voxel_schema import VoxelSchema
from core.dense_mapping.voxel_store import StoreIsFullError, VoxelStatus, VoxelStore


def make_voxel(
    key: tuple[int, int, int],
    *,
    hits: float,
    color: tuple[float, float, float],
    center: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> np.ndarray:
    """Create one VoxelSchema row for store tests."""
    voxel = np.zeros((1, VoxelSchema.count()), dtype=np.float32)
    voxel[0, VoxelSchema.VOXEL_KEY] = np.array(key, dtype=np.float32)
    voxel[0, VoxelSchema.VOXEL_HITS] = hits
    voxel[0, VoxelSchema.VOXEL_COLOR] = np.array(color, dtype=np.float32)
    voxel[0, VoxelSchema.VOXEL_CENTER] = np.array(center, dtype=np.float32)
    return voxel


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


def test_add_voxels_updates_existing_slot_and_averages_color_by_hits() -> None:
    """Repeated voxel keys should update stats in place and blend color by hit count."""
    store = VoxelStore(store_size=4)
    first_indices = store.add_voxels(
        make_voxel((1, 2, 3), hits=2.0, color=(10.0, 20.0, 30.0), center=(1.5, 2.5, 3.5))
    )

    second_indices = store.add_voxels(make_voxel((1, 2, 3), hits=6.0, color=(50.0, 60.0, 70.0)))
    stored_voxels = store.active_voxel_view()

    assert np.array_equal(first_indices, np.array([0], dtype=np.int32))
    assert np.array_equal(second_indices, np.array([0], dtype=np.int32))
    assert store.size == 1
    assert stored_voxels[0, VoxelSchema.VOXEL_HITS] == 8.0
    assert stored_voxels[0, VoxelSchema.VOXEL_OBSERVATIONS] == 2.0
    assert np.allclose(stored_voxels[0, VoxelSchema.VOXEL_COLOR], np.array([40.0, 50.0, 60.0]))
    assert np.allclose(stored_voxels[0, VoxelSchema.VOXEL_CENTER], np.array([1.5, 2.5, 3.5]))


def test_voxel_stats_and_status_update_use_indices() -> None:
    """Store query helpers should operate on explicit slot indices."""
    store = VoxelStore(store_size=4)
    indices = store.add_voxels(
        np.vstack(
            [
                make_voxel((0, 0, 0), hits=1.0, color=(10.0, 20.0, 30.0)),
                make_voxel((1, 0, 0), hits=4.0, color=(40.0, 50.0, 60.0)),
            ]
        ).astype(np.float32, copy=False)
    )

    hits, observations = store.voxel_stats(indices)
    store.update_voxel_status(indices, np.array([VoxelStatus.TENTATIVE.value, VoxelStatus.CONFIRMED.value]))
    selected = store.get_voxels_by_indices(indices)

    assert np.array_equal(hits, np.array([1.0, 4.0], dtype=np.float32))
    assert np.array_equal(observations, np.array([1.0, 1.0], dtype=np.float32))
    assert np.array_equal(
        selected[:, VoxelSchema.VOXEL_STATUS],
        np.array([VoxelStatus.TENTATIVE.value, VoxelStatus.CONFIRMED.value], dtype=np.float32),
    )


def test_add_voxels_raises_when_store_is_full_for_new_key() -> None:
    """VoxelStore should fail loudly when a new key cannot be allocated."""
    store = VoxelStore(store_size=1)
    store.add_voxels(make_voxel((0, 0, 0), hits=1.0, color=(10.0, 20.0, 30.0)))

    with pytest.raises(StoreIsFullError, match="Voxel store is full"):
        store.add_voxels(make_voxel((1, 0, 0), hits=1.0, color=(40.0, 50.0, 60.0)))
