import pytest

from pipeline.nodes.mapping import VoxelConfig


def test_voxel_size_is_loaded_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mapping voxel resolution should be configurable per node."""
    monkeypatch.setenv("VOXEL_SIZE_M", "0.2")

    assert VoxelConfig().voxel_size_m == 0.2
