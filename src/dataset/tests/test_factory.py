from pathlib import Path
from typing import Any

import pytest

from dataset.euroc import EurocDataset
from dataset.factory import DatasetFactory
from dataset.interfaces import MonocularDataset, StereoDataset, VioDataset
from dataset.manifest import DatasetManifest
from datasets import Dataset


def _minimal_euroc_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "timestamp": [1],
            "stereo": [[None, None]],
            "gyro": [[0.0, 0.0, 0.0]],
            "acc": [[0.0, 0.0, 0.0]],
            "gt_position": [[1.0, 2.0, 3.0]],
            "gt_orientation": [[0.0, 0.0, 0.0, 1.0]],
            "gt_velocity": [[0.0, 0.0, 0.0]],
            "gt_gyro_bias": [[0.0, 0.0, 0.0]],
            "gt_acc_bias": [[0.0, 0.0, 0.0]],
        }
    )


class FakeManifestLoader:
    """Canonical HF dataset loader test double."""

    def __init__(self, dataset: Dataset) -> None:
        """Create a fake loader."""
        self.dataset = dataset
        self.seen_manifest: DatasetManifest | None = None

    def load_by_manifest(self, manifest: DatasetManifest) -> Dataset:
        """Capture the manifest and return a fake HF dataset."""
        self.seen_manifest = manifest
        return self.dataset


class TestDatasetFactory:
    """Tests for capability-based dataset adapter creation."""

    def test_load_vio_dataset_returns_euroc_adapter(self) -> None:
        """EuRoC should satisfy the VIO and stereo dataset contracts."""
        hf_dataset = _minimal_euroc_dataset()
        loader = FakeManifestLoader(hf_dataset)
        factory = DatasetFactory(repo_root=Path.cwd(), loader=loader)

        dataset = factory.load_vio_dataset("euroc_mh_01")

        assert isinstance(dataset, VioDataset)
        assert isinstance(dataset, StereoDataset)
        assert not isinstance(dataset, MonocularDataset)
        assert isinstance(dataset, EurocDataset)
        assert dataset.ds is hf_dataset
        assert loader.seen_manifest is not None
        assert loader.seen_manifest.name == "euroc_mh_01"

    def test_load_stereo_dataset_accepts_vio_dataset(self) -> None:
        """A VIO dataset should also satisfy the stereo contract."""
        hf_dataset = _minimal_euroc_dataset()
        factory = DatasetFactory(repo_root=Path.cwd(), loader=FakeManifestLoader(hf_dataset))

        dataset = factory.load_stereo_dataset("euroc")

        assert isinstance(dataset, StereoDataset)

    def test_load_monocular_dataset_rejects_euroc_adapter(self) -> None:
        """EuRoC currently has no monocular dataset contract implementation."""
        hf_dataset = _minimal_euroc_dataset()
        factory = DatasetFactory(repo_root=Path.cwd(), loader=FakeManifestLoader(hf_dataset))

        with pytest.raises(TypeError, match="does not support monocular"):
            factory.load_monocular_dataset("euroc_mh_01")

    def test_load_vio_dataset_rejects_adapter_without_vio_contract(self) -> None:
        """Factory should fail clearly when an adapter does not implement VIO."""
        hf_dataset = _minimal_euroc_dataset()
        factory = DatasetFactory(
            repo_root=Path.cwd(),
            loader=FakeManifestLoader(hf_dataset),
            adapters={"euroc": lambda _dataset, _rig: object()},
        )

        with pytest.raises(TypeError, match="does not support VIO"):
            factory.load_vio_dataset("euroc_mh_01")

    def test_unsupported_dataset_type_raises(self) -> None:
        """Factory should report unsupported manifest types clearly."""
        hf_dataset = _minimal_euroc_dataset()
        captured: dict[str, Any] = {}

        def fake_adapter(dataset: Dataset, _rig: object) -> object:
            captured["dataset"] = dataset
            return object()

        factory = DatasetFactory(
            repo_root=Path.cwd(),
            loader=FakeManifestLoader(hf_dataset),
            adapters={"kitti": fake_adapter},
        )

        with pytest.raises(ValueError, match="Unsupported dataset type 'euroc'"):
            factory.load_dataset("euroc_mh_01")
        assert captured == {}
