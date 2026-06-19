from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from dataset.euroc import EurocDataset
from dataset.interfaces import GroundTruthDataset, MonocularDataset, StereoDataset, VioDataset
from dataset.loader import DatasetLoader
from dataset.registry import DatasetRegistry

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dataset.manifest import DatasetManifest, DatasetRigConfig
    from datasets import Dataset


class ManifestDatasetLoader(Protocol):
    """Loader that can materialize a canonical HF dataset from a manifest."""

    def load_by_manifest(self, manifest: DatasetManifest) -> Dataset:
        """Load a canonical HF dataset from an already resolved manifest."""


class DatasetFactory:
    """Create domain dataset adapters from registry manifests."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        dataset_dir: Path | None = None,
        loader: ManifestDatasetLoader | None = None,
        adapters: Mapping[str, Callable[[Dataset, DatasetRigConfig], object]] | None = None,
    ) -> None:
        """Create a dataset factory."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.registry = DatasetRegistry(repo_root=self.repo_root, dataset_dir=dataset_dir)
        self.loader = loader or DatasetLoader(repo_root=self.repo_root, dataset_dir=dataset_dir)
        self.adapters = {"euroc": EurocDataset.from_dataset} if adapters is None else dict(adapters)

    def load_dataset(self, name: str) -> object:
        """Load a dataset adapter by registry selector."""
        resolved = self.registry.resolve(name)
        adapter_factory = self.adapters.get(resolved.dataset.type)
        if adapter_factory is None:
            supported_types = ", ".join(sorted(self.adapters)) or "<none>"
            msg = f"Unsupported dataset type '{resolved.dataset.type}'. Supported types: {supported_types}"
            raise ValueError(msg)

        hf_dataset = self.loader.load_by_manifest(resolved.dataset)
        return adapter_factory(hf_dataset, resolved.rig)

    def load_stereo_dataset(self, name: str) -> StereoDataset:
        """Load a dataset that supports stereo frames."""
        dataset = self.load_dataset(name)
        if not isinstance(dataset, StereoDataset):
            msg = f"Dataset '{name}' does not support stereo"
            raise TypeError(msg)
        return dataset

    def load_vio_dataset(self, name: str) -> VioDataset:
        """Load a dataset that supports VIO stereo+IMU frames."""
        dataset = self.load_dataset(name)
        if not isinstance(dataset, VioDataset):
            msg = f"Dataset '{name}' does not support VIO stereo+IMU"
            raise TypeError(msg)
        return dataset

    def load_monocular_dataset(self, name: str) -> MonocularDataset:
        """Load a dataset that supports monocular frames."""
        dataset = self.load_dataset(name)
        if not isinstance(dataset, MonocularDataset):
            msg = f"Dataset '{name}' does not support monocular"
            raise TypeError(msg)
        return dataset

    def load_ground_truth_dataset(self, name: str) -> GroundTruthDataset:
        """Load a dataset that supports ground truth data."""
        dataset = self.load_dataset(name)
        if not isinstance(dataset, GroundTruthDataset):
            msg = f"Dataset '{name}' does not support ground truth"
            raise TypeError(msg)
        return dataset
