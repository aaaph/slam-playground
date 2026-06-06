from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dataset.euroc import EurocDatasetBuilder
from dataset.registry import DatasetRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataset.builder import DatasetBuilder
    from dataset.manifest import DatasetManifest
    from datasets import Dataset


class DatasetLoader:
    """Load HuggingFace datasets from named dataset manifests."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        dataset_dir: Path | None = None,
        builders: Mapping[str, DatasetBuilder] | None = None,
    ) -> None:
        """Create a dataset loader."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.registry = DatasetRegistry(repo_root=self.repo_root, dataset_dir=dataset_dir)
        self.builders = (
            {"euroc": EurocDatasetBuilder(repo_root=self.repo_root)} if builders is None else dict(builders)
        )

    def load(self, name: str) -> Dataset:
        """Load a HuggingFace dataset by manifest name."""
        manifest = self.registry.find(name)
        return self.load_by_manifest(manifest)

    def load_by_manifest(self, manifest: DatasetManifest) -> Dataset:
        """Load a HuggingFace dataset from an already loaded manifest."""
        builder = self.builders.get(manifest.type)
        if builder is None:
            supported_types = ", ".join(sorted(self.builders)) or "<none>"
            msg = f"Unsupported dataset type '{manifest.type}'. Supported types: {supported_types}"
            raise ValueError(msg)
        return builder.build(manifest)
