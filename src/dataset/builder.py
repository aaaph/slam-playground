from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dataset.manifest import DatasetManifest
    from datasets import Dataset


class DatasetBuilder(Protocol):
    """Build or load a HuggingFace dataset for a resolved manifest."""

    def build(self, manifest: DatasetManifest) -> Dataset:
        """Build or load a HuggingFace dataset."""


class DatasetAdapter(Protocol):
    """Materialize canonical HuggingFace datasets from raw stream collections."""

    def materialize(self, streams: RawStreamBundle) -> Dataset:
        """Map raw streams to a canonical HuggingFace dataset."""


class StreamLoader(Protocol):
    """Load raw stream collections for a dataset manifest."""

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Load raw stream files into a collection."""


@dataclass(frozen=True)
class RawStream:
    """A resolved raw stream file and its loaded dataframe."""

    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class RawStreamBundle:
    """Raw dataset streams loaded from a manifest."""

    manifest: DatasetManifest
    root: Path
    cache: Path
    streams: Mapping[str, RawStream]

    def require(self, name: str) -> RawStream:
        """Return a required stream or raise a clear error."""
        stream = self.streams.get(name)
        if stream is None:
            msg = f"Dataset manifest '{self.manifest.name}' must define streams.{name}"
            raise ValueError(msg)
        return stream


class RawStreamLoader:
    """Load raw stream CSV files described by a dataset manifest."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        """Create a raw stream loader."""
        self.repo_root = (repo_root or Path.cwd()).resolve()

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Load manifest streams into raw dataframes without joining them."""
        root = self.resolve_path(manifest.root)
        cache = self.resolve_path(manifest.cache) if manifest.cache is not None else root / "cache"
        streams = {
            name: RawStream(path=path, frame=pd.read_csv(path))
            for name, path in self.resolve_stream_paths(manifest, root).items()
        }
        return RawStreamBundle(manifest=manifest, root=root, cache=cache, streams=streams)

    def resolve_stream_paths(self, manifest: DatasetManifest, root: Path | None = None) -> dict[str, Path]:
        """Resolve all stream paths without opening them."""
        dataset_root = root or self.resolve_path(manifest.root)
        raw_streams = manifest.streams.model_dump(exclude_none=True)
        return {name: self.resolve_stream_path(dataset_root, Path(path)) for name, path in raw_streams.items()}

    def resolve_path(self, path: Path) -> Path:
        """Resolve a manifest-level path against the repo root."""
        return path if path.is_absolute() else self.repo_root / path

    def resolve_stream_path(self, dataset_root: Path, path: Path) -> Path:
        """Resolve a stream path against the dataset root."""
        return path if path.is_absolute() else dataset_root / path
