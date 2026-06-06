from pathlib import Path
from typing import Any

from pydantic import BaseModel
from yaml import safe_load

from dataset.manifest import DatasetManifest, DatasetRigConfig, ResolvedDatasetManifest


class DatasetLocalStatus(BaseModel):
    """Local availability status for a dataset manifest."""

    exists: bool
    verified: bool
    issues: list[Path]


class DatasetRegistry:
    """Registry for supported dataset manifests and local availability."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        dataset_dir: Path | None = None,
    ) -> None:
        """Create a dataset registry."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.dataset_dir = self._resolve_path(dataset_dir or Path("datasets"))

    def list(self) -> list[DatasetManifest]:
        """List dataset manifests available in the dataset registry directory."""
        manifests = []
        for manifest_path in sorted(self.dataset_dir.glob("*.yaml")):
            raw_manifest = self._load_yaml(manifest_path)
            raw_manifest.setdefault("name", manifest_path.stem)
            manifests.append(DatasetManifest.model_validate(raw_manifest))
        return manifests

    def find(self, selector: str) -> DatasetManifest:
        """Find a dataset manifest by name or unique dataset type."""
        manifests = self.list()
        by_name = [manifest for manifest in manifests if manifest.name == selector]
        if by_name:
            return by_name[0]

        by_type = [manifest for manifest in manifests if manifest.type == selector]
        if len(by_type) == 1:
            return by_type[0]
        if len(by_type) > 1:
            matches = ", ".join(manifest.name for manifest in by_type)
            msg = f"Dataset selector '{selector}' is ambiguous. Matching datasets: {matches}"
            raise ValueError(msg)

        msg = f"Unknown dataset '{selector}'"
        raise FileNotFoundError(msg)

    def resolve(self, selector: str) -> ResolvedDatasetManifest:
        """Resolve a dataset selector into its manifest and sensor rig."""
        dataset = self.find(selector)
        rig = self.load_rig(dataset.rig)
        return ResolvedDatasetManifest(dataset=dataset, rig=rig)

    def local_status(self, manifest: DatasetManifest) -> DatasetLocalStatus:
        """Inspect whether a dataset manifest is available on local disk."""
        root = self._resolve_path(manifest.root)
        missing_streams = [
            path for path in self._resolve_stream_paths(manifest, root).values() if not path.exists()
        ]
        exists = root.exists()
        issues = ([] if exists else [root]) + missing_streams
        return DatasetLocalStatus(
            exists=exists,
            verified=not issues,
            issues=issues,
        )

    def load_rig(self, path: Path) -> DatasetRigConfig:
        """Load normalized sensor rig config."""
        rig_path = self._resolve_path(path)
        return DatasetRigConfig.model_validate(self._load_yaml(rig_path))

    def resolve_path(self, path: Path) -> Path:
        """Resolve a manifest-level path against the repository root."""
        return self._resolve_path(path)

    def _load_yaml(self, config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = safe_load(f) or {}
        if not isinstance(raw_config, dict):
            msg = f"Dataset config must be a mapping: {config_path}"
            raise TypeError(msg)
        return raw_config

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path

    @staticmethod
    def _resolve_stream_path(root: Path, path: Path) -> Path:
        return path if path.is_absolute() else root / path

    def _resolve_stream_paths(self, manifest: DatasetManifest, root: Path) -> dict[str, Path]:
        raw_streams = manifest.streams.model_dump(exclude_none=True)
        return {name: self._resolve_stream_path(root, Path(path)) for name, path in raw_streams.items()}
