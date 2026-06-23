from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from yaml import safe_load

from artifacts.manifest import ArtifactFileConfig, ArtifactManifest

type ArtifactManifestList = list[ArtifactManifest]
type PathList = list[Path]


class ArtifactFileLocalStatus(BaseModel):
    """Local availability status for one artifact file."""

    path: Path
    exists: bool
    verified: bool
    optional: bool
    size_bytes: int | None = None
    actual_size_bytes: int | None = None
    sha256: str | None = None
    actual_sha256: str | None = None
    issue: str | None = None


class ArtifactLocalStatus(BaseModel):
    """Local availability status for an artifact manifest."""

    exists: bool
    verified: bool
    issues: list[Path]
    files: dict[str, ArtifactFileLocalStatus]


class ArtifactRegistry:
    """Registry for local research artifact manifests and payload availability."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        artifact_dir: Path | None = None,
    ) -> None:
        """Create an artifact registry."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.artifact_dir = self._resolve_path(artifact_dir or Path("artifacts"))

    def list(self) -> ArtifactManifestList:
        """List artifact manifests available in the artifact registry directory."""
        manifests: ArtifactManifestList = []
        for manifest_path in sorted([*self.artifact_dir.glob("*.yaml"), *self.artifact_dir.glob("*.yml")]):
            raw_manifest = self._load_yaml(manifest_path)
            raw_manifest.setdefault("name", manifest_path.stem)
            manifests.append(ArtifactManifest.model_validate(raw_manifest))
        return manifests

    def find(self, selector: str) -> ArtifactManifest:
        """Find an artifact manifest by name or unique artifact type."""
        manifests = self.list()
        by_name = [manifest for manifest in manifests if manifest.name == selector]
        if by_name:
            return by_name[0]

        by_type = [manifest for manifest in manifests if manifest.type == selector]
        if len(by_type) == 1:
            return by_type[0]
        if len(by_type) > 1:
            matches = ", ".join(manifest.name for manifest in by_type)
            msg = f"Artifact selector '{selector}' is ambiguous. Matching artifacts: {matches}"
            raise ValueError(msg)

        msg = f"Unknown artifact '{selector}'"
        raise FileNotFoundError(msg)

    def local_status(self, manifest: ArtifactManifest, *, verify_hashes: bool = False) -> ArtifactLocalStatus:
        """Inspect whether an artifact payload exists and optionally validate checksums."""
        root = self.resolve_path(manifest.root)
        issues: PathList = [] if root.exists() else [root]
        file_statuses = {
            file_id: self._file_status(
                file_config,
                self.resolve_file_path(manifest, file_id),
                verify_hashes=verify_hashes,
            )
            for file_id, file_config in manifest.files.items()
        }
        issues.extend(status.path for status in file_statuses.values() if status.issue is not None)
        return ArtifactLocalStatus(
            exists=root.exists(),
            verified=not issues,
            issues=issues,
            files=file_statuses,
        )

    def resolve_file_path(self, manifest: ArtifactManifest, file_id: str) -> Path:
        """Resolve one artifact file path against the artifact root."""
        file_config = manifest.files[file_id]
        root = self.resolve_path(manifest.root)
        path = file_config.path
        return path if path.is_absolute() else root / path

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
            msg = f"Artifact config must be a mapping: {config_path}"
            raise TypeError(msg)
        return raw_config

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path

    def _file_status(
        self,
        file_config: ArtifactFileConfig,
        path: Path,
        *,
        verify_hashes: bool,
    ) -> ArtifactFileLocalStatus:
        exists = path.exists()
        issue: str | None = None
        actual_size_bytes: int | None = None
        actual_sha256: str | None = None

        if not exists:
            if not file_config.optional:
                issue = "missing"
        elif not path.is_file():
            issue = "not a file"
        else:
            actual_size_bytes = path.stat().st_size
            if file_config.size_bytes is not None and actual_size_bytes != file_config.size_bytes:
                issue = "size mismatch"
            if issue is None and verify_hashes and file_config.sha256 is not None:
                actual_sha256 = _sha256_file(path)
                if actual_sha256 != file_config.sha256:
                    issue = "sha256 mismatch"

        return ArtifactFileLocalStatus(
            path=path,
            exists=exists,
            verified=issue is None,
            optional=file_config.optional,
            size_bytes=file_config.size_bytes,
            actual_size_bytes=actual_size_bytes,
            sha256=file_config.sha256,
            actual_sha256=actual_sha256,
            issue=issue,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
