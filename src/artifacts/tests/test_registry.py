from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from yaml import safe_dump

from artifacts.registry import ArtifactRegistry


class TestArtifactRegistry:
    """Tests for artifact registry lookup and local status."""

    def test_list_artifacts_returns_manifest_models(self, tmp_path: Path) -> None:
        """List artifact manifests from the artifact registry directory."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry_dir, root)

        artifacts = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir).list()

        artifacts_by_name = {artifact.name: artifact for artifact in artifacts}
        assert "orb_vocabulary" in artifacts_by_name
        assert artifacts_by_name["orb_vocabulary"].type == "vocabulary"
        assert artifacts_by_name["orb_vocabulary"].root == root
        assert artifacts_by_name["orb_vocabulary"].files["dbow3"].path == Path("ORBvoc.dbow3")

    def test_find_can_select_artifact_type_when_unique(self, tmp_path: Path) -> None:
        """Artifact selectors can target a unique artifact type."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry_dir, root)

        artifact = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir).find("vocabulary")

        assert artifact.name == "orb_vocabulary"

    def test_find_ambiguous_artifact_type_raises(self, tmp_path: Path) -> None:
        """Artifact type selectors should be unique."""
        registry_dir = tmp_path / "registry"
        _write_manifest(registry_dir, tmp_path / "first", name="first_vocabulary")
        _write_manifest(registry_dir, tmp_path / "second", name="second_vocabulary")

        registry = ArtifactRegistry(repo_root=tmp_path, artifact_dir=registry_dir)

        with pytest.raises(ValueError, match="ambiguous"):
            registry.find("vocabulary")

    def test_missing_artifact_manifest_raises(self, tmp_path: Path) -> None:
        """Unknown artifact names should fail before path resolution."""
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        registry = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir)

        with pytest.raises(FileNotFoundError, match="missing_artifact"):
            registry.find("missing_artifact")

    def test_local_status_marks_existing_payload_verified(self, tmp_path: Path) -> None:
        """An artifact is available locally when root and all required files exist."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry_dir, root)
        _write_file(root / "ORBvoc.dbow3", b"dbow3")
        registry = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir)
        manifest = registry.find("orb_vocabulary")

        status = registry.local_status(manifest)

        assert status.exists is True
        assert status.verified is True
        assert status.issues == []
        assert status.files["dbow3"].exists is True

    def test_local_status_lists_missing_required_files(self, tmp_path: Path) -> None:
        """Missing required files should make an artifact incomplete."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry_dir, root)
        registry = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir)
        manifest = registry.find("orb_vocabulary")

        status = registry.local_status(manifest)

        assert status.exists is False
        assert status.verified is False
        assert status.issues == [root, root / "ORBvoc.dbow3"]
        assert status.files["dbow3"].issue == "missing"

    def test_optional_missing_files_do_not_fail_verification(self, tmp_path: Path) -> None:
        """Optional files can be absent without making the artifact incomplete."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(
            registry_dir,
            root,
            files={"dbow3": {"path": "ORBvoc.dbow3", "optional": True}},
        )
        root.mkdir()
        registry = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir)
        manifest = registry.find("orb_vocabulary")

        status = registry.local_status(manifest)

        assert status.verified is True
        assert status.issues == []
        assert status.files["dbow3"].exists is False
        assert status.files["dbow3"].verified is True

    def test_hash_verification_is_explicit(self, tmp_path: Path) -> None:
        """Checksum validation should run only when requested."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(
            registry_dir,
            root,
            files={"dbow3": {"path": "ORBvoc.dbow3", "sha256": _sha256(b"expected")}},
        )
        _write_file(root / "ORBvoc.dbow3", b"actual")
        registry = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir)
        manifest = registry.find("orb_vocabulary")

        unchecked_status = registry.local_status(manifest)
        checked_status = registry.local_status(manifest, verify_hashes=True)

        assert unchecked_status.verified is True
        assert checked_status.verified is False
        assert checked_status.files["dbow3"].issue == "sha256 mismatch"

    def test_resolve_file_path_returns_absolute_payload_path(self, tmp_path: Path) -> None:
        """Artifact file paths should resolve relative to the manifest root."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "vocabulary"
        _write_manifest(registry_dir, root)
        registry = ArtifactRegistry(repo_root=Path.cwd(), artifact_dir=registry_dir)
        manifest = registry.find("orb_vocabulary")

        assert registry.resolve_file_path(manifest, "dbow3") == root / "ORBvoc.dbow3"


def _write_manifest(
    registry: Path,
    root: Path,
    *,
    name: str = "orb_vocabulary",
    files: dict[str, dict[str, object]] | None = None,
) -> None:
    registry.mkdir(exist_ok=True)
    manifest = {
        "name": name,
        "type": "vocabulary",
        "root": str(root),
        "description": "ORB vocabulary fixture.",
        "tags": ["orb", "dbow3"],
        "files": files or {"dbow3": {"path": "ORBvoc.dbow3"}},
    }
    (registry / f"{name}.yaml").write_text(safe_dump(manifest), encoding="utf-8")


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
