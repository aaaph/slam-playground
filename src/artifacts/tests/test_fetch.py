from __future__ import annotations

import hashlib
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING, Self

import pytest
from yaml import safe_dump

from artifacts.fetch import ArtifactFetchError, ArtifactFetchManager, OrbSlam3VocabularyFetchStrategy
from artifacts.registry import ArtifactRegistry

if TYPE_CHECKING:
    from pathlib import Path
    from urllib.request import Request


def test_orb_slam3_strategy_downloads_extracts_and_converts_vocabulary(tmp_path: Path, monkeypatch) -> None:
    """The ORB-SLAM3 strategy should download the text vocabulary and produce the configured binary file."""
    archive_bytes = _tar_gz_bytes("ORBvoc.txt", b"text-vocabulary")
    registry_dir = tmp_path / "registry"
    root = tmp_path / "vocabulary"
    _write_manifest(registry_dir, root, archive_bytes=archive_bytes, binary_bytes=b"dbow3")
    registry = ArtifactRegistry(repo_root=tmp_path, artifact_dir=registry_dir)
    manifest = registry.find("orb_vocabulary")
    calls: list[str] = []
    progress: list[tuple[str, int, int | None]] = []

    def fake_urlopen(request: Request):
        calls.append(request.full_url)
        return _BytesResponse(archive_bytes, headers={"Content-Length": str(len(archive_bytes))})

    def fake_convert(text_path: Path, output_path: Path) -> None:
        assert text_path.read_bytes() == b"text-vocabulary"
        output_path.write_bytes(b"dbow3")

    monkeypatch.setattr("artifacts.fetch.urlopen", fake_urlopen)
    monkeypatch.setattr("artifacts.fetch._convert_txt_to_dbow3", fake_convert)

    result = ArtifactFetchManager(
        registry=registry,
        downloads_dir=tmp_path / "downloads",
        progress_callback=lambda label, done, total: progress.append((label, done, total)),
    ).ensure_artifact(manifest)

    assert result.fetched is True
    assert result.extracted is True
    assert result.converted is True
    assert result.ready is True
    assert calls == ["https://example.test/ORBvoc.txt.tar.gz"]
    assert (tmp_path / "downloads/orb_vocabulary/ORBvoc.txt.tar.gz").read_bytes() == archive_bytes
    assert (tmp_path / "downloads/orb_vocabulary/ORBvoc.txt").read_bytes() == b"text-vocabulary"
    assert (root / "ORBvoc.dbow3").read_bytes() == b"dbow3"
    assert progress[0] == ("Downloading source artifact", 0, len(archive_bytes))
    assert progress[-1] == ("Downloading source artifact", len(archive_bytes), len(archive_bytes))


def test_orb_slam3_plan_skips_when_binary_vocabulary_is_verified(tmp_path: Path) -> None:
    """A verified final binary vocabulary should make fetch a no-op."""
    registry_dir = tmp_path / "registry"
    root = tmp_path / "vocabulary"
    _write_manifest(registry_dir, root, archive_bytes=b"unused", binary_bytes=b"dbow3")
    root.mkdir()
    (root / "ORBvoc.dbow3").write_bytes(b"dbow3")
    registry = ArtifactRegistry(repo_root=tmp_path, artifact_dir=registry_dir)
    manifest = registry.find("orb_vocabulary")

    plan = ArtifactFetchManager(registry=registry, downloads_dir=tmp_path / "downloads").plan_artifact(manifest)
    result = ArtifactFetchManager(
        registry=registry,
        downloads_dir=tmp_path / "downloads",
    ).ensure_artifact(manifest)

    assert plan.actions == ()
    assert result.fetched is False
    assert result.extracted is False
    assert result.converted is False
    assert result.ready is True


def test_orb_slam3_strategy_rejects_bad_source_checksum(tmp_path: Path, monkeypatch) -> None:
    """Configured source checksums should be enforced before conversion."""
    archive_bytes = _tar_gz_bytes("ORBvoc.txt", b"text-vocabulary")
    registry_dir = tmp_path / "registry"
    root = tmp_path / "vocabulary"
    _write_manifest(registry_dir, root, archive_bytes=b"expected", binary_bytes=b"dbow3")
    registry = ArtifactRegistry(repo_root=tmp_path, artifact_dir=registry_dir)
    manifest = registry.find("orb_vocabulary")

    def fake_urlopen(_request: Request):
        return _BytesResponse(archive_bytes)

    monkeypatch.setattr("artifacts.fetch.urlopen", fake_urlopen)

    with pytest.raises(ArtifactFetchError, match="SHA-256 mismatch"):
        OrbSlam3VocabularyFetchStrategy(registry=registry, downloads_dir=tmp_path / "downloads").ensure(manifest)


def _write_manifest(registry: Path, root: Path, *, archive_bytes: bytes, binary_bytes: bytes) -> None:
    registry.mkdir()
    manifest = {
        "name": "orb_vocabulary",
        "type": "vocabulary",
        "root": str(root),
        "files": {
            "dbow3": {
                "path": "ORBvoc.dbow3",
                "size_bytes": len(binary_bytes),
                "sha256": _sha256(binary_bytes),
            },
        },
        "source": {
            "download_url": "https://example.test/ORBvoc.txt.tar.gz",
            "filename": "ORBvoc.txt.tar.gz",
            "size_bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
        },
        "metadata": {
            "fetch_strategy": "orb_slam3_vocabulary",
            "archive_member": "ORBvoc.txt",
            "output_file_id": "dbow3",
        },
    }
    (registry / "orb_vocabulary.yaml").write_text(safe_dump(manifest), encoding="utf-8")


def _tar_gz_bytes(member_name: str, data: bytes) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _BytesResponse:
    def __init__(self, data: bytes, *, headers: dict[str, str] | None = None) -> None:
        self._buffer = BytesIO(data)
        self.headers = headers or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self._buffer.close()

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)
