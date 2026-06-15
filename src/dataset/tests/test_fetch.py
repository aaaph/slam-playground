from __future__ import annotations

from email.message import Message
from io import BytesIO
from typing import TYPE_CHECKING, cast
from urllib.error import HTTPError
from zipfile import ZipFile

import pytest

from dataset.fetch import DatasetCacheManager, DatasetFetchError, EurocFetchStrategy, fetch_download_headers
from dataset.manifest import DatasetManifest
from dataset.registry import DatasetRegistry

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self
    from urllib.request import Request

    from datasets import Dataset


class FakeBuilder:
    """Dataset cache builder test double."""

    def __init__(self, repo_root: Path) -> None:
        """Create a fake cache builder."""
        self.repo_root = repo_root
        self.seen_manifest: DatasetManifest | None = None

    def build(self, manifest: DatasetManifest) -> Dataset:
        """Write the minimal files DatasetRegistry uses to verify a cache."""
        self.seen_manifest = manifest
        cache_root = manifest.cache or manifest.root / "cache"
        cache_root = self.repo_root / cache_root if not cache_root.is_absolute() else cache_root
        cache_path = cache_root / "full"
        cache_path.mkdir(parents=True)
        (cache_path / "dataset_info.json").write_text("{}", encoding="utf-8")
        (cache_path / "state.json").write_text("{}", encoding="utf-8")
        return cast("Dataset", object())


def test_euroc_fetch_downloads_extracts_and_builds_cache(tmp_path: Path) -> None:
    """EuRoC fetch should download the archive, extract the configured root, and build cache."""
    archive_path = tmp_path / "source.zip"
    _write_euroc_archive(archive_path)
    manifest = _euroc_manifest(archive_path)
    registry = DatasetRegistry(repo_root=tmp_path)
    builder = FakeBuilder(tmp_path)

    result = DatasetCacheManager(
        repo_root=tmp_path,
        registry=registry,
        downloads_dir=tmp_path / "downloads",
        builders={"euroc": builder},
    ).ensure_dataset(manifest)

    assert result.raw_fetched is True
    assert result.raw_ready is True
    assert result.cache_built is True
    assert result.cache_ready is True
    assert builder.seen_manifest == manifest
    assert (tmp_path / "datasets/euroc_mh_01/cam0/data.csv").exists()
    assert (tmp_path / "datasets/euroc_mh_01/cam0/data/1.png").exists()
    assert (tmp_path / "downloads/euroc.zip").exists()


def test_euroc_fetch_extracts_nested_sequence_zip(tmp_path: Path) -> None:
    """EuRoC fetch should handle archives that contain per-sequence zip files."""
    archive_path = tmp_path / "outer.zip"
    _write_nested_euroc_archive(archive_path, tmp_path / "inner.zip")
    manifest = _euroc_manifest(archive_path)
    registry = DatasetRegistry(repo_root=tmp_path)

    result = DatasetCacheManager(
        repo_root=tmp_path,
        registry=registry,
        downloads_dir=tmp_path / "downloads",
    ).ensure_dataset(manifest, materialize_cache=False)

    assert result.raw_fetched is True
    assert result.raw_ready is True
    assert result.cache_built is False
    assert (tmp_path / "datasets/euroc_mh_01/cam1/data/1.png").exists()
    assert (tmp_path / "downloads/MH_01_easy.zip").exists()


def test_cache_manager_can_require_existing_raw_files_without_fetching(tmp_path: Path) -> None:
    """Pipeline pre-cache should be able to fail locally without downloading large archives."""
    archive_path = tmp_path / "source.zip"
    _write_euroc_archive(archive_path)
    manifest = _euroc_manifest(archive_path)

    with pytest.raises(DatasetFetchError, match="raw files are incomplete"):
        DatasetCacheManager(repo_root=tmp_path).ensure_dataset(manifest, fetch_raw=False)


def test_euroc_download_falls_back_to_dspace_content_endpoint(tmp_path: Path, monkeypatch) -> None:
    """ETH frontend bitstream download failures should fall back to the DSpace content URL."""
    manifest = _euroc_manifest_from_url(
        "https://www.research-collection.ethz.ch/bitstreams/test-bitstream/download"
    )
    calls: list[str] = []
    archive_bytes = _euroc_archive_bytes()

    def fake_urlopen(request: Request):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 500, "Internal Server Error", Message(), None)
        return _BytesResponse(archive_bytes)

    monkeypatch.setattr("dataset.fetch.urlopen", fake_urlopen)

    archive_path = EurocFetchStrategy(repo_root=tmp_path, downloads_dir=tmp_path / "downloads")._download_archive(  # noqa: SLF001
        manifest
    )

    assert archive_path.exists()
    assert calls == [
        "https://www.research-collection.ethz.ch/bitstreams/test-bitstream/download",
        "https://www.research-collection.ethz.ch/server/api/core/bitstreams/test-bitstream/content",
    ]


def test_euroc_download_reports_progress(tmp_path: Path, monkeypatch) -> None:
    """Archive downloads should report downloaded and total byte counts."""
    manifest = _euroc_manifest_from_url("https://example.test/euroc.zip")
    archive_bytes = _euroc_archive_bytes()
    progress: list[tuple[str, int, int | None]] = []

    def fake_urlopen(_request: Request):
        return _BytesResponse(archive_bytes, headers={"Content-Length": str(len(archive_bytes))})

    monkeypatch.setattr("dataset.fetch.urlopen", fake_urlopen)

    strategy = EurocFetchStrategy(
        repo_root=tmp_path,
        downloads_dir=tmp_path / "downloads",
        progress_callback=lambda label, downloaded, total: progress.append((label, downloaded, total)),
    )
    strategy._download_archive(manifest)  # noqa: SLF001

    assert progress
    assert progress[0] == ("Downloading source archive", 0, len(archive_bytes))
    assert progress[-1] == ("Downloading source archive", len(archive_bytes), len(archive_bytes))


def test_euroc_download_probes_range_total_when_content_length_missing(tmp_path: Path, monkeypatch) -> None:
    """Downloads should still report percentage totals when DSpace omits Content-Length."""
    manifest = _euroc_manifest_from_url("https://example.test/euroc.zip")
    archive_bytes = _euroc_archive_bytes()
    progress: list[tuple[str, int, int | None]] = []
    calls: list[tuple[str, str | None]] = []

    def fake_urlopen(request: Request):
        calls.append((request.get_method(), request.get_header("Range")))
        if request.get_method() == "HEAD":
            return _HeadResponse(status=200, reason="OK", headers={})
        if request.get_header("Range") == "bytes=0-0":
            return _HeadResponse(
                status=206,
                reason="Partial Content",
                headers={"Content-Range": f"bytes 0-0/{len(archive_bytes)}"},
            )
        return _BytesResponse(archive_bytes)

    monkeypatch.setattr("dataset.fetch.urlopen", fake_urlopen)

    strategy = EurocFetchStrategy(
        repo_root=tmp_path,
        downloads_dir=tmp_path / "downloads",
        progress_callback=lambda label, downloaded, total: progress.append((label, downloaded, total)),
    )
    strategy._download_archive(manifest)  # noqa: SLF001

    assert calls == [("GET", None), ("HEAD", None), ("GET", "bytes=0-0")]
    assert progress
    assert progress[0] == ("Downloading source archive", 0, len(archive_bytes))
    assert progress[-1] == ("Downloading source archive", len(archive_bytes), len(archive_bytes))


def test_euroc_download_uses_manifest_size_when_content_length_missing(tmp_path: Path, monkeypatch) -> None:
    """Configured manifest sizes should drive progress without extra size probes."""
    archive_bytes = _euroc_archive_bytes()
    configured_size = len(archive_bytes) + 10
    manifest = _euroc_manifest_from_url("https://example.test/euroc.zip", size_bytes=configured_size)
    progress: list[tuple[str, int, int | None]] = []
    calls: list[str] = []

    def fake_urlopen(request: Request):
        calls.append(request.get_method())
        return _BytesResponse(archive_bytes)

    monkeypatch.setattr("dataset.fetch.urlopen", fake_urlopen)

    strategy = EurocFetchStrategy(
        repo_root=tmp_path,
        downloads_dir=tmp_path / "downloads",
        progress_callback=lambda label, downloaded, total: progress.append((label, downloaded, total)),
    )
    strategy._download_archive(manifest)  # noqa: SLF001

    assert calls == ["GET"]
    assert progress
    assert progress[0] == ("Downloading source archive", 0, configured_size)
    assert progress[-1] == ("Downloading source archive", len(archive_bytes), configured_size)


def test_fetch_download_headers_reports_error_and_fallback_headers(monkeypatch) -> None:
    """HEAD metadata should include HTTP error headers and DSpace fallback candidates."""
    calls: list[tuple[str, str]] = []

    def fake_urlopen(request: Request):
        calls.append((request.full_url, request.get_method()))
        if len(calls) == 1:
            headers = Message()
            headers["Content-Length"] = "494241"
            raise HTTPError(request.full_url, 500, "Internal Server Error", headers, None)
        return _HeadResponse(status=200, reason="OK", headers={"Content-Length": "42"})

    monkeypatch.setattr("dataset.fetch.urlopen", fake_urlopen)

    results = fetch_download_headers("https://www.research-collection.ethz.ch/bitstreams/test-bitstream/download")

    assert calls == [
        ("https://www.research-collection.ethz.ch/bitstreams/test-bitstream/download", "HEAD"),
        (
            "https://www.research-collection.ethz.ch/server/api/core/bitstreams/test-bitstream/content",
            "HEAD",
        ),
    ]
    assert results[0].status == 500
    assert results[0].headers["Content-Length"] == "494241"
    assert results[1].status == 200
    assert results[1].headers["Content-Length"] == "42"


def test_euroc_download_wraps_http_errors(tmp_path: Path, monkeypatch) -> None:
    """Download failures should surface as DatasetFetchError instead of rich tracebacks."""
    manifest = _euroc_manifest_from_url("https://example.test/euroc.zip")

    def fake_urlopen(request: Request):
        raise HTTPError(request.full_url, 500, "Internal Server Error", Message(), None)

    monkeypatch.setattr("dataset.fetch.urlopen", fake_urlopen)

    with pytest.raises(DatasetFetchError, match="HTTP 500 Internal Server Error"):
        EurocFetchStrategy(repo_root=tmp_path, downloads_dir=tmp_path / "downloads")._download_archive(manifest)  # noqa: SLF001


def _euroc_manifest(archive_path: Path) -> DatasetManifest:
    return _euroc_manifest_from_url(archive_path.as_uri())


def _euroc_manifest_from_url(download_url: str, *, size_bytes: int | None = None) -> DatasetManifest:
    raw_manifest = {
        "name": "euroc_mh_01",
        "type": "euroc",
        "root": "datasets/euroc_mh_01",
        "rig": "config/dataset_rig/euroc.yaml",
        "cache": "datasets/euroc_mh_01/cache",
        "source": {
            "page_url": "https://example.test/euroc",
            "download_url": download_url,
            "filename": "euroc.zip",
            "size_bytes": size_bytes,
            "archive": {
                "format": "zip",
                "nested": "MH_01_easy.zip",
                "root": "MH_01_easy/mav0",
            },
        },
        "streams": {
            "cam0": "cam0/data.csv",
            "cam1": "cam1/data.csv",
            "imu0": "imu0/data.csv",
            "ground_truth": "state_groundtruth_estimate0/data.csv",
        },
    }
    return DatasetManifest.model_validate(raw_manifest)


class _BytesResponse:
    def __init__(self, data: bytes, *, headers: dict[str, str] | None = None) -> None:
        self._buffer = BytesIO(data)
        self.headers = headers or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self._buffer.close()

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)


class _HeadResponse:
    def __init__(self, *, status: int, reason: str, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers or {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status


def _euroc_archive_bytes() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        prefix = "bundle/MH_01_easy/mav0"
        zip_file.writestr(f"{prefix}/cam0/data.csv", "#timestamp [ns],filename\n1,1.png\n")
    return buffer.getvalue()


def _write_euroc_archive(path: Path) -> None:
    with ZipFile(path, "w") as zip_file:
        prefix = "bundle/MH_01_easy/mav0"
        zip_file.writestr(f"{prefix}/cam0/data.csv", "#timestamp [ns],filename\n1,1.png\n")
        zip_file.writestr(f"{prefix}/cam1/data.csv", "#timestamp [ns],filename\n1,1.png\n")
        zip_file.writestr(f"{prefix}/imu0/data.csv", "#timestamp [ns],w_RS_S_x [rad s^-1]\n1,0\n")
        zip_file.writestr(f"{prefix}/state_groundtruth_estimate0/data.csv", "#timestamp\n1\n")
        zip_file.writestr(f"{prefix}/cam0/data/1.png", b"")
        zip_file.writestr(f"{prefix}/cam1/data/1.png", b"")


def _write_nested_euroc_archive(path: Path, inner_path: Path) -> None:
    _write_euroc_archive(inner_path)
    with ZipFile(path, "w") as zip_file:
        zip_file.write(inner_path, "archives/MH_01_easy.zip")
