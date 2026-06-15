from pathlib import Path

from typer.testing import CliRunner
from yaml import safe_dump, safe_load

from dataset.cli import app
from dataset.fetch import DatasetFetchPlan, DatasetFetchPlanAction, DownloadHeadResult


class TestDatasetCli:
    """Dataset CLI tests."""

    def test_list_outputs_dataset_manifest_table(self, tmp_path: Path) -> None:
        """List dataset manifests through the dataset CLI."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)
        _write_streams(root)

        result = CliRunner().invoke(app, ["list", "--dataset-dir", str(registry)])

        assert result.exit_code == 0
        lines = result.output.splitlines()
        assert lines[0].split() == ["NAME", "TYPE", "EXISTS", "VERIFIED", "CACHE", "ISSUES", "ROOT"]
        assert lines[1].split(maxsplit=6) == ["euroc_mh_01", "euroc", "true", "true", "false", "-", str(root)]

    def test_list_outputs_dataset_manifest_summaries_as_yaml(self, tmp_path: Path) -> None:
        """List dataset manifests through the dataset CLI as structured YAML."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)
        _write_streams(root)

        result = CliRunner().invoke(app, ["list", "--dataset-dir", str(registry), "--format", "yaml"])

        assert result.exit_code == 0
        output = safe_load(result.output)
        assert output == [
            {
                "name": "euroc_mh_01",
                "type": "euroc",
                "root": str(root),
                "rig": "config/dataset_rig/euroc.yaml",
                "cache": str(root / "cache"),
                "local": {
                    "exists": True,
                    "verified": True,
                    "issues": [],
                    "cache": {
                        "path": str(root / "cache/full"),
                        "exists": False,
                        "verified": False,
                        "issues": [str(root / "cache/full")],
                    },
                },
            }
        ]

    def test_list_table_outputs_dataset_issues(self, tmp_path: Path) -> None:
        """Table output should include compact issue names."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)

        result = CliRunner().invoke(app, ["list", "--dataset-dir", str(registry)])

        assert result.exit_code == 0
        assert "false" in result.output
        assert "root, cam0/data.csv, cam1/data.csv, imu0/data.csv, state_groundtruth_estimate0/data.csv" in (
            result.output
        )

    def test_open_dataset_uses_source_page_url(self, tmp_path: Path, monkeypatch) -> None:
        """Open should resolve the manifest and pass the official page URL to the browser."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)
        opened: list[str] = []
        monkeypatch.setattr("dataset.cli.webbrowser.open", lambda url: opened.append(url) or True)

        result = CliRunner().invoke(app, ["open", "euroc_mh_01", "--dataset-dir", str(registry)])

        assert result.exit_code == 0
        assert opened == ["https://example.test/euroc"]
        assert "https://example.test/euroc" in result.output

    def test_fetch_dataset_uses_cache_manager(self, tmp_path: Path, monkeypatch) -> None:
        """Fetch command should delegate raw and materialized cache preparation."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)

        class FakeCacheManager:
            def __init__(self, **kwargs) -> None:
                self.registry = kwargs["registry"]
                self.progress_callback = kwargs["progress_callback"]

            def plan_dataset(self, manifest, **kwargs):
                assert manifest.name == "euroc_mh_01"
                assert kwargs == {"force_raw": False, "force_cache": False, "materialize_cache": True}
                return DatasetFetchPlan(
                    dataset=manifest.name,
                    dataset_page_url="https://example.test/euroc",
                    open_command=f"just dataset open {manifest.name}",
                    actions=(
                        DatasetFetchPlanAction(
                            name="Download source archive",
                            detail="https://example.test/euroc.zip -> downloads/euroc.zip",
                            download_bytes=100,
                        ),
                        DatasetFetchPlanAction(
                            name="Extract dataset files",
                            detail="MH_01_easy/mav0 -> datasets/euroc_mh_01",
                        ),
                        DatasetFetchPlanAction(
                            name="Build dataset cache",
                            detail=str(root / "cache/full"),
                        ),
                    ),
                )

            def ensure_dataset(self, manifest, **kwargs):
                assert manifest.name == "euroc_mh_01"
                assert kwargs == {"force_raw": False, "force_cache": False, "materialize_cache": True}
                self.progress_callback("Downloading source archive", 0, 100)
                self.progress_callback("Downloading source archive", 50, 100)
                self.progress_callback("Downloading source archive", 100, 100)
                self.progress_callback("Building dataset cache", 0, 1)
                self.progress_callback("Building dataset cache", 1, 1)
                return type(
                    "Result",
                    (),
                    {
                        "dataset": manifest.name,
                        "raw_fetched": True,
                        "cache_built": True,
                    },
                )()

        monkeypatch.setattr("dataset.cli.DatasetCacheManager", FakeCacheManager)

        result = CliRunner().invoke(app, ["fetch", "euroc_mh_01", "--dataset-dir", str(registry)], input="y\n")

        assert result.exit_code == 0
        assert "Dataset: euroc_mh_01" in result.output
        assert "Download: 100.0 B" in result.output
        assert "Dataset link page: https://example.test/euroc" in result.output
        assert "To open dataset link page - cli command: just dataset open euroc_mh_01" in result.output
        assert "1. Download source archive (100.0 B download)" in result.output
        assert "2. Extract dataset files" in result.output
        assert "3. Build dataset cache" in result.output
        assert "Continue? [Y/n]" in result.output
        assert "Downloading source archive: [--------------------------------]   0%" in result.output
        assert "Downloading source archive: [################################] 100%" in result.output
        assert "Building dataset cache..." in result.output
        assert "Building dataset cache: done" in result.output
        assert "euroc_mh_01: raw=fetched cache=built" in result.output

    def test_fetch_head_outputs_download_headers(self, tmp_path: Path, monkeypatch) -> None:
        """Fetch-head should print HEAD status and headers for the dataset download URL."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)
        seen_urls: list[str] = []

        def fake_fetch_download_headers(download_url: str) -> list[DownloadHeadResult]:
            seen_urls.append(download_url)
            return [
                DownloadHeadResult(
                    url=download_url,
                    status=500,
                    reason="Internal Server Error",
                    headers={"Content-Length": "494241"},
                ),
                DownloadHeadResult(
                    url="https://example.test/euroc-content.zip",
                    status=200,
                    reason="OK",
                    headers={"Content-Length": "42"},
                ),
            ]

        monkeypatch.setattr("dataset.cli.fetch_download_headers", fake_fetch_download_headers)

        result = CliRunner().invoke(app, ["fetch-head", "euroc_mh_01", "--dataset-dir", str(registry)])

        assert result.exit_code == 0
        assert seen_urls == ["https://example.test/euroc.zip"]
        assert "URL: https://example.test/euroc.zip" in result.output
        assert "Status: 500 Internal Server Error" in result.output
        assert "Content-Length: 494241" in result.output
        assert "URL: https://example.test/euroc-content.zip" in result.output
        assert "Status: 200 OK" in result.output
        assert "Content-Length: 42" in result.output

    def test_cache_clear_removes_all_manifest_cache_roots(self, tmp_path: Path) -> None:
        """Cache clear should delete cache roots while preserving raw streams."""
        registry = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry, root)
        _write_streams(root)
        cache_root = root / "cache"
        cache_file = cache_root / "full/data-00000-of-00001.arrow"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("cache", encoding="utf-8")

        result = CliRunner().invoke(app, ["cache", "clear", "--dataset-dir", str(registry)])

        assert result.exit_code == 0
        assert "euroc_mh_01: cleared" in result.output
        assert not cache_root.exists()
        assert (root / "cam0/data.csv").exists()

    def test_cache_clear_can_target_one_dataset(self, tmp_path: Path) -> None:
        """Cache clear should support a single dataset selector."""
        registry = tmp_path / "registry"
        first_root = tmp_path / "euroc_mh_01"
        second_root = tmp_path / "euroc_mh_02"
        _write_manifest(registry, first_root)
        _write_manifest(registry, second_root, name="euroc_mh_02", mkdir=False)
        for cache_root in [first_root / "cache", second_root / "cache"]:
            cache_file = cache_root / "full/data-00000-of-00001.arrow"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_text("cache", encoding="utf-8")

        result = CliRunner().invoke(app, ["cache", "clear", "euroc_mh_01", "--dataset-dir", str(registry)])

        assert result.exit_code == 0
        assert not (first_root / "cache").exists()
        assert (second_root / "cache").exists()


def _write_manifest(registry: Path, root: Path, *, name: str = "euroc_mh_01", mkdir: bool = True) -> None:
    if mkdir:
        registry.mkdir()
    manifest = {
        "name": name,
        "type": "euroc",
        "root": str(root),
        "rig": "config/dataset_rig/euroc.yaml",
        "cache": str(root / "cache"),
        "source": {
            "page_url": "https://example.test/euroc",
            "download_url": "https://example.test/euroc.zip",
            "filename": "euroc.zip",
            "archive": {
                "format": "zip",
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
    (registry / f"{name}.yaml").write_text(safe_dump(manifest), encoding="utf-8")


def _write_streams(root: Path) -> None:
    for stream_path in [
        root / "cam0/data.csv",
        root / "cam1/data.csv",
        root / "imu0/data.csv",
        root / "state_groundtruth_estimate0/data.csv",
    ]:
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_text("", encoding="utf-8")
