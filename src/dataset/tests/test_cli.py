from pathlib import Path

from typer.testing import CliRunner
from yaml import safe_dump, safe_load

from dataset.cli import app


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
        assert lines[0].split() == ["NAME", "TYPE", "EXISTS", "VERIFIED", "ISSUES", "ROOT"]
        assert lines[1].split(maxsplit=5) == ["euroc_mh_01", "euroc", "true", "true", "-", str(root)]

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


def _write_manifest(registry: Path, root: Path) -> None:
    registry.mkdir()
    manifest = {
        "name": "euroc_mh_01",
        "type": "euroc",
        "root": str(root),
        "rig": "config/dataset_rig/euroc.yaml",
        "cache": str(root / "cache"),
        "streams": {
            "cam0": "cam0/data.csv",
            "cam1": "cam1/data.csv",
            "imu0": "imu0/data.csv",
            "ground_truth": "state_groundtruth_estimate0/data.csv",
        },
    }
    (registry / "euroc_mh_01.yaml").write_text(safe_dump(manifest), encoding="utf-8")


def _write_streams(root: Path) -> None:
    for stream_path in [
        root / "cam0/data.csv",
        root / "cam1/data.csv",
        root / "imu0/data.csv",
        root / "state_groundtruth_estimate0/data.csv",
    ]:
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_text("", encoding="utf-8")
