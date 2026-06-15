from pathlib import Path

import pytest
from yaml import safe_dump

from dataset.registry import DatasetRegistry


class TestDatasetRegistry:
    """Tests for dataset registry lookup, rig loading, and local status."""

    def test_resolve_euroc_manifest_with_rig(self, tmp_path: Path) -> None:
        """Load a dataset manifest and its normalized sensor rig."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry_dir, root)

        resolved = DatasetRegistry(repo_root=Path.cwd(), dataset_dir=registry_dir).resolve("euroc_mh_01")

        assert resolved.dataset.name == "euroc_mh_01"
        assert resolved.dataset.type == "euroc"
        assert resolved.dataset.root == root
        assert resolved.dataset.streams.cam0 == Path("cam0/data.csv")
        assert resolved.dataset.rig == Path("config/dataset_rig/euroc.yaml")

        assert resolved.rig.name == "euroc"
        assert resolved.rig.cam0.resolution == (752, 480)
        assert resolved.rig.cam0.body_sensor_transform.rows == 4
        assert resolved.rig.imu0.rate_hz == 200

    def test_list_datasets_returns_dataset_manifests(self, tmp_path: Path) -> None:
        """List dataset manifests from the dataset registry directory."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry_dir, root)

        datasets = DatasetRegistry(repo_root=Path.cwd(), dataset_dir=registry_dir).list()

        datasets_by_name = {dataset.name: dataset for dataset in datasets}
        assert "euroc_mh_01" in datasets_by_name
        assert datasets_by_name["euroc_mh_01"].type == "euroc"
        assert datasets_by_name["euroc_mh_01"].root == root

    def test_local_status_marks_verified_dataset(self, tmp_path: Path) -> None:
        """A dataset exists locally when root and all stream files exist."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "dataset"
        _write_manifest(registry_dir, root)
        _write_streams(root)
        registry = DatasetRegistry(repo_root=Path.cwd(), dataset_dir=registry_dir)
        manifest = registry.find("euroc_mh_01")

        status = registry.local_status(manifest)

        assert status.exists is True
        assert status.verified is True
        assert status.issues == []

    def test_local_status_lists_dataset_issues(self, tmp_path: Path) -> None:
        """Missing root and stream files should make a dataset incomplete."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "dataset"
        _write_manifest(registry_dir, root)
        registry = DatasetRegistry(repo_root=Path.cwd(), dataset_dir=registry_dir)
        manifest = registry.find("euroc_mh_01")

        status = registry.local_status(manifest)

        assert status.exists is False
        assert status.verified is False
        assert status.issues == [
            root,
            root / "cam0/data.csv",
            root / "cam1/data.csv",
            root / "imu0/data.csv",
            root / "state_groundtruth_estimate0/data.csv",
        ]

    def test_missing_dataset_manifest_raises(self, tmp_path: Path) -> None:
        """Unknown dataset names should fail before pipeline resolution."""
        registry_dir = tmp_path / "registry"
        registry_dir.mkdir()
        registry = DatasetRegistry(repo_root=Path.cwd(), dataset_dir=registry_dir)

        with pytest.raises(FileNotFoundError, match="missing_dataset"):
            registry.resolve("missing_dataset")

    def test_find_can_select_dataset_name(self, tmp_path: Path) -> None:
        """Dataset selectors can target a dataset manifest name."""
        registry_dir = tmp_path / "registry"
        root = tmp_path / "euroc_mh_01"
        _write_manifest(registry_dir, root)

        dataset = DatasetRegistry(repo_root=Path.cwd(), dataset_dir=registry_dir).find("euroc_mh_01")

        assert dataset.name == "euroc_mh_01"

    def test_find_ambiguous_dataset_type_raises(self, tmp_path: Path) -> None:
        """Dataset type selectors should be unique."""
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        for name in ["euroc_a", "euroc_b"]:
            (dataset_dir / f"{name}.yaml").write_text(
                f"""
name: {name}
type: euroc
root: datasets/{name}
rig: config/dataset_rig/euroc.yaml
streams:
  cam0: cam0/data.csv
  cam1: cam1/data.csv
  imu0: imu0/data.csv
  ground_truth: state_groundtruth_estimate0/data.csv
""".lstrip(),
                encoding="utf-8",
            )

        registry = DatasetRegistry(repo_root=tmp_path, dataset_dir=dataset_dir)

        with pytest.raises(ValueError, match="ambiguous"):
            registry.find("euroc")


def _write_manifest(registry: Path, root: Path, *, name: str = "euroc_mh_01") -> None:
    registry.mkdir(exist_ok=True)
    manifest = {
        "name": name,
        "type": "euroc",
        "root": str(root),
        "rig": "config/dataset_rig/euroc.yaml",
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
