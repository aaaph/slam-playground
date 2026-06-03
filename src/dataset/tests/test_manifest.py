from pathlib import Path

import pytest

from dataset.manifest import DatasetManifestLoader


class TestDatasetManifestLoader:
    """Tests for dataset manifest and rig loading."""

    def test_resolve_euroc_manifest_with_rig(self) -> None:
        """Load a dataset manifest and its normalized sensor rig."""
        resolved = DatasetManifestLoader(repo_root=Path.cwd()).resolve("euroc_mh_01")

        assert resolved.dataset.name == "euroc_mh_01"
        assert resolved.dataset.type == "euroc"
        assert resolved.dataset.root == Path("datasets/euroc_mh_01")
        assert resolved.dataset.streams.cam0 == Path("cam0/data.csv")
        assert resolved.dataset.rig == Path("config/dataset_rig/euroc.yaml")

        assert resolved.rig.name == "euroc"
        assert resolved.rig.cam0.resolution == (752, 480)
        assert resolved.rig.cam0.body_sensor_transform.rows == 4
        assert resolved.rig.imu0.rate_hz == 200

    def test_list_datasets_returns_dataset_manifests(self) -> None:
        """List dataset manifests from the dataset registry directory."""
        datasets = DatasetManifestLoader(repo_root=Path.cwd()).list_datasets()

        assert [dataset.name for dataset in datasets] == ["euroc_mh_01"]
        assert datasets[0].type == "euroc"
        assert datasets[0].root == Path("datasets/euroc_mh_01")

    def test_local_status_marks_verified_dataset(self, tmp_path: Path) -> None:
        """A dataset exists locally when root and all stream files exist."""
        root = tmp_path / "dataset"
        manifest = (
            DatasetManifestLoader(repo_root=Path.cwd())
            .load_dataset("euroc_mh_01")
            .model_copy(update={"root": root})
        )
        for stream_path in [
            root / "cam0/data.csv",
            root / "cam1/data.csv",
            root / "imu0/data.csv",
            root / "state_groundtruth_estimate0/data.csv",
        ]:
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            stream_path.write_text("", encoding="utf-8")

        status = DatasetManifestLoader(repo_root=Path.cwd()).local_status(manifest)

        assert status.exists is True
        assert status.verified is True
        assert status.issues == []

    def test_local_status_lists_dataset_issues(self, tmp_path: Path) -> None:
        """Missing root and stream files should make a dataset incomplete."""
        root = tmp_path / "dataset"
        manifest = (
            DatasetManifestLoader(repo_root=Path.cwd())
            .load_dataset("euroc_mh_01")
            .model_copy(update={"root": root})
        )

        status = DatasetManifestLoader(repo_root=Path.cwd()).local_status(manifest)

        assert status.exists is False
        assert status.verified is False
        assert status.issues == [
            root,
            root / "cam0/data.csv",
            root / "cam1/data.csv",
            root / "imu0/data.csv",
            root / "state_groundtruth_estimate0/data.csv",
        ]

    def test_missing_dataset_manifest_raises(self) -> None:
        """Unknown dataset names should fail before pipeline resolution."""
        loader = DatasetManifestLoader(repo_root=Path.cwd())

        with pytest.raises(FileNotFoundError, match="missing_dataset"):
            loader.resolve("missing_dataset")
