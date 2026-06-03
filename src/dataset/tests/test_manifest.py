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
        assert resolved.dataset.root == Path("datasets/euroc_v_01_easy")
        assert resolved.dataset.streams.cam0 == Path("cam0/data.csv")
        assert resolved.dataset.rig == Path("config/dataset_rig/euroc.yaml")

        assert resolved.rig.name == "euroc"
        assert resolved.rig.cam0.resolution == (752, 480)
        assert resolved.rig.cam0.body_sensor_transform.rows == 4
        assert resolved.rig.imu0.rate_hz == 200

    def test_missing_dataset_manifest_raises(self) -> None:
        """Unknown dataset names should fail before pipeline resolution."""
        loader = DatasetManifestLoader(repo_root=Path.cwd())

        with pytest.raises(FileNotFoundError, match="missing_dataset"):
            loader.resolve("missing_dataset")
