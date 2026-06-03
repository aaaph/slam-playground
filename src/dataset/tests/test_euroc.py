from pathlib import Path
from typing import Any

from dataset.builder import DatasetLoader
from dataset.euroc import EurocDataset
from dataset.manifest import DatasetManifestLoader
from datasets import Dataset


def _minimal_euroc_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "timestamp": [1],
            "stereo": [[None, None]],
            "gyro": [[0.0, 0.0, 0.0]],
            "acc": [[0.0, 0.0, 0.0]],
            "gt_position": [[1.0, 2.0, 3.0]],
            "gt_orientation": [[0.0, 0.0, 0.0, 1.0]],
            "gt_velocity": [[0.0, 0.0, 0.0]],
            "gt_gyro_bias": [[0.0, 0.0, 0.0]],
            "gt_acc_bias": [[0.0, 0.0, 0.0]],
        }
    )


class TestEurocDatasetManifestFactory:
    """Tests for manifest-backed EuRoC dataset construction."""

    def test_data_paths_from_manifest_resolves_legacy_paths(self, tmp_path: Path) -> None:
        """Manifest paths should map to the old EurocDataPaths shape."""
        resolved = DatasetManifestLoader(repo_root=Path.cwd()).resolve("euroc_mh_01")
        manifest = resolved.dataset.model_copy(update={"cache": tmp_path / "cache"})

        data_paths = EurocDataset.data_paths_from_manifest(manifest, repo_root=Path.cwd())

        assert data_paths.cam0 == Path.cwd() / "datasets/euroc_mh_01/cam0/data.csv"
        assert data_paths.cam1 == Path.cwd() / "datasets/euroc_mh_01/cam1/data.csv"
        assert data_paths.imu0 == Path.cwd() / "datasets/euroc_mh_01/imu0/data.csv"
        assert data_paths.gth0 == Path.cwd() / "datasets/euroc_mh_01/state_groundtruth_estimate0/data.csv"
        assert data_paths.cache == tmp_path / "cache"

    def test_from_manifest_wraps_hf_dataset_and_rig_config(self, tmp_path: Path) -> None:
        """A materialized HF dataset should become an EurocDataset wrapper with rig config."""
        resolved = DatasetManifestLoader(repo_root=Path.cwd()).resolve("euroc_mh_01")
        cache = tmp_path / "cache"
        cache.mkdir()
        manifest = resolved.dataset.model_copy(update={"cache": cache})

        euroc = EurocDataset.from_manifest(
            manifest=manifest,
            rig=resolved.rig,
            dataset=_minimal_euroc_dataset(),
            repo_root=Path.cwd(),
        )

        assert euroc.data_paths.cache == cache
        assert euroc.config.cam0.resolution == (752, 480)
        assert euroc.config.cam0.body_sensor_transform.shape == (4, 4)
        assert euroc.config.imu0.body_sensor_transform.shape == (4, 4)

        mapped = euroc.map(lambda dataset: dataset)
        assert mapped.config.cam0 is euroc.config.cam0
        assert mapped.config.cam1 is euroc.config.cam1
        assert mapped.config.imu0 is euroc.config.imu0

    def test_from_name_uses_dataset_loader_and_manifest_rig(self, monkeypatch: Any) -> None:
        """from_name should route through manifest resolution and DatasetLoader."""
        loaded_dataset = _minimal_euroc_dataset()
        captured: dict[str, Any] = {}
        sentinel = object()

        def fake_load_manifest(self: DatasetLoader, manifest: object) -> Dataset:
            captured["loaded_manifest"] = manifest
            return loaded_dataset

        def fake_from_manifest(
            cls: type[EurocDataset],
            *,
            manifest: object,
            rig: object,
            dataset: Dataset,
            repo_root: Path,
        ) -> object:
            captured["wrapped_manifest"] = manifest
            captured["wrapped_rig"] = rig
            captured["wrapped_dataset"] = dataset
            captured["repo_root"] = repo_root
            return sentinel

        monkeypatch.setattr(DatasetLoader, "load_manifest", fake_load_manifest)
        monkeypatch.setattr(EurocDataset, "from_manifest", classmethod(fake_from_manifest))

        result = EurocDataset.from_name("euroc_mh_01", repo_root=Path.cwd())

        assert result is sentinel
        assert captured["loaded_manifest"] is captured["wrapped_manifest"]
        assert captured["wrapped_dataset"] is loaded_dataset
        assert captured["wrapped_rig"].name == "euroc"
        assert captured["repo_root"] == Path.cwd()
