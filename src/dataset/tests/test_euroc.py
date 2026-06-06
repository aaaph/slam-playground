from pathlib import Path
from typing import Any

from dataset.euroc import EurocConfig, EurocDataset, EurocDatasetBuilder
from dataset.registry import DatasetRegistry
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

    def test_config_from_rig_config(self) -> None:
        """EurocConfig should be constructable from a normalized rig config."""
        resolved = DatasetRegistry(repo_root=Path.cwd()).resolve("euroc_mh_01")

        config = EurocConfig.from_rig_config(resolved.rig)

        assert config.cam0.resolution == (752, 480)
        assert config.cam1.resolution == (752, 480)
        assert config.imu0.body_sensor_transform.shape == (4, 4)

    def test_from_dataset_wraps_hf_dataset_and_rig_config(self) -> None:
        """A materialized HF dataset should become an EurocDataset wrapper with rig config."""
        resolved = DatasetRegistry(repo_root=Path.cwd()).resolve("euroc_mh_01")
        hf_dataset = _minimal_euroc_dataset()

        euroc = EurocDataset.from_dataset(hf_dataset, resolved.rig)

        assert euroc.ds is hf_dataset
        assert euroc.config.cam0.resolution == (752, 480)
        assert euroc.config.cam0.body_sensor_transform.shape == (4, 4)
        assert euroc.config.imu0.body_sensor_transform.shape == (4, 4)

    def test_from_name_uses_registry_and_euroc_builder(self, monkeypatch: Any) -> None:
        """from_name should route through DatasetRegistry resolution and EurocDatasetBuilder."""
        loaded_dataset = _minimal_euroc_dataset()
        captured: dict[str, Any] = {}
        sentinel = object()

        def fake_build(self: EurocDatasetBuilder, manifest: object) -> Dataset:
            captured["built_manifest"] = manifest
            return loaded_dataset

        def fake_from_dataset(
            cls: type[EurocDataset],
            dataset: Dataset,
            rig: object,
        ) -> object:
            captured["wrapped_dataset"] = dataset
            captured["wrapped_rig"] = rig
            return sentinel

        monkeypatch.setattr(EurocDatasetBuilder, "build", fake_build)
        monkeypatch.setattr(EurocDataset, "from_dataset", classmethod(fake_from_dataset))

        result = EurocDataset.from_name("euroc_mh_01", repo_root=Path.cwd())

        assert result is sentinel
        assert captured["built_manifest"].name == "euroc_mh_01"
        assert captured["wrapped_dataset"] is loaded_dataset
        assert captured["wrapped_rig"].name == "euroc"

    def test_from_name_can_resolve_unique_dataset_type_selector(self, monkeypatch: Any) -> None:
        """from_name should accept a registry selector like 'euroc'."""
        loaded_dataset = _minimal_euroc_dataset()

        monkeypatch.setattr(EurocDatasetBuilder, "build", lambda _self, _manifest: loaded_dataset)

        euroc = EurocDataset.from_name("euroc", repo_root=Path.cwd())

        assert isinstance(euroc, EurocDataset)
        assert euroc.ds is loaded_dataset
        assert euroc.config.cam0.resolution == (752, 480)
