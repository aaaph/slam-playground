from pathlib import Path

import pytest

from dataset.builder import DatasetBuilder
from dataset.loader import DatasetLoader
from dataset.manifest import DatasetManifest
from datasets import Dataset


def _euroc_manifest(**overrides: object) -> DatasetManifest:
    raw_manifest: dict[str, object] = {
        "name": "euroc_test",
        "type": "euroc",
        "root": "datasets/euroc_test",
        "rig": "config/dataset_rig/euroc.yaml",
        "cache": "datasets/euroc_test/cache",
        "streams": {
            "cam0": "cam0/data.csv",
            "cam1": "cam1/data.csv",
            "imu0": "imu0/data.csv",
            "ground_truth": "state_groundtruth_estimate0/data.csv",
        },
    }
    return DatasetManifest.model_validate({**raw_manifest, **overrides})


class FakeDatasetBuilder(DatasetBuilder):
    """Dataset builder test double."""

    def __init__(self, dataset: Dataset) -> None:
        """Create a fake builder."""
        self.dataset = dataset
        self.seen_manifest: DatasetManifest | None = None

    def build(self, manifest: DatasetManifest) -> Dataset:
        """Capture the manifest and return the fake dataset."""
        self.seen_manifest = manifest
        return self.dataset


class TestDatasetLoader:
    """Tests for generic dataset loading from manifests."""

    def test_load_dispatches_by_manifest_type(self, tmp_path: Path) -> None:
        """DatasetLoader should choose a builder based on manifest.type."""
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir()
        (dataset_dir / "euroc_test.yaml").write_text(
            """
name: euroc_test
type: euroc
root: datasets/euroc_test
rig: config/dataset_rig/euroc.yaml
streams:
  cam0: cam0/data.csv
  cam1: cam1/data.csv
  imu0: imu0/data.csv
  ground_truth: state_groundtruth_estimate0/data.csv
""".lstrip(),
            encoding="utf-8",
        )
        expected_dataset = Dataset.from_dict({"timestamp": [1]})
        builder = FakeDatasetBuilder(expected_dataset)
        loader = DatasetLoader(repo_root=tmp_path, dataset_dir=dataset_dir, builders={"euroc": builder})

        dataset = loader.load("euroc_test")

        assert dataset is expected_dataset
        assert builder.seen_manifest is not None
        assert builder.seen_manifest.name == "euroc_test"

        dataset = loader.load("euroc")

        assert dataset is expected_dataset
        assert builder.seen_manifest.name == "euroc_test"

    def test_unknown_manifest_type_raises(self) -> None:
        """DatasetLoader should report unsupported dataset families clearly."""
        loader = DatasetLoader(builders={})

        with pytest.raises(ValueError, match="Unsupported dataset type 'euroc'"):
            loader.load_by_manifest(_euroc_manifest())
