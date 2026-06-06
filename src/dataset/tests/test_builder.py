from pathlib import Path

import pandas as pd
import pytest
from PIL import Image as PilImage

from dataset.builder import (
    DatasetAdapter,
    DatasetBuilder,
    DatasetLoader,
    EurocDatasetAdapter,
    EurocDatasetBuilder,
    RawStream,
    RawStreamBundle,
    RawStreamLoader,
    StreamLoader,
)
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


class FakeStreamLoader(StreamLoader):
    """Raw stream loader test double."""

    def __init__(self, bundle: RawStreamBundle) -> None:
        """Create a fake stream loader."""
        self.bundle = bundle
        self.seen_manifest: DatasetManifest | None = None

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Capture the manifest and return the fake bundle."""
        self.seen_manifest = manifest
        return self.bundle


class FakeAdapter(DatasetAdapter):
    """Dataset adapter test double."""

    def __init__(self, dataset: Dataset) -> None:
        """Create a fake adapter."""
        self.dataset = dataset
        self.seen_bundle: RawStreamBundle | None = None

    def materialize(self, streams: RawStreamBundle) -> Dataset:
        """Capture the stream bundle and return the fake dataset."""
        self.seen_bundle = streams
        return self.dataset


class FailStreamLoader(StreamLoader):
    """Stream loader that should not be called."""

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Fail if a cache hit still tries to load raw streams."""
        raise AssertionError("raw streams should not be loaded on cache hit")


class FailAdapter(DatasetAdapter):
    """Adapter that should not be called."""

    def materialize(self, streams: RawStreamBundle) -> Dataset:
        """Fail if a cache hit still tries to materialize raw streams."""
        raise AssertionError("adapter should not be called on cache hit")


class TestEurocDatasetBuilder:
    """Tests for EuRoC HuggingFace dataset building."""

    def test_cache_miss_loads_streams_materializes_and_saves_dataset(self, tmp_path: Path) -> None:
        """The builder should orchestrate streams -> adapter -> HF cache on cache miss."""
        manifest = _euroc_manifest()
        bundle = RawStreamBundle(
            manifest=manifest,
            root=tmp_path / "datasets/euroc_test",
            cache=tmp_path / "datasets/euroc_test/cache",
            streams={},
        )
        expected_dataset = Dataset.from_dict({"timestamp": [1]})
        stream_loader = FakeStreamLoader(bundle)
        adapter = FakeAdapter(expected_dataset)
        builder = EurocDatasetBuilder(repo_root=tmp_path, stream_loader=stream_loader, adapter=adapter)

        dataset = builder.build(manifest)

        assert dataset is expected_dataset
        assert stream_loader.seen_manifest == manifest
        assert adapter.seen_bundle == bundle
        assert (tmp_path / "datasets/euroc_test/cache/full").exists()

    def test_cache_hit_loads_hf_dataset_without_raw_streams(self, tmp_path: Path) -> None:
        """The builder should return cached HF data before touching raw streams."""
        cached_dataset = Dataset.from_dict({"timestamp": [42]})
        cached_dataset.save_to_disk(tmp_path / "datasets/euroc_test/cache/full")
        builder = EurocDatasetBuilder(repo_root=tmp_path, stream_loader=FailStreamLoader(), adapter=FailAdapter())

        dataset = builder.build(_euroc_manifest())

        assert list(dataset["timestamp"]) == [42]


class TestRawStreamLoader:
    """Tests for raw stream loading without stream joins."""

    def test_load_reads_manifest_streams_into_named_collection(self, tmp_path: Path) -> None:
        """RawStreamLoader should load stream files independently."""
        root = tmp_path / "datasets/euroc_test"
        for stream_dir in [
            root / "cam0",
            root / "cam1",
            root / "imu0",
            root / "state_groundtruth_estimate0",
        ]:
            stream_dir.mkdir(parents=True)
        (root / "cam0/data.csv").write_text("#timestamp [ns],filename\n1,left.png\n", encoding="utf-8")
        (root / "cam1/data.csv").write_text("#timestamp [ns],filename\n1,right.png\n", encoding="utf-8")
        (root / "imu0/data.csv").write_text("#timestamp [ns],w_RS_S_x [rad s^-1]\n1,0.1\n", encoding="utf-8")
        (root / "state_groundtruth_estimate0/data.csv").write_text(
            "#timestamp, p_RS_R_x [m]\n1,2.0\n",
            encoding="utf-8",
        )

        bundle = RawStreamLoader(repo_root=tmp_path).load(_euroc_manifest())

        assert set(bundle.streams) == {"cam0", "cam1", "imu0", "ground_truth"}
        assert bundle.streams["cam0"].path == root / "cam0/data.csv"
        assert bundle.streams["cam0"].frame["filename"].to_list() == ["left.png"]


class TestEurocDatasetAdapter:
    """Tests for EuRoC stream mapping and join semantics."""

    def test_materialize_maps_and_joins_raw_streams(self, tmp_path: Path) -> None:
        """The adapter should own EuRoC fields, timestamp columns, and joins."""
        captured_frames: list[pd.DataFrame] = []
        expected_dataset = Dataset.from_dict({"timestamp": [1]})

        def fake_pandas_to_dataset(frame: pd.DataFrame) -> Dataset:
            captured_frames.append(frame)
            return expected_dataset

        manifest = _euroc_manifest()
        root = tmp_path / "datasets/euroc_test"
        bundle = RawStreamBundle(
            manifest=manifest,
            root=root,
            cache=root / "cache",
            streams={
                "cam0": RawStream(
                    path=root / "cam0/data.csv",
                    frame=pd.DataFrame({"#timestamp [ns]": [1], "filename": ["left.png"]}),
                ),
                "cam1": RawStream(
                    path=root / "cam1/data.csv",
                    frame=pd.DataFrame({"#timestamp [ns]": [1], "filename": ["right.png"]}),
                ),
                "imu0": RawStream(
                    path=root / "imu0/data.csv",
                    frame=pd.DataFrame(
                        {
                            "#timestamp [ns]": [1, 2],
                            "w_RS_S_x [rad s^-1]": [0.1, 0.2],
                            "w_RS_S_y [rad s^-1]": [0.3, 0.4],
                            "w_RS_S_z [rad s^-1]": [0.5, 0.6],
                            "a_RS_S_x [m s^-2]": [1.1, 1.2],
                            "a_RS_S_y [m s^-2]": [1.3, 1.4],
                            "a_RS_S_z [m s^-2]": [1.5, 1.6],
                        }
                    ),
                ),
                "ground_truth": RawStream(
                    path=root / "state_groundtruth_estimate0/data.csv",
                    frame=pd.DataFrame(
                        {
                            "#timestamp": [1],
                            " p_RS_R_x [m]": [2.0],
                            " p_RS_R_y [m]": [3.0],
                            " p_RS_R_z [m]": [4.0],
                            " q_RS_w []": [1.0],
                            " q_RS_x []": [0.0],
                            " q_RS_y []": [0.0],
                            " q_RS_z []": [0.0],
                            " v_RS_R_x [m s^-1]": [0.0],
                            " v_RS_R_y [m s^-1]": [0.0],
                            " v_RS_R_z [m s^-1]": [0.0],
                            " b_w_RS_S_x [rad s^-1]": [0.0],
                            " b_w_RS_S_y [rad s^-1]": [0.0],
                            " b_w_RS_S_z [rad s^-1]": [0.0],
                            " b_a_RS_S_x [m s^-2]": [0.0],
                            " b_a_RS_S_y [m s^-2]": [0.0],
                            " b_a_RS_S_z [m s^-2]": [0.0],
                        }
                    ),
                ),
            },
        )

        dataset = EurocDatasetAdapter(pandas_to_dataset=fake_pandas_to_dataset).materialize(bundle)

        assert dataset is expected_dataset
        assert len(captured_frames) == 1
        frame = captured_frames[0].sort_values("timestamp").reset_index(drop=True)
        assert frame["timestamp"].to_list() == [1, 2]
        assert frame.loc[0, "left_image"] == f"{root}/cam0/data/left.png"
        assert frame.loc[0, "right_image"] == f"{root}/cam1/data/right.png"
        assert frame.loc[1, "gyro_x"] == 0.2
        assert "gt_position_x" in frame.columns

    def test_default_conversion_builds_canonical_hf_dataset(self, tmp_path: Path) -> None:
        """The adapter should convert joined EuRoC data into the canonical HF schema."""
        manifest = _euroc_manifest()
        root = tmp_path / "datasets/euroc_test"
        for image_path in [root / "cam0/data/left.png", root / "cam1/data/right.png"]:
            image_path.parent.mkdir(parents=True)
            PilImage.new("L", (1, 1), 0).save(image_path)
        dataset = EurocDatasetAdapter().materialize(
            RawStreamBundle(
                manifest=manifest,
                root=root,
                cache=root / "cache",
                streams={
                    "cam0": RawStream(
                        path=root / "cam0/data.csv",
                        frame=pd.DataFrame({"#timestamp [ns]": [1], "filename": ["left.png"]}),
                    ),
                    "cam1": RawStream(
                        path=root / "cam1/data.csv",
                        frame=pd.DataFrame({"#timestamp [ns]": [1], "filename": ["right.png"]}),
                    ),
                    "imu0": RawStream(
                        path=root / "imu0/data.csv",
                        frame=pd.DataFrame(
                            {
                                "#timestamp [ns]": [1],
                                "w_RS_S_x [rad s^-1]": [0.1],
                                "w_RS_S_y [rad s^-1]": [0.2],
                                "w_RS_S_z [rad s^-1]": [0.3],
                                "a_RS_S_x [m s^-2]": [1.1],
                                "a_RS_S_y [m s^-2]": [1.2],
                                "a_RS_S_z [m s^-2]": [1.3],
                            }
                        ),
                    ),
                    "ground_truth": RawStream(
                        path=root / "state_groundtruth_estimate0/data.csv",
                        frame=pd.DataFrame(
                            {
                                "#timestamp": [1],
                                " p_RS_R_x [m]": [2.1],
                                " p_RS_R_y [m]": [2.2],
                                " p_RS_R_z [m]": [2.3],
                                " q_RS_w []": [1.0],
                                " q_RS_x []": [0.0],
                                " q_RS_y []": [0.0],
                                " q_RS_z []": [0.0],
                                " v_RS_R_x [m s^-1]": [3.1],
                                " v_RS_R_y [m s^-1]": [3.2],
                                " v_RS_R_z [m s^-1]": [3.3],
                                " b_w_RS_S_x [rad s^-1]": [0.0],
                                " b_w_RS_S_y [rad s^-1]": [0.0],
                                " b_w_RS_S_z [rad s^-1]": [0.0],
                                " b_a_RS_S_x [m s^-2]": [0.0],
                                " b_a_RS_S_y [m s^-2]": [0.0],
                                " b_a_RS_S_z [m s^-2]": [0.0],
                            }
                        ),
                    ),
                },
            )
        )

        assert dataset.column_names == [
            "timestamp",
            "stereo",
            "gyro",
            "acc",
            "gt_position",
            "gt_orientation",
            "gt_velocity",
            "gt_gyro_bias",
            "gt_acc_bias",
            "has_imu",
            "has_ground_truth",
        ]
        assert dataset["has_imu"] == [True]
        assert dataset["has_ground_truth"] == [True]
        assert dataset["gt_orientation"] == [[0.0, 0.0, 0.0, 1.0]]

    def test_missing_ground_truth_stream_raises(self, tmp_path: Path) -> None:
        """The EuRoC adapter should report missing required streams clearly."""
        manifest = _euroc_manifest(
            streams={
                "cam0": "cam0/data.csv",
                "cam1": "cam1/data.csv",
                "imu0": "imu0/data.csv",
            }
        )
        bundle = RawStreamBundle(
            manifest=manifest,
            root=tmp_path / "datasets/euroc_test",
            cache=tmp_path / "datasets/euroc_test/cache",
            streams={},
        )

        with pytest.raises(ValueError, match=r"streams\.cam0"):
            EurocDatasetAdapter().materialize(bundle)


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
            loader.load_manifest(_euroc_manifest())
