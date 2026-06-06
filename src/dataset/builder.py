from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd

from dataset.registry import DatasetRegistry
from datasets import Dataset, Image, Sequence, Value, load_from_disk

if TYPE_CHECKING:
    from dataset.manifest import DatasetManifest


class DatasetBuilder(Protocol):
    """Build or load a HuggingFace dataset for a resolved manifest."""

    def build(self, manifest: DatasetManifest) -> Dataset:
        """Build or load a HuggingFace dataset."""


class DatasetAdapter(Protocol):
    """Materialize canonical HuggingFace datasets from raw stream collections."""

    def materialize(self, streams: RawStreamBundle) -> Dataset:
        """Map raw streams to a canonical HuggingFace dataset."""


class StreamLoader(Protocol):
    """Load raw stream collections for a dataset manifest."""

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Load raw stream files into a collection."""


@dataclass(frozen=True)
class RawStream:
    """A resolved raw stream file and its loaded dataframe."""

    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class RawStreamBundle:
    """Raw dataset streams loaded from a manifest."""

    manifest: DatasetManifest
    root: Path
    cache: Path
    streams: Mapping[str, RawStream]

    def require(self, name: str) -> RawStream:
        """Return a required stream or raise a clear error."""
        stream = self.streams.get(name)
        if stream is None:
            msg = f"Dataset manifest '{self.manifest.name}' must define streams.{name}"
            raise ValueError(msg)
        return stream


type PandasToDataset = Callable[[pd.DataFrame], Dataset]


class RawStreamLoader:
    """Load raw stream CSV files described by a dataset manifest."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        """Create a raw stream loader."""
        self.repo_root = (repo_root or Path.cwd()).resolve()

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Load manifest streams into raw dataframes without joining them."""
        root = self.resolve_path(manifest.root)
        cache = self.resolve_path(manifest.cache) if manifest.cache is not None else root / "cache"
        streams = {
            name: RawStream(path=path, frame=pd.read_csv(path))
            for name, path in self.resolve_stream_paths(manifest, root).items()
        }
        return RawStreamBundle(manifest=manifest, root=root, cache=cache, streams=streams)

    def resolve_stream_paths(self, manifest: DatasetManifest, root: Path | None = None) -> dict[str, Path]:
        """Resolve all stream paths without opening them."""
        dataset_root = root or self.resolve_path(manifest.root)
        raw_streams = manifest.streams.model_dump(exclude_none=True)
        return {name: self.resolve_stream_path(dataset_root, Path(path)) for name, path in raw_streams.items()}

    def resolve_path(self, path: Path) -> Path:
        """Resolve a manifest-level path against the repo root."""
        return path if path.is_absolute() else self.repo_root / path

    def resolve_stream_path(self, dataset_root: Path, path: Path) -> Path:
        """Resolve a stream path against the dataset root."""
        return path if path.is_absolute() else dataset_root / path


class EurocDatasetAdapter:
    """Map raw EuRoC streams into the canonical HuggingFace dataset schema."""

    def __init__(self, *, pandas_to_dataset: PandasToDataset | None = None) -> None:
        """Create an EuRoC dataset adapter."""
        self.pandas_to_dataset = pandas_to_dataset or self._pandas_to_dataset_with_euroc_wrapper

    def materialize(self, streams: RawStreamBundle) -> Dataset:
        """Join and map raw EuRoC streams into the canonical HuggingFace dataset."""
        left_cam = self._map_cam_stream(streams.require("cam0"), image_column="left_image")
        right_cam = self._map_cam_stream(streams.require("cam1"), image_column="right_image")
        imu = self._map_imu_stream(streams.require("imu0"))
        ground_truth = self._map_ground_truth_stream(streams.require("ground_truth"))

        cam = left_cam.merge(right_cam, on="timestamp", how="inner")
        imu_cam = cam.merge(imu, on="timestamp", how="outer")
        return self.pandas_to_dataset(imu_cam.merge(ground_truth, on="timestamp", how="outer"))

    def _map_cam_stream(self, stream: RawStream, *, image_column: str) -> pd.DataFrame:
        image_prefix = stream.path.parent / "data"
        frame = stream.frame.rename(columns={"#timestamp [ns]": "timestamp", "filename": image_column}).copy()
        frame[image_column] = frame[image_column].map(lambda filename: f"{image_prefix}/{filename}")
        return frame

    def _map_imu_stream(self, stream: RawStream) -> pd.DataFrame:
        return stream.frame.rename(
            columns={
                "#timestamp [ns]": "timestamp",
                "w_RS_S_x [rad s^-1]": "gyro_x",
                "w_RS_S_y [rad s^-1]": "gyro_y",
                "w_RS_S_z [rad s^-1]": "gyro_z",
                "a_RS_S_x [m s^-2]": "acc_x",
                "a_RS_S_y [m s^-2]": "acc_y",
                "a_RS_S_z [m s^-2]": "acc_z",
            }
        )

    def _map_ground_truth_stream(self, stream: RawStream) -> pd.DataFrame:
        return stream.frame.rename(
            columns={
                "#timestamp": "timestamp",
                " p_RS_R_x [m]": "gt_position_x",
                " p_RS_R_y [m]": "gt_position_y",
                " p_RS_R_z [m]": "gt_position_z",
                " q_RS_w []": "gt_q_w",
                " q_RS_x []": "gt_q_x",
                " q_RS_y []": "gt_q_y",
                " q_RS_z []": "gt_q_z",
                " v_RS_R_x [m s^-1]": "gt_velocity_x",
                " v_RS_R_y [m s^-1]": "gt_velocity_y",
                " v_RS_R_z [m s^-1]": "gt_velocity_z",
                " b_w_RS_S_x [rad s^-1]": "gt_gyro_bias_x",
                " b_w_RS_S_y [rad s^-1]": "gt_gyro_bias_y",
                " b_w_RS_S_z [rad s^-1]": "gt_gyro_bias_z",
                " b_a_RS_S_x [m s^-2]": "gt_acc_bias_x",
                " b_a_RS_S_y [m s^-2]": "gt_acc_bias_y",
                " b_a_RS_S_z [m s^-2]": "gt_acc_bias_z",
            }
        )

    @staticmethod
    def _pandas_to_dataset_with_euroc_wrapper(frame: pd.DataFrame) -> Dataset:
        frame["stereo"] = frame[["left_image", "right_image"]].apply(
            lambda row: [row.left_image, row.right_image], axis=1
        )
        frame["gyro"] = frame[["gyro_x", "gyro_y", "gyro_z"]].apply(
            lambda row: [row.gyro_x, row.gyro_y, row.gyro_z], axis=1
        )
        frame["acc"] = frame[["acc_x", "acc_y", "acc_z"]].apply(
            lambda row: [row.acc_x, row.acc_y, row.acc_z], axis=1
        )
        frame["gt_position"] = frame[["gt_position_x", "gt_position_y", "gt_position_z"]].apply(
            lambda row: [row.gt_position_x, row.gt_position_y, row.gt_position_z], axis=1
        )
        frame["gt_orientation"] = frame[["gt_q_w", "gt_q_x", "gt_q_y", "gt_q_z"]].apply(
            lambda row: [row.gt_q_w, row.gt_q_x, row.gt_q_y, row.gt_q_z], axis=1
        )
        frame["gt_velocity"] = frame[["gt_velocity_x", "gt_velocity_y", "gt_velocity_z"]].apply(
            lambda row: [row.gt_velocity_x, row.gt_velocity_y, row.gt_velocity_z], axis=1
        )
        frame["gt_gyro_bias"] = frame[["gt_gyro_bias_x", "gt_gyro_bias_y", "gt_gyro_bias_z"]].apply(
            lambda row: [row.gt_gyro_bias_x, row.gt_gyro_bias_y, row.gt_gyro_bias_z],
            axis=1,
        )
        frame["gt_acc_bias"] = frame[["gt_acc_bias_x", "gt_acc_bias_y", "gt_acc_bias_z"]].apply(
            lambda row: [row.gt_acc_bias_x, row.gt_acc_bias_y, row.gt_acc_bias_z], axis=1
        )
        frame = frame.drop(
            columns=[
                "left_image",
                "right_image",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "acc_x",
                "acc_y",
                "acc_z",
                "gt_position_x",
                "gt_position_y",
                "gt_position_z",
                "gt_q_w",
                "gt_q_x",
                "gt_q_y",
                "gt_q_z",
                "gt_velocity_x",
                "gt_velocity_y",
                "gt_velocity_z",
                "gt_gyro_bias_x",
                "gt_gyro_bias_y",
                "gt_gyro_bias_z",
                "gt_acc_bias_x",
                "gt_acc_bias_y",
                "gt_acc_bias_z",
            ]
        )
        frame["timestamp"] = frame["timestamp"].astype("int64")
        dataset = Dataset.from_pandas(frame)
        new_features = dataset.features.copy()
        new_features["stereo"] = Sequence(Image(), 2)
        new_features["timestamp"] = Value("int64")

        dataset = dataset.cast(new_features)
        dataset = dataset.map(lambda x: {**x, "has_imu": x["gyro"][0] is not None})
        dataset = dataset.map(lambda x: {**x, "has_ground_truth": x["gt_position"][0] is not None})

        return dataset.map(
            lambda x: {
                **x,
                "gt_orientation": [
                    x["gt_orientation"][1],
                    x["gt_orientation"][2],
                    x["gt_orientation"][3],
                    x["gt_orientation"][0],
                ],
            }
        )


class EurocDatasetBuilder:
    """Cache-first EuRoC builder from manifest streams and an EuRoC adapter."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        stream_loader: StreamLoader | None = None,
        adapter: DatasetAdapter | None = None,
    ) -> None:
        """Create an EuRoC dataset builder."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.stream_loader = stream_loader or RawStreamLoader(repo_root=self.repo_root)
        self.adapter = adapter or EurocDatasetAdapter()

    def build(self, manifest: DatasetManifest) -> Dataset:
        """Load a cached dataset or materialize it from raw EuRoC streams."""
        cache_path = self.cache_path(manifest)
        if cache_path.exists():
            return cast("Dataset", load_from_disk(cache_path))

        streams = self.stream_loader.load(manifest)
        dataset = self.adapter.materialize(streams)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(cache_path)
        return dataset

    def cache_path(self, manifest: DatasetManifest) -> Path:
        """Return the HuggingFace cache directory for a manifest."""
        root = self._resolve_path(manifest.root)
        cache = self._resolve_path(manifest.cache) if manifest.cache is not None else root / "cache"
        return cache / "full"

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path


class DatasetLoader:
    """Load HuggingFace datasets from named dataset manifests."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        dataset_dir: Path | None = None,
        builders: Mapping[str, DatasetBuilder] | None = None,
    ) -> None:
        """Create a dataset loader."""
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.registry = DatasetRegistry(repo_root=self.repo_root, dataset_dir=dataset_dir)
        self.builders = (
            {"euroc": EurocDatasetBuilder(repo_root=self.repo_root)} if builders is None else dict(builders)
        )

    def load(self, name: str) -> Dataset:
        """Load a HuggingFace dataset by manifest name."""
        manifest = self.registry.find(name)
        return self.load_manifest(manifest)

    def load_manifest(self, manifest: DatasetManifest) -> Dataset:
        """Load a HuggingFace dataset from an already loaded manifest."""
        builder = self.builders.get(manifest.type)
        if builder is None:
            supported_types = ", ".join(sorted(self.builders)) or "<none>"
            msg = f"Unsupported dataset type '{manifest.type}'. Supported types: {supported_types}"
            raise ValueError(msg)
        return builder.build(manifest)
