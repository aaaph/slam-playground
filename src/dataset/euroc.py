from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import cv2
import numpy as np
import pandas as pd
from pyarrow import compute as pa_compute
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import ImuContext, VioContext
from core.transformations.special_euclidian_3_dim import SE3
from dataset.builder import (
    DatasetAdapter,
    RawStream,
    RawStreamBundle,
    RawStreamLoader,
    StreamLoader,
)
from dataset.interfaces import VioDataset
from dataset.manifest import DatasetManifest, DatasetRigConfig
from dataset.registry import DatasetRegistry
from dataset.sensor_config import CameraSensor, IMUSensor
from datasets import Array2D, Dataset, Image, Sequence, Value, load_from_disk
from logger import log

# `pyarrow.compute` dynamically exposes compute kernels, but its type information is incomplete
# (type checkers may flag valid kernels as missing). Use an `Any` alias for static analysis.
pc: Any = pa_compute


def decode_stereo_image(image: object) -> np.ndarray:
    """Decode a EuRoC stereo image sample into a grayscale ndarray."""
    image_dict = cast("dict[str, object]", image) if isinstance(image, dict) else None
    if image_dict is not None and isinstance(image_dict.get("bytes"), bytes):
        encoded = np.frombuffer(image_dict["bytes"], dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            msg = f"Failed to decode stereo image {image_dict.get('path', '<unknown>')}"
            raise ValueError(msg)
        return decoded
    return np.asarray(image, dtype=np.uint8)


def decode_stereo_pair(stereo: tuple[object, object] | list[object]) -> tuple[np.ndarray, np.ndarray]:
    """Decode a EuRoC stereo pair into grayscale ndarrays."""
    return decode_stereo_image(stereo[0]), decode_stereo_image(stereo[1])


@dataclass
class EurocConfig:
    """Euroc configuration."""

    cam0: CameraSensor
    cam1: CameraSensor
    imu0: IMUSensor

    def body_sensor_transforms(self) -> tuple[SE3, SE3]:
        """Get the body sensor transforms."""
        t_body_cam0 = self.cam0.body_sensor_transform
        t_body_cam0_rot = Rotation.from_matrix(t_body_cam0[:3, :3])
        t_body_cam0_translation = t_body_cam0[:3, 3]
        t_body_cam0_se3 = SE3(t_body_cam0_rot, t_body_cam0_translation)
        t_body_cam1 = self.cam1.body_sensor_transform
        t_body_cam1_rot = Rotation.from_matrix(t_body_cam1[:3, :3])
        t_body_cam1_translation = t_body_cam1[:3, 3]
        t_body_cam1_se3 = SE3(t_body_cam1_rot, t_body_cam1_translation)
        return (t_body_cam0_se3, t_body_cam1_se3)

    def as_vio_ctx(self) -> VioContext:
        """Get the VIO context."""
        camera_model = StereoCameraModel.from_cameras_config(self.cam0, self.cam1)
        stereo_ctx = camera_model.as_stereo_ctx()
        imu_ctx = ImuContext.from_imu_config(self.imu0)
        return VioContext.from_stereo_and_imu_config(stereo_ctx, imu_ctx)

    @classmethod
    def from_rig_config(cls, rig: DatasetRigConfig) -> Self:
        """Create EuRoC config from a normalized sensor rig."""
        return cls(
            CameraSensor.from_rig_config(rig.cam0),
            CameraSensor.from_rig_config(rig.cam1),
            IMUSensor.from_rig_config(rig.imu0),
        )


type PandasToDataset = Callable[[pd.DataFrame], Dataset]


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


class EurocDataset(VioDataset):
    """Euroc dataset."""

    config: EurocConfig
    logger = log.bind(app="euroc_dataset")
    ground_truth_ds: Dataset

    def __init__(self, dataset: Dataset, config: EurocConfig) -> None:
        """Initialize the Euroc dataset."""
        self.ds = dataset
        self.config = config
        # self.ground_truth_map: dict[float, GroundTruth] = {}
        # self._create_and_save_ground_truth_map()
        # self.ground_truth_sorted_timestamps = sorted(self.ground_truth_map.keys())

    """def _create_and_save_ground_truth_map(self) -> None:
        Create and save the ground truth map to avoid mapping the ground truth dataset from disk every time.
        ground_truth_cache = self.data_paths.cache / "ground_truth.pkl"
        if ground_truth_cache.exists():
            with Path.open(ground_truth_cache, "rb") as f:
                self.ground_truth_map = pickle.load(f)  # noqa: S301
        else:
            self.logger.info("Ground truth map not found, creating from scratch...")
            ground_truth_ds = self.ground_truth().with_format("numpy")
            self.ground_truth_map = {float(x["timestamp"]): x for x in ground_truth_ds}
            self.logger.info(f"Saved ground truth map to {ground_truth_cache}")
            with Path.open(ground_truth_cache, "wb") as f:
                pickle.dump(self.ground_truth_map, f)"""

    def stereo(self) -> Dataset:
        """Get the stereo dataset."""
        ds = self.ds.remove_columns(
            [
                "gyro",
                "acc",
                "gt_position",
                "gt_orientation",
                "gt_velocity",
                "gt_gyro_bias",
                "gt_acc_bias",
            ]
        )
        return ds.filter(lambda x: x["stereo"][0] is not None).sort("timestamp")

    def imu_and_stereo(self, *, decode_images: bool = True) -> Dataset:
        """Get the imu and stereo dataset. Guarantee to have one stereo frame and N IMU frames."""
        imu_ds = self.ds.remove_columns(
            ["gt_position", "gt_orientation", "gt_velocity", "gt_gyro_bias", "gt_acc_bias", "stereo"]
        )
        imu_ds = imu_ds.filter(lambda x: x["gyro"][0] is not None or x["acc"][0] is not None).flatten_indices()
        imu_table = imu_ds.data
        imu_timestamps = imu_table["timestamp"].to_numpy().astype(np.int64)

        stereo_ds = self.stereo()
        stereo_ds = stereo_ds.remove_columns(["has_imu", "has_ground_truth"])

        def map_fn(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
            result = {"imu_ts": [], "gyro_data": [], "acc_data": []}
            for idx, t1 in zip(indices, batch["timestamp"], strict=True):
                # print(idx, t1)
                t0 = stereo_ds[idx - 1]["timestamp"] if idx > 0 else imu_timestamps[0]
                if t0 == t1:
                    mask = pc.and_(
                        pc.greater_equal(imu_table["timestamp"], t0), pc.less_equal(imu_table["timestamp"], t1)
                    )
                else:
                    mask = pc.and_(
                        pc.greater(imu_table["timestamp"], t0), pc.less_equal(imu_table["timestamp"], t1)
                    )
                imu_chunk = imu_table.filter(mask)
                n = len(imu_chunk)
                acc_data = np.zeros((n, 3), dtype=np.float32)
                gyro_data = np.zeros((n, 3), dtype=np.float32)
                imu_ts_column = np.zeros(n, dtype=np.int64)
                if n:
                    gyro_data = np.vstack(imu_chunk["gyro"].to_pylist()).astype(np.float32)
                    acc_data = np.vstack(imu_chunk["acc"].to_pylist()).astype(np.float32)
                    imu_ts_column = imu_chunk["timestamp"].to_numpy().astype(np.int64)

                result["gyro_data"].append(gyro_data)
                result["acc_data"].append(acc_data)
                result["imu_ts"].append(imu_ts_column)

            return result

        features = stereo_ds.features.copy()
        features["stereo"] = Sequence(Image(decode=decode_images), length=2)
        features["gyro_data"] = Array2D(shape=(None, 3), dtype="float32")
        features["acc_data"] = Array2D(shape=(None, 3), dtype="float32")
        features["imu_ts"] = Sequence(Value("int64"))
        new_ds = stereo_ds.map(
            map_fn,
            with_indices=True,
            batched=True,
            batch_size=100,
            features=features,
            desc="Sync IMU and Stereo",
        )
        return new_ds.with_format("numpy")

    @classmethod
    def from_name(cls, name: str, *, repo_root: Path | None = None) -> Self:
        """Load an EuRoC dataset by manifest name."""
        repo_root = (repo_root or Path(__file__).parent.parent.parent).resolve()

        registry = DatasetRegistry(repo_root=repo_root)
        resolved = registry.resolve(name)
        manifest = resolved.dataset
        rig = resolved.rig

        hf_dataset = EurocDatasetBuilder(repo_root=repo_root).build(manifest)

        return cls.from_dataset(hf_dataset, rig)

    @classmethod
    def from_dataset(cls, dataset: Dataset, rig: DatasetRigConfig) -> Self:
        """Wrap a materialized HuggingFace dataset with EuRoC config and data paths."""
        return cls(dataset, EurocConfig.from_rig_config(rig))

    @staticmethod
    def mh_01_easy() -> "EurocDataset":
        """Load the MH_01_easy dataset."""
        return EurocDataset.from_name("euroc_mh_01")

    @classmethod
    def mh_01(cls) -> Self:
        """Load the MH_01 dataset."""
        return cls.from_name("euroc_mh_01")

    @classmethod
    def mh_02(cls) -> Self:
        """Load the MH_02 dataset."""
        return cls.from_name("euroc_mh_02")
