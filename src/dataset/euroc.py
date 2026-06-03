import bisect
import pickle
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypedDict, cast

import cv2
import numpy as np
from pyarrow import compute as pa_compute
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import ImuContext, VioContext
from core.transformations.special_euclidian_3_dim import SE3
from dataset.builder import DatasetLoader
from dataset.manifest import DatasetManifest, DatasetManifestLoader, DatasetRigConfig
from dataset.sensor_config import CameraSensor, IMUSensor
from datasets import Array2D, Dataset, Image, Sequence, Value, load_from_disk
from logger import log

if TYPE_CHECKING:
    from dataset.sensor_interfaces import CameraConfigOptions, IMUConfigOptions

# `pyarrow.compute` dynamically exposes compute kernels, but its type information is incomplete
# (type checkers may flag valid kernels as missing). Use an `Any` alias for static analysis.
pc: Any = pa_compute


class EurocDatasetSample(TypedDict):
    """Euroc dataset sample."""

    timestamp: float
    stereo: tuple[np.ndarray, np.ndarray]
    gyro: tuple[float, float, float]
    acc: tuple[float, float, float]
    gt_position: tuple[float, float, float]
    gt_orientation: tuple[float, float, float, float]
    gt_velocity: tuple[float, float, float]
    gt_gyro_bias: tuple[float, float, float]
    gt_acc_bias: tuple[float, float, float]


class GroundTruth(TypedDict):
    """Ground truth."""

    timestamp: float
    gt_position: tuple[float, float, float]
    gt_orientation: tuple[float, float, float, float]
    gt_velocity: tuple[float, float, float]
    gt_gyro_bias: tuple[float, float, float]
    gt_acc_bias: tuple[float, float, float]


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


@dataclass
class EurocDataPaths:
    """Euroc data paths."""

    cam0: Path
    cam1: Path
    imu0: Path
    gth0: Path
    cache: Path


class EurocDataset:
    """Euroc dataset."""

    config: EurocConfig
    logger = log.bind(app="euroc_dataset")
    _static_logger = log.bind(app="euroc_dataset_static")
    ground_truth_ds: Dataset

    def __init__(
        self, dataset: Dataset, config: dict[str, CameraSensor | IMUSensor], data_paths: EurocDataPaths
    ) -> None:
        """Initialize the Euroc dataset."""
        self.ds = dataset
        self.data_paths = data_paths
        self.config = EurocConfig(
            cast("CameraSensor", config["cam0"]),
            cast("CameraSensor", config["cam1"]),
            cast("IMUSensor", config["imu0"]),
        )
        self.ground_truth_map: dict[float, GroundTruth] = {}
        self._create_and_save_ground_truth_map()
        self.ground_truth_sorted_timestamps = sorted(self.ground_truth_map.keys())

    def _create_and_save_ground_truth_map(self) -> None:
        """Create and save the ground truth map to avoid mapping the ground truth dataset from disk every time."""
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
                pickle.dump(self.ground_truth_map, f)

    def map(self, func: Callable[[Dataset], Dataset]) -> "EurocDataset":
        """Map a function over the Euroc dataset."""
        return EurocDataset(
            func(self.ds),
            {
                "cam0": self.config.cam0,
                "cam1": self.config.cam1,
                "imu0": self.config.imu0,
            },
            self.data_paths,
        )

    def ground_truth(self) -> Dataset:
        """Get the ground truth dataset."""
        ds = self.ds.remove_columns(
            [
                "gyro",
                "acc",
                "stereo",
            ]
        )
        return ds.filter(lambda x: x["gt_position"][0] is not None)

    def imu(self) -> Dataset:
        """Get the imu dataset."""
        gt_ds = self.ground_truth()
        first_gt = gt_ds[0]
        first_gt_timestamp = first_gt["timestamp"]
        ds = self.ds.remove_columns(
            [
                "stereo",
                "gt_position",
                "gt_orientation",
                "gt_velocity",
                "gt_gyro_bias",
                "gt_acc_bias",
            ]
        )
        return ds.filter(lambda x: x["gyro"][0] is not None).filter(lambda x: x["timestamp"] > first_gt_timestamp)

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

    def imu_and_ground_truth(self) -> Dataset:
        """Get the imu and ground truth dataset."""
        gt_ds = self.ground_truth()
        first_gt = gt_ds[0]
        ds = self.ds.remove_columns(
            [
                "stereo",
            ]
        )
        return ds.filter(lambda x: x["gyro"][0] is not None).filter(
            lambda x: x["timestamp"] > first_gt["timestamp"]
        )

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

    def all(self) -> Dataset:
        """Get the all dataset."""
        return self.ds

    def iterate_stereo(self) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
        """Iterate over the Euroc dataset with only stereo images."""
        ds = self.ds.filter(lambda x: x["stereo"][0] is not None)
        ds = ds.remove_columns(
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
        ds = ds.with_format("numpy")
        iterable = ds.to_iterable_dataset()
        for sample in iterable:
            ts = float(sample["timestamp"])
            left = np.array(sample["stereo"][0])
            right = np.array(sample["stereo"][1])
            yield ts, left, right

    def first_ground_truth(self) -> GroundTruth:
        """Get the first ground truth."""
        ds = self.ground_truth()
        return cast("GroundTruth", ds[0])

    def find_nearest_ground_truth_by_timestamp(self, timestamp: float) -> GroundTruth:
        """Find the nearest ground truth by timestamp."""
        gth = self.ground_truth_map.get(timestamp)
        if gth is not None:
            return gth
        pos = bisect.bisect_left(self.ground_truth_sorted_timestamps, timestamp)
        candidates = []
        if pos > 0:
            candidates.append(self.ground_truth_sorted_timestamps[pos - 1])
        if pos < len(self.ground_truth_sorted_timestamps):
            candidates.append(self.ground_truth_sorted_timestamps[pos])
        if not candidates:
            msg = f"No ground truth found for timestamp {timestamp}"
            raise ValueError(msg)

        def distance(ts: float) -> float:
            return abs(ts - timestamp)

        closest_ts = min(candidates, key=distance)
        return self.ground_truth_map[closest_ts]

    def find_nearest_ground_truth_by_timestamp_se3(self, timestamp: float) -> SE3:
        """Find the nearest ground truth by timestamp and return the SE3 transform."""
        gth = self.find_nearest_ground_truth_by_timestamp(timestamp)
        if gth is None:
            msg = f"No ground truth found for timestamp {timestamp}"
            raise ValueError(msg)
        return SE3.from_quat_and_translation(np.array(gth["gt_orientation"]), np.array(gth["gt_position"]))

    @classmethod
    def from_name(cls, name: str = "euroc_mh_01", *, repo_root: Path | None = None) -> Self:
        """Load an EuRoC dataset by manifest name."""
        repo_root = (repo_root or Path(__file__).parent.parent.parent).resolve()

        manifest_loader = DatasetManifestLoader(repo_root=repo_root)
        resolved = manifest_loader.resolve(name)

        hf_dataset = DatasetLoader(repo_root=repo_root).load_manifest(resolved.dataset)

        return cls.from_manifest(
            manifest=resolved.dataset,
            rig=resolved.rig,
            dataset=hf_dataset,
            repo_root=repo_root,
        )

    @classmethod
    def from_manifest(
        cls,
        *,
        manifest: DatasetManifest,
        rig: DatasetRigConfig,
        dataset: Dataset,
        repo_root: Path,
    ) -> Self:
        """Wrap a materialized HuggingFace dataset with EuRoC config and data paths."""
        data_paths = cls.data_paths_from_manifest(manifest, repo_root=repo_root)
        cam0_config = cast(
            "CameraConfigOptions",
            rig.cam0.model_dump(by_alias=True, exclude={"rate_hz"}),
        )
        cam1_config = cast(
            "CameraConfigOptions",
            rig.cam1.model_dump(by_alias=True, exclude={"rate_hz"}),
        )
        imu0_config = cast("IMUConfigOptions", rig.imu0.model_dump(by_alias=True))

        return cls(
            dataset,
            {
                "cam0": CameraSensor(cam0_config),
                "cam1": CameraSensor(cam1_config),
                "imu0": IMUSensor(imu0_config),
            },
            data_paths,
        )

    @staticmethod
    def data_paths_from_manifest(manifest: DatasetManifest, *, repo_root: Path) -> EurocDataPaths:
        """Resolve an EuRoC manifest into legacy data paths used by this wrapper."""
        if manifest.streams.ground_truth is None:
            msg = f"EuRoC dataset manifest '{manifest.name}' must define streams.ground_truth"
            raise ValueError(msg)

        root = EurocDataset._resolve_repo_path(manifest.root, repo_root)
        cache = (
            EurocDataset._resolve_repo_path(manifest.cache, repo_root)
            if manifest.cache is not None
            else root / "cache"
        )
        return EurocDataPaths(
            cam0=EurocDataset._resolve_stream_path(root, manifest.streams.cam0),
            cam1=EurocDataset._resolve_stream_path(root, manifest.streams.cam1),
            imu0=EurocDataset._resolve_stream_path(root, manifest.streams.imu0),
            gth0=EurocDataset._resolve_stream_path(root, manifest.streams.ground_truth),
            cache=cache,
        )

    @staticmethod
    def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    @staticmethod
    def _resolve_stream_path(root: Path, path: Path) -> Path:
        return path if path.is_absolute() else root / path

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

    def feat_db_iterate(
        self,
    ) -> Iterator[tuple[int, float, dict[int, tuple[tuple[float, float], tuple[float, float] | None]]]]:
        """Get the feature database dataset."""

        def is_valid_scalar(value: float) -> bool:
            return value is not None and not np.isnan(value)

        feat_ds = cast("Dataset", load_from_disk(self.data_paths.cache / "feat_db"))
        feat_ds = feat_ds.to_iterable_dataset()
        for item in feat_ds:
            frame_id = item["frame_id"]
            timestamp = item["timestamp"]
            feat_ids = item["feat_ids"]
            ul = item["uL"]
            vl = item["vL"]
            ur = item["uR"]
            vr = item["vR"]
            feat_in_frame = {}
            for index, feat_id in enumerate(feat_ids):
                uv_left = (ul[index], vl[index])
                uv_right = (ur[index], vr[index]) if is_valid_scalar(ur[index]) else None
                feat_in_frame[feat_id] = (uv_left, uv_right)
            yield frame_id, timestamp, feat_in_frame

    def load_stereo_by_ts(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        """Load the stereo image by timestamp."""
        left_cam_path = self.data_paths.cam0.parent / "data" / f"{timestamp:.0f}.png"
        right_cam_path = self.data_paths.cam1.parent / "data" / f"{timestamp:.0f}.png"
        left_cam = cv2.imread(left_cam_path, cv2.IMREAD_GRAYSCALE)
        right_cam = cv2.imread(right_cam_path, cv2.IMREAD_GRAYSCALE)
        return np.array(left_cam), np.array(right_cam)
