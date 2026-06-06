from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import cv2
import numpy as np
from pyarrow import compute as pa_compute
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import ImuContext, VioContext
from core.transformations.special_euclidian_3_dim import SE3
from dataset.builder import DatasetLoader
from dataset.manifest import DatasetRigConfig
from dataset.registry import DatasetRegistry
from dataset.sensor_config import CameraSensor, IMUSensor
from datasets import Array2D, Dataset, Image, Sequence, Value
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


class EurocDataset:
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

        hf_dataset = DatasetLoader(repo_root=repo_root).load_manifest(manifest)

        return cls.from_dataset(hf_dataset, rig)

    @classmethod
    def from_dataset(cls, dataset: Dataset, rig: DatasetRigConfig) -> Self:
        """Wrap a materialized HuggingFace dataset with EuRoC config and data paths."""
        euroc_config = EurocConfig(
            CameraSensor.from_rig_config(rig.cam0),
            CameraSensor.from_rig_config(rig.cam1),
            IMUSensor.from_rig_config(rig.imu0),
        )
        return cls(dataset, euroc_config)

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
