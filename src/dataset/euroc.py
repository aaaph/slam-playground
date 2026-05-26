import bisect
import pickle
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import cv2
import numpy as np
import pandas as pd
from pyarrow import compute as pa_compute
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import ImuContext, VioContext
from core.transformations.special_euclidian_3_dim import SE3
from dataset.dataset_config import CameraConfig, IMUConfig
from datasets import Array2D, Dataset, Image, Sequence, Value, load_from_disk
from logger import log

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

    cam0: CameraConfig
    cam1: CameraConfig
    imu0: IMUConfig

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
        self, dataset: Dataset, config: dict[str, CameraConfig | IMUConfig], data_paths: EurocDataPaths
    ) -> None:
        """Initialize the Euroc dataset."""
        self.ds = dataset
        self.data_paths = data_paths
        self.config = EurocConfig(
            cast("CameraConfig", config["cam0"]),
            cast("CameraConfig", config["cam1"]),
            cast("IMUConfig", config["imu0"]),
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
        # Get the project root directory (assuming this file is in src/dataset/)
        project_root = Path(__file__).parent.parent.parent
        datasets_dir = project_root / "datasets" / "euroc_v_01_easy"

        return EurocDataset(
            func(self.ds),
            {
                "cam0": CameraConfig.from_yaml(str(datasets_dir / "cam0" / "sensor.yaml")),
                "cam1": CameraConfig.from_yaml(str(datasets_dir / "cam1" / "sensor.yaml")),
                "imu0": IMUConfig.from_yaml(str(datasets_dir / "imu0" / "sensor.yaml")),
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

    @staticmethod
    def pandas_to_dataset(euroc: pd.DataFrame) -> Dataset:
        """Convert a pandas dataframe to a dataset."""
        euroc["stereo"] = euroc[["left_image", "right_image"]].apply(
            lambda row: [row.left_image, row.right_image], axis=1
        )
        euroc["gyro"] = euroc[["gyro_x", "gyro_y", "gyro_z"]].apply(
            lambda row: [row.gyro_x, row.gyro_y, row.gyro_z], axis=1
        )
        euroc["acc"] = euroc[["acc_x", "acc_y", "acc_z"]].apply(
            lambda row: [row.acc_x, row.acc_y, row.acc_z], axis=1
        )
        euroc["gt_position"] = euroc[["gt_position_x", "gt_position_y", "gt_position_z"]].apply(
            lambda row: [row.gt_position_x, row.gt_position_y, row.gt_position_z], axis=1
        )
        euroc["gt_orientation"] = euroc[["gt_q_w", "gt_q_x", "gt_q_y", "gt_q_z"]].apply(
            lambda row: [row.gt_q_w, row.gt_q_x, row.gt_q_y, row.gt_q_z], axis=1
        )
        euroc["gt_velocity"] = euroc[["gt_velocity_x", "gt_velocity_y", "gt_velocity_z"]].apply(
            lambda row: [row.gt_velocity_x, row.gt_velocity_y, row.gt_velocity_z], axis=1
        )
        euroc["gt_gyro_bias"] = euroc[["gt_gyro_bias_x", "gt_gyro_bias_y", "gt_gyro_bias_z"]].apply(
            lambda row: [row.gt_gyro_bias_x, row.gt_gyro_bias_y, row.gt_gyro_bias_z],
            axis=1,
        )
        euroc["gt_acc_bias"] = euroc[["gt_acc_bias_x", "gt_acc_bias_y", "gt_acc_bias_z"]].apply(
            lambda row: [row.gt_acc_bias_x, row.gt_acc_bias_y, row.gt_acc_bias_z], axis=1
        )
        euroc = euroc.drop(
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
        # Convert timestamp to float64 before creating dataset
        euroc["timestamp"] = euroc["timestamp"].astype("int64")
        ds = Dataset.from_pandas(euroc)
        new_features = ds.features.copy()
        new_features["stereo"] = Sequence(Image(), 2)
        new_features["timestamp"] = Value("int64")

        ds = ds.cast(new_features)
        ds = ds.map(lambda x: {**x, "has_imu": x["gyro"][0] is not None})
        ds = ds.map(lambda x: {**x, "has_ground_truth": x["gt_position"][0] is not None})

        return ds.map(
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

    @staticmethod
    def _try_to_load_from_disk(data_paths: EurocDataPaths) -> Dataset | None:
        path = data_paths.cache / "full"
        try:
            EurocDataset._static_logger.info(f"Loading dataset from disk at {path}")
            return cast("Dataset", load_from_disk(path))
        except FileNotFoundError:
            return None

    @staticmethod
    def _try_to_save_to_disk(data_paths: EurocDataPaths, ds: Dataset) -> None:
        path = Path(str(data_paths.cache) + "/full")
        ds.save_to_disk(path)

    @staticmethod
    def load_and_cache(data_paths: EurocDataPaths) -> Dataset:
        """Load and cache the Euroc dataset."""
        ds = EurocDataset._try_to_load_from_disk(data_paths)

        if ds is not None:
            return ds

        EurocDataset._static_logger.info("Dataset not found on disk, creating from scratch...")
        EurocDataset._static_logger.info("Creating panda dataframe from csv files...")
        EurocDataset._static_logger.info("Loading left camera dataframe...", extra={"path": data_paths.cam0})
        left_cam_df = pd.read_csv(data_paths.cam0)
        left_cam_prefix = data_paths.cam0.parent / "data"
        left_cam_df = left_cam_df.rename(columns={"#timestamp [ns]": "timestamp", "filename": "left_image"})
        left_cam_df["left_image"] = left_cam_df["left_image"].map(lambda x: f"{left_cam_prefix}/{x}")

        EurocDataset._static_logger.info("Loading right camera dataframe...", extra={"path": data_paths.cam1})
        right_cam_df = pd.read_csv(data_paths.cam1)
        right_cam_prefix = data_paths.cam1.parent / "data"
        right_cam_df = right_cam_df.rename(columns={"#timestamp [ns]": "timestamp", "filename": "right_image"})
        right_cam_df["right_image"] = right_cam_df["right_image"].map(lambda x: f"{right_cam_prefix}/{x}")

        EurocDataset._static_logger.info("Loading imu dataframe...", extra={"path": data_paths.imu0})
        imu_df = pd.read_csv(data_paths.imu0)
        imu_df = imu_df.rename(
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

        EurocDataset._static_logger.info("Loading ground truth dataframe...", extra={"path": data_paths.gth0})
        gth_df = pd.read_csv(data_paths.gth0)
        gth_df = gth_df.rename(
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

        EurocDataset._static_logger.info("Creating dataset from dataframe...")
        cam_df = left_cam_df.merge(right_cam_df, on="timestamp", how="inner")
        imu_cam_df = cam_df.merge(imu_df, on="timestamp", how="outer")
        full_df = imu_cam_df.merge(gth_df, on="timestamp", how="outer")
        ds = EurocDataset.pandas_to_dataset(full_df)
        EurocDataset._try_to_save_to_disk(data_paths, ds)
        return ds

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

    @staticmethod
    def mh_01_easy() -> "EurocDataset":
        """Load the MH_01_easy dataset."""
        # Get the project root directory (assuming this file is in src/dataset/)
        project_root = Path(__file__).parent.parent.parent
        datasets_dir = project_root / "datasets" / "euroc_v_01_easy"

        data_paths = EurocDataPaths(
            cam0=datasets_dir / "cam0" / "data.csv",
            cam1=datasets_dir / "cam1" / "data.csv",
            imu0=datasets_dir / "imu0" / "data.csv",
            gth0=datasets_dir / "state_groundtruth_estimate0" / "data.csv",
            cache=datasets_dir / "cache",
        )
        dataset = EurocDataset.load_and_cache(data_paths)

        return EurocDataset(
            dataset,
            {
                "cam0": CameraConfig.from_yaml(str(datasets_dir / "cam0" / "sensor.yaml")),
                "cam1": CameraConfig.from_yaml(str(datasets_dir / "cam1" / "sensor.yaml")),
                "imu0": IMUConfig.from_yaml(str(datasets_dir / "imu0" / "sensor.yaml")),
            },
            data_paths,
        )

    def feat_db_iterate(
        self,
    ) -> Iterator[tuple[int, float, dict[int, tuple[tuple[float, float], tuple[float, float] | None]]]]:
        """Get the feature database dataset."""

        def is_valid_scalar(value: float) -> bool:
            return value is not None and not np.isnan(value)

        path = self.data_paths.cache / "feat_db"
        feat_ds = load_from_disk(path)
        feat_ds = cast("Dataset", feat_ds)
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
                ul_val = ul[index]
                vl_val = vl[index]
                ur_val = ur[index]
                vr_val = vr[index]
                uv_left = (ul_val, vl_val)
                uv_right = (ur_val, vr_val) if is_valid_scalar(ur_val) else None
                feat_in_frame[feat_id] = (uv_left, uv_right)
            yield frame_id, timestamp, feat_in_frame

    def load_stereo_by_ts(self, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
        """Load the stereo image by timestamp."""
        left_cam_path = self.data_paths.cam0.parent / "data" / f"{timestamp:.0f}.png"
        right_cam_path = self.data_paths.cam1.parent / "data" / f"{timestamp:.0f}.png"
        left_cam = cv2.imread(left_cam_path, cv2.IMREAD_GRAYSCALE)
        right_cam = cv2.imread(right_cam_path, cv2.IMREAD_GRAYSCALE)
        return np.array(left_cam), np.array(right_cam)
