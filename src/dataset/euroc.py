from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import jax
import pandas as pd

from dataset.dataset_config import CameraConfig, IMUConfig
from datasets import Dataset, Image, Sequence, Value, load_from_disk
from logger import log


class EurocDatasetSample(TypedDict):
    """Euroc dataset sample."""

    timestamp: jax.Array
    stereo: tuple[jax.Array, jax.Array]
    gyro: tuple[jax.Array, jax.Array, jax.Array]
    acc: tuple[jax.Array, jax.Array, jax.Array]
    gt_position: tuple[jax.Array, jax.Array, jax.Array]
    gt_orientation: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    gt_velocity: tuple[jax.Array, jax.Array, jax.Array]
    gt_gyro_bias: tuple[jax.Array, jax.Array, jax.Array]
    gt_acc_bias: tuple[jax.Array, jax.Array, jax.Array]


@dataclass
class EurocConfig:
    """Euroc configuration."""

    cam0: CameraConfig
    cam1: CameraConfig
    imu0: IMUConfig


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

    def __init__(self, dataset: Dataset, config: dict[str, CameraConfig | IMUConfig]) -> None:
        """Initialize the Euroc dataset."""
        self.ds = dataset
        self.config = EurocConfig(
            cast("CameraConfig", config["cam0"]),
            cast("CameraConfig", config["cam1"]),
            cast("IMUConfig", config["imu0"]),
        )

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
        return ds.filter(lambda x: x["gyro"][0] is not None)

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
        return ds.filter(lambda x: x["stereo"][0] is not None)

    def iterate_all(self) -> Iterator[EurocDatasetSample]:
        """Iterate over the Euroc dataset."""
        self.ds = self.ds.with_format("jax")
        iterable = self.ds.to_iterable_dataset()
        yield from iterable

    def iterate_stereo(self) -> Iterator[tuple[float, jax.Array, jax.Array]]:
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
        ds = ds.with_format("jax")
        iterable = ds.to_iterable_dataset()
        for sample in iterable:
            ts = jax.numpy.array(sample["timestamp"]).astype(jax.numpy.float32)
            left = jax.numpy.array(sample["stereo"][0])
            right = jax.numpy.array(sample["stereo"][1])
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
        euroc["timestamp"] = euroc["timestamp"].astype("float64")
        ds = Dataset.from_pandas(euroc)
        new_features = ds.features.copy()
        new_features["stereo"] = Sequence(Image(), 2)
        new_features["timestamp"] = Value("float64")

        ds = ds.cast(new_features)

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

    @staticmethod
    def mh_01_easy() -> "EurocDataset":
        """Load the MH_01_easy dataset."""
        # Get the project root directory (assuming this file is in src/dataset/)
        project_root = Path(__file__).parent.parent.parent
        datasets_dir = project_root / "datasets" / "euroc_v_01_easy"

        dataset = EurocDataset.load_and_cache(
            EurocDataPaths(
                cam0=datasets_dir / "cam0" / "data.csv",
                cam1=datasets_dir / "cam1" / "data.csv",
                imu0=datasets_dir / "imu0" / "data.csv",
                gth0=datasets_dir / "state_groundtruth_estimate0" / "data.csv",
                cache=datasets_dir / "cache",
            )
        )

        return EurocDataset(
            dataset,
            {
                "cam0": CameraConfig.from_yaml(str(datasets_dir / "cam0" / "sensor.yaml")),
                "cam1": CameraConfig.from_yaml(str(datasets_dir / "cam1" / "sensor.yaml")),
                "imu0": IMUConfig.from_yaml(str(datasets_dir / "imu0" / "sensor.yaml")),
            },
        )
