import time
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import ImuContext, VioContext
from core.front_end.keyframe import ActiveTrackSchema
from core.transformations.special_euclidian_3_dim import SE3
from dataset.dataset_config import CameraConfig


@pytest.fixture
def cam_config_0() -> CameraConfig:
    """Create a camera configuration."""
    return CameraConfig(
        {
            "resolution": (752, 480),
            "camera_model": "pinhole",
            "intrinsics": (458.654, 457.296, 367.215, 248.375),
            "distortion_model": "radial-tangential",
            "distortion_coefficients": (-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05),
            "T_BS": {
                "cols": 4,
                "rows": 4,
                "data": [
                    0.0148655429818,
                    -0.999880929698,
                    0.00414029679422,
                    -0.0216401454975,
                    0.999557249008,
                    0.0149672133247,
                    0.025715529948,
                    -0.064676986768,
                    -0.0257744366974,
                    0.00375618835797,
                    0.999660727178,
                    0.00981073058949,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        }
    )


@pytest.fixture
def cam_config_1() -> CameraConfig:
    """Create a camera configuration."""
    return CameraConfig(
        {
            "resolution": (752, 480),
            "camera_model": "pinhole",
            "intrinsics": (457.587, 456.134, 379.999, 255.238),
            "distortion_model": "radial-tangential",
            "distortion_coefficients": (-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05),
            "T_BS": {
                "cols": 4,
                "rows": 4,
                "data": [
                    0.0125552670891,
                    -0.999755099723,
                    0.0182237714554,
                    -0.0198435579556,
                    0.999598781151,
                    0.0130119051815,
                    0.0251588363115,
                    0.0453689425024,
                    -0.0253898008918,
                    0.0179005838253,
                    0.999517347078,
                    0.00786212447038,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
        }
    )


@pytest.fixture
def camera_model(cam_config_0: CameraConfig, cam_config_1: CameraConfig) -> StereoCameraModel:
    """Create a camera model."""
    return StereoCameraModel.from_cameras_config(cam_config_0, cam_config_1)


@pytest.fixture
def stereo_ctx() -> StereoContext:
    """Create a stereo context."""
    return StereoContext(
        resolution=(100, 100),
        stereo_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
        cam0_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
        cam1_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
        baseline=1.0,
        cam0_in_body_se3=SE3.identity(),
        cam1_in_body_se3=SE3.identity(),
    )


@pytest.fixture
def imu_ctx() -> ImuContext:
    """Create an IMU context."""
    return ImuContext(
        frequency=200.0,
        accel_noise_destiny=1.9393e-05,
        gyro_noise_destiny=1.6968e-04,
        accel_random_walk=3.0000e-3,
        gyro_random_walk=1.9393e-05,
        gravity=np.array([0, 0, -9.81]),
    )


@pytest.fixture
def vio_ctx(stereo_ctx: StereoContext, imu_ctx: ImuContext) -> VioContext:
    """Create a VIO context."""
    return VioContext(stereo_ctx, imu_ctx)


@pytest.fixture
def first_active_track() -> NDArray[np.float32]:
    """First active track."""
    payload = np.array(
        [
            [104.88, 151.02, 88.46, 151.18, -1.74, -0.71, 3.06],
            [108.85, 150.04, 92.44, 150.21, -1.71, -0.72, 3.06],
            [106.91, 155.06, 90.53, 155.20, -1.73, -0.68, 3.07],
            [75.90, 85.04, 58.38, 84.91, -1.81, -1.08, 2.87],
            [58.84, 33.07, 40.89, 33.06, -1.87, -1.37, 2.80],
            [74.83, 81.98, 57.30, 81.89, -1.82, -1.10, 2.87],
            [205.97, 80.98, 187.72, 80.91, -0.96, -1.06, 2.75],
            [260.92, 80.00, 243.72, 80.15, -0.66, -1.13, 2.92],
            [210.95, 82.97, 192.71, 82.91, -0.93, -1.05, 2.76],
            [197.87, 151.05, 179.66, 150.97, -1.01, -0.64, 2.76],
            [262.92, 78.00, 245.74, 78.07, -0.65, -1.15, 2.93],
            [247.76, 81.00, 230.06, 81.16, -0.73, -1.09, 2.84],
            [551.86, 86.98, 529.09, 86.85, 0.91, -0.82, 2.21],
            [552.85, 98.99, 530.02, 98.83, 0.91, -0.76, 2.20],
            [557.86, 92.00, 535.14, 91.88, 0.94, -0.80, 2.21],
            [551.87, 90.00, 529.11, 89.87, 0.91, -0.81, 2.21],
            [554.83, 97.01, 532.02, 96.85, 0.92, -0.77, 2.20],
            [554.86, 101.99, 532.08, 101.82, 0.92, -0.75, 2.21],
            [577.85, 120.01, 556.25, 119.95, 1.09, -0.70, 2.33],
            [566.86, 60.82, 543.96, 59.95, 0.97, -0.94, 2.19],
            [568.91, 48.90, 545.49, 48.51, 0.96, -0.98, 2.15],
            [571.95, 65.97, 548.97, 65.63, 0.99, -0.91, 2.19],
            [168.95, 316.01, 141.45, 315.81, -0.78, 0.24, 1.83],
            [161.95, 316.99, 134.64, 316.84, -0.82, 0.24, 1.84],
            [166.96, 315.99, 139.54, 315.81, -0.79, 0.24, 1.83],
            [185.89, 226.03, 167.70, 226.13, -1.08, -0.19, 2.76],
            [151.86, 315.00, 124.18, 314.87, -0.85, 0.23, 1.82],
            [371.92, 286.99, 348.51, 286.94, 0.03, 0.14, 2.15],
            [362.90, 276.99, 340.08, 276.86, -0.01, 0.09, 2.20],
            [374.90, 285.99, 351.58, 285.91, 0.05, 0.14, 2.15],
            [364.90, 280.99, 341.85, 280.91, 0.00, 0.11, 2.18],
            [372.89, 300.00, 347.56, 299.85, 0.04, 0.19, 1.98],
            [366.91, 280.98, 343.89, 280.90, 0.01, 0.11, 2.18],
            [394.89, 307.01, 369.15, 306.86, 0.13, 0.21, 1.95],
            [390.88, 309.02, 365.01, 308.91, 0.11, 0.22, 1.94],
            [394.91, 317.00, 368.39, 317.01, 0.13, 0.25, 1.90],
            [385.92, 316.99, 359.46, 317.03, 0.09, 0.25, 1.90],
            [383.92, 308.98, 357.90, 308.93, 0.08, 0.22, 1.93],
            [388.90, 298.00, 363.64, 297.77, 0.11, 0.18, 1.99],
            [613.91, 300.97, 589.34, 300.98, 1.12, 0.20, 2.05],
            [642.90, 309.97, 617.30, 309.81, 1.20, 0.23, 1.96],
            [644.92, 315.95, 619.16, 315.90, 1.20, 0.25, 1.95],
            [656.85, 314.96, 630.67, 314.83, 1.23, 0.24, 1.92],
            [626.91, 301.98, 602.26, 301.96, 1.17, 0.20, 2.04],
            [629.90, 309.97, 604.84, 309.97, 1.17, 0.23, 2.01],
            [153.88, 338.03, 124.25, 337.94, -0.78, 0.30, 1.70],
            [127.90, 341.02, 97.42, 340.99, -0.85, 0.30, 1.65],
            [143.86, 344.03, 114.08, 343.85, -0.82, 0.32, 1.69],
            [116.89, 358.02, 85.12, 357.95, -0.86, 0.35, 1.58],
            [188.89, 324.01, 161.10, 323.81, -0.70, 0.27, 1.81],
            [194.92, 334.03, 167.00, 333.90, -0.67, 0.30, 1.80],
            [187.92, 328.02, 160.01, 327.89, -0.70, 0.28, 1.80],
            [249.91, 348.00, 219.17, 347.92, -0.41, 0.33, 1.64],
            [250.94, 354.06, 220.11, 354.03, -0.41, 0.35, 1.63],
            [254.88, 350.99, 224.18, 350.94, -0.39, 0.34, 1.64],
            [422.91, 341.99, 393.82, 341.91, 0.22, 0.32, 1.73],
            [403.93, 335.98, 375.15, 335.94, 0.15, 0.30, 1.75],
            [678.82, 323.94, 651.70, 323.83, 1.28, 0.27, 1.85],
            [707.85, 342.96, 679.08, 342.90, 1.31, 0.33, 1.75],
            [678.90, 329.97, 651.58, 329.92, 1.27, 0.29, 1.84],
            [721.85, 342.94, 692.52, 342.72, 1.34, 0.32, 1.71],
            [458.89, 384.00, np.nan, np.nan, np.nan, np.nan, np.nan],
            [450.94, 378.99, np.nan, np.nan, np.nan, np.nan, np.nan],
            [511.94, 437.98, np.nan, np.nan, np.nan, np.nan, np.nan],
            [469.93, 402.99, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    data = np.full((payload.shape[0], ActiveTrackSchema.count()), np.nan, dtype=np.float32)
    data[:, ActiveTrackSchema.FEAT_ID] = np.arange(payload.shape[0])
    data[:, ActiveTrackSchema.TIMESTAMP] = time.time()
    data[:, ActiveTrackSchema.LEFT_U : ActiveTrackSchema.RIGHT_V + 1] = payload[:, 0:4]
    data[:, ActiveTrackSchema.AGE] = 10
    data[:, ActiveTrackSchema.STEREO_SCORE] = 10
    data[:, ActiveTrackSchema.X : ActiveTrackSchema.Z + 1] = payload[:, 4:7]
    is_unstable = np.isnan(payload[:, 4])
    data[:, ActiveTrackSchema.STATE] = 1.0
    data[is_unstable, ActiveTrackSchema.STATE] = 5.0
    return data.astype(np.float32)


@pytest.fixture
def active_track_x7() -> NDArray[np.float32]:
    """Active track x7."""
    csv_path = Path(__file__).with_name("active_track_kf_000007.csv")
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=np.float32, ndmin=2)
    if data.shape[1] == ActiveTrackSchema.count():
        return data
    if data.shape[1] != ActiveTrackSchema.count() - 1:
        msg = f"Unexpected active track width: {data.shape[1]}"
        raise ValueError(msg)

    expanded = np.full((data.shape[0], ActiveTrackSchema.count()), np.nan, dtype=np.float32)
    expanded[:, : ActiveTrackSchema.STEREO_SCORE] = data[:, : ActiveTrackSchema.STEREO_SCORE]
    expanded[:, ActiveTrackSchema.STEREO_SCORE] = 10
    expanded[:, ActiveTrackSchema.AGE] = 10
    expanded[:, ActiveTrackSchema.X : ActiveTrackSchema.Z + 1] = data[:, ActiveTrackSchema.STEREO_SCORE :]
    return expanded


@pytest.fixture
def state_x7() -> NDArray[np.float32]:
    """State x7."""
    payload = np.array(
        [0.8298639, -0.00889524, 0.5577379, 0.01323531, 0.0, 0.0, 0.0, 0.0, -0.00240924, 0.02035533, 0.07797173]
    )
    return payload.astype(np.float32)
