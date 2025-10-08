from typing import Literal, TypedDict


class TransformMatrix(TypedDict, total=False):
    """Typed dictionary for transform matrix."""

    cols: int
    rows: int
    data: list[float]


IMUConfigOptionsKeys = Literal[
    "T_BS",
    "gyroscope_noise_density",
    "gyroscope_random_walk",
    "accelerometer_noise_density",
    "accelerometer_random_walk",
]


class IMUConfigOptions(TypedDict, total=False):
    """Typed dictionary for IMU configuration."""

    # Sensor extrinsics wrt. the body-frame.
    T_BS: TransformMatrix

    # inertial sensor noise model parameters (static)
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float


CameraConfigOptionsKeys = Literal[
    "resolution",
    "camera_model",
    "intrinsics",
    "distortion_model",
    "distortion_coefficients",
    "T_BS",
]


class CameraConfigOptions(TypedDict, total=False):
    """Typed dictionary for camera configuration."""

    resolution: tuple[int, int]
    camera_model: Literal["pinhole"]
    intrinsics: tuple[float, float, float, float]  # fu, fv, cu, cv
    distortion_model: Literal["radial-tangential"]
    distortion_coefficients: tuple[float, float, float, float]
    T_BS: TransformMatrix


class StereoConfigOptions(TypedDict, total=False):
    """Typed dictionary for stereo configuration."""
