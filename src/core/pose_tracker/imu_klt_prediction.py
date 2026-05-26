from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

NANOSECONDS_IN_SECOND = 1e-9
MIN_RAY_DEPTH_ABS = 1e-12
POINTS_NDIM = 2
POINT2_WIDTH = 2
CAMERA_MATRIX_SHAPE = (3, 3)


class GyroKltPredictionContext(NamedTuple):
    """Input context for gyro-based temporal KLT prediction."""

    gyro_batch: NDArray[np.float32]
    imu_ts_batch: NDArray[np.float64]
    start_ts: float
    k_matrix: NDArray[np.float64]
    camera_in_body_rotation: Rotation
    gyro_bias: NDArray[np.float64] | None = None


def integrate_gyro_delta_rotation(
    gyro_batch: NDArray[np.float32],
    imu_ts_batch: NDArray[np.float64],
    start_ts: float,
    gyro_bias: NDArray[np.float64] | None = None,
) -> Rotation:
    """Integrate a gyro batch into body-frame delta rotation."""
    gyro_batch = np.asarray(gyro_batch, dtype=np.float64)
    imu_ts_batch = np.asarray(imu_ts_batch, dtype=np.float64)
    if gyro_batch.shape[0] != imu_ts_batch.shape[0]:
        msg = "gyro_batch and imu_ts_batch must have the same number of rows"
        raise ValueError(msg)
    if gyro_batch.shape[0] == 0:
        return Rotation.identity()

    bias = np.zeros(3, dtype=np.float64) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)
    if bias.shape != (3,):
        msg = "gyro_bias must have shape (3,)"
        raise ValueError(msg)

    delta_rotation = Rotation.identity()
    previous_ts = float(start_ts)
    for gyro, timestamp in zip(gyro_batch, imu_ts_batch, strict=True):
        if not np.isfinite(timestamp) or not np.all(np.isfinite(gyro)):
            continue

        dt_sec = (float(timestamp) - previous_ts) * NANOSECONDS_IN_SECOND
        previous_ts = float(timestamp)
        if dt_sec <= 0:
            continue

        delta_rotation *= Rotation.from_rotvec((gyro - bias) * dt_sec)

    return delta_rotation


def predict_points_by_gyro(
    points: NDArray[np.float32],
    context: GyroKltPredictionContext,
) -> NDArray[np.float32]:
    """Predict next-frame image points from gyro-only camera rotation."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != POINTS_NDIM or points.shape[1] != POINT2_WIDTH:
        msg = "points must have shape (N, 2)"
        raise ValueError(msg)
    if points.shape[0] == 0:
        return np.empty((0, POINT2_WIDTH), dtype=np.float32)

    k_matrix = np.asarray(context.k_matrix, dtype=np.float64)
    if k_matrix.shape != CAMERA_MATRIX_SHAPE:
        msg = "k_matrix must have shape (3, 3)"
        raise ValueError(msg)

    delta_body = integrate_gyro_delta_rotation(
        gyro_batch=context.gyro_batch,
        imu_ts_batch=context.imu_ts_batch,
        start_ts=context.start_ts,
        gyro_bias=context.gyro_bias,
    )
    camera_next_from_prev = (
        context.camera_in_body_rotation.inv() * delta_body.inv() * context.camera_in_body_rotation
    )

    hom_points = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    rays_prev = (np.linalg.inv(k_matrix) @ hom_points.T).T
    rays_next = camera_next_from_prev.apply(rays_prev)

    prediction = np.full(points.shape, np.nan, dtype=np.float64)
    finite_depth_mask = np.isfinite(rays_next).all(axis=1) & (np.abs(rays_next[:, 2]) > MIN_RAY_DEPTH_ABS)
    if np.any(finite_depth_mask):
        projected = (k_matrix @ rays_next[finite_depth_mask].T).T
        prediction[finite_depth_mask] = projected[:, :2] / projected[:, 2:3]

    return prediction.astype(np.float32)
