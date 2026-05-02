from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

import gtsam
from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3

type ActiveTrack = NDArray[np.float32]  # [feat_id, left_u, left_v, right_u, right_v]
type ImuBatch = NDArray[np.float32]  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
type OptimizedPose = SE3
type FeatureId = int

X = gtsam.symbol_shorthand.X


class StereoMeasurement(NamedTuple):
    """Stereo measurement."""

    kfid: int
    ul: float
    ur: float
    v: float

    @property
    def pose_key(self) -> int:
        """Get the pose key."""
        return X(self.kfid)


class FeatureStatus(IntEnum):
    """Feature status."""

    EMPTY = 0
    SMART_FACTOR = 1
    MARGNILIZED = 2
    SMART_TO_EXPLICIT = 3
    SMART_TO_MARGNILIZED = 4
    EXPLICIT_LANDMARK = 5


class OptKeyframe(NamedTuple):
    """Optimized keyframe."""

    keyframe_id: int
    select_reason: SelectReason
    timestamp: float
    pose: SE3
    active_track: ActiveTrack


@dataclass(slots=True)
class FeatureTrack:
    """Feature track."""

    feat_id: FeatureId
    status: FeatureStatus = FeatureStatus.EMPTY
    history: deque[StereoMeasurement] = field(default_factory=lambda: deque(maxlen=25))
    slot: int = -1
    cached_point: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan, np.nan, np.nan]))


class FactorType(IntEnum):
    """Factor slot type."""

    SMART_FACTOR = 1
    LANDMARK = 2
    BETWEEN_FACTOR = 3
    PRIOR_FACTOR = 4
    IMU_FACTOR = 5


class PredictionMode(IntEnum):
    """Pose guess source."""

    BOOTSTRAP = auto()
    PNP = auto()
    IMU_PREDICTION = auto()


class VioKeyframe(NamedTuple):
    """VIO keyframe."""

    keyframe_id: int
    select_reason: list[SelectReason]
    timestamp: float
    # Nx12 (feat_id, timestamp, left_u, left_v, right_u, right_v, state, age, stereo_score, x, y, z)
    active_track: NDArray[np.float32]
    imu_batch: NDArray[np.float64]  # [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt]

    prediction_mode: PredictionMode
    pose_guess: SE3 | None = None
    velocity_guess: NDArray[np.float32] | None = None  # [vx, vy, vz]
    bias_guess: NDArray[np.float32] | None = None  # [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
    zupt: bool = False
