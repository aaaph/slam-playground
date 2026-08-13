from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import NamedTuple, Protocol, cast, overload, runtime_checkable

import gtsam
import gtsam_unstable
import numpy as np
from numpy.typing import NDArray

from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3

type StereoFrame = NDArray[np.float32]
type LandmarkFrame = NDArray[np.float64]
type ImuBatch = NDArray[np.float32]  # [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
type OptimizedPose = SE3
type FeatureId = int

X = gtsam.symbol_shorthand.X


@runtime_checkable
class SmartStereoProjectionPoseFactor(Protocol):
    """Typed surface of the smart stereo factor missing from gtsam_unstable stubs."""

    def add(self, measurement: gtsam.StereoPoint2, pose_key: int, calibration: gtsam.Cal3_S2Stereo) -> None:
        """Add a stereo observation associated with a pose."""
        ...

    @overload
    def point(self) -> gtsam.TriangulationResult: ...

    @overload
    def point(self, values: gtsam.Values) -> gtsam.TriangulationResult: ...

    def keys(self) -> list[int]:
        """Return factor keys."""
        ...

    def isValid(self) -> bool:  # noqa: N802
        """Return whether the factor can triangulate a valid point."""
        ...

    def print(self, prefix: str) -> None:
        """Print the factor with a text prefix."""
        ...


_SmartStereoFactorFactory = Callable[..., SmartStereoProjectionPoseFactor]
_smart_stereo_factor_factory = cast(
    "_SmartStereoFactorFactory",
    vars(gtsam_unstable)["SmartStereoProjectionPoseFactor"],
)


def create_smart_stereo_projection_pose_factor(
    shared_noise_model: gtsam.noiseModel.Base,
    params: gtsam.SmartProjectionParams,
    body_p_sensor: gtsam.Pose3,
) -> SmartStereoProjectionPoseFactor:
    """Create the smart stereo factor exposed dynamically by gtsam_unstable."""
    return _smart_stereo_factor_factory(
        sharedNoiseModel=shared_noise_model,
        params=params,
        body_P_sensor=body_p_sensor,
    )


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
    landmark_frame: LandmarkFrame


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

    PNP = auto()
    PIM_RAMP_IN = auto()
    PIM = auto()


class VioKeyframe(NamedTuple):
    """VIO keyframe."""

    keyframe_id: int
    select_reason: list[SelectReason]
    timestamp: float
    stereo_frame: StereoFrame
    imu_batch: NDArray[np.float64]  # [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt]

    prediction_mode: PredictionMode
    pose_guess: SE3
    velocity_guess: NDArray[np.float32]  # [vx, vy, vz]
    bias_guess: NDArray[np.float32]  # [ba_x, ba_y, ba_z, bg_x, bg_y, bg_z]
    zupt: bool = False
