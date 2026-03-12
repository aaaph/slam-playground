from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3

type ActiveTrack = NDArray[np.float32]  # [feat_id, left_u, left_v, right_u, right_v]
type OptimizedPose = SE3
type FeatureId = int


class StereoMeasurement(NamedTuple):
    """Stereo measurement."""

    pose_key: int
    ul: float
    ur: float
    v: float


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
