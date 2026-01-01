from dataclasses import dataclass

import numpy as np

from core.feature_tracker.feature import Feature
from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3


@dataclass
class Keyframe:
    """Front-End keyframe."""

    keyframe_id: int

    select_reason: SelectReason
    timestamp: float
    pose: SE3
    active_features: dict[int, Feature]
    active_landmarks: dict[int, np.ndarray]
