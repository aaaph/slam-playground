from dataclasses import dataclass

import numpy as np

from core.feature_tracker.feature import Feature
from core.front_end.keyframe import Keyframe
from core.transformations.special_euclidian_3_dim import SE3


@dataclass
class FrontendResult:
    """
    The result of the frame processing by the SLAM frontend.

    Contains all data to send to the optimizer and visualizer.
    """

    result_id: int
    timestamp: float

    camera_in_world_se3: SE3
    new_landmarks: dict[int, np.ndarray]
    active_features: dict[int, Feature]
    left: np.ndarray
    right: np.ndarray

    keyframe: Keyframe | None = None
