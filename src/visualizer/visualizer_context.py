from dataclasses import dataclass, field

import numpy as np

from core.front_end.keyframe import Keyframe
from core.transformations.special_euclidian_3_dim import SE3


@dataclass(frozen=True)
class VisualizerContext:
    """Visualizer context message DTO."""

    timestamp: float | None = None
    body_in_world_se3: SE3 | None = None
    pointcloud: dict[int, np.ndarray] | None = None
    active_feat_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    optimized_pose_se3: SE3 | None = None
    ground_truth_se3: SE3 | None = None
    frame: np.ndarray | None = None
    selected_keyframes: list[Keyframe] | None = None
    debug_features: list[int] | None = None
    pose_history: list[SE3] | None = None
    errors: np.ndarray | None = None
