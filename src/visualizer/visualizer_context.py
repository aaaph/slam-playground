from dataclasses import dataclass, field

import numpy as np

from core.front_end.keyframe import Keyframe
from core.transformations.special_euclidian_3_dim import SE3


@dataclass(frozen=True)
class VisualizerContext:
    """Visualizer context message DTO."""

    body_in_world_se3: SE3 | None = None
    pointcloud: dict[int, np.ndarray] | None = None
    active_feat_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)

    ground_truth_se3: SE3 | None = None
    frame: np.ndarray | None = None
    selected_keyframes: list[Keyframe] | None = None
