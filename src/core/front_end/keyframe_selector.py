from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.feature_tracker.feature import Feature
from core.transformations.special_euclidian_3_dim import SE3

Timestamp = float
ActiveFeatures = dict[int, Feature]
SelectReason = Literal["initial", "big_distance", "big_angle", "low_connectivity"]


@dataclass
class KeyFrameSelectThresholds:
    """Keyframe selection thresholds."""

    min_distance: float = 0.1
    min_angle: float = 10.0
    max_time_delta: float = 0.2
    min_connectivity_ratio: float = 0.5


class KeyframeSelector:
    """Keyframe selector."""

    def __init__(self, thresholds: KeyFrameSelectThresholds) -> None:
        """Initialize the keyframe selector."""
        self.thresholds = thresholds
        self.last_keyframe_ts: Timestamp = -1.0
        self.pose_history: OrderedDict[Timestamp, SE3] = OrderedDict()
        self.last_feat_ids: set[int] = set()

    @classmethod
    def default_factory(cls) -> "KeyframeSelector":
        """Create a default `KeyframeSelector`."""
        return cls(thresholds=KeyFrameSelectThresholds())

    def check(self, current_pose: SE3, feat_ids: set[int]) -> tuple[bool, SelectReason | None]:
        """Check if a keyframe should be selected."""
        if self.last_keyframe_ts == -1.0:
            return True, "initial"
        prev_pose = self.pose_history.get(self.last_keyframe_ts)
        pose_diff = prev_pose.inverse() * current_pose
        distance = np.linalg.norm(pose_diff.translation())
        trace_val = np.trace(pose_diff.rotation().as_matrix())
        trace_val = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = np.arccos(trace_val)
        angle_deg = np.rad2deg(angle_rad)

        # Cast NumPy comparison results to built-in bools for type-checkers and consumers.
        too_big_distance: bool = bool(distance > self.thresholds.min_distance)
        if too_big_distance:
            return True, "big_distance"

        too_big_angle: bool = bool(angle_deg > self.thresholds.min_angle)
        if too_big_angle:
            return True, "big_angle"

        common_feat_ids = feat_ids.intersection(self.last_feat_ids)
        common_feat_count = len(common_feat_ids)
        last_feat_count = len(self.last_feat_ids)
        connectivity_ratio = common_feat_count / last_feat_count if last_feat_count > 0 else 0.0
        too_low_connectivity = connectivity_ratio < self.thresholds.min_connectivity_ratio
        if too_low_connectivity:
            return True, "low_connectivity"

        return False, None

    def update(self, ts: Timestamp, current_pose: SE3, feat_ids: set[int]) -> None:
        """Update the keyframe selector."""
        self.pose_history[ts] = current_pose
        self.last_feat_ids = feat_ids
        self.last_keyframe_ts = ts
