from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from core.feature_tracker.feature import Feature
from core.transformations.special_euclidian_3_dim import SE3

Timestamp = float
ActiveFeatures = dict[int, Feature]
# SelectReason = Literal["initial", "big_distance", "big_angle", "low_connectivity"]


class SelectMetrics(TypedDict):
    """Select metrics."""

    keyframe_time_diff: float
    keyframe_median_parallax: float
    keyframe_connectivity_ratio: float
    keyframe_common_feat_count: int
    keyframe_distance_diff: float
    keyframe_angle_diff: float


class SelectReason(Enum):
    """Select reason."""

    INITIAL = auto()
    LOW_CONNECTIVITY = auto()
    PARALLAX = auto()
    TIME_ELAPSED = auto()
    TIME_IGNORED = auto()


@dataclass
class KeyFrameSelectThresholds:
    """Keyframe selection thresholds."""

    ignore_time_until_sec: float = (
        0.2  # the keyframe should not be selected if the delta is lower than this threshold
    )
    max_time_delta_sec: float = 5.0  # the keyframe should be selected if the delta is higher than this threshold
    min_connectivity_ratio: float = 0.4
    min_parallax_pts: int = 15  # avg parallax 15 pixels - threshold
    min_common_feat_for_parallax: int = 5


class KeyframeSelector:
    """Keyframe selector."""

    def __init__(self, thresholds: KeyFrameSelectThresholds, capacity: int = 500) -> None:
        """Initialize the keyframe selector."""
        self.capacity = capacity
        self.thresholds = thresholds
        self.keyframe_ts: Timestamp = -1.0
        self.keyframe_pose_window: deque[SE3] = deque(maxlen=5)
        self.keyframe_ids = np.full(capacity, -1, dtype=np.int32)
        self.keyframe_left_points = np.full((capacity, 2), np.nan, dtype=np.float32)
        self.keyframe_feat_count = 0

    @classmethod
    def default_factory(cls) -> "KeyframeSelector":
        """Create a default `KeyframeSelector`."""
        return cls(thresholds=KeyFrameSelectThresholds())

    def calc_selector_metrics(
        self, ts: Timestamp, current_pose: SE3, active_track: NDArray[np.float32]
    ) -> SelectMetrics:
        """Calculate the selector metrics."""
        prev_pose = self.keyframe_pose_window[-1]
        if prev_pose is None:
            raise KeyError("Previous pose is not found")
        common_feat_ids, idx_kf, idx_cur = np.intersect1d(
            self.keyframe_ids[: self.keyframe_feat_count], active_track[:, 0].astype(int), return_indices=True
        )
        common_feat_count = len(common_feat_ids)
        connectivity = common_feat_count / self.keyframe_feat_count if self.keyframe_feat_count > 0 else 0.0
        parallax = 0.0
        if common_feat_count >= self.thresholds.min_common_feat_for_parallax:
            diffs = active_track[idx_cur, 1:3] - self.keyframe_left_points[idx_kf]
            parallax = np.sqrt(np.median(np.sum(np.square(diffs), axis=1)))

        pose_diff = prev_pose.inverse() * current_pose
        distance = np.linalg.norm(pose_diff.translation())

        trace_val = np.trace(pose_diff.rotation().as_matrix())
        angle_deg = np.rad2deg(np.acos(np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)))

        return SelectMetrics(
            keyframe_time_diff=(ts - self.keyframe_ts) / 1e9,
            keyframe_median_parallax=parallax,
            keyframe_connectivity_ratio=connectivity,
            keyframe_common_feat_count=common_feat_count,
            keyframe_distance_diff=float(distance),
            keyframe_angle_diff=float(angle_deg),
        )

    def check(
        self, ts: Timestamp, current_pose: SE3, active_track: NDArray[np.float32]
    ) -> tuple[bool, list[SelectReason], SelectMetrics]:
        """Check if a keyframe should be selected."""
        if self.keyframe_ts == -1.0:
            reasons = [SelectReason.INITIAL]
            return (True, reasons, self._zero_metrics())

        metrics = self.calc_selector_metrics(ts, current_pose, active_track)
        if metrics["keyframe_time_diff"] <= self.thresholds.ignore_time_until_sec:
            reasons = [SelectReason.TIME_IGNORED]
            return (False, reasons, metrics)

        good_keyframe = False
        reasons: list[SelectReason] = []
        if metrics["keyframe_time_diff"] > self.thresholds.max_time_delta_sec:
            reasons.append(SelectReason.TIME_ELAPSED)
            good_keyframe = True
        if metrics["keyframe_median_parallax"] > self.thresholds.min_parallax_pts:
            reasons.append(SelectReason.PARALLAX)
            good_keyframe = True
        if metrics["keyframe_connectivity_ratio"] < self.thresholds.min_connectivity_ratio:
            reasons.append(SelectReason.LOW_CONNECTIVITY)
            good_keyframe = True
        return (good_keyframe, reasons, metrics)

    def set_new_keyframe(self, ts: Timestamp, current_pose: SE3, active_track: NDArray[np.float32]) -> None:
        """Set the new keyframe."""
        self.keyframe_pose_window.append(current_pose)
        self.keyframe_ts = ts

        n = np.minimum(active_track.shape[0], self.capacity)
        self.keyframe_ids[:] = -1
        self.keyframe_left_points[:] = np.nan
        self.keyframe_feat_count = n
        keyframe_ids = active_track[:, 0].astype(int)
        keyframe_left_points = active_track[:, 1:3]
        self.keyframe_ids[:n] = keyframe_ids
        self.keyframe_left_points[:n] = keyframe_left_points

    @staticmethod
    def _zero_metrics() -> SelectMetrics:
        """Zero metrics."""
        return SelectMetrics(
            keyframe_time_diff=0.0,
            keyframe_median_parallax=0.0,
            keyframe_connectivity_ratio=0.0,
            keyframe_common_feat_count=0,
            keyframe_distance_diff=0.0,
            keyframe_angle_diff=0.0,
        )
