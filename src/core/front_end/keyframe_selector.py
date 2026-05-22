from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from core.feature_tracker.feature_schema import FeatureSchema

Timestamp = float


class SelectReason(Enum):
    """Select reason."""

    LOW_CONNECTIVITY = auto()
    PARALLAX = auto()
    TIME_ELAPSED = auto()
    TIME_IGNORED = auto()
    STATIC_INITIALIZATION = auto()
    MOTION_INITIALIZATION = auto()
    WAITING_FOR_INITIALIZATION = auto()


select_metrics_schema = pa.schema(
    [
        pa.field("keyframe_time_diff", pa.float64()),
        pa.field("keyframe_median_parallax", pa.float64()),
        pa.field("keyframe_connectivity_ratio", pa.float64()),
        pa.field("keyframe_common_feat_count", pa.int32()),
        pa.field("keyframe_time_diff_min_threshold", pa.float64()),
        pa.field("keyframe_time_diff_max_threshold", pa.float64()),
        pa.field("keyframe_median_parallax_threshold", pa.float64()),
        pa.field("keyframe_connectivity_ratio_threshold", pa.float64()),
    ]
)


@dataclass
class SelectMetrics:
    """Select metrics."""

    keyframe_time_diff: float
    keyframe_time_diff_min_threshold: float
    keyframe_time_diff_max_threshold: float
    keyframe_median_parallax: float
    keyframe_median_parallax_threshold: float
    keyframe_connectivity_ratio: float
    keyframe_connectivity_ratio_threshold: float
    keyframe_common_feat_count: int

    @staticmethod
    def schema() -> pa.Schema:
        """Get the schema of the select metrics."""
        return select_metrics_schema

    def as_arrow(self) -> pa.RecordBatch:
        """Convert the select metrics to a record batch."""
        return pa.RecordBatch.from_pydict(
            {
                "keyframe_time_diff": [self.keyframe_time_diff],
                "keyframe_median_parallax": [self.keyframe_median_parallax],
                "keyframe_connectivity_ratio": [self.keyframe_connectivity_ratio],
                "keyframe_common_feat_count": [self.keyframe_common_feat_count],
                "keyframe_time_diff_min_threshold": [self.keyframe_time_diff_min_threshold],
                "keyframe_time_diff_max_threshold": [self.keyframe_time_diff_max_threshold],
                "keyframe_median_parallax_threshold": [self.keyframe_median_parallax_threshold],
                "keyframe_connectivity_ratio_threshold": [self.keyframe_connectivity_ratio_threshold],
            },
            schema=select_metrics_schema,
        )

    @classmethod
    def from_arrow(cls, arrow: pa.RecordBatch) -> "SelectMetrics":
        """Convert the record batch to a select metrics."""
        return cls(
            keyframe_time_diff=arrow.column("keyframe_time_diff")[0],
            keyframe_median_parallax=arrow.column("keyframe_median_parallax")[0],
            keyframe_connectivity_ratio=arrow.column("keyframe_connectivity_ratio")[0],
            keyframe_common_feat_count=arrow.column("keyframe_common_feat_count")[0],
            keyframe_time_diff_min_threshold=arrow.column("keyframe_time_diff_min_threshold")[0],
            keyframe_time_diff_max_threshold=arrow.column("keyframe_time_diff_max_threshold")[0],
            keyframe_median_parallax_threshold=arrow.column("keyframe_median_parallax_threshold")[0],
            keyframe_connectivity_ratio_threshold=arrow.column("keyframe_connectivity_ratio_threshold")[0],
        )


@dataclass
class KeyFrameSelectThresholds:
    """Keyframe selection thresholds."""

    ignore_time_until_sec: float = (
        0.4  # the keyframe should not be selected if the delta is lower than this threshold
    )
    max_time_delta_sec: float = 5.0  # the keyframe should be selected if the delta is higher than this threshold
    min_connectivity_ratio: float = 0.75
    min_parallax_pts: int = 150
    min_common_feat_for_parallax: int = 5


class KeyframeSelector:
    """Keyframe selector."""

    def __init__(self, thresholds: KeyFrameSelectThresholds, capacity: int = 500) -> None:
        """Initialize the keyframe selector."""
        self.capacity = capacity
        self.thresholds = thresholds
        self.initialized = False
        self.keyframe_ts: Timestamp = -1.0
        self.keyframe_ids = np.full(capacity, -1, dtype=np.int32)
        self.keyframe_left_points = np.full((capacity, 2), np.nan, dtype=np.float32)
        self.keyframe_feat_count = 0

    @classmethod
    def default_factory(cls) -> "KeyframeSelector":
        """Create a default `KeyframeSelector`."""
        return cls(thresholds=KeyFrameSelectThresholds())

    @classmethod
    def from_thresholds(cls, thresholds: KeyFrameSelectThresholds) -> "KeyframeSelector":
        """Create a `KeyframeSelector` from thresholds."""
        return cls(thresholds=thresholds)

    def switch_thresholds(self, thresholds: KeyFrameSelectThresholds) -> None:
        """Switch the thresholds."""
        self.thresholds = thresholds

    def calc_selector_metrics(self, ts: Timestamp, active_track: NDArray[np.float32]) -> SelectMetrics:
        """Calculate the selector metrics."""
        common_feat_ids, idx_kf, idx_cur = np.intersect1d(
            self.keyframe_ids[: self.keyframe_feat_count], active_track[:, 0].astype(int), return_indices=True
        )
        common_feat_count = len(common_feat_ids)
        connectivity = common_feat_count / self.keyframe_feat_count if self.keyframe_feat_count > 0 else 0.0
        parallax = 0.0
        if common_feat_count >= self.thresholds.min_common_feat_for_parallax:
            diffs = (
                active_track[idx_cur, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1]
                - self.keyframe_left_points[idx_kf]
            )
            parallax = np.sqrt(np.median(np.sum(np.square(diffs), axis=1)))

        return SelectMetrics(
            keyframe_time_diff=(ts - self.keyframe_ts) / 1e9,
            keyframe_median_parallax=parallax,
            keyframe_connectivity_ratio=connectivity,
            keyframe_common_feat_count=common_feat_count,
            keyframe_time_diff_min_threshold=self.thresholds.ignore_time_until_sec,
            keyframe_time_diff_max_threshold=self.thresholds.max_time_delta_sec,
            keyframe_median_parallax_threshold=self.thresholds.min_parallax_pts,
            keyframe_connectivity_ratio_threshold=self.thresholds.min_connectivity_ratio,
        )

    def check(
        self, ts: Timestamp, active_track: NDArray[np.float32]
    ) -> tuple[bool, list[SelectReason], SelectMetrics]:
        """Check if a keyframe should be selected."""
        if self.keyframe_ts == -1.0:
            reasons = [SelectReason.WAITING_FOR_INITIALIZATION]
            return (False, reasons, self._zero_metrics(self.thresholds))

        metrics = self.calc_selector_metrics(ts, active_track)

        if not self.initialized:
            reasons = [SelectReason.WAITING_FOR_INITIALIZATION]
            return (False, reasons, metrics)

        if metrics.keyframe_time_diff <= self.thresholds.ignore_time_until_sec:
            reasons = [SelectReason.TIME_IGNORED]
            return (False, reasons, metrics)

        good_keyframe = False
        reasons: list[SelectReason] = []
        if metrics.keyframe_time_diff > self.thresholds.max_time_delta_sec:
            reasons.append(SelectReason.TIME_ELAPSED)
            good_keyframe = True
        if metrics.keyframe_median_parallax > self.thresholds.min_parallax_pts:
            reasons.append(SelectReason.PARALLAX)
            good_keyframe = True
        if metrics.keyframe_connectivity_ratio < self.thresholds.min_connectivity_ratio:
            reasons.append(SelectReason.LOW_CONNECTIVITY)
            good_keyframe = True
        return (good_keyframe, reasons, metrics)

    def set_new_keyframe(self, ts: Timestamp, active_track: NDArray[np.float32]) -> None:
        """Set the new keyframe."""
        self.keyframe_ts = ts

        n = np.minimum(active_track.shape[0], self.capacity)
        self.keyframe_ids[:] = -1
        self.keyframe_left_points[:] = np.nan
        self.keyframe_feat_count = n
        keyframe_ids = active_track[:, FeatureSchema.FEAT_ID].astype(int)
        keyframe_left_points = active_track[:, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1]
        self.keyframe_ids[:n] = keyframe_ids
        self.keyframe_left_points[:n] = keyframe_left_points

    def initialize(self) -> None:
        """Initialize the keyframe selector."""
        self.initialized = True

    @staticmethod
    def _zero_metrics(thresholds: KeyFrameSelectThresholds) -> SelectMetrics:
        """Zero metrics."""
        return SelectMetrics(
            keyframe_time_diff=0.0,
            keyframe_median_parallax=0.0,
            keyframe_connectivity_ratio=1.0,
            keyframe_common_feat_count=0,
            keyframe_time_diff_min_threshold=thresholds.ignore_time_until_sec,
            keyframe_time_diff_max_threshold=thresholds.max_time_delta_sec,
            keyframe_median_parallax_threshold=thresholds.min_parallax_pts,
            keyframe_connectivity_ratio_threshold=thresholds.min_connectivity_ratio,
        )
