from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from core.feature_tracker.feature_schema import FeatureSchema
from logger import spawn_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.transform import Rotation

    from core.feature_tracker.feature_frame import FeatureFrame
    from core.feature_tracker.feature_metrics_schema import FeatureTrackerMetrics
    from core.pose_tracker.inertial_integration import ImuBatch

MATURE_TRACK_MIN_AGE = 2.0
TRIANGULATED_STATUS_WIDTH = 5
EPSILON = 1e-9
MIN_ACCEL_DIRECTION_SAMPLES = 2


class FrontEndBootstrapDecision(IntEnum):
    """Bootstrap decision for the current evidence window."""

    UNKNOWN = 0
    STATIC = 1
    DYNAMIC = 2
    VISION_DEGRADED = 3


@dataclass(slots=True, frozen=True)
class FrontEndBootstrapConfig:
    """Thresholds for the frontend bootstrap classifier."""

    min_window_frames: int = 8
    window_size_frames: int = 15
    min_window_duration_sec: float = 0.2

    min_good_features: int = 30
    min_triangulated_features: int = 20
    min_stereo_ok_ratio: float = 0.35

    static_median_parallax_px: float = 0.75
    static_p90_parallax_px: float = 2.0
    dynamic_median_parallax_px: float = 2.5
    dynamic_p90_parallax_px: float = 6.0

    static_gyro_std_norm_max: float = 0.015
    static_accel_norm_std_max: float = 0.08
    static_accel_direction_std_rad_max: float = 0.02

    dynamic_gyro_std_norm_min: float = 0.04
    dynamic_accel_norm_std_min: float = 0.20
    dynamic_accel_direction_std_rad_min: float = 0.05


@dataclass(slots=True, frozen=True)
class FeatureBootstrapMetrics:
    """Feature-tracker metrics consumed by the frontend bootstrapper."""

    active_count: int
    good_count: int
    stereo_ok_count: int
    triangulated_count: int
    tracked_count: int
    mature_track_count: int
    temporal_parallax_px: float
    temporal_parallax_p90_px: float = 0.0

    @property
    def stereo_ok_ratio(self) -> float:
        """Return the ratio of good tracks that have a stereo match."""
        if self.good_count == 0:
            return 0.0
        return self.stereo_ok_count / self.good_count

    @property
    def track_survival_ratio(self) -> float:
        """Return the ratio of good tracks that survived from a previous frame."""
        if self.good_count == 0:
            return 0.0
        return self.tracked_count / self.good_count

    @classmethod
    def from_frame(
        cls,
        frame: FeatureFrame,
        *,
        temporal_parallax_px: float,
        temporal_parallax_p90_px: float | None = None,
        triangulated_points: NDArray[np.float32] | None = None,
    ) -> FeatureBootstrapMetrics:
        """Create feature bootstrap metrics from the active feature frame."""
        good_features = frame.good_features()
        good_count = int(good_features.shape[0])
        if good_count == 0:
            stereo_ok_count = 0
            tracked_count = 0
            mature_track_count = 0
        else:
            right_uv = good_features[:, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1]
            stereo_ok_count = int(np.count_nonzero(np.all(np.isfinite(right_uv), axis=1)))
            ages = good_features[:, FeatureSchema.AGE]
            tracked_count = int(np.count_nonzero(ages > 0.0))
            mature_track_count = int(np.count_nonzero(ages >= MATURE_TRACK_MIN_AGE))

        if triangulated_points is None or triangulated_points.size == 0:
            triangulated_count = 0
        elif triangulated_points.shape[1] >= TRIANGULATED_STATUS_WIDTH:
            triangulated_count = int(np.count_nonzero(triangulated_points[:, 4].astype(bool)))
        else:
            xyz = triangulated_points[:, 1:4]
            triangulated_count = int(np.count_nonzero(np.all(np.isfinite(xyz), axis=1)))

        return cls(
            active_count=frame.count(),
            good_count=good_count,
            stereo_ok_count=stereo_ok_count,
            triangulated_count=triangulated_count,
            tracked_count=tracked_count,
            mature_track_count=mature_track_count,
            temporal_parallax_px=float(temporal_parallax_px),
            temporal_parallax_p90_px=float(
                temporal_parallax_px if temporal_parallax_p90_px is None else temporal_parallax_p90_px
            ),
        )


@dataclass(slots=True, frozen=True)
class ImuBootstrapMetrics:
    """IMU batch metrics consumed by the frontend bootstrapper."""

    sample_count: int
    duration_sec: float
    gyro_mean: NDArray[np.float64]
    gyro_std: NDArray[np.float64]
    gyro_norm_mean: float
    gyro_std_norm: float
    accel_mean: NDArray[np.float64]
    accel_std: NDArray[np.float64]
    accel_norm_mean: float
    accel_norm_std: float
    accel_direction_std_rad: float

    @classmethod
    def from_batch(
        cls,
        accel: NDArray[np.float64],
        gyro: NDArray[np.float64],
        timestamps_ns: NDArray[np.float64],
    ) -> ImuBootstrapMetrics:
        """Create IMU bootstrap metrics from one image-frame IMU batch."""
        sample_count = int(min(accel.shape[0], gyro.shape[0], timestamps_ns.shape[0]))
        if sample_count == 0:
            return cls.empty()

        accel = np.asarray(accel[:sample_count], dtype=np.float64)
        gyro = np.asarray(gyro[:sample_count], dtype=np.float64)
        timestamps_ns = np.asarray(timestamps_ns[:sample_count], dtype=np.float64)
        duration_sec = float((timestamps_ns[-1] - timestamps_ns[0]) * 1e-9) if sample_count > 1 else 0.0

        gyro_norms = np.linalg.norm(gyro, axis=1)
        accel_norms = np.linalg.norm(accel, axis=1)
        accel_direction_std_rad = _accel_direction_std_rad(accel)

        return cls(
            sample_count=sample_count,
            duration_sec=duration_sec,
            gyro_mean=np.mean(gyro, axis=0),
            gyro_std=np.std(gyro, axis=0),
            gyro_norm_mean=float(np.mean(gyro_norms)),
            gyro_std_norm=float(np.linalg.norm(np.std(gyro, axis=0))),
            accel_mean=np.mean(accel, axis=0),
            accel_std=np.std(accel, axis=0),
            accel_norm_mean=float(np.mean(accel_norms)),
            accel_norm_std=float(np.std(accel_norms)),
            accel_direction_std_rad=accel_direction_std_rad,
        )

    @classmethod
    def empty(cls) -> ImuBootstrapMetrics:
        """Create empty IMU metrics."""
        zeros = np.zeros(3, dtype=np.float64)
        return cls(
            sample_count=0,
            duration_sec=0.0,
            gyro_mean=zeros.copy(),
            gyro_std=zeros.copy(),
            gyro_norm_mean=0.0,
            gyro_std_norm=0.0,
            accel_mean=zeros.copy(),
            accel_std=zeros.copy(),
            accel_norm_mean=0.0,
            accel_norm_std=0.0,
            accel_direction_std_rad=0.0,
        )


@dataclass(slots=True, frozen=True)
class FrontEndBootstrapInput:
    """Input sample for the frontend bootstrapper."""

    frame_id: int
    timestamp_ns: float
    feature_metrics: FeatureBootstrapMetrics
    imu_metrics: ImuBootstrapMetrics

    @classmethod
    def from_frame_and_imu(  # noqa: PLR0913
        cls,
        *,
        frame_id: int,
        timestamp_ns: float,
        frame: FeatureFrame,
        temporal_parallax_px: float,
        accel: NDArray[np.float64],
        gyro: NDArray[np.float64],
        imu_timestamps_ns: NDArray[np.float64],
        temporal_parallax_p90_px: float | None = None,
        triangulated_points: NDArray[np.float32] | None = None,
    ) -> FrontEndBootstrapInput:
        """Create a bootstrap input sample from tracker and IMU outputs."""
        return cls(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            feature_metrics=FeatureBootstrapMetrics.from_frame(
                frame,
                temporal_parallax_px=temporal_parallax_px,
                temporal_parallax_p90_px=temporal_parallax_p90_px,
                triangulated_points=triangulated_points,
            ),
            imu_metrics=ImuBootstrapMetrics.from_batch(accel, gyro, imu_timestamps_ns),
        )


@dataclass(slots=True, frozen=True)
class FrontEndBootstrapWindowMetrics:
    """Aggregated bootstrap metrics over the rolling evidence window."""

    frame_count: int
    duration_sec: float
    median_temporal_parallax_px: float
    p90_temporal_parallax_px: float
    median_good_feature_count: float
    median_triangulated_feature_count: float
    median_stereo_ok_ratio: float
    median_gyro_std_norm: float
    median_accel_norm_std: float
    median_accel_direction_std_rad: float

    @classmethod
    def empty(cls) -> FrontEndBootstrapWindowMetrics:
        """Create an empty window metrics."""
        return cls(
            frame_count=0,
            duration_sec=0.0,
            median_temporal_parallax_px=0.0,
            p90_temporal_parallax_px=0.0,
            median_good_feature_count=0.0,
            median_triangulated_feature_count=0.0,
            median_stereo_ok_ratio=0.0,
            median_gyro_std_norm=0.0,
            median_accel_norm_std=0.0,
            median_accel_direction_std_rad=0.0,
        )


@dataclass(slots=True, frozen=True)
class FrontEndBootstrapResult:
    """Bootstrapper output for the current evidence window."""

    decision: FrontEndBootstrapDecision
    confidence: float
    reasons: tuple[str, ...]
    window_metrics: FrontEndBootstrapWindowMetrics

    rotation: Rotation | None = None

    @property
    def rotation_ready(self) -> bool:
        """Return whether the bootstrapper selected a usable rotation."""
        return self.rotation is not None

    @property
    def rotation_quat(self) -> NDArray[np.float64]:
        """Return the rotation as a quaternion."""
        if self.rotation is None:
            raise ValueError("Rotation is not ready")
        return self.rotation.as_quat()

    @property
    def ready(self) -> bool:
        """Return whether the bootstrapper selected a usable bootstrap path."""
        return self.decision in (FrontEndBootstrapDecision.STATIC, FrontEndBootstrapDecision.DYNAMIC)

    @classmethod
    def empty(cls) -> FrontEndBootstrapResult:
        """Create an empty bootstrap result."""
        return cls(
            decision=FrontEndBootstrapDecision.UNKNOWN,
            confidence=0.0,
            reasons=(),
            window_metrics=FrontEndBootstrapWindowMetrics.empty(),
        )


class FrontEndBootstrap:
    """Windowed bootstrap classifier for the VIO frontend."""

    def __init__(self, config: FrontEndBootstrapConfig | None = None) -> None:
        """Construct the frontend bootstrap classifier."""
        self.logger = spawn_logger(__name__)
        self.config = config or FrontEndBootstrapConfig()
        self._window: deque[FrontEndBootstrapInput] = deque(maxlen=self.config.window_size_frames)
        self.latest_result = self._result(FrontEndBootstrapDecision.UNKNOWN, 0.0, ("no_samples",))
        self.rotation_initialization = False

    def feed(
        self,
        frame_id: int,
        _timestamp_ns: float,
        visual_metrics: FeatureTrackerMetrics,
        imu_batch: ImuBatch,
    ) -> FrontEndBootstrapResult:
        """Feed one synchronized image/IMU sample and return the current bootstrap decision."""
        self.logger.info(f"[FE:FRAME_ID]: {frame_id}")
        self.logger.info(f"[FE:VISUAL_METRICS]: {visual_metrics}")
        self.logger.info(f"[FE:IMU_METRICS]: {imu_batch.metrics()}")

        rotation: Rotation | None = None

        if not self.rotation_initialization and imu_batch.sample_count > 0:
            rotation = imu_batch.gram_schmidt()
            self.rotation_initialization = True

        return FrontEndBootstrapResult(
            decision=FrontEndBootstrapDecision.UNKNOWN,
            confidence=0.0,
            reasons=(),
            window_metrics=FrontEndBootstrapWindowMetrics.empty(),
            rotation=rotation,
        )

    def reset(self) -> None:
        """Clear accumulated bootstrap evidence."""
        self._window.clear()
        self.latest_result = self._result(FrontEndBootstrapDecision.UNKNOWN, 0.0, ("reset",))

    @property
    def frame_count(self) -> int:
        """Return the number of samples currently in the evidence window."""
        return len(self._window)

    def _classify(self) -> FrontEndBootstrapResult:
        metrics = self._window_metrics()
        config = self.config
        reasons: list[str] = []

        enough_frames = metrics.frame_count >= config.min_window_frames
        enough_duration = metrics.duration_sec >= config.min_window_duration_sec
        if not enough_frames or not enough_duration:
            reasons.append("window_not_ready")
            return self._result(FrontEndBootstrapDecision.UNKNOWN, 0.0, tuple(reasons), metrics)

        latest_feature_metrics = self._window[-1].feature_metrics
        latest_vision_degraded = (
            latest_feature_metrics.good_count < config.min_good_features
            or latest_feature_metrics.triangulated_count < config.min_triangulated_features
            or latest_feature_metrics.stereo_ok_ratio < config.min_stereo_ok_ratio
        )
        window_vision_degraded = (
            metrics.median_good_feature_count < config.min_good_features
            or metrics.median_triangulated_feature_count < config.min_triangulated_features
            or metrics.median_stereo_ok_ratio < config.min_stereo_ok_ratio
        )
        if latest_vision_degraded or window_vision_degraded:
            reasons.append("vision_degraded")
            return self._result(FrontEndBootstrapDecision.VISION_DEGRADED, 1.0, tuple(reasons), metrics)

        visual_static = (
            metrics.median_temporal_parallax_px <= config.static_median_parallax_px
            and metrics.p90_temporal_parallax_px <= config.static_p90_parallax_px
        )
        imu_static = (
            metrics.median_gyro_std_norm <= config.static_gyro_std_norm_max
            and metrics.median_accel_norm_std <= config.static_accel_norm_std_max
            and metrics.median_accel_direction_std_rad <= config.static_accel_direction_std_rad_max
        )
        if visual_static and imu_static:
            reasons.extend(("visual_static", "imu_static"))
            confidence = self._static_confidence(metrics)
            return self._result(FrontEndBootstrapDecision.STATIC, confidence, tuple(reasons), metrics)

        visual_dynamic = (
            metrics.median_temporal_parallax_px >= config.dynamic_median_parallax_px
            or metrics.p90_temporal_parallax_px >= config.dynamic_p90_parallax_px
        )
        imu_dynamic = (
            metrics.median_gyro_std_norm >= config.dynamic_gyro_std_norm_min
            or metrics.median_accel_norm_std >= config.dynamic_accel_norm_std_min
            or metrics.median_accel_direction_std_rad >= config.dynamic_accel_direction_std_rad_min
        )
        if visual_dynamic or imu_dynamic:
            if visual_dynamic:
                reasons.append("visual_dynamic")
            if imu_dynamic:
                reasons.append("imu_dynamic")
            confidence = self._dynamic_confidence(metrics)
            return self._result(FrontEndBootstrapDecision.DYNAMIC, confidence, tuple(reasons), metrics)

        reasons.append("ambiguous_motion")
        return self._result(FrontEndBootstrapDecision.UNKNOWN, 0.2, tuple(reasons), metrics)

    def _static_confidence(self, metrics: FrontEndBootstrapWindowMetrics) -> float:
        config = self.config
        parallax_score = 1.0 - _safe_ratio(metrics.median_temporal_parallax_px, config.static_median_parallax_px)
        gyro_score = 1.0 - _safe_ratio(metrics.median_gyro_std_norm, config.static_gyro_std_norm_max)
        accel_score = 1.0 - _safe_ratio(metrics.median_accel_norm_std, config.static_accel_norm_std_max)
        return float(np.clip(np.mean([parallax_score, gyro_score, accel_score]), 0.0, 1.0))

    def _dynamic_confidence(self, metrics: FrontEndBootstrapWindowMetrics) -> float:
        config = self.config
        parallax_score = _safe_ratio(metrics.median_temporal_parallax_px, config.dynamic_median_parallax_px)
        p90_score = _safe_ratio(metrics.p90_temporal_parallax_px, config.dynamic_p90_parallax_px)
        gyro_score = _safe_ratio(metrics.median_gyro_std_norm, config.dynamic_gyro_std_norm_min)
        accel_score = _safe_ratio(metrics.median_accel_norm_std, config.dynamic_accel_norm_std_min)
        return float(np.clip(max(parallax_score, p90_score, gyro_score, accel_score), 0.0, 1.0))

    def _window_metrics(self) -> FrontEndBootstrapWindowMetrics:
        if not self._window:
            return FrontEndBootstrapWindowMetrics(
                frame_count=0,
                duration_sec=0.0,
                median_temporal_parallax_px=0.0,
                p90_temporal_parallax_px=0.0,
                median_good_feature_count=0.0,
                median_triangulated_feature_count=0.0,
                median_stereo_ok_ratio=0.0,
                median_gyro_std_norm=0.0,
                median_accel_norm_std=0.0,
                median_accel_direction_std_rad=0.0,
            )

        samples = list(self._window)
        duration_sec = (samples[-1].timestamp_ns - samples[0].timestamp_ns) * 1e-9
        feature_metrics = [sample.feature_metrics for sample in samples]
        imu_metrics = [sample.imu_metrics for sample in samples]
        parallaxes = np.array([m.temporal_parallax_px for m in feature_metrics], dtype=np.float64)
        p90_parallaxes = np.array([m.temporal_parallax_p90_px for m in feature_metrics], dtype=np.float64)

        return FrontEndBootstrapWindowMetrics(
            frame_count=len(samples),
            duration_sec=max(float(duration_sec), 0.0),
            median_temporal_parallax_px=_median(parallaxes),
            p90_temporal_parallax_px=_median(p90_parallaxes),
            median_good_feature_count=_median([m.good_count for m in feature_metrics]),
            median_triangulated_feature_count=_median([m.triangulated_count for m in feature_metrics]),
            median_stereo_ok_ratio=_median([m.stereo_ok_ratio for m in feature_metrics]),
            median_gyro_std_norm=_median([m.gyro_std_norm for m in imu_metrics]),
            median_accel_norm_std=_median([m.accel_norm_std for m in imu_metrics]),
            median_accel_direction_std_rad=_median([m.accel_direction_std_rad for m in imu_metrics]),
        )

    def _result(
        self,
        decision: FrontEndBootstrapDecision,
        confidence: float,
        reasons: tuple[str, ...],
        metrics: FrontEndBootstrapWindowMetrics | None = None,
    ) -> FrontEndBootstrapResult:
        return FrontEndBootstrapResult(
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            window_metrics=metrics or self._window_metrics(),
        )


def _accel_direction_std_rad(accel: NDArray[np.float64]) -> float:
    norms = np.linalg.norm(accel, axis=1)
    valid = norms > EPSILON
    if np.count_nonzero(valid) < MIN_ACCEL_DIRECTION_SAMPLES:
        return 0.0
    directions = accel[valid] / norms[valid, None]
    mean_direction = np.mean(directions, axis=0)
    mean_norm = np.linalg.norm(mean_direction)
    if mean_norm <= EPSILON:
        return float(np.pi)
    mean_direction /= mean_norm
    cosines = np.clip(directions @ mean_direction, -1.0, 1.0)
    return float(np.std(np.arccos(cosines)))


def _median(values: list[float] | list[int] | NDArray[np.float64]) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def _percentile(values: list[float] | list[int] | NDArray[np.float64], q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, q))


def _safe_ratio(value: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 0.0
    return float(value / threshold)
