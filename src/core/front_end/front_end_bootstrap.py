from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.pose_tracker.inertial_integration import ImuBatch, ImuBuffer, ImuSchema
from logger import spawn_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.spatial.transform import Rotation

    from core.feature_tracker.feature_metrics_schema import FeatureTrackerMetrics

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


class _BootstrapSlidingWindow:
    """Sliding window for the bootstrap classifier."""

    def __init__(self, window_size: int = 10) -> None:
        """Construct the sliding window."""
        self.window_size = window_size
        self._timestamps_ns = np.full(window_size, fill_value=-1, dtype=np.int64)
        self._frame_ids = np.full(window_size, fill_value=-1, dtype=np.int32)
        self._features = np.full((window_size, 400, 6), fill_value=np.nan, dtype=np.float64)
        self._imu = np.full((window_size, 100, 8), fill_value=np.nan, dtype=np.float64)
        self._feature_counts = np.zeros(window_size, dtype=np.int32)
        self._imu_counts = np.zeros(window_size, dtype=np.int32)
        self.head = 0
        self.size = 0

    def _chronological_indices(self) -> NDArray[np.intp]:
        """Return valid storage indices from oldest to newest."""
        if self.size < self.window_size:
            return np.arange(self.size)
        return np.concatenate((np.arange(self.head, self.window_size), np.arange(self.head)))

    @property
    def timestamps_ns(self) -> NDArray[np.int64]:
        """Return valid timestamps from oldest to newest."""
        return self._timestamps_ns[self._chronological_indices()]

    @property
    def frame_ids(self) -> NDArray[np.int32]:
        """Return valid frame IDs from oldest to newest."""
        return self._frame_ids[self._chronological_indices()]

    @property
    def features(self) -> NDArray[np.float64]:
        """Return valid feature payloads from oldest to newest."""
        return self._features[self._chronological_indices()]

    @property
    def imu(self) -> NDArray[np.float64]:
        """Return valid IMU payloads from oldest to newest."""
        return self._imu[self._chronological_indices()]

    @property
    def feature_counts(self) -> NDArray[np.int32]:
        """Return valid feature counts from oldest to newest."""
        return self._feature_counts[self._chronological_indices()]

    @property
    def imu_counts(self) -> NDArray[np.int32]:
        """Return valid IMU counts from oldest to newest."""
        return self._imu_counts[self._chronological_indices()]

    def add(self, timestamp_ns: float, frame_id: int, features: np.ndarray, imu_buffer: ImuBuffer) -> None:
        """Add a sample to the sliding window."""
        self._timestamps_ns[self.head] = timestamp_ns
        self._frame_ids[self.head] = frame_id
        self._features[self.head].fill(np.nan)
        self._imu[self.head].fill(np.nan)
        frame_count = features.shape[0]
        self._features[self.head, :frame_count, :] = features
        self._feature_counts[self.head] = frame_count
        imu_rows = imu_buffer.get_full_buffer().rows.copy()
        imu_count = imu_rows.shape[0]
        self._imu[self.head, :imu_count, :] = imu_rows
        self._imu_counts[self.head] = imu_count
        self.head = (self.head + 1) % self.window_size
        self.size = min(self.size + 1, self.window_size)

    def clear(self) -> None:
        """Clear the sliding window."""
        self._timestamps_ns.fill(-1)
        self._frame_ids.fill(-1)
        self._features.fill(np.nan)
        self._imu.fill(np.nan)
        self._feature_counts.fill(0)
        self._imu_counts.fill(0)
        self.head = 0

    def stacked_imu_data(self) -> NDArray[np.float64]:
        """Return the stacked IMU data."""
        imu = self.imu
        valid = np.arange(imu.shape[1])[None, :] < self.imu_counts[:, None]
        return imu[valid]


@dataclass(frozen=True, slots=True)
class FrontEndBootstrapResult:
    """Result of the frontend bootstrap classifier."""

    decision: FrontEndBootstrapDecision
    initial_rotation: Rotation | None = None
    gyro_bias: NDArray[np.float64] | None = None

    @classmethod
    def unknown(
        cls, initial_rotation: Rotation | None = None, gyro_bias: NDArray[np.float64] | None = None
    ) -> FrontEndBootstrapResult:
        """Return an unknown result."""
        return cls(
            decision=FrontEndBootstrapDecision.UNKNOWN, initial_rotation=initial_rotation, gyro_bias=gyro_bias
        )


class FrontEndBootstrap:
    """Windowed bootstrap classifier for the VIO frontend."""

    def __init__(
        self,
        initial_decision: FrontEndBootstrapDecision = FrontEndBootstrapDecision.UNKNOWN,
        sample_stride: int = 4,
        mininal_window_size: int = 5,
    ) -> None:
        """Construct the frontend bootstrap classifier."""
        if mininal_window_size < 1:
            msg = "mininal_window_size must be at least 1"
            raise ValueError(msg)
        if sample_stride < 1:
            msg = "sample_stride must be at least 1"
            raise ValueError(msg)
        self.logger = spawn_logger(__name__)
        self.mininal_window_size = mininal_window_size
        self.initial_decision = initial_decision
        self.rotation_initialization = False
        self.sample_stride = sample_stride
        self.imu_buffer = ImuBuffer(capacity=(self.sample_stride + 1) * 10)
        self.frames_seen = 0
        self.sliding_window = _BootstrapSlidingWindow(window_size=10)
        self.zupt_queue = deque(maxlen=self.mininal_window_size)
        self.initial_rotation_value: Rotation | None = None

    def feed(
        self,
        frame_id: int,
        timestamp_ns: float,
        stereo_frame: NDArray[np.float32],
        visual_metrics: FeatureTrackerMetrics,
        imu_batch: ImuBatch,
    ) -> None:
        """Feed one synchronized image/IMU sample and return the current bootstrap decision."""
        # imu_metrics = imu_batch.metrics()
        self.zupt_queue.append(visual_metrics.zero_velocity_state)
        should_store = self.frames_seen % self.sample_stride == 0
        self.frames_seen += 1
        # we have imu_batch of imu_measurements (from i to j) -> need to put in local buffer (from anchor to j)
        accel_batch = imu_batch.accel()
        gyro_batch = imu_batch.gyro()
        timestamp_batch = imu_batch.timestamps()
        self.imu_buffer.add_batch(accel_batch, gyro_batch, timestamp_batch)
        if not should_store:
            return

        features_payload = np.full((stereo_frame.shape[0], 6), fill_value=np.nan, dtype=np.float64)
        features_payload[:, 0] = stereo_frame[:, StereoTriangulationSchema.FEAT_ID]
        features_payload[:, 1] = stereo_frame[:, StereoTriangulationSchema.LEFT_U]
        features_payload[:, 2] = stereo_frame[:, StereoTriangulationSchema.LEFT_V]
        features_payload[:, 3] = stereo_frame[:, StereoTriangulationSchema.LEFT_BEARING_X]
        features_payload[:, 4] = stereo_frame[:, StereoTriangulationSchema.LEFT_BEARING_Y]
        features_payload[:, 5] = stereo_frame[:, StereoTriangulationSchema.LEFT_BEARING_Z]

        self.sliding_window.add(timestamp_ns, frame_id, features_payload, self.imu_buffer)
        self.imu_buffer.reset(timestamp_ns)

    def make_decision(self) -> FrontEndBootstrapDecision:
        """Classify the current evidence window."""
        if len(self.zupt_queue) < self.mininal_window_size:
            return FrontEndBootstrapDecision.UNKNOWN

        visual_stationary = all(zupt == ZeroVelocityTrackerState.ZERO_VELOCITY for zupt in self.zupt_queue)
        if visual_stationary:
            return FrontEndBootstrapDecision.STATIC

        return FrontEndBootstrapDecision.UNKNOWN

    def evaluate(self) -> FrontEndBootstrapResult:
        """Evaluate the current evidence window."""
        decision = self.make_decision()
        if decision != FrontEndBootstrapDecision.STATIC:
            return FrontEndBootstrapResult.unknown(initial_rotation=self.initial_rotation_once())
        initial_rotation = self.initial_rotation()
        gyro_bias = self.gyro_bias_from_imu()
        return FrontEndBootstrapResult(decision=decision, initial_rotation=initial_rotation, gyro_bias=gyro_bias)

    def initial_rotation(self) -> Rotation:
        """Compute the initial rotation from the sliding window."""
        imu_rows = np.concatenate(
            (
                self.sliding_window.stacked_imu_data(),
                self.imu_buffer.get_full_buffer().rows,
            ),
            axis=0,
        )
        batch = ImuBatch(imu_rows)
        return batch.gram_schmidt()

    def initial_rotation_once(self) -> Rotation | None:
        """Compute the initial rotation from the sliding window."""
        if self.initial_rotation_value is not None:
            return None
        self.initial_rotation_value = self.initial_rotation()
        return self.initial_rotation_value

    def gyro_bias_from_imu(self) -> NDArray[np.float64]:  # (3,) -> bias gyro
        """Compute the gyro bias from the sliding window."""
        imu_rows = np.concatenate(
            (
                self.sliding_window.stacked_imu_data(),
                self.imu_buffer.get_full_buffer().rows,
            ),
            axis=0,
        )
        if imu_rows.shape[0] == 0:
            return np.zeros(3, dtype=np.float64)

        return imu_rows[:, ImuSchema.GYRO_SLICE].mean(axis=0)

    def commit(self, timestamp_ns: float) -> None:
        """Commit the current evidence window."""
        self.sliding_window.clear()
        self.zupt_queue.clear()
        self.initial_rotation_value = None
        self.imu_buffer.reset(timestamp_ns)
        self.frames_seen = 0
