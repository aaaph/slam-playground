from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pyarrow as pa

from core.feature_tracker.feature_schema import FeatureSchema
from core.front_end.keyframe_selector import SelectReason
from core.graph_optimizer.optimizer_types import PredictionMode, VioKeyframe
from core.transformations.special_euclidian_3_dim import SE3


class ActiveTrackSchema(IntEnum):
    """Active track schema."""

    FEAT_ID = FeatureSchema.FEAT_ID
    TIMESTAMP = FeatureSchema.TIMESTAMP
    LEFT_U = FeatureSchema.LEFT_U
    LEFT_V = FeatureSchema.LEFT_V
    RIGHT_U = FeatureSchema.RIGHT_U
    RIGHT_V = FeatureSchema.RIGHT_V
    STATE = FeatureSchema.LIFECYCLE
    AGE = FeatureSchema.AGE
    STEREO_SCORE = FeatureSchema.STEREO_SCORE
    X = FeatureSchema.count()
    Y = FeatureSchema.count() + 1
    Z = FeatureSchema.count() + 2

    @classmethod
    def count(cls) -> int:
        """Get the count of the active track schema."""
        return FeatureSchema.count() + 3


keyframe_schema = pa.schema(
    [
        pa.field("keyframe_id", pa.int64(), nullable=False),
        pa.field("timestamp", pa.float64(), nullable=False),
        pa.field("select_reasons", pa.list_(pa.int32()), nullable=False),
        pa.field("state", pa.list_(pa.float32(), 16), nullable=False),  # quat(4) + t(3) + v(3) + ba(3) + bg(3)
        pa.field("imu_batch", pa.list_(pa.list_(pa.float64(), 8)), nullable=False),
        pa.field("active_track", pa.list_(pa.list_(pa.float32(), ActiveTrackSchema.count())), nullable=False),
        pa.field("vibration_detected", pa.bool_(), nullable=False),
        pa.field("non_zero_velocity_detected", pa.bool_(), nullable=False),
    ]
)


class StateSchema:
    """State schema."""

    QX = 0
    QY = 1
    QZ = 2
    QW = 3

    PX = 4
    PY = 5
    PZ = 6

    VX = 7
    VY = 8
    VZ = 9

    BAX = 10
    BAY = 11
    BAZ = 12

    BGX = 13
    BGY = 14
    BGZ = 15

    QUAT = slice(QX, QW + 1)
    VEC = slice(PX, PZ + 1)
    POSE = slice(PX, PZ + 1)
    VEL = slice(VX, VZ + 1)
    ACCEL_BIAS = slice(BAX, BAZ + 1)
    GYRO_BIAS = slice(BGX, BGZ + 1)

    @classmethod
    def count(cls) -> int:
        """Get the count of the state schema."""
        return 16


class ImuBatchSchema:
    """Imu batch schema."""

    TIMESTAMP = 0
    ACCEL_X = 1
    ACCEL_Y = 2
    ACCEL_Z = 3
    GYRO_X = 4
    GYRO_Y = 5
    GYRO_Z = 6
    DT = 7

    ACCEL = slice(ACCEL_X, ACCEL_Z + 1)
    GYRO = slice(GYRO_X, GYRO_Z + 1)
    SIX_DOF = slice(ACCEL_X, GYRO_Z + 1)

    @classmethod
    def count(cls) -> int:
        """Get the count of the imu batch schema."""
        return 8


@dataclass()
class KF:
    """Keyframe V2."""

    keyframe_id: int
    timestamp: float
    select_reasons: list[SelectReason]

    state: np.ndarray  # quat(4) + t(3) + v(3) + ba(3) + bg(3) = 16
    imu_batch: np.ndarray  # [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, dt]
    # [feat_id, timestamp, left_u, left_v, right_u, right_v, state, age, stereo_score, x, y, z]
    active_track: np.ndarray

    vibration_detected: bool = False
    non_zero_velocity_detected: bool = False

    def __repr__(self) -> str:
        """Return a string with keyframe information."""
        kf_id = self.keyframe_id
        nanosec = self.timestamp
        select_reasons = [reason.name for reason in self.select_reasons]
        imu_buffer_size = self.imu_batch.shape[0]
        zero_velocity = not self.non_zero_velocity_detected
        se3 = SE3.from_quat_and_translation(self.state[:4], self.state[4:7])
        velocity = self.state[7:10]
        accel_bias = self.state[10:13]
        gyro_bias = self.state[13:16]
        return (
            f"KF(id={kf_id}, ts={nanosec:.0f}, reasons={select_reasons}, pose={se3}, velocity={velocity}, "
            f"accel_bias={accel_bias}, gyro_bias={gyro_bias}, "
            f"imu_buffer_size={imu_buffer_size}, "
            f"active_track_size={self.active_track.shape[0]}, "
            f"zero_velocity={zero_velocity})"
        )

    def as_vio_kf(self) -> VioKeyframe:
        """Convert a front end keyframe to a vio keyframe."""
        front_end_pose = SE3.from_quat_and_translation(self.state[:4], self.state[4:7])
        front_end_velocity = self.state[7:10]
        front_end_bias = self.state[10:16]
        return VioKeyframe(
            keyframe_id=self.keyframe_id,
            select_reason=self.select_reasons,
            timestamp=self.timestamp,
            active_track=self.active_track,
            imu_batch=self.imu_batch,
            prediction_mode=PredictionMode.BOOTSTRAP,
            pose_guess=front_end_pose,
            velocity_guess=front_end_velocity,
            bias_guess=front_end_bias,
            zupt=not self.non_zero_velocity_detected,
        )

    def as_arrow(self) -> pa.RecordBatch:
        """Convert the keyframe to a record batch."""
        return type(self).to_record_batch([self])

    @classmethod
    def to_record_batch(cls, keyframes: list["KF"]) -> pa.RecordBatch:
        """Convert a list of keyframes to a record batch."""
        if len(keyframes) == 0:
            empty_columns = [pa.array([], type=field.type) for field in keyframe_schema]
            return pa.RecordBatch.from_arrays(empty_columns, schema=keyframe_schema)

        return pa.RecordBatch.from_pydict(
            {
                "keyframe_id": [kf.keyframe_id for kf in keyframes],
                "timestamp": [kf.timestamp for kf in keyframes],
                "select_reasons": [[reason.value for reason in kf.select_reasons] for kf in keyframes],
                "state": [kf.state.astype(np.float32).tolist() for kf in keyframes],
                "imu_batch": [kf.imu_batch.astype(np.float64).tolist() for kf in keyframes],
                "active_track": [kf.active_track.astype(np.float32).tolist() for kf in keyframes],
                "vibration_detected": [kf.vibration_detected for kf in keyframes],
                "non_zero_velocity_detected": [kf.non_zero_velocity_detected for kf in keyframes],
            },
            schema=keyframe_schema,
        )

    @classmethod
    def list_from_arrow(cls, arrow: pa.RecordBatch) -> list["KF"]:
        """Create keyframes from all rows in a record batch."""
        return [cls.from_arrow_row(arrow, row_idx) for row_idx in range(arrow.num_rows)]

    @classmethod
    def from_arrow(cls, arrow: pa.RecordBatch) -> "KF":
        """Create a keyframe from the first row of a record batch."""
        return cls.from_arrow_row(arrow, 0)

    @classmethod
    def from_arrow_row(cls, arrow: pa.RecordBatch, row_idx: int) -> "KF":
        """Create a keyframe from a specific row of a record batch."""
        imu_batch = np.array(arrow.column("imu_batch")[row_idx].as_py(), dtype=np.float64)
        if imu_batch.size == 0:
            imu_batch = np.empty((0, 8), dtype=np.float64)
        active_track = np.array(arrow.column("active_track")[row_idx].as_py(), dtype=np.float32)
        if active_track.size == 0:
            active_track = np.empty((0, ActiveTrackSchema.count()), dtype=np.float32)
        return cls(
            keyframe_id=arrow.column("keyframe_id")[row_idx].as_py(),
            timestamp=arrow.column("timestamp")[row_idx].as_py(),
            select_reasons=[SelectReason(reason) for reason in arrow.column("select_reasons")[row_idx].as_py()],
            state=np.array(arrow.column("state")[row_idx].as_py(), dtype=np.float32),
            imu_batch=imu_batch,
            active_track=active_track,
            vibration_detected=arrow.column("vibration_detected")[row_idx].as_py(),
            non_zero_velocity_detected=arrow.column("non_zero_velocity_detected")[row_idx].as_py(),
        )
