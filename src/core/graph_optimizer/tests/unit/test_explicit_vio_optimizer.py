from typing import cast

import gtsam
import numpy as np
import pytest
from numpy.typing import NDArray

from core.camera_model.vio_context import VioContext
from core.front_end.keyframe_selector import SelectReason
from core.front_end.landmark_cache import LandmarkCacheStatus
from core.front_end.landmark_initialization import LandmarkInitializationFrameSchema
from core.graph_optimizer.explicit_vio_optimizer import ExplicitVIOOptimizer, VioKeyframe
from core.graph_optimizer.optimizer_types import PredictionMode
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B
L = gtsam.symbol_shorthand.L


def as_stereo_frame(frame: NDArray[np.float64]) -> NDArray[np.float32]:
    """Strip legacy landmark columns from graph optimizer fixtures."""
    return frame[:, : StereoTriangulationSchema.count()].astype(np.float32)


def make_static_imu_batch(
    start_ts: float,
    dt: float,
    sample_count: int,
) -> NDArray[np.float64]:
    """Create a static IMU batch for identity pose and zero bias."""
    imu_batch = np.zeros((sample_count, 8), dtype=np.float64)
    for i in range(sample_count):
        imu_batch[i, 0] = start_ts + i * dt
        imu_batch[i, 1] = 0.0
        imu_batch[i, 2] = 0.0
        imu_batch[i, 3] = 9.81
        imu_batch[i, 4] = 0.0
        imu_batch[i, 5] = 0.0
        imu_batch[i, 6] = 0.0
        imu_batch[i, 7] = dt
    return imu_batch


def make_one_feature_landmark_frame(
    feat_id: int,
    stereo: tuple[float, float, float],
    point: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Create a single-row landmark frame with one stable stereo feature."""
    left_u, left_v, right_u = stereo
    landmark_frame = np.full((1, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
    landmark_frame[0, LandmarkInitializationFrameSchema.FEAT_ID] = feat_id
    landmark_frame[0, LandmarkInitializationFrameSchema.TIMESTAMP] = 10.0
    landmark_frame[0, LandmarkInitializationFrameSchema.LEFT_U] = left_u
    landmark_frame[0, LandmarkInitializationFrameSchema.LEFT_V] = left_v
    landmark_frame[0, LandmarkInitializationFrameSchema.RIGHT_U] = right_u
    landmark_frame[0, LandmarkInitializationFrameSchema.RIGHT_V] = left_v
    landmark_frame[0, LandmarkInitializationFrameSchema.LIFECYCLE] = 1.0
    landmark_frame[0, LandmarkInitializationFrameSchema.AGE] = 10.0
    landmark_frame[0, LandmarkInitializationFrameSchema.STEREO_SCORE] = 10.0
    landmark_frame[0, LandmarkInitializationFrameSchema.STEREO_STATUS] = (
        StereoTriangulationStatus.TRIANGULATED.value
    )
    landmark_frame[0, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = LandmarkCacheStatus.COMPLETED.value
    landmark_frame[0, LandmarkInitializationFrameSchema.TRACKED] = 1.0
    landmark_frame[0, LandmarkInitializationFrameSchema.LANDMARK_XYZ] = point
    landmark_frame[0, LandmarkInitializationFrameSchema.STEREO_XYZ] = point
    return landmark_frame


class TestExplicitVIOOptimizer:
    """Test the ExplicitVIOOptimizer."""

    @pytest.fixture
    def optimizer(self, vio_ctx: VioContext) -> ExplicitVIOOptimizer:
        """Create an ExplicitVIOOptimizer."""
        return ExplicitVIOOptimizer.from_vio_ctx(vio_ctx)

    def test_first_keyframe(
        self, optimizer: ExplicitVIOOptimizer, first_landmark_frame: NDArray[np.float64]
    ) -> None:
        """Test the first keyframe."""
        keyframe = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
        )
        optimizer.add_keyframe(keyframe)
        assert optimizer.last_keyframe_id == 0
        assert optimizer.result.atPose3(X(0)) is not None
        assert optimizer.result.atVector(V(0)) is not None
        assert optimizer.result.atConstantBias(B(0)) is not None

        for i in range(first_landmark_frame.shape[0]):
            feat_id = int(first_landmark_frame[i, 0])
            is_stereo = np.isfinite(first_landmark_frame[i, 4])
            if is_stereo:
                assert optimizer.result.exists(L(feat_id))

    def test_imu_factor_creation(
        self, optimizer: ExplicitVIOOptimizer, first_landmark_frame: NDArray[np.float64]
    ) -> None:
        """Test the IMU factor creation for the next keyframe."""
        first_kf = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
        )
        optimizer.add_keyframe(first_kf)

        imu_batch = make_static_imu_batch(start_ts=10.0, dt=0.05, sample_count=100)

        second_kf = VioKeyframe(
            keyframe_id=1,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=15.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=imu_batch,
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
        )

        subgraph = optimizer.keyframe_to_subgraph(second_kf)
        assert subgraph.keyframe_id == 1
        # how to check that exists imu factors between keyframes?
        imu_factor_index = -1
        for i in range(subgraph.factors.size()):
            factor = subgraph.factors.at(i)
            if isinstance(factor, gtsam.ImuFactor):
                imu_factor_index = i
                break
        assert imu_factor_index > -1
        imu_factor = subgraph.factors.at(imu_factor_index)
        imu_factor = cast("gtsam.ImuFactor", imu_factor)
        assert imu_factor.keys() == [X(0), V(0), X(1), V(1), B(0)]
        bias_factor_index = -1
        for i in range(subgraph.factors.size()):
            factor = subgraph.factors.at(i)
            if isinstance(factor, gtsam.BetweenFactorConstantBias):
                bias_factor_index = i
                break
        assert bias_factor_index > -1
        bias_factor = subgraph.factors.at(bias_factor_index)
        bias_factor = cast("gtsam.BetweenFactorConstantBias", bias_factor)
        assert bias_factor.keys() == [B(0), B(1)]

    def test_new_explicit_landmark_has_no_point_prior(self, optimizer: ExplicitVIOOptimizer) -> None:
        """A validated stereo landmark should be constrained only by its stereo factor."""
        feat_id = 14352
        landmark_frame = make_one_feature_landmark_frame(
            feat_id=feat_id,
            stereo=(100.0, 50.0, 50.0),
            point=(2.0, 1.0, 20.0),
        )
        keyframe = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(landmark_frame),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
        )

        subgraph = optimizer.keyframe_to_subgraph(keyframe)

        landmark_prior_count = 0
        for i in range(subgraph.factors.size()):
            factor = subgraph.factors.at(i)
            if factor is None:
                continue
            if list(factor.keys()) == [L(feat_id)]:
                landmark_prior_count += 1
        assert landmark_prior_count == 0
        assert any(
            set(subgraph.factors.at(i).keys()) == {X(0), L(feat_id)}
            for i in range(subgraph.factors.size())
            if subgraph.factors.at(i) is not None
        )

    def test_keyframes_to_subgraph(
        self, optimizer: ExplicitVIOOptimizer, first_landmark_frame: NDArray[np.float64]
    ) -> None:
        """Test that multiple keyframes are merged into one optimization subgraph."""
        first_kf = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
        )
        imu_batch = make_static_imu_batch(start_ts=10.0, dt=0.05, sample_count=100)
        second_kf = VioKeyframe(
            keyframe_id=1,
            select_reason=[SelectReason.PARALLAX],
            timestamp=15.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=imu_batch,
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
        )

        with pytest.raises(ValueError, match="More then one keyframe is not supported for now"):
            optimizer.keyframes_to_subgraph([first_kf, second_kf])

    def test_static_resolve(
        self, optimizer: ExplicitVIOOptimizer, first_landmark_frame: NDArray[np.float64]
    ) -> None:
        """Test the static resolve."""
        first_kf = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        )
        optimizer.add_keyframe(first_kf)
        sample_count = 100
        dt = 0.01
        start_ts = 10.0
        imu_batch = np.zeros((sample_count, 8), dtype=np.float64)
        for i in range(sample_count):
            imu_batch[i, 0] = start_ts + i * dt
            imu_batch[i, 1] = 0.0
            imu_batch[i, 2] = 0.0
            imu_batch[i, 3] = 9.81
            imu_batch[i, 4] = 0.0
            imu_batch[i, 5] = 0.0
            imu_batch[i, 6] = 0.0
            imu_batch[i, 7] = dt

        second_kf = VioKeyframe(
            keyframe_id=1,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=11.0,
            stereo_frame=as_stereo_frame(first_landmark_frame),
            imu_batch=imu_batch,
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.identity(),
            velocity_guess=np.array([0, 0, 0]),
            bias_guess=np.array([0, 0, 0, 0, 0, 0]),
            zupt=True,
        )
        optimizer.add_keyframe(second_kf)
        assert optimizer.last_keyframe_id == 1
        assert optimizer.result.exists(X(1))
        assert optimizer.result.exists(V(1))
        assert optimizer.result.exists(B(1))

    def test_x7_keyframe(
        self,
        optimizer: ExplicitVIOOptimizer,
        state_x7: NDArray[np.float64],
        landmark_frame_x7: NDArray[np.float64],
    ) -> None:
        """Test the x7 keyframe."""
        keyframe = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(landmark_frame_x7),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.from_quat_and_translation(state_x7[:4], state_x7[4:7]),
            velocity_guess=state_x7[7:10].astype(np.float32),
            bias_guess=state_x7[10:16].astype(np.float32),
            zupt=True,
        )
        optimizer.add_keyframe(keyframe)
        assert optimizer.last_keyframe_id == 0
        assert optimizer.result.exists(X(0))
        assert optimizer.result.exists(V(0))
        assert optimizer.result.exists(B(0))

    def test_get_landmarks_ndarray(
        self,
        optimizer: ExplicitVIOOptimizer,
        state_x7: NDArray[np.float64],
        landmark_frame_x7: NDArray[np.float64],
    ) -> None:
        """Test the get landmarks ndarray."""
        keyframe = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=as_stereo_frame(landmark_frame_x7),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.from_quat_and_translation(state_x7[:4], state_x7[4:7]),
            velocity_guess=state_x7[7:10].astype(np.float32),
            bias_guess=state_x7[10:16].astype(np.float32),
            zupt=True,
        )
        optimizer.add_keyframe(keyframe)
        landmarks = optimizer.get_landmarks_ndarray()
        assert landmarks.shape[1] == 5
        stereo_landmark_frame = landmark_frame_x7[
            (
                landmark_frame_x7[:, LandmarkInitializationFrameSchema.STEREO_STATUS]
                == StereoTriangulationStatus.TRIANGULATED.value
            )
            & (
                landmark_frame_x7[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS]
                == LandmarkCacheStatus.COMPLETED.value
            )
            & (landmark_frame_x7[:, LandmarkInitializationFrameSchema.TRACKED] > 0)
        ]
        stereo_landmark_frame_length = stereo_landmark_frame.shape[0]
        assert stereo_landmark_frame_length == landmarks.shape[0]

    def test_get_nav_state_ndarray(
        self,
        optimizer: ExplicitVIOOptimizer,
        state_x7: NDArray[np.float64],
    ) -> None:
        """Test the get nav_state ndarray."""
        keyframe = VioKeyframe(
            keyframe_id=0,
            select_reason=[SelectReason.STATIC_INITIALIZATION],
            timestamp=10.0,
            stereo_frame=np.empty((0, StereoTriangulationSchema.count()), dtype=np.float32),
            imu_batch=np.empty((0, 8)),
            prediction_mode=PredictionMode.PNP,
            pose_guess=SE3.from_quat_and_translation(state_x7[:4], state_x7[4:7]),
            velocity_guess=state_x7[7:10].astype(np.float32),
            bias_guess=state_x7[10:16].astype(np.float32),
            zupt=True,
        )
        optimizer.add_keyframe(keyframe)

        nav_state = optimizer.get_nav_state_ndarray()
        assert nav_state.shape[0] == 10
