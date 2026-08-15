import numpy as np

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.front_end.front_end_bootstrap import FrontEndBootstrapDecision
from core.front_end.keyframe_selector import KeyFrameSelectThresholds, SelectMetrics
from core.graph_optimizer.optimizer_types import PredictionMode
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from pipeline.context import PipelineContext
from pipeline.nodes.vio_frontend import FrontEndMode, VIOFrontend


class _SelectorShouldNotBeCalled:
    """Selector stub for bootstrap gating tests."""

    thresholds = KeyFrameSelectThresholds()

    def check(self, *args, **kwargs):
        raise AssertionError("selector.check should not run before frontend initialization")

    def set_new_keyframe(self, *args, **kwargs) -> None:
        raise AssertionError("selector.set_new_keyframe should not run before frontend initialization")

    def initialize(self) -> None:
        raise AssertionError("selector.initialize should not run before frontend initialization")


class KeyFrameSelectorStub:
    """Minimal selector stub for committed-bootstrap keyframe tests."""

    thresholds = KeyFrameSelectThresholds()

    def __init__(self) -> None:
        """Initialize call counters."""
        self.new_keyframe_called = False
        self.initialized = False

    def set_new_keyframe(self, *args, **kwargs) -> None:
        """Record keyframe baseline updates."""
        self.new_keyframe_called = True

    def initialize(self) -> None:
        """Record selector initialization."""
        self.initialized = True

    def check(self, *args, **kwargs):
        """Reject nominal keyframes after initialization."""
        return False, [], SelectMetrics.zero(self.thresholds)


class _ImuBufferStub:
    """Minimal IMU buffer stub for keyframe construction tests."""

    size = 0
    buffer = np.empty((0, 8), dtype=np.float64)


class TestVIOFrontend:
    """VIO frontend unit tests."""

    def test_backend_feedback_corrects_current_vo_pose_without_rewinding_time(self, mocker) -> None:
        """A delayed backend correction should preserve VO motion accumulated after its keyframe."""
        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.state = np.zeros(16, dtype=np.float32)
        frontend.vo_state = np.zeros(11, dtype=np.float64)
        current_vo_pose = SE3.from_rpy_xyz(np.zeros(3), np.array([2.0, 0.0, 0.0]))
        frontend.vo_state[:7] = current_vo_pose.as_flat_ndarray()
        frontend.vo_state[7:10] = [1.0, 0.0, 0.0]
        frontend.vo_state[10] = 123.0
        frontend.logger = mocker.Mock()
        frontend.apply_new_bias_and_reintegrate = mocker.Mock()

        optimized_pose = SE3.from_rpy_xyz(np.zeros(3), np.array([0.5, 0.0, 0.0]))
        correction = SE3.from_rpy_xyz(np.array([0.0, 0.0, np.pi / 2.0]), np.array([1.0, 0.0, 0.0]))
        feedback = (
            PipelineContext.from_timestamp(100.0)
            .set_ndarray("actual_bias", np.zeros(6))
            .set_ndarray("pose_matrix", optimized_pose.as_matrix())
            .set_ndarray("vo_pose_correction", correction.as_matrix())
            .set_ndarray("optimized_velocity", np.zeros(3))
            .set_scalar("prediction_mode", PredictionMode.PNP.value)
            .reassemble()
        )

        frontend.handle_backend_feedback(feedback)

        corrected_vo_pose = SE3.from_flat_ndarray(frontend.vo_state[:7])
        np.testing.assert_allclose(
            corrected_vo_pose.as_matrix(),
            (correction * current_vo_pose).as_matrix(),
            atol=1e-12,
        )
        np.testing.assert_allclose(frontend.vo_state[7:10], [0.0, 1.0, 0.0], atol=1e-12)
        assert frontend.vo_state[10] == 123.0

    def test_select_keyframes_waits_for_bootstrap_commit(self) -> None:
        """Bootstrap mode should not publish keyframes or call the selector."""
        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.mode = FrontEndMode.BOOTSTRAP
        frontend.kf_selector = _SelectorShouldNotBeCalled()  # ty: ignore[invalid-assignment]
        stereo_frame = np.full((2, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        keyframe_state = np.zeros(16, dtype=np.float32)

        keyframes, metrics = frontend.select_keyframes(
            frame_id=1,
            timestamp=1000.0,
            stereo_frame=stereo_frame,
            keyframe_state=keyframe_state,
            zero_velocity_state=ZeroVelocityTrackerState.UNKNOWN,
        )

        assert keyframes == []
        assert metrics.keyframe_time_diff == 0.0
        assert metrics.keyframe_common_feat_count == 0

    def test_select_keyframes_uses_stereo_frame_payload(self) -> None:
        """Committed bootstrap keyframe should carry the tracked stereo frame."""
        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.mode = FrontEndMode.NOMINAL
        frontend.bootstrap_outcome = FrontEndBootstrapDecision.STATIC
        frontend.kf_selector = KeyFrameSelectorStub()  # ty: ignore[invalid-assignment]
        frontend.state = np.zeros(16, dtype=np.float32)
        frontend.state[:4] = [0.0, 0.0, 0.0, 1.0]
        frontend.imu_buffer = _ImuBufferStub()  # ty: ignore[invalid-assignment]
        stereo_frame = np.full((1, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = [10.0]
        keyframe_state = frontend.state.copy()
        keyframe_state[4:7] = [1.0, 2.0, 3.0]

        keyframes, _metrics = frontend.select_keyframes(
            frame_id=7,
            timestamp=1000.0,
            stereo_frame=stereo_frame,
            keyframe_state=keyframe_state,
            zero_velocity_state=ZeroVelocityTrackerState.UNKNOWN,
        )

        assert len(keyframes) == 1
        np.testing.assert_allclose(keyframes[0].stereo_frame, stereo_frame, equal_nan=True)
        np.testing.assert_array_equal(keyframes[0].state, keyframe_state)
        assert keyframes[0].non_zero_velocity_detected

        repeated_keyframes, _metrics = frontend.select_keyframes(
            frame_id=8,
            timestamp=2000.0,
            stereo_frame=stereo_frame,
            keyframe_state=keyframe_state,
            zero_velocity_state=ZeroVelocityTrackerState.ZERO_VELOCITY,
        )

        assert repeated_keyframes == []

    def test_build_stereo_points_for_visualization_uses_triangulated_stereo_rows(self) -> None:
        """Stereo pointcloud rows should come from one-shot stereo XYZ columns."""
        stereo_frame = np.full((3, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        stereo_frame[:, StereoTriangulationSchema.XYZ] = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        stereo_mask = np.array([True, False, True], dtype=np.bool_)

        stereo_points = VIOFrontend.build_stereo_points_for_visualization(stereo_mask, stereo_frame)

        np.testing.assert_allclose(
            stereo_points,
            np.array(
                [
                    [10.0, 1.0, 2.0, 3.0],
                    [30.0, 7.0, 8.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )
