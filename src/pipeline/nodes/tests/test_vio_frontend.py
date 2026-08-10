from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np

from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.front_end.front_end_bootstrap import FrontEndBootstrapDecision
from core.front_end.keyframe_selector import KeyFrameSelectThresholds, SelectMetrics
from core.front_end.landmark_cache import LandmarkCache, LandmarkCacheSchema, LandmarkCacheStatus
from core.front_end.landmark_initialization import LandmarkInitialization, LandmarkInitializationFrameSchema
from core.front_end.observation_store import ObservationSchema, ObservationStore
from core.graph_optimizer.optimizer_types import PredictionMode
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from pipeline.nodes.vio_frontend import FrontEndMode, VIOFrontend

if TYPE_CHECKING:
    from core.front_end.landmark_triangulation import LandmarkTriangulatorProtocol


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

    def test_pnp_to_pim_feedback_resets_landmark_initialization(self) -> None:
        """A pose-source switch should discard observations built in the old frame chain."""
        store = ObservationStore(k_inv=np.eye(3), capacity=1)
        cache = LandmarkCache(capacity=1)
        landmark_init = LandmarkInitialization(
            store,
            cache,
            cast(
                "LandmarkTriangulatorProtocol",
                SimpleNamespace(stereo_k=np.eye(3), rect0_from_rect1=np.eye(4)),
            ),
        )
        observation = np.full((1, ObservationSchema.size()), np.nan)
        observation[0, ObservationSchema.FEAT_ID] = 34
        observation[0, ObservationSchema.LEFT_UV] = [10.0, 20.0]
        observation[0, ObservationSchema.RIGHT_UV] = [9.0, 20.0]
        observation[0, ObservationSchema.CAM0_MATRIX] = np.eye(4).reshape(-1)
        slots = store.add_observations(observation)
        cache.commit(
            np.array([34.0]),
            slots.astype(np.float64),
            np.array([LandmarkCacheStatus.COMPLETED.value]),
            np.array([1.0]),
            np.array([[1.0, 2.0, 3.0]]),
        )

        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.estimation_mode = PredictionMode.PNP
        frontend.landmark_init = landmark_init
        frontend.state = np.zeros(16, dtype=np.float32)
        frontend.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
        frontend.apply_new_bias_and_reintegrate = lambda _bias: None
        arrays = {
            "actual_bias": np.zeros(6),
            "pose_matrix": np.eye(4),
            "optimized_velocity": np.zeros(3),
        }
        ctx = SimpleNamespace(
            get_ndarray=lambda name, _shape: arrays[name],
            get_scalar=lambda _name: PredictionMode.PIM.value,
        )

        frontend.handle_backend_feedback(ctx)

        assert frontend.estimation_mode == PredictionMode.PIM
        assert store.get_feat_history(34).shape[0] == 0
        assert cache._data[slots[0], LandmarkCacheSchema.STATUS] == LandmarkCacheStatus.EMPTY.value  # noqa: SLF001

    def test_select_keyframes_waits_for_bootstrap_commit(self) -> None:
        """Bootstrap mode should not publish keyframes or call the selector."""
        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.mode = FrontEndMode.BOOTSTRAP
        frontend.kf_selector = _SelectorShouldNotBeCalled()
        landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
        keyframe_mask = np.zeros(2, dtype=np.bool_)
        keyframe_state = np.zeros(16, dtype=np.float32)

        keyframes, metrics = frontend.select_keyframes(
            frame_id=1,
            timestamp=1000.0,
            landmark_frame=landmark_frame,
            keyframe_mask=keyframe_mask,
            keyframe_state=keyframe_state,
        )

        assert keyframes == []
        assert metrics.keyframe_time_diff == 0.0
        assert metrics.keyframe_common_feat_count == 0

    def test_select_keyframes_uses_landmark_frame_payload(self) -> None:
        """Committed bootstrap keyframe should carry the landmark frame payload."""
        frontend = VIOFrontend.__new__(VIOFrontend)
        frontend.mode = FrontEndMode.NOMINAL
        frontend.bootstrap_outcome = FrontEndBootstrapDecision.STATIC
        frontend.kf_selector = KeyFrameSelectorStub()
        frontend.state = np.zeros(16, dtype=np.float32)
        frontend.state[:4] = [0.0, 0.0, 0.0, 1.0]
        frontend.imu_buffer = _ImuBufferStub()
        frontend.ft = SimpleNamespace(
            metrics=SimpleNamespace(zero_velocity_state=ZeroVelocityTrackerState.ZERO_VELOCITY)
        )
        landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
        landmark_frame[:, LandmarkInitializationFrameSchema.FEAT_ID] = [10.0, 20.0]
        landmark_frame[:, LandmarkInitializationFrameSchema.TRACKED] = [1.0, 0.0]
        keyframe_mask = np.array([True, False])
        keyframe_state = frontend.state.copy()
        keyframe_state[4:7] = [1.0, 2.0, 3.0]

        keyframes, _metrics = frontend.select_keyframes(
            frame_id=7,
            timestamp=1000.0,
            landmark_frame=landmark_frame,
            keyframe_mask=keyframe_mask,
            keyframe_state=keyframe_state,
        )

        assert len(keyframes) == 1
        np.testing.assert_allclose(keyframes[0].landmark_frame, landmark_frame[keyframe_mask], equal_nan=True)
        np.testing.assert_array_equal(keyframes[0].state, keyframe_state)

        repeated_keyframes, _metrics = frontend.select_keyframes(
            frame_id=8,
            timestamp=2000.0,
            landmark_frame=landmark_frame,
            keyframe_mask=keyframe_mask,
            keyframe_state=keyframe_state,
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

    def test_build_landmarks_for_visualization_uses_landmark_xyz_in_current_cam0_frame(self) -> None:
        """Landmark pointcloud rows should use cached landmark XYZ, not stereo-frame prefix columns."""
        landmark_frame = np.full((2, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
        landmark_frame[:, LandmarkInitializationFrameSchema.FEAT_ID] = np.array([10.0, 20.0])
        landmark_frame[:, LandmarkInitializationFrameSchema.TIMESTAMP] = np.array([1000.0, 2000.0])
        landmark_frame[:, LandmarkInitializationFrameSchema.LEFT_UV] = np.array([[110.0, 111.0], [120.0, 121.0]])
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_XYZ] = np.array(
            [
                [11.0, 2.0, 3.0],
                [18.0, -1.0, 0.5],
            ],
            dtype=np.float64,
        )
        success_mask = np.array([True, False], dtype=np.bool_)
        cam0_in_world = SE3(t=np.array([10.0, 0.0, 0.0], dtype=np.float64))

        landmarks = VIOFrontend.build_landmarks_for_visualization(
            success_mask,
            landmark_frame,
            cam0_in_world,
        )

        np.testing.assert_allclose(
            landmarks,
            np.array([[10.0, 1.0, 2.0, 3.0]], dtype=np.float32),
        )
