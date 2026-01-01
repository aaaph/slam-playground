import numpy as np
import pytest

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Feature
from core.front_end.keyframe import Keyframe
from core.graph_optimizer.isam2_optimizer import ISam2Optimizer
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class TestISAM2Optimizer:
    """Unit test for ISAM2 optimizer."""

    @pytest.fixture
    def stereo_ctx(self) -> StereoContext:
        """Create a stereo context."""
        return StereoContext(
            stereo_k=np.array([[456.715, 0.0, 364.44123459], [0.0, 456.715, 256.95167542], [0.0, 0.0, 1.0]]),
            cam0_k=np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]]),
            cam1_k=np.array([[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1.0]]),
            baseline=0.11,
            cam0_in_body_se3=SE3.identity(),
            cam1_in_body_se3=SE3.identity(),
        )

    @pytest.fixture
    def optimizer(self, stereo_ctx: StereoContext) -> ISam2Optimizer:
        """Create an ISAM2 optimizer."""
        return ISam2Optimizer.from_stereo_ctx(stereo_ctx)

    def test_should_be_possible_to_create_from_initial_pose(self, optimizer: ISam2Optimizer) -> None:
        """Test that the ISAM2 optimizer can be created from an initial pose."""
        assert optimizer is not None
        assert optimizer.isam_params is not None
        assert optimizer.ctx.stereo_k is not None
        assert optimizer.ctx.mono_k is not None

    def test_isam2_init_by_first_keyframe(self, optimizer: ISam2Optimizer) -> None:
        """Test that the ISAM2 optimizer can be initialized by a first keyframe."""
        keyframe = Keyframe(
            keyframe_id=0,
            select_reason="initial",
            timestamp=0.0,
            pose=SE3.identity(),
            active_features={},
            active_landmarks={},
        )
        optimizer.update_by_keyframe(keyframe)
        assert optimizer.last_keyframe_id == 0
        assert optimizer.result.exists(X(0))

    def test_isam2_update_two_keyframes(
        self,
        optimizer: ISam2Optimizer,
    ) -> None:
        """Test that the ISAM2 optimizer can update two keyframes."""
        stereo_k = optimizer.ctx.stereo_k
        pose_cam = gtsam.Pose3()
        camera = gtsam.StereoCamera(pose_cam, stereo_k)

        active_features = {}
        active_landmarks = {}

        for i in range(-2, 3):
            y = float(i % 2)
            point_3d = gtsam.Point3(i, y, 10)
            stereo_point = camera.project(point_3d)
            u_l = stereo_point.uL()
            v = stereo_point.v()
            u_r = stereo_point.uR()
            active_features[i + 2] = Feature.spawn_from_left_and_right(i + 2, 0.0, (u_l, v), (u_r, v))
            active_landmarks[i + 2] = np.array([i, 0, 10])

        keyframe1 = Keyframe(
            keyframe_id=0,
            select_reason="initial",
            timestamp=0.0,
            pose=SE3(t=np.array([0, 0, 0])),
            active_features=active_features,
            active_landmarks=active_landmarks,
        )
        optimizer.update_by_keyframe(keyframe1)

        assert optimizer.result.exists(X(0))
        assert optimizer.result.exists(L(0))
        assert optimizer.result.exists(L(1))
        assert optimizer.result.exists(L(2))
        assert optimizer.result.exists(L(3))
        assert optimizer.result.exists(L(4))

        pose_cam = gtsam.Pose3(gtsam.Rot3.Ypr(0, 0, 0), gtsam.Point3(0.2, 0, 0))
        camera = gtsam.StereoCamera(pose_cam, stereo_k)
        active_features = {}
        for i in range(-2, 3):
            y = float(i % 2)
            point_3d = gtsam.Point3(i, y, 10)
            stereo_point = camera.project(point_3d)
            u_l = stereo_point.uL()
            v = stereo_point.v()
            u_r = stereo_point.uR()
            active_features[i + 2] = Feature.spawn_from_left_and_right(i + 2, 10.0, (u_l, v), (u_r, v))

        keyframe2 = Keyframe(
            keyframe_id=10,
            select_reason="big_distance",
            timestamp=10.0,
            pose=SE3(t=np.array([0.2, 0, 0])),
            active_features=active_features,
            active_landmarks=active_landmarks,
        )
        optimizer.update_by_keyframe(keyframe2)

        assert optimizer.result.exists(X(0))
        assert optimizer.result.exists(L(0))
        assert optimizer.result.exists(L(1))
        assert optimizer.result.exists(L(2))
        assert optimizer.result.exists(L(3))
        assert optimizer.result.exists(L(4))
        assert optimizer.result.exists(X(10))
