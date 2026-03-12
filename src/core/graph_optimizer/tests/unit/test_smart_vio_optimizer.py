from typing import cast

import numpy as np
import pytest
from gtsam_unstable import SmartStereoProjectionPoseFactor
from numpy.typing import NDArray

import gtsam
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.keyframe_selector import SelectReason
from core.graph_optimizer.smart_vio_optimizer import OptKeyframe, SmartVIOOptimizer
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class TestSmartVIOOptimizer:
    """Unit test for SmartVIOOptimizer."""

    @pytest.fixture
    def optimizer(self, camera_model: StereoCameraModel) -> SmartVIOOptimizer:
        """Create a SmartVIOOptimizer."""
        return SmartVIOOptimizer.from_stereo_ctx(camera_model.as_stereo_ctx())

    def test_first_keyframe(self, optimizer: SmartVIOOptimizer, first_active_track: NDArray[np.float32]) -> None:
        """Test the first keyframe."""
        keyframe = OptKeyframe(
            keyframe_id=0,
            select_reason=SelectReason.INITIAL,
            active_track=first_active_track,
            timestamp=10.0,
            pose=SE3.identity(),
        )

        opt_pose = optimizer.add_new_keyframe(keyframe)
        np.testing.assert_allclose(opt_pose.translation(), np.zeros(3))
        np.testing.assert_allclose(opt_pose.rotation().as_matrix(), np.eye(3))
        assert optimizer.last_keyframe_id == 0

    def test_sequential_keyframes(
        self, optimizer: SmartVIOOptimizer, first_active_track: NDArray[np.float32]
    ) -> None:
        """Test the sequential keyframes. The smart factor should marginilize on out of the horizon."""
        keyframe_one = OptKeyframe(
            keyframe_id=0,
            select_reason=SelectReason.INITIAL,
            active_track=first_active_track,
            timestamp=10.0,
            pose=SE3.identity(),
        )
        opt_pose = optimizer.add_new_keyframe(keyframe_one)
        np.testing.assert_allclose(opt_pose.translation(), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(opt_pose.rotation().as_matrix(), np.eye(3), atol=1e-6)

        keyframe_two = OptKeyframe(
            keyframe_id=1,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=15.0,
            pose=SE3.identity(),
        )
        opt_pose = optimizer.add_new_keyframe(keyframe_two)
        np.testing.assert_allclose(opt_pose.translation(), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(opt_pose.rotation().as_matrix(), np.eye(3), atol=1e-6)

        zero_feat_slot = optimizer.tracks[0].slot
        assert zero_feat_slot > 50
        factor = optimizer.smoother.getFactors().at(zero_feat_slot)
        factor = cast("SmartStereoProjectionPoseFactor", factor)
        assert factor is not None
        assert isinstance(factor, SmartStereoProjectionPoseFactor)
        assert factor.keys() == [X(0), X(1)]
        assert factor.point(optimizer.result) is not None  # ty: ignore
        assert factor.point(optimizer.result).valid()  # ty: ignore

        assert optimizer.result.exists(X(0))

        keyframe_three = OptKeyframe(
            keyframe_id=2,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=21.0,
            pose=SE3.identity(),
        )
        opt_pose = optimizer.add_new_keyframe(keyframe_three)
        np.testing.assert_allclose(opt_pose.translation(), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(opt_pose.rotation().as_matrix(), np.eye(3), atol=1e-6)

        assert optimizer.tracks[0].slot == -1
        assert optimizer.result.exists(X(2))
        assert optimizer.result.exists(X(1))
        assert not optimizer.result.exists(X(0))

        keyframe_four = OptKeyframe(
            keyframe_id=3,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=22.0,
            pose=SE3.identity(),
        )
        opt_pose = optimizer.add_new_keyframe(keyframe_four)
        np.testing.assert_allclose(opt_pose.translation(), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(opt_pose.rotation().as_matrix(), np.eye(3), atol=1e-6)

        assert optimizer.tracks[0].slot == -1

        keyframe_five = OptKeyframe(
            keyframe_id=4,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=23.0,
            pose=SE3.identity(),
        )
        opt_pose = optimizer.add_new_keyframe(keyframe_five)
        np.testing.assert_allclose(opt_pose.translation(), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(opt_pose.rotation().as_matrix(), np.eye(3), atol=1e-6)

        assert optimizer.tracks[0].slot == -1

    def test_sliding_window_control(
        self, optimizer: SmartVIOOptimizer, first_active_track: NDArray[np.float32]
    ) -> None:
        """Test the sliding window control. Optimizer from fixture has 10 lag."""
        keyframe_one = OptKeyframe(
            keyframe_id=0,
            select_reason=SelectReason.INITIAL,
            active_track=first_active_track,
            timestamp=10.0,
            pose=SE3.identity(),
        )
        keyframe_two = OptKeyframe(
            keyframe_id=1,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=15.0,
            pose=SE3.identity(),
        )
        keyframe_three = OptKeyframe(
            keyframe_id=2,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=21.0,
            pose=SE3.identity(),
        )

        optimizer.add_new_keyframe(keyframe_one)
        assert optimizer.sliding_window_poses == {X(0): 10.0}
        for value in optimizer.sliding_window_poses_dq:
            assert value == X(0)
        optimizer.add_new_keyframe(keyframe_two)
        assert optimizer.sliding_window_poses == {X(0): 10.0, X(1): 15.0}
        optimizer.add_new_keyframe(keyframe_three)
        assert optimizer.sliding_window_poses == {X(1): 15.0, X(2): 21.0}

    def test_get_points(self, optimizer: SmartVIOOptimizer, first_active_track: NDArray[np.float32]) -> None:
        """Test the get points method."""
        keyframe_one = OptKeyframe(
            keyframe_id=0,
            select_reason=SelectReason.INITIAL,
            active_track=first_active_track,
            timestamp=10.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_one)
        points_dict = optimizer.get_points()
        assert len(points_dict) == len(first_active_track)
        points_ndarray = optimizer.get_points_ndarray()
        assert points_ndarray.shape == (len(first_active_track), 5)

        keyframe_two = OptKeyframe(
            keyframe_id=1,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=25.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_two)
        points_ndarray = optimizer.get_points_ndarray()
        assert points_ndarray.shape == (len(first_active_track), 5)
        keyframe_three = OptKeyframe(
            keyframe_id=2,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=26.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_three)
        points_ndarray = optimizer.get_points_ndarray()
        assert points_ndarray.shape == (len(first_active_track), 5)

        keyframe_four = OptKeyframe(
            keyframe_id=3,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=27.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_four)
        points_ndarray = optimizer.get_points_ndarray()
        assert points_ndarray.shape == (len(first_active_track), 5)

    def test_get_graph_arrow(self, optimizer: SmartVIOOptimizer, first_active_track: NDArray[np.float32]) -> None:
        """Test the get graph arrow method."""
        keyframe_one = OptKeyframe(
            keyframe_id=0,
            select_reason=SelectReason.INITIAL,
            active_track=first_active_track,
            timestamp=10.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_one)
        keyframe_two = OptKeyframe(
            keyframe_id=1,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=15.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_two)
        keyframe_three = OptKeyframe(
            keyframe_id=2,
            select_reason=SelectReason.BIG_DISTANCE,
            active_track=first_active_track.copy(),
            timestamp=21.0,
            pose=SE3.identity(),
        )
        optimizer.add_new_keyframe(keyframe_three)
        graph_arrow = optimizer.get_graph_arrow()
        assert "nodes" in graph_arrow
