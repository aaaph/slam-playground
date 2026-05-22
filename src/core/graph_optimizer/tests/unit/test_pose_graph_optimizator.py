import numpy as np

import gtsam
from core.graph_optimizer.pose_graph_optimizator import (
    EdgeType,
    LoopClosure,
    PoseGraphOptimizator,
    PoseGraphSnapshot,
    trajectory_arrow_schema,
)
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X


class TestPoseGraphOptimizator:
    """Unit test for PoseGraphOptimizator."""

    def test_update_by_pose(self) -> None:
        """Test the update_by_pose method."""
        pose_graph_optimizator = PoseGraphOptimizator()
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        assert pose_graph_optimizator.values.size() == 1
        assert pose_graph_optimizator.factors.size() > 0

    def test_update_should_add_odometry_factor(self) -> None:
        """Test the update_by_pose method should add odometry factor."""
        pose_graph_optimizator = PoseGraphOptimizator()
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(1, SE3.identity())
        assert pose_graph_optimizator.factors.size() == 2
        assert pose_graph_optimizator.factors.at(0).keys() == [X(0)]
        assert pose_graph_optimizator.factors.at(1).keys() == [X(0), X(1)]

    def test_optimize(self) -> None:
        """Test the optimize method."""
        pose_graph_optimizator = PoseGraphOptimizator()
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(
            1, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([1, 0, 0]))
        )
        result = pose_graph_optimizator.optimize()
        assert result.exists(X(0))
        assert result.exists(X(1))

    def test_pose_dict(self) -> None:
        """Test the pose dict."""
        pose_graph_optimizator = PoseGraphOptimizator()
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(
            1, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([1, 0, 0]))
        )
        assert pose_graph_optimizator.pose_dict.get(0, None) is not None
        assert pose_graph_optimizator.pose_dict.get(1, None) is not None

    def test_poses_ndarray(self) -> None:
        """Test the poses ndarray."""
        pose_graph_optimizator = PoseGraphOptimizator()
        assert hasattr(pose_graph_optimizator, "poses_ndarray")
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(
            1, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([1, 0, 0]))
        )
        poses_array = pose_graph_optimizator.poses_ndarray()
        assert poses_array.shape == (2, 8)
        assert poses_array[0, 0] == 0
        assert poses_array[1, 0] == 1

    def test_edges_ndarray(self) -> None:
        """Test the edges ndarray."""
        pose_graph_optimizator = PoseGraphOptimizator()
        assert hasattr(pose_graph_optimizator, "edges_ndarray")
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(
            1, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([1, 0, 0]))
        )
        pose_graph_optimizator.update_by_pose(
            2, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([2, 0, 0]))
        )
        edges_array = pose_graph_optimizator.edges_ndarray()
        assert edges_array.shape == (2, 3)
        assert edges_array[0, 0] == 0
        assert edges_array[0, 1] == 1
        assert edges_array[0, 2] == EdgeType.ODOMETRY.value
        assert edges_array[1, 0] == 1
        assert edges_array[1, 1] == 2
        assert edges_array[1, 2] == EdgeType.ODOMETRY.value

    def test_update_by_loop_closure_should_add_loop_factor(self) -> None:
        """Loop closure should add a Pose3 between factor and edge metadata."""
        pose_graph_optimizator = PoseGraphOptimizator()
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(
            1, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([1, 0, 0]))
        )
        pose_graph_optimizator.update_by_loop_closure(
            LoopClosure(
                from_key=0,
                to_key=1,
                transform=SE3.from_quat_and_translation(
                    quat=np.array([0, 0, 0, 1]),
                    translation=np.array([1, 0, 0]),
                ),
                cam0_in_body=SE3.identity(),
            )
        )

        loop_factor = pose_graph_optimizator.factors.at(2)
        edges_array = pose_graph_optimizator.edges_ndarray()

        assert pose_graph_optimizator.factors.size() == 3
        assert loop_factor.keys() == [X(0), X(1)]
        assert edges_array[-1, 0] == 0
        assert edges_array[-1, 1] == 1
        assert edges_array[-1, 2] == EdgeType.LOOP_CLOSURE.value

    def test_update_by_loop_closure_transform_should_follow_gtsam_between_convention(self) -> None:
        """Loop closure transform should be X(from).between(X(to))."""
        pose_graph_optimizator = PoseGraphOptimizator()
        from_pose = SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([5, 0, 0]))
        to_pose = SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([6, 0, 0]))
        loop_transform = from_pose.inverse() * to_pose
        pose_graph_optimizator.update_by_pose(0, from_pose)
        pose_graph_optimizator.update_by_pose(1, to_pose)
        pose_graph_optimizator.update_by_loop_closure(
            LoopClosure(from_key=0, to_key=1, transform=loop_transform, cam0_in_body=SE3.identity())
        )

        result = pose_graph_optimizator.optimize()

        np.testing.assert_allclose(result.atPose3(X(0)).translation(), np.array([5.0, 0.0, 0.0]), atol=1e-7)
        np.testing.assert_allclose(result.atPose3(X(1)).translation(), np.array([6.0, 0.0, 0.0]), atol=1e-7)

    def test_to_trajectory_arrow(self) -> None:
        """Test the to_trajectory_arrow method."""
        pose_graph_optimizator = PoseGraphOptimizator()
        pose_graph_optimizator.update_by_pose(0, SE3.identity())
        pose_graph_optimizator.update_by_pose(
            1, SE3.from_quat_and_translation(quat=np.array([0, 0, 0, 1]), translation=np.array([1, 0, 0]))
        )
        trajectory_arrow = pose_graph_optimizator.to_trajectory().to_arrow()
        assert trajectory_arrow.schema == trajectory_arrow_schema
        trajectory = PoseGraphSnapshot.from_arrow(trajectory_arrow)
        np.testing.assert_array_equal(trajectory.poses, pose_graph_optimizator.poses_ndarray())
        np.testing.assert_array_equal(trajectory.edges, pose_graph_optimizator.edges_ndarray())
