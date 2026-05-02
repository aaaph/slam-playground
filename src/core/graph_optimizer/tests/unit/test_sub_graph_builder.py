from collections import deque

import numpy as np
import pytest

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.vio_context import ImuContext, VioContext
from core.graph_optimizer.optimizer_types import FactorType, StereoMeasurement
from core.graph_optimizer.sub_graph_builder import GraphContext, SubGraphBuilder
from core.transformations.special_euclidian_3_dim import SE3

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X


class TestSubGraphBuilder:
    """Unit test for SubGraphBuilder."""

    @pytest.fixture
    def stereo_ctx(self) -> StereoContext:
        """Create a stereo context."""
        return StereoContext(
            resolution=(100, 100),
            stereo_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam0_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam1_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            baseline=1.0,
            cam0_in_body_se3=SE3.identity(),
            cam1_in_body_se3=SE3.identity(),
        )

    @pytest.fixture
    def imu_ctx(self) -> ImuContext:
        """Create an IMU context."""
        return ImuContext(
            frequency=200.0,
            accel_noise_destiny=1.9393e-05,
            gyro_noise_destiny=1.6968e-04,
            accel_random_walk=3.0000e-3,
            gyro_random_walk=1.9393e-05,
            gravity=np.array([0, 0, -9.81]),
        )

    @pytest.fixture
    def vio_ctx(self, stereo_ctx: StereoContext, imu_ctx: ImuContext) -> VioContext:
        """Create a VIO context."""
        return VioContext(stereo_ctx, imu_ctx)

    @pytest.fixture
    def graph_context(self, vio_ctx: VioContext) -> GraphContext:
        """Create a graph context."""
        return GraphContext(vio_ctx)

    def test_build_sub_graph(self, graph_context: GraphContext) -> None:
        """Test the build_sub_graph method."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_stereo_factor(0, 1, gtsam.StereoPoint2(100.0, 99.5, 50.0))
        sub_graph_builder.add_freeze_prior(0)
        factors, values, _, _ = sub_graph_builder.build()
        assert factors.size() == 2  # stereo landmark factor and pose freeze prior
        assert values.size() == 2  # pose and landmark
        np.testing.assert_allclose(values.atPoint3(L(1)), np.array([0, 0, 0]))
        np.testing.assert_allclose(values.atPose3(X(0)).matrix(), SE3.identity().as_matrix())

    def test_build_sub_graph_with_diff_pose_types(self, graph_context: GraphContext) -> None:
        """Test the build_sub_graph method with different pose types."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        _, values, _, _ = sub_graph_builder.build()
        assert values.size() == 2

    def test_add_freeze_prior_in_all_poses(self, graph_context: GraphContext) -> None:
        """Test the add_freeze_prior_in_all_poses method."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.with_landmark(0, np.array([0, 0, 0]))
        sub_graph_builder.add_freeze_prior_in_all_poses()
        factors, _, _, _ = sub_graph_builder.build()
        assert factors.size() == 2  # 2 poses with freeze prior
        for i in range(factors.size()):
            factor = factors.at(i)
            keys = factor.keys()
            for key in keys:
                assert key != L(0)

    def test_add_between_keyframe_prior(self, graph_context: GraphContext) -> None:
        """Test the add_between_keyframe_prior method."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())
        factors, _, _, _ = sub_graph_builder.build()
        assert factors.size() == 1  # between keyframe prior

    def test_slots_increment(self, graph_context: GraphContext) -> None:
        """Test the slots increment."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())

        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))

        sub_graph_builder.add_smart_factor(2, deque([StereoMeasurement(0, 0, 0, 0)]))

        assert sub_graph_builder.factor_slot(2) == 1

    def test_factor_slots_types(self, graph_context: GraphContext) -> None:
        """Test the factor slots types."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_smart_factor(2, deque([StereoMeasurement(0, 0, 0, 0)]))
        assert sub_graph_builder._subgraph_factor_types[0] == FactorType.BETWEEN_FACTOR  # noqa: SLF001
        assert sub_graph_builder._subgraph_factor_types[1] == FactorType.SMART_FACTOR  # noqa: SLF001

    def test_upper_graph_slots_from_empty(self, graph_context: GraphContext) -> None:
        """Test the upper graph slots."""
        sub_graph_builder = graph_context.new_builder(null_slots=deque(), factors_graph_size=0, timestamp=0.0)
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_smart_factor(2, deque([StereoMeasurement(0, 0, 0, 0)]))

        assert sub_graph_builder._upper_factor_slots[2] == 1  # noqa: SLF001
        assert sub_graph_builder._upper_factor_slots[1] == 0  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[1] == FactorType.SMART_FACTOR  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[0] == FactorType.BETWEEN_FACTOR  # noqa: SLF001

    def test_upper_graph_slots_from_not_empty(self, graph_context: GraphContext) -> None:
        """Test the upper graph slots from not empty."""
        null_slots = deque([1])
        factors_graph_size = 10
        sub_graph_builder = graph_context.new_builder(
            null_slots=null_slots.copy(), factors_graph_size=factors_graph_size, timestamp=0.0
        )
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_smart_factor(2, deque([StereoMeasurement(0, 0, 0, 0)]))
        sub_graph_builder.add_smart_factor(3, deque([StereoMeasurement(0, 0, 0, 0)]))

        assert sub_graph_builder._upper_factor_types[1] == FactorType.BETWEEN_FACTOR  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[10] == FactorType.SMART_FACTOR  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[11] == FactorType.SMART_FACTOR  # noqa: SLF001

        assert sub_graph_builder._upper_factor_slots[1] == 1  # noqa: SLF001
        assert sub_graph_builder._upper_factor_slots[2] == 10  # noqa: SLF001
        assert sub_graph_builder._upper_factor_slots[3] == 11  # noqa: SLF001

    def test_add_pose_prior(self, graph_context: GraphContext) -> None:
        """Test the add_pose_prior method."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        noise_model = gtsam.noiseModel.Constrained.All(6)
        sub_graph_builder.add_pose_prior(0, gtsam.Pose3(), noise_model)
        factors, _, _, _ = sub_graph_builder.build()
        assert factors.size() == 1  # pose prior
        assert factors.at(0).keys() == [X(0)]

    def test_merge_subgraphs(self, graph_context: GraphContext) -> None:
        """Test that two subgraphs can be merged into one."""
        first_builder = graph_context.new_builder(timestamp=10.0, keyframe_id=0)
        first_builder.with_pose(0, SE3.identity())
        first_builder.with_velocity(0, np.zeros(3))
        first_builder.push_delete_slot(1)
        first_subgraph = first_builder.build_subgraph()

        second_builder = graph_context.new_builder(timestamp=15.0, keyframe_id=1)
        second_builder.with_pose(1, SE3.identity())
        second_builder.with_velocity(1, np.ones(3))
        second_builder.with_landmark(5, np.array([1.0, 2.0, 3.0]))
        second_builder.push_delete_slot(2)
        second_subgraph = second_builder.build_subgraph()

        merged = first_subgraph.merge(second_subgraph)

        assert merged.timestamp == second_subgraph.timestamp
        assert merged.keyframe_id == second_subgraph.keyframe_id
        assert merged.factors.size() == first_subgraph.factors.size() + second_subgraph.factors.size()
        assert merged.values.exists(X(0))
        assert merged.values.exists(X(1))
        assert merged.values.exists(L(5))
        assert len(merged.timestamp_map) == len(first_subgraph.timestamp_map) + len(second_subgraph.timestamp_map)
        assert merged.delete_slots == [1, 2]

    def test_merge_subgraphs_rejects_duplicate_values(self, graph_context: GraphContext) -> None:
        """Merging should fail fast when both subgraphs contain the same value key."""
        first_builder = graph_context.new_builder(timestamp=10.0, keyframe_id=0)
        first_builder.with_pose(0, SE3.identity())
        first_subgraph = first_builder.build_subgraph()

        second_builder = graph_context.new_builder(timestamp=15.0, keyframe_id=1)
        second_builder.with_pose(0, SE3.identity())
        second_subgraph = second_builder.build_subgraph()

        with pytest.raises(ValueError, match="duplicate value keys"):
            first_subgraph.merge(second_subgraph)
