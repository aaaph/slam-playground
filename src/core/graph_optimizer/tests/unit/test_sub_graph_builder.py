from collections import deque

import numpy as np
import pytest

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Measurement
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
    def graph_context(self, stereo_ctx: StereoContext) -> GraphContext:
        """Create a graph context."""
        return GraphContext(stereo_ctx)

    def test_build_sub_graph(self, graph_context: GraphContext) -> None:
        """Test the build_sub_graph method."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_meas(0, 1, Measurement(0, (0, 0), (0, 0)))
        sub_graph_builder.add_freeze_prior(0)
        factors, values, _, _ = sub_graph_builder.build()
        assert factors.size() == 2  # prior and measurement
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
        sub_graph_builder.add_meas(0, 1, Measurement(0, (0, 0), (0, 0)))

        sub_graph_builder.add_smart_factor(1, deque([StereoMeasurement(X(0), 0, 0, 0)]))

        assert sub_graph_builder.factor_slot(1) == 2

    def test_factor_slots_types(self, graph_context: GraphContext) -> None:
        """Test the factor slots types."""
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_meas(0, 1, Measurement(0, (0, 0), (0, 0)))
        sub_graph_builder.add_smart_factor(1, deque([StereoMeasurement(X(0), 0, 0, 0)]))
        assert sub_graph_builder._subgraph_factor_types[0] == FactorType.BETWEEN_FACTOR  # noqa: SLF001
        assert sub_graph_builder._subgraph_factor_types[1] == FactorType.LANDMARK  # noqa: SLF001
        assert sub_graph_builder._subgraph_factor_types[2] == FactorType.SMART_FACTOR  # noqa: SLF001

    def test_upper_graph_slots_from_empty(self, graph_context: GraphContext) -> None:
        """Test the upper graph slots."""
        sub_graph_builder = graph_context.new_builder(null_slots=deque(), factors_graph_size=0, timestamp=0.0)
        sub_graph_builder = SubGraphBuilder(graph_context)
        sub_graph_builder.with_pose(0, SE3.identity())
        sub_graph_builder.with_pose(1, gtsam.Pose3())
        sub_graph_builder.add_between_keyframe_prior(1, 0, gtsam.Pose3())
        sub_graph_builder.with_landmark(1, np.array([0, 0, 0]))
        sub_graph_builder.add_meas(0, 1, Measurement(0, (0, 0), (0, 0)))
        sub_graph_builder.add_smart_factor(1, deque([StereoMeasurement(X(0), 0, 0, 0)]))

        assert sub_graph_builder._upper_factor_slots[1] == 2  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[2] == FactorType.SMART_FACTOR  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[1] == FactorType.LANDMARK  # noqa: SLF001
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
        sub_graph_builder.add_meas(0, 1, Measurement(0, (0, 0), (0, 0)))
        sub_graph_builder.add_smart_factor(1, deque([StereoMeasurement(X(0), 0, 0, 0)]))
        sub_graph_builder.add_smart_factor(2, deque([StereoMeasurement(X(0), 0, 0, 0)]))

        assert sub_graph_builder._upper_factor_types[1] == FactorType.BETWEEN_FACTOR  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[10] == FactorType.LANDMARK  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[11] == FactorType.SMART_FACTOR  # noqa: SLF001
        assert sub_graph_builder._upper_factor_types[12] == FactorType.SMART_FACTOR  # noqa: SLF001

        assert sub_graph_builder._upper_factor_slots[1] == 11  # noqa: SLF001
        assert sub_graph_builder._upper_factor_slots[2] == 12  # noqa: SLF001
