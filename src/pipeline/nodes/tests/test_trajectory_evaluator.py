from unittest.mock import Mock

import numpy as np
import pytest

from core.transformations.special_euclidian_3_dim import SE3
from dataset.ground_truth import GroundTruthIndex
from pipeline.context import PipelineContext
from pipeline.nodes.trajectory_evaluator import TrajectoryEvaluator


def _slam_ctx(timestamp: int, pose: SE3, *, mode: int = 2) -> PipelineContext:
    return (
        PipelineContext.from_timestamp(float(timestamp))
        .set_scalar("init_mode", mode)
        .set_ndarray("slam_pose", pose.as_matrix())
        .reassemble()
    )


def _trajectory_evaluator() -> TrajectoryEvaluator:
    node = TrajectoryEvaluator.__new__(TrajectoryEvaluator)
    node.ground_truth_index = GroundTruthIndex(
        timestamps_ns=np.array([100, 200], dtype=np.int64),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        orientations=np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        velocities=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    )
    node.logger = Mock()
    node.init = False
    node.offset = SE3.identity()
    node.ape_translation_sq_sum = 0.0
    node.ape_rotation_sq_sum = 0.0
    node.ape_samples_count = 0
    return node


class TestTrajectoryEvaluator:
    """Trajectory evaluator unit tests."""

    def test_handle_slam_frame_publishes_trajectory_metrics(self) -> None:
        """Trajectory evaluator should publish APE and running RMSE as a RecordBatch."""
        evaluator = _trajectory_evaluator()

        first_ctx = evaluator.handle_slam_frame(_slam_ctx(100, SE3.identity()))
        second_ctx = evaluator.handle_slam_frame(_slam_ctx(200, SE3(t=np.array([2.0, 0.0, 0.0]))))

        assert first_ctx is not None
        first_metrics = first_ctx.reassemble().get_record_batch("trajectory_metrics")
        assert first_metrics.column("ape_translation_m")[0].as_py() == 0.0
        assert first_metrics.column("rmse_translation_m")[0].as_py() == 0.0
        assert first_metrics.column("samples_count")[0].as_py() == 1

        assert second_ctx is not None
        second_metrics = second_ctx.reassemble().get_record_batch("trajectory_metrics")
        assert second_metrics.column("ape_translation_m")[0].as_py() == 1.0
        assert second_metrics.column("rmse_translation_m")[0].as_py() == pytest.approx(2**-0.5)
        assert second_metrics.column("ape_rotation_deg")[0].as_py() == 0.0
        assert second_metrics.column("rmse_rotation_deg")[0].as_py() == 0.0
        assert second_metrics.column("gt_timestamp_abs_diff_ms")[0].as_py() == 0.0
        assert second_metrics.column("samples_count")[0].as_py() == 2
