import numpy as np
from dora import Node

from core.camera_model.vio_context import VioContext
from core.front_end.keyframe import KF, keyframe_schema
from core.graph_optimizer.explicit_vio_optimizer import ExplicitVIOOptimizer
from core.graph_optimizer.optimizer_types import PredictionMode, VioKeyframe
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, send_pipeline_context_output, to_output
from pipeline.nodes.base import PipelineNode


@reactive
class FixedLagSmoother(PipelineNode):
    """Fixed lag smoother."""

    def __init__(self, vio_ctx: VioContext) -> None:
        """Initialize the fixed lag smoother."""
        self.mode = PredictionMode.PNP
        self.node = Node()
        self.logger = spawn_logger(app="fixed_lag")
        self.vio_ctx = vio_ctx
        self.explicit_vio_opt = ExplicitVIOOptimizer.from_vio_ctx(self.vio_ctx, 10.0 * 1e9)

    @on_input("keyframes")
    @to_output("frame")
    def handle_keyframes(self, ctx: Ctx, metadata: Metadata) -> Ctx:
        """Handle the keyframes event."""
        timestamp = ctx.get_scalar("timestamp")
        front_end_keyframes = KF.list_from_arrow(ctx.get_record_batch("keyframes", keyframe_schema))
        prediction_mode = self.mode
        vio_keyframes: list[VioKeyframe] = [kf.as_vio_kf(prediction_mode) for kf in front_end_keyframes]
        kfid = vio_keyframes[0].keyframe_id

        self.explicit_vio_opt.apply_subgraph(self.explicit_vio_opt.keyframes_to_subgraph(vio_keyframes))

        if self.mode == PredictionMode.PNP:
            accel_bias_sigma = self.explicit_vio_opt.get_accel_bias_sigma()
            accel_bias_converged = np.all(accel_bias_sigma < self.explicit_vio_opt.ctx.sigma_ba_value / 10.0)
            self.logger.info(
                f"[BE:AFTER]: kfid={kfid}, Accel bias sigma: {accel_bias_sigma}, converged: {accel_bias_converged}"
            )
            if accel_bias_converged:
                self.mode = PredictionMode.PIM

        points = self.explicit_vio_opt.get_landmarks_ndarray()
        pose_matrix = self.explicit_vio_opt.get_nav_state().pose().matrix()
        actual_bias = self.explicit_vio_opt.get_actual_bias_ndarray()
        actual_velocity = self.explicit_vio_opt.get_nav_state().velocity()
        self.logger.info(
            f"[BE:AFTER]: kfid={kfid}, Actual pose: {SE3.from_matrix(pose_matrix)}, ",
            f"Actual velocity: {actual_velocity}, Actual bias: {actual_bias}",
        )

        (
            ctx.set_ndarray("optimized_points", points)
            .set_scalar("optimized_points_size", points.shape[0])
            .set_ndarray("cam0_in_body", self.vio_ctx.stereo.cam0_in_body_se3.as_matrix())
            .set_ndarray("optimized_pose", pose_matrix)
            .set_ndarray("optimized_bias", actual_bias)
            .set_ndarray("optimized_accel_bias", actual_bias[:3])
            .set_ndarray("optimized_gyro_bias", actual_bias[3:])
            .set_ndarray("optimized_velocity", actual_velocity)
            .set_scalar("prediction_mode", prediction_mode.value)
            .set_scalar("optimizer_post_fit_error", self.explicit_vio_opt.post_fit_avg_error())
        )

        feedback_ctx = PipelineContext.from_timestamp(timestamp)
        feedback_ctx.set_ndarray("optimized_points", points)
        feedback_ctx.set_scalar("optimized_points_size", points.shape[0])
        feedback_ctx.set_scalar("prediction_mode", self.mode.value)
        feedback_ctx.set_ndarray("actual_bias", actual_bias)
        feedback_ctx.set_ndarray("pose_matrix", pose_matrix)
        feedback_ctx.set_ndarray("optimized_velocity", actual_velocity)

        send_pipeline_context_output(self.node, "feedback", feedback_ctx, metadata)

        return ctx


if __name__ == "__main__":
    FixedLagSmoother(FixedLagSmoother.create_vio_ctx()).run()
