import numpy as np
from dora import Node

from core.front_end.keyframe import KF, keyframe_schema
from core.graph_optimizer.explicit_vio_optimizer import ExplicitVIOOptimizer
from core.graph_optimizer.optimizer_types import PredictionMode, VioKeyframe
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output


@reactive
class VIOBackend:
    """VIO backend."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the VIO backend."""
        self.mode = PredictionMode.PNP
        self.node = Node()
        self.logger = spawn_logger(app="vio_backend")
        euroc = EurocDataset.mh_01_easy()
        self.vio_ctx = euroc.config.as_vio_ctx()
        self.explicit_vio_opt = ExplicitVIOOptimizer.from_vio_ctx(self.vio_ctx, 10.0 * 1e9)

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> Ctx:
        """Handle the ctx event."""
        if not ctx.exists("keyframes"):
            return ctx
        timestamp = ctx.get_scalar("timestamp")
        front_end_keyframes = KF.list_from_arrow(ctx.get_record_batch("keyframes", keyframe_schema))
        prediction_mode = self.mode
        vio_keyframes: list[VioKeyframe] = [kf.as_vio_kf(prediction_mode) for kf in front_end_keyframes]

        subgraph = self.explicit_vio_opt.keyframes_to_subgraph(vio_keyframes)
        self.explicit_vio_opt.apply_subgraph(subgraph)

        if self.mode == PredictionMode.PNP:
            accel_bias_sigma = self.explicit_vio_opt.get_accel_bias_sigma()
            accel_bias_converged = np.all(accel_bias_sigma < self.explicit_vio_opt.ctx.sigma_ba_value / 10.0)
            self.logger.info(
                f"[BE:AFTER]: Accel bias sigma: {accel_bias_sigma}, converged: {accel_bias_converged}"
            )
            if accel_bias_converged:
                self.mode = PredictionMode.PIM

        points = self.explicit_vio_opt.get_landmarks_ndarray()
        pose_matrix = self.explicit_vio_opt.get_nav_state().pose().matrix()
        actual_bias = self.explicit_vio_opt.get_actual_bias_ndarray()
        actual_velocity = self.explicit_vio_opt.get_nav_state().velocity()
        self.logger.info(f"[BE:AFTER]: Actual velocity: {actual_velocity}")
        self.logger.info(f"[BE:AFTER]: Actual pose: {SE3.from_matrix(pose_matrix)}")
        self.logger.info(f"[BE:AFTER]: Actual bias: {actual_bias}")
        (
            ctx.set_ndarray("optimized_points", points)
            .set_scalar("optimized_points_size", points.shape[0])
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

        self.node.send_output("feedback", feedback_ctx.reassemble().get_struct())

        return ctx


if __name__ == "__main__":
    VIOBackend().run()
