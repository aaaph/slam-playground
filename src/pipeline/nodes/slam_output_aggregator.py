from types import NoneType

from dora import Node

from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output
from pipeline.nodes.base import PipelineNode


@reactive
class SlamOutputAggregator(PipelineNode):
    """Slam output aggregator."""

    def __init__(self) -> None:
        """Initialize the slam output aggregator."""
        self.node = Node()
        self.logger = spawn_logger(app="slam_output_aggregator")
        self.pgo_T_odom = SE3.identity()
        self.odomentry = SE3.identity()
        self.init_mode = 0
        self.pgo_T_odom_version = 0

    @on_input("frontend_frame")
    @to_output("slam_frame")
    def handle_frontend_frame(self, ctx: Ctx) -> Ctx:
        """Handle the frontend frame."""
        timestamp = ctx.get_scalar("timestamp")
        self.init_mode = int(ctx.get_scalar("front_end_mode"))
        pose_estimate = ctx.get_ndarray("pose_estimate", (4, 4))
        pose_estimate_se3 = SE3.from_matrix(pose_estimate)
        self.odomentry = pose_estimate_se3
        slam_pose = self.pgo_T_odom * self.odomentry
        self.logger.debug(f"Slam pose: {slam_pose}")

        new_ctx = PipelineContext.from_timestamp(timestamp)
        new_ctx.set_ndarray("slam_pose", slam_pose.as_matrix())
        new_ctx.set_scalar("init_mode", self.init_mode)
        new_ctx.set_scalar("pgo_T_odom_version", self.pgo_T_odom_version)
        return new_ctx

    @on_input("pgo_frame")
    def handle_pgo_frame(self, ctx: Ctx) -> NoneType:
        """Handle the pgo frame."""
        self.pgo_T_odom = SE3.from_matrix(ctx.get_ndarray("pgo_T_odom", (4, 4)))
        self.pgo_T_odom_version += 1


if __name__ == "__main__":
    SlamOutputAggregator().run()
