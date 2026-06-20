from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, reactive, to_output
from pipeline.nodes.base import PipelineNode


@reactive
class SlamOutputAggregator(PipelineNode):
    """Slam output aggregator."""

    def __init__(self) -> None:
        """Initialize the slam output aggregator."""
        self.logger = spawn_logger(app="slam_output_aggregator")
        self.pgo_T_odom = SE3.identity()
        self.pgo_T_odom_version = 0

    @on_input("frontend_frame")
    @to_output("slam_output")
    def handle_frontend_frame(self, ctx: Ctx) -> Ctx:
        """Handle the frontend frame."""
        return ctx

    @on_input("pgo_frame")
    def handle_pgo_frame(self, ctx: Ctx) -> Ctx:
        """Handle the pgo frame."""
        return ctx


if __name__ == "__main__":
    SlamOutputAggregator().run()
