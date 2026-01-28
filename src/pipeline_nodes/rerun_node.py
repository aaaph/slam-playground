import rerun as rr
from dora import Node

from logger import node_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, reactive


@reactive
class RerunNode:
    """Rerun vizualization node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the rerun node."""
        self.node = Node()
        self.logger = node_logger(app="rerun_node")
        self.rr = rr.init(self.node.dataflow_id(), spawn=True)

    @on_input("ctx")
    def handle_ctx(self, ctx: Ctx) -> None:
        """Handle the ctx event. Should visualize the context."""
        timestamp = ctx.get_scalar("timestamp")
        self.logger.debug(f"Timestamp: {timestamp:.0f}")
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        rr.log("stereo/left", rr.Image(left))
        rr.log("stereo/right", rr.Image(right))


if __name__ == "__main__":
    RerunNode().run()
