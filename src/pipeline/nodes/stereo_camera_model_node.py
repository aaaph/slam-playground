from core.camera_model.stereo_camera_model import StereoCameraModel
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, reactive, to_output
from pipeline.nodes.base import PipelineNode


@reactive
class StereoCameraModelNode(PipelineNode):
    """Stereo camera model node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self, camera_model: StereoCameraModel) -> None:
        """Initialize the stereo camera model node."""
        self.logger = spawn_logger(app="stereo_camera_model_node")
        self.model = camera_model

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> Ctx:
        """Handle the ctx event."""
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        rectified_left, rectified_right = self.model.process_stereo(left, right)
        return (
            ctx.set_image("rectified_left", rectified_left)
            .set_image("rectified_right", rectified_right)
            .reassemble()
        )


if __name__ == "__main__":
    StereoCameraModelNode(StereoCameraModelNode.create_stereo_camera_model()).run()
