from core.camera_model.stereo_camera_model import StereoCameraModel
from dataset.euroc import EurocDataset
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, reactive, to_output


@reactive
class StereoCameraModelNode:
    """Stereo camera model node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the stereo camera model node."""
        self.logger = spawn_logger(app="stereo_camera_model_node")
        euroc = EurocDataset.mh_01_easy()
        self.model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.model.as_stereo_ctx()

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
    StereoCameraModelNode().run()
