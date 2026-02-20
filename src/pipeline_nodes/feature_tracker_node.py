from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from dataset.euroc import EurocDataset
from logger import node_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, on_stop, reactive, to_output


@reactive
class FeatureTrackerNode:
    """Feature tracker node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the feature tracker node."""
        self.logger = node_logger(app="feature_tracker_node")
        euroc = EurocDataset.mh_01_easy()
        self.camera_model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.camera_model.as_stereo_ctx()
        self.ft = FeatureTracker.default_factory(
            self.stereo_ctx,
            feat_amount_per_region=6,
            feat_retrack_threshold=4,
            region_amount=12,
            mode=FeatureTrackerMode.MONOCULAR,
        )

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> None:
        """Handle the ctx event."""
        has_stereo = bool(ctx.get_scalar("has_stereo"))
        if not has_stereo:
            return ctx
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        timestamp = ctx.get_scalar("timestamp")
        left, right = self.camera_model.process_stereo(left, right)
        self.ft.feed(timestamp, (left, right))
        return (
            ctx.set_image("left_rect", left)
            .set_record_batch("active_feat", self.ft.tensor.as_arrow())
            .reassemble()
        )

    @on_stop
    def graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Feature tracker node stopped")

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")


if __name__ == "__main__":
    FeatureTrackerNode().run()
