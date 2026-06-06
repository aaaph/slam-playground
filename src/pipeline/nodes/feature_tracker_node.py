import os
from pathlib import Path

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from dataset.manifest import DatasetRigConfig
from dataset.registry import DatasetRegistry
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.context import PipelineContext
from pipeline.decorators import handle, on_input, on_stop, reactive
from pipeline.nodes.base import PipelineNode


@reactive
class FeatureTrackerNode(PipelineNode):
    """Feature tracker node."""

    def __init__(self, camera_model: StereoCameraModel) -> None:
        """Initialize the feature tracker node."""
        self.logger = spawn_logger(app="feature_tracker_node")
        self.camera_model = camera_model
        self.stereo_ctx = self.camera_model.as_stereo_ctx()
        self.ft = FeatureTracker.default_factory(
            self.stereo_ctx,
            feat_amount_per_region=6,
            feat_retrack_threshold=4,
            region_amount=12,
            mode=FeatureTrackerMode.STEREO,
        )

    @handle("sensor_frame", "tracker_frame")
    def handle_sensor_frame(self, ctx: Ctx, metadata: Metadata) -> Ctx:
        """Handle the ctx event."""
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        timestamp = ctx.get_scalar("timestamp")
        left, right = self.camera_model.process_stereo(left, right)
        self.ft.feed(timestamp, (left, right))

        return (
            PipelineContext.from_timestamp(timestamp)
            .set_record_batch("active_feat", self.ft.tensor.as_arrow())
            .set_scalar("features_count", self.ft.tensor.active_frame.count())
            .set_scalar("width", width)
            .set_scalar("height", height)
            .set_scalar("timestamp", timestamp)
            .set_scalar("frame_id", metadata.get("frame_id", 0))
            .set_image("left_rect", left)
        )

    @on_stop
    def graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Feature tracker node stopped")

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")


def load_dataset_rig_from_env() -> DatasetRigConfig:
    """Load dataset rig config declared by the pipeline launcher env."""
    dataset_rig_path = os.getenv("DATASET_RIG_PATH")
    if dataset_rig_path is None:
        raise ValueError("DATASET_RIG_PATH is not set")
    repo_root = os.getenv("REPO_ROOT")
    registry = DatasetRegistry(repo_root=Path(repo_root) if repo_root is not None else None)
    return registry.load_rig(Path(dataset_rig_path))


if __name__ == "__main__":
    FeatureTrackerNode(FeatureTrackerNode.create_stereo_camera_model()).run()
