import numpy as np
from dora import Node
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel
from dataset.euroc import EurocDataset
from logger import spawn_logger
from pipeline.annotations import (
    EXECUTION_TIME_MS_METADATA_FIELD,
    Ctx,
    ExecutionTimeMetadata,
)
from pipeline.decorators import on_input, on_stop, reactive
from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.loaders import RerunConfigLoader
from visualizer.rerun.schemas import RerunConfigSchema

type Vector3 = NDArray[np.float32]


@reactive
class RerunNode:
    """Rerun vizualization node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self, config: RerunConfigSchema) -> None:
        """Initialize the rerun node."""
        self.node = Node()
        self.config = config
        self.config.app_name = f"rerun_{self.node.dataflow_id()}"

        euroc = EurocDataset.mh_01_easy()
        self.camera_model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.camera_model.as_stereo_ctx()
        self.config.resolution = self.camera_model.resolution

        self.vizualizer = RerunConfigFactory.from_config(self.config)

        self.logger = spawn_logger(app="rerun_node")
        self.logger.info(self.vizualizer.info())
        self.vizualize = self.vizualizer.pipeline_generator()

    @on_input("dataset_frame")
    def handle_dataset_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "dataset_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("frontend_frame")
    def handle_frontend_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "frontend_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("fixedlag_frame")
    def handle_fixedlag_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "fixedlag_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("tracker_frame")
    def handle_tracker_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "frontend_frame",
            ctx,
            execution_time_metadata,
        )

    @on_input("loopclosure_frame")
    def handle_loopclosure_frame(
        self,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Handle the ctx event. Should visualize the context."""
        self.visualize_branch(
            "loopclosure_frame",
            ctx,
            execution_time_metadata,
        )

    def visualize_branch(
        self,
        branch: str,
        ctx: Ctx,
        execution_time_metadata: ExecutionTimeMetadata,
    ) -> None:
        """Materialize reactive metadata into the context and visualize one branch."""
        ctx.set_record_batch(EXECUTION_TIME_MS_METADATA_FIELD, execution_time_metadata)
        self.vizualize.send((branch, ctx.reassemble()))

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")

    @on_stop
    def handle_shutdown(self) -> None:
        """Handle the shutdown event."""
        self.logger.info("Rerun node stopping...")
        self.vizualize.close()
        self.logger.info("Rerun node stopped")


if __name__ == "__main__":
    RerunNode(RerunConfigLoader.from_env_path("VISUALIZE_CONFIG")).run()
