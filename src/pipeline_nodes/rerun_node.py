import numpy as np
from dora import Node
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel
from dataset.euroc import EurocDataset
from logger import node_logger
from pipeline.annotations import Ctx
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

        self.logger.info(self.vizualizer.info())
        self.vizualize = self.vizualizer.pipeline_generator()
        self.logger = node_logger(app="rerun_node")

    @on_input("ctx")
    def handle_ctx(self, ctx: Ctx) -> None:
        """Handle the ctx event. Should visualize the context."""
        timestamp = ctx.get_scalar("timestamp")
        self.logger.debug(f"Timestamp: {timestamp:.0f}")
        self.vizualize.send(ctx)

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
