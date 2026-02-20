import json
import os

import numpy as np
from dora import Node
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel
from dataset.euroc import EurocDataset
from logger import node_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, on_stop, reactive
from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.rerun_viz_config import VisualizerConfig

type Vector3 = NDArray[np.float32]


class RerunNodeConfigProvider:
    """Rerun node configuration."""

    def __init__(self) -> None:
        """Initialize the rerun node configuration."""
        viz_image_streams_str = os.getenv("VISUALIZE_IMAGE_STREAMS", "{}")
        viz_features_streams_str = os.getenv("VISUALIZE_FEATURES_STREAMS", "{}")
        viz_imu_stream_str = os.getenv("VISUALIZE_IMU_STREAM", "{}")
        self.viz_image_streams: dict[str, str] = json.loads(viz_image_streams_str)
        self.viz_features_streams: dict[str, str] = json.loads(viz_features_streams_str)
        self.viz_imu_streams: dict[str, str] = json.loads(viz_imu_stream_str)

    @property
    def image_stream_names(self) -> list[str]:
        """Get the image streams."""
        return list(self.viz_image_streams.keys())

    @property
    def features_stream_names(self) -> list[str]:
        """Get the features streams."""
        return list(self.viz_features_streams.keys())

    @property
    def imu_stream_names(self) -> list[str]:
        """Get the imu streams."""
        return self.viz_imu_streams["fields"]

    def to_visualizer_config(self, app_name: str, image_resolution: tuple[int, int]) -> VisualizerConfig:
        """Convert the rerun node configuration to a visualizer config."""
        return VisualizerConfig(
            app_name=app_name,
            image_streams=self.viz_image_streams,
            features_streams=self.viz_features_streams,
            image_resolution=image_resolution,
            imu_path=self.viz_imu_streams["entity_path"],
            imu_streams=self.imu_stream_names,
        )


@reactive
class RerunNode:
    """Rerun vizualization node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self, config: RerunNodeConfigProvider) -> None:
        """Initialize the rerun node."""
        self.config = config
        self.node = Node()
        self.logger = node_logger(app="rerun_node")
        euroc = EurocDataset.mh_01_easy()
        self.camera_model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.camera_model.as_stereo_ctx()
        app_name = f"rerun_{self.node.dataflow_id()}"
        self.viz_config = self.config.to_visualizer_config(app_name, self.camera_model.resolution)
        self.vizualizer = RerunConfigFactory.from_config(self.viz_config)

        self.logger.info(self.vizualizer.info())
        self.vizualize = self.vizualizer.pipeline_generator()

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
    RerunNode(config=RerunNodeConfigProvider()).run()
