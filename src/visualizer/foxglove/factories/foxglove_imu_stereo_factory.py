from collections.abc import Generator

from visualizer.foxglove.foxglove_visualizer import FoxgloveVisualizer
from visualizer.foxglove.modules.image_module import ImageModule
from visualizer.foxglove.modules.vector_3_module import Vector3Module
from visualizer.visualizer_context import VisualizerContext


class FoxgloveImuStereoFactory:
    """Foxglove IMU and stereo factory."""

    def __init__(self) -> None:
        """Initialize the Foxglove IMU and stereo factory."""

    def create_imu_stereo_viz(
        self, viz_type: str = "websocket", mcap_file_name: str = "default", *, wait_for_client: bool = False
    ) -> Generator[None, VisualizerContext]:
        """Create a Foxglove IMU and stereo viz."""
        foxglove_visualizer = FoxgloveVisualizer(wait_for_client=wait_for_client)
        foxglove_visualizer.add_module(ImageModule(log_warning=False))
        foxglove_visualizer.add_module(Vector3Module(channel_name="/gyro", property_name="gyro"))
        foxglove_visualizer.add_module(Vector3Module(channel_name="/acc", property_name="acc"))
        if viz_type == "websocket":
            return foxglove_visualizer.websocket_viz_gen()
        if viz_type == "mcap":
            return foxglove_visualizer.mcap_gen(mcap_file_name)
        msg = f"Invalid viz type: {viz_type}"
        raise ValueError(msg)
