from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

from core.transformations.special_euclidian_3_dim import SE3
from visualizer.foxglove.foxglove_visualizer import FoxgloveVisualizer
from visualizer.foxglove.modules.image_module import ImageModule
from visualizer.foxglove.modules.vector_3_module import Vector3Module


@dataclass(frozen=True)
class ImuAndImageContext:
    """Some new context."""

    timestamp: float | None = None
    gyro: np.ndarray | None = None
    acc: np.ndarray | None = None
    frame: np.ndarray | None = None
    ground_truth_se3: SE3 | None = None


class FoxgloveImuStereoFactory:
    """Foxglove IMU and stereo factory."""

    def __init__(self) -> None:
        """Initialize the Foxglove IMU and stereo factory."""

    def create_imu_stereo_viz(
        self, viz_type: str = "websocket", mcap_file_name: str = "default", *, wait_for_client: bool = False
    ) -> Generator[None, ImuAndImageContext]:
        """Create a Foxglove IMU and stereo viz."""
        foxglove_visualizer = FoxgloveVisualizer[ImuAndImageContext](wait_for_client=wait_for_client)
        foxglove_visualizer.add_module(ImageModule(log_warning=False))
        foxglove_visualizer.add_module(Vector3Module(channel_name="/gyro", property_name="gyro"))
        foxglove_visualizer.add_module(Vector3Module(channel_name="/acc", property_name="acc"))
        if viz_type == "websocket":
            return foxglove_visualizer.websocket_viz_gen()
        if viz_type == "mcap":
            return foxglove_visualizer.mcap_gen(mcap_file_name)
        msg = f"Invalid viz type: {viz_type}"
        raise ValueError(msg)
