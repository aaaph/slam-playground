import numpy as np
from foxglove.channels import Vector3Channel
from foxglove.schemas import FrameTransform, Vector3

from logger import spawn_logger
from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class Vector3Module(IVizModule):
    """Foxglove Vector 3 module."""

    def __init__(self, channel_name: str, property_name: str) -> None:
        """Initialize the Foxglove Vector 3 module."""
        self.logger = spawn_logger(app="foxglove_vector_3_module")
        self.channel_name = channel_name
        self.property_name = property_name

    def setup(self) -> None:
        """Set up the Foxglove Vector 3 module."""
        self.vector_3_channel = Vector3Channel(self.channel_name)

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the Foxglove Vector 3 module."""
        if getattr(context, self.property_name) is None:
            msg = f"Property {self.property_name} not found in data"
            self.logger.warning(msg)
            raise ValueError(msg)
        vector_3 = getattr(context, self.property_name)
        vector_3 = np.array(vector_3)
        vector_3_message = Vector3(x=vector_3[0], y=vector_3[1], z=vector_3[2])
        sec = None
        if context.timestamp:
            sec = int(context.timestamp // 1_000_000_000)

        self.vector_3_channel.log(vector_3_message, log_time=sec)

        return []
