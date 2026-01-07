from foxglove.channels import Vector2Channel
from foxglove.schemas import FrameTransform, Vector2

from logger import spawn_logger
from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class ErrorModule(IVizModule):
    """Foxglove Error module."""

    def __init__(self) -> None:
        """Initialize the Foxglove Image module."""
        self.logger = spawn_logger(app="foxglove_error_module")

    def setup(self) -> None:
        """Set up the Foxglove Image module."""
        self.absoluter_error_channel = Vector2Channel("/absolute_error")
        self.relative_error_channel = Vector2Channel("/relative_error")

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the Foxglove Error module."""
        if context.errors is None:
            return []
        ate, are, rte, rre = context.errors
        absolute_error_message = Vector2(x=float(ate), y=float(are))
        self.absoluter_error_channel.log(absolute_error_message)
        relative_error_message = Vector2(x=float(rte), y=float(rre))
        self.relative_error_channel.log(relative_error_message)
        return []
