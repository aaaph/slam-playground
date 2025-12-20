from abc import ABC, abstractmethod

from foxglove.schemas import FrameTransform

from visualizer.visualizer_context import VisualizerContext


class IVizModule(ABC):
    """Interface for a visualizer module."""

    @abstractmethod
    def setup(self) -> None:
        """Abstract method to setup the visualizer module."""

    @abstractmethod
    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Abstract method to process the data."""
