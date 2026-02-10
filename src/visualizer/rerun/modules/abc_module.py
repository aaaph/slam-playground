from abc import ABC, abstractmethod

from visualizer.visualizer_context import VisualizerContext


class IVizModule(ABC):
    """Interface for a visualizer module."""

    @abstractmethod
    def setup(self) -> None:
        """Abstract method to setup the visualizer module."""

    @abstractmethod
    def process(self, context: VisualizerContext) -> None:
        """Abstract method to process the data."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return the string representation of the visualizer module."""
        return f"Module: {self.__class__.__name__}"
