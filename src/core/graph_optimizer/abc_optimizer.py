from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):
    """Abstract optimizer."""

    @abstractmethod
    def optimize(self) -> None:
        """Optimize the graph."""
