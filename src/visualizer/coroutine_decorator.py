from collections.abc import Callable, Generator
from functools import wraps

from visualizer.visualizer_context import VisualizerContext


def coroutine(
    func: Callable[..., Generator[None, VisualizerContext]],
) -> Callable[..., Generator[None, VisualizerContext]]:
    """Coroutine decorator."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Generator[None, VisualizerContext]:  # noqa: ANN002, ANN003
        """Wrap the coroutine."""
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return wrapper
