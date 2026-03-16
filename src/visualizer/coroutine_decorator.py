from collections.abc import Callable, Generator
from functools import wraps

from pipeline.context import PipelineContext


def coroutine(
    func: Callable[..., Generator[None, PipelineContext]],
) -> Callable[..., Generator[None, PipelineContext]]:
    """Coroutine decorator."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Generator[None, PipelineContext]:  # noqa: ANN002, ANN003
        """Wrap the coroutine."""
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return wrapper
