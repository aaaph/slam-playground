from collections.abc import Callable, Generator
from functools import wraps


def coroutine[TSend](
    func: Callable[..., Generator[None, TSend]],
) -> Callable[..., Generator[None, TSend]]:
    """Coroutine decorator."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Generator[None, TSend]:  # noqa: ANN002, ANN003
        """Wrap the coroutine."""
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return wrapper
