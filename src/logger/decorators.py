import time
from collections.abc import Callable
from functools import wraps
from types import FunctionType
from typing import Any


def timeit(func: FunctionType) -> Callable[..., Any]:
    """Decorate to time a function and print the duration."""

    @wraps(func)
    def wrapper(*args: tuple, **kwargs: dict) -> Callable:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000

        instance = args[0] if args and hasattr(args[0], "__class__") else None
        class_name = instance.__class__.__name__ if instance else ""
        func_name = func.__name__
        msg = f"{class_name}.{func_name} took {duration:.2f} ms"

        logger = getattr(instance, "logger", None)
        if logger and hasattr(logger, "debug"):
            logger.debug(msg)
        else:
            print(f"DEBUG: {msg}")  # noqa: T201

        return result

    return wrapper
