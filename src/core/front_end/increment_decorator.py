from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from core.front_end.front_end_result import FrontendResult

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime circular imports.
    from core.front_end.front_end import FrontEnd


def increment_counter(
    prop_name: str = "iteration_id",
) -> Callable[[Callable[..., FrontendResult]], Callable[..., FrontendResult]]:
    """Increment the counter property of the instance before returning the result."""

    def decorator(func: Callable[..., FrontendResult]) -> Callable[..., FrontendResult]:
        """Inner decorator."""

        @wraps(func)
        def wrapper(self: "FrontEnd", *args: object, **kwargs: object) -> FrontendResult:
            result = func(self, *args, **kwargs)

            if hasattr(self, prop_name):
                current_value = getattr(self, prop_name)
                setattr(self, prop_name, current_value + 1)
            else:
                msg = f"Property {prop_name} not found in {self}"
                raise ValueError(msg)
            return result

        return wrapper

    return decorator
