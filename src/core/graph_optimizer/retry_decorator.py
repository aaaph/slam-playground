import re
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Literal
from warnings import deprecated

import gtsam

L = gtsam.symbol_shorthand.L
Symbol = gtsam.Symbol

if TYPE_CHECKING:
    from core.graph_optimizer.fixed_lag_optimizer import FixedLagOptimizer

ErrorType = Literal["LANDMARK_ISSUE", "STATE_ISSUE", "UNKNOWN_ISSUE"]


def extract_error_type(error_msg: str) -> tuple[ErrorType, Symbol | None]:
    """Extract the error type and the landmark id from the error message."""
    if "Indeterminant linear system" not in error_msg:
        return "UNKNOWN_ISSUE", None
    match = re.search(r"variable\s+(\d+)\s+\(Symbol:\s+([^)]+)\)", error_msg)
    l_char = ord("l")
    x_char = ord("x")
    if match:
        bad_key = int(match.group(1))
        symbol = Symbol(bad_key)
        chr_value = symbol.chr()
        if chr_value == l_char:
            return "LANDMARK_ISSUE", symbol
        if chr_value == x_char:
            return "STATE_ISSUE", symbol
        return "UNKNOWN_ISSUE", None

    return "UNKNOWN_ISSUE", None


@deprecated("GTSAM smoother is not transactional")
def with_retry(max_attemps: int = 3) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """
    Retry decorator for the optimize method.

    RuntimeError:
    Indeterminant linear system detected while working near variable
    7782220156096217587 (Symbol: l499).

    Thrown when a linear system is ill-posed.  The most common cause for this
    error is having underconstrained variables.  Mathematically, the system is
    underdetermined.  See the GTSAM Doxygen documentation at
    http://borg.cc.gatech.edu/ on gtsam::IndeterminantLinearSystemException for
    more information.
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @wraps(func)
        def wrapper(self: "FixedLagOptimizer", *args: object, **kwargs: object) -> None:
            last_exception = None
            current_args = list(args)
            for _ in range(max_attemps + 1):
                try:
                    return func(self, *current_args, **kwargs)
                except RuntimeError as e:
                    last_exception = e
                    error_msg = str(e)
                    self.logger.exception(error_msg)
                    error_type, landmark_id = extract_error_type(error_msg)
                    self.logger.warning(f"Error type: {error_type}, landmark id: {landmark_id}")
                    if error_type == "LANDMARK_ISSUE" and landmark_id is not None:
                        # gtsam smoother has hidden state...
                        landmark_id = landmark_id.index()
                        self.logger.debug(f"Invalidate landmark {landmark_id} from smoother")
                        self.ignoring_list.append(landmark_id)
                        self.result = self.smoother.calculateEstimate()
                        return None
                    raise
                except IndexError:
                    raise
            msg = f"Failed to execute {func} after {max_attemps} attempts"
            raise last_exception or Exception(msg)

        return wrapper

    return decorator
