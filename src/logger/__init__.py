import os
import sys
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

import loguru

if TYPE_CHECKING:
    from loguru import Record

LOG_LEVEL = os.getenv("LOG_LEVEL", "TRACE")
TRACE_ID_METADATA_FIELD = "trace_id"

_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def current_trace_id() -> str | None:
    """Get the current trace id."""
    return _current_trace_id.get()


def set_current_trace_id(trace_id: Any) -> Token[str | None]:  # noqa: ANN401
    """Set the current trace id."""
    return _current_trace_id.set(str(trace_id) if trace_id is not None else None)


def reset_current_trace_id(token: Token[str | None]) -> None:
    """Reset the current trace id."""
    _current_trace_id.reset(token)


def bind_trace_id(metadata: dict[str, Any], trace_id: Any) -> str:  # noqa: ANN401
    """Bind trace id to metadata and the current logging context."""
    value = str(trace_id)
    metadata[TRACE_ID_METADATA_FIELD] = value
    _current_trace_id.set(value)
    return value


def _inject_trace_id(record: "Record") -> None:
    trace_id = current_trace_id()
    if trace_id is not None:
        record["extra"][TRACE_ID_METADATA_FIELD] = trace_id


def _dynamic_format(record: "Record") -> str:
    trace = (
        f" | <magenta>trace={{extra[{TRACE_ID_METADATA_FIELD}]}}</magenta>"
        if record["extra"].get(TRACE_ID_METADATA_FIELD)
        else ""
    )
    return f"<level>{{level: <7}}</level>{trace} | <level>{{message}}</level>\n"


loguru.logger.remove()
_logger = loguru.logger.patch(_inject_trace_id)
_logger.add(
    sink=sys.stderr,
    level=LOG_LEVEL,
    format=_dynamic_format,
    enqueue=True,
)
log = _logger.bind(app="vins-rnd")


def spawn_logger(app: str) -> "loguru.Logger":
    """Spawn a logger for an application."""
    return _logger.bind(app=app)
