import os
import sys
from typing import TYPE_CHECKING

import loguru

if TYPE_CHECKING:
    from loguru import Record

LOG_LEVEL = os.getenv("LOG_LEVEL", "TRACE")


def _dynamic_format(record: "Record") -> str:
    if record["extra"].get("is_node"):
        return "<level>{level: <7}</level> | <level>{message}</level>\n"

    return "<green>{time:HH:mm:ss}</green> | <cyan>{extra[app]: <15}</cyan> | <level>{message}</level>\n"


loguru.logger.remove()
loguru.logger.add(
    sink=sys.stderr,
    level=LOG_LEVEL,
    format=_dynamic_format,
    enqueue=True,
)
log = loguru.logger.bind(app="vins-rnd")


def spawn_logger(app: str) -> "loguru.Logger":
    """Spawn a logger for an application."""
    return loguru.logger.bind(app=app)


def node_logger(app: str) -> "loguru.Logger":
    """Spawn a logger for a node."""
    return loguru.logger.bind(app=app, is_node=True)
