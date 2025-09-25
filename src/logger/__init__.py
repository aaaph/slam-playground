import sys

import loguru

loguru.logger.remove()
loguru.logger.add(
    sink=sys.stdout,
    level="TRACE",
    format="{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{message}</lvl>",
)

log = loguru.logger.bind(app="vins-rnd")


def spawn_logger(app: str) -> "loguru.Logger":
    """Spawn a logger for an application."""
    return loguru.logger.bind(app=app)
