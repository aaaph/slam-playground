import sys

from loguru import logger

logger.remove()
logger.add(
    sink=sys.stdout,
    level="TRACE",
    format="{time:YYYY-MM-DD at HH:mm:ss} | <lvl>{message}</lvl>",
)

log = logger.bind(app="vins-rnd")
