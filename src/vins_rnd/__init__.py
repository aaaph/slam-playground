from loguru import logger


def main() -> None:
    """Run the main vins-rnd application."""
    logger.info({"123": "Hello from vins-rnd!"})
    logger.debug("Hello from vins-rnd!")
    logger.info("Hello from vins-rnd!")
    logger.warning("Hello from vins-rnd!")
    logger.error("Hello from vins-rnd!")
    logger.critical("Hello from vins-rnd!")
