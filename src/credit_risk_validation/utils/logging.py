"""Logging configuration."""

from loguru import logger


def configure_logging(verbose: bool = False) -> None:
    """Configura logging seguro y conciso."""

    logger.remove()
    logger.add(lambda message: print(message, end=""), level="DEBUG" if verbose else "INFO")
