"""
Logging configuration for sevenrad-stills.

Provides structured logging setup with optional file output.
"""

import logging
import sys
from pathlib import Path

from sevenrad_stills.settings.models import LoggingSettings


def setup_logging(settings: LoggingSettings) -> logging.Logger:
    """
    Set up logging based on settings.

    Args:
        settings: Logging configuration

    Returns:
        Configured logger instance

    """
    logger = logging.getLogger("sevenrad_stills")
    logger.setLevel(settings.level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if settings.log_file:
        log_file = Path(settings.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(settings.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Get the sevenrad-stills logger.

    Returns:
        Logger instance

    """
    return logging.getLogger("sevenrad_stills")
