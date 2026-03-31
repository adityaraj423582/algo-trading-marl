"""
Logging utility for the Algorithmic Trading MARL project.

Provides a consistent, configurable logger that writes to both
console (with colour where supported) and a rotating log file.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Download started")
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.utils.config import LOG_DIR


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = "project.log",
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB per file
    backup_count: int = 3,
) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int
        Minimum log level (default ``logging.INFO``).
    log_file : str
        Name of the log file inside ``LOG_DIR``.
    max_bytes : int
        Max size of a single log file before rotation.
    backup_count : int
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_DIR / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
