"""
ProScale — Structured Logger
-----------------------------
Provides a consistent logging setup across all modules.
Logs are written to both the console and a rotating file in logs/.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Training started")
    log.info("Episode %d | reward=%.2f | epsilon=%.4f", episode, reward, eps)
    log.warning("Seed file not found: %s", path)
    log.error("Config file missing")
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_LOG_FILE = _LOG_DIR / "proscale.log"

# Format: 2026-04-22 14:03:01 | INFO     | agents.ddqn     | message
_FMT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with console + rotating file handlers attached."""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    # Rotating file handler — DEBUG and above, 5 MB max, 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
