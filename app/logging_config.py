"""Centralized logging setup using stdlib logging with consistent formatting."""
from __future__ import annotations

import logging
import sys


_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger. Idempotent."""
    root = logging.getLogger()
    if getattr(root, "_eth_configured", False):
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet down chatty libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    setattr(root, "_eth_configured", True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
