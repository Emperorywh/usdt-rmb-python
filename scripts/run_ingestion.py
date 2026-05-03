"""Standalone ingestion runner.

Usage:
    python -m scripts.run_ingestion

Useful when you want to deploy ingestion separately from the API process.
"""
from __future__ import annotations

import asyncio
import signal

from app.config import get_settings
from app.container import AppContainer
from app.logging_config import get_logger, setup_logging


async def _main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("scripts.run_ingestion")

    container = await AppContainer.create(settings)
    if container.ingestion_runner is None:
        raise RuntimeError("Ingestion runner not configured")

    logger.info("Starting standalone ingestion process")
    await container.ingestion_runner.start()

    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        logger.info("Stop signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows: signal handlers via add_signal_handler are not supported.
            signal.signal(sig, lambda *_: _request_stop())

    try:
        await stop_event.wait()
    finally:
        await container.shutdown()
        logger.info("Standalone ingestion stopped")


if __name__ == "__main__":
    asyncio.run(_main())
