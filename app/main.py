"""FastAPI entry point.

Lifespan:
* Build :class:`AppContainer` (DB pool + clients + services).
* Start ingestion runner (WebSocket + REST + on-chain pollers).
* Start periodic signal generation loop.
* Tear everything down cleanly on shutdown.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api_service.routes import router as api_router
from app.config import get_settings
from app.container import AppContainer
from app.logging_config import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("app.main")
    logger.info("Starting ETH Trading Analysis Platform")

    container = await AppContainer.create(settings)
    app.state.container = container

    if container.ingestion_runner is not None:
        await container.ingestion_runner.start()
    await container.signal_service.start_periodic(
        symbols=settings.symbols,
        interval_seconds=settings.signal_interval_seconds,
    )

    try:
        yield
    finally:
        logger.info("Shutting down ETH Trading Analysis Platform")
        await container.signal_service.stop()
        await container.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ETH Trading Analysis Platform",
        description=(
            "Real-time analysis platform for ETH market data. "
            "Outputs structured trading suggestions (advisory only)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
