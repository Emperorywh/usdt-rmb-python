"""FastAPI entry point.

Lifespan:
* Build :class:`AppContainer` (DB pool + clients + services).
* Start ingestion runner (WebSocket + REST pollers).
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
    logger.info("正在启动 ETH 行情分析平台")

    container = await AppContainer.create(settings)
    app.state.container = container

    if container.ingestion_runner is not None:
        await container.ingestion_runner.start()
    # 启动多周期 K 线增量聚合器：1m/5m 每秒一次，15m/1h 每 10 秒，4h/1d 每分钟。
    if container.kline_aggregator is not None:
        await container.kline_aggregator.start(symbols=list(settings.ingestion.symbols))
    # LLM-First 架构下不再启动 IC 校准 / 生命周期跟踪 / 信号评估后台任务——
    # 这三个模块整体删除（plan 第 1.1 / 1.5 节）。
    await container.signal_service.start_periodic(
        symbols=settings.ingestion.symbols,
        interval_seconds=settings.signal.signal_interval_seconds,
    )

    try:
        yield
    finally:
        logger.info("正在关闭 ETH 行情分析平台")
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
