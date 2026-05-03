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
    logger.info("正在启动 ETH 行情分析平台")

    container = await AppContainer.create(settings)
    app.state.container = container

    if container.ingestion_runner is not None:
        await container.ingestion_runner.start()
    # P0：启动多周期 K 线增量聚合器（仅在 enable_mtf_factors=True 时创建实例）
    # 1m / 5m 每秒一次；15m / 1h 每 10 秒一次；4h / 1d 每分钟一次。
    if container.kline_aggregator is not None:
        await container.kline_aggregator.start(symbols=list(settings.symbols))
    # P2：启动 IC 校准 + 信号生命周期跟踪后台任务
    # （任一开关关闭则容器里对应字段为 None，这里跳过启动；既不查表也不写表）。
    if container.ic_calibrator is not None:
        await container.ic_calibrator.start()
    if container.lifecycle_tracker is not None:
        await container.lifecycle_tracker.start()
    await container.signal_service.start_periodic(
        symbols=settings.symbols,
        interval_seconds=settings.signal_interval_seconds,
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
