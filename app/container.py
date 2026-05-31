"""轻量依赖注入容器。

从 :class:`Settings` 构建并装配所有长生命周期组件。容器在 FastAPI 的
lifespan 中实例化一次并挂载到 ``app.state``；路由通过
:mod:`app.api_service.deps` 拉取依赖。

Phase 2 重构后：
- 使用各独立 Repo（TradeRepo / KlineRepo / ...）替代 Repositories facade
- Worker（TradeBufferWorker / RestWatchdog / ...）由容器创建并注入 Runner
- LLM 拆为 LLMClient + LLMThrottleManager + LLMAgent 门面
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import Settings
from app.data_ingestion.okx_rest import OKXRestClient
from app.data_ingestion.okx_ws import OKXWebSocketClient
from app.data_ingestion.runner import IngestionRunner
from app.data_ingestion.workers.trade_buffer import TradeBufferWorker
from app.data_ingestion.workers.orderbook_writer import OrderbookWriter
from app.data_ingestion.workers.rest_watchdog import RestWatchdog
from app.data_ingestion.workers.retention import RetentionCleaner
from app.data_storage.database import Database
from app.data_storage.repos.trade_repo import TradeRepo
from app.data_storage.repos.kline_repo import KlineRepo
from app.data_storage.repos.orderbook_repo import OrderbookRepo
from app.data_storage.repos.derivatives_repo import DerivativesRepo
from app.data_storage.repos.signal_repo import SignalRepo
from app.data_storage.repos.email_repo import EmailRepo
from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.factor_engine.klines import KlineAggregator
from app.logging_config import get_logger
from app.notification.email_sender import EmailSender
from app.signal_engine.llm_client import LLMClient
from app.signal_engine.llm_throttle import LLMThrottleManager
from app.signal_engine.llm_agent import LLMAgent
from app.signal_engine.service import SignalService

logger = get_logger(__name__)

# 启动时如果第一次 instruments 拉取失败，后台 refresh 任务的重试节奏
_INSTRUMENT_REFRESH_INTERVAL_SECONDS = 5 * 60.0


@dataclass
class AppContainer:
    """全局依赖容器。"""

    settings: Settings
    db: Database
    # 独立 Repo
    trade_repo: TradeRepo
    kline_repo: KlineRepo
    ob_repo: OrderbookRepo
    deriv_repo: DerivativesRepo
    signal_repo: SignalRepo
    email_repo: EmailRepo
    # 向后兼容 facade（aggregator / klines / service / routes 仍使用）
    repos: Repositories
    # Ingestion
    okx_rest: OKXRestClient
    ingestion_runner: Optional[IngestionRunner] = field(default=None)
    # Factor
    factor_aggregator: Optional[FactorAggregator] = field(default=None)
    kline_aggregator: Optional[KlineAggregator] = field(default=None)
    # Signal
    llm_agent: Optional[LLMAgent] = field(default=None)
    email_sender: Optional[EmailSender] = field(default=None)
    signal_service: Optional[SignalService] = field(default=None)
    # Background
    instrument_refresh_task: Optional[asyncio.Task] = field(default=None)

    @classmethod
    async def create(cls, settings: Settings) -> "AppContainer":
        # ---- Storage ----
        db = Database(
            dsn=settings.db.database_url,
            min_size=settings.db.db_pool_min_size,
            max_size=settings.db.db_pool_max_size,
            max_inactive_connection_lifetime=settings.db.db_max_inactive_connection_lifetime,
            acquire_timeout=settings.db.db_pool_acquire_timeout,
            write_max_retries=settings.db.db_write_max_retries,
            write_retry_backoff=settings.db.db_write_retry_backoff,
        )
        await db.connect()

        # 独立 Repo 实例
        trade_repo = TradeRepo(db)
        kline_repo = KlineRepo(db)
        ob_repo = OrderbookRepo(db)
        deriv_repo = DerivativesRepo(db)
        signal_repo = SignalRepo(db)
        email_repo = EmailRepo(db)
        # 向后兼容 facade
        repos = Repositories(db)

        # ---- Ingestion clients ----
        okx_rest = OKXRestClient(
            base_url=settings.ingestion.okx_rest_url,
            timeout=settings.ingestion.okx_rest_timeout,
            max_retries=settings.ingestion.okx_rest_max_retries,
            retry_backoff=settings.ingestion.okx_rest_retry_backoff,
            trust_env=settings.ingestion.okx_rest_trust_env,
            proxy=settings.ingestion.okx_rest_proxy or None,
            breaker_base_cooldown=settings.ingestion.breaker_base_cooldown_seconds,
            breaker_max_cooldown=settings.ingestion.breaker_max_cooldown_seconds,
        )

        contract_values: Dict[str, float] = {
            sym: settings.ingestion.default_contract_value
            for sym in settings.ingestion.symbols
        }

        ws_clients = [
            OKXWebSocketClient(
                ws_url=settings.ingestion.okx_ws_url,
                symbols=settings.ingestion.symbols,
                depth=settings.ingestion.orderbook_depth,
                contract_values=contract_values,
                default_contract_value=settings.ingestion.default_contract_value,
                ping_interval=settings.ingestion.ws_ping_interval_seconds,
            )
        ]

        instrument_refresh_task = asyncio.create_task(
            cls._refresh_instruments_loop(
                okx_rest=okx_rest,
                symbols=list(settings.ingestion.symbols),
                ws_clients=ws_clients,
                fallback_value=settings.ingestion.default_contract_value,
            ),
            name="instrument-refresh",
        )

        # ---- Workers ----
        stopping = asyncio.Event()
        trade_buffer = TradeBufferWorker(
            trade_repo=trade_repo,
            deriv_repo=deriv_repo,
            settings=settings,
            stopping=stopping,
        )
        ob_writer = OrderbookWriter(ob_repo=ob_repo, settings=settings)
        watchdog = RestWatchdog(
            deriv_repo=deriv_repo,
            rest_client=okx_rest,
            settings=settings,
            stopping=stopping,
            # ws_event_at 将在 Runner 启动后由共享引用传入
        )
        retention = RetentionCleaner(
            trade_repo=trade_repo,
            ob_repo=ob_repo,
            signal_repo=signal_repo,
            deriv_repo=deriv_repo,
            settings=settings,
            stopping=stopping,
        )

        # ---- Runner（精简编排器）----
        runner = IngestionRunner(
            settings=settings,
            deriv_repo=deriv_repo,
            ws_clients=ws_clients,
            rest_client=okx_rest,
            trade_buffer=trade_buffer,
            ob_writer=ob_writer,
            watchdog=watchdog,
            retention=retention,
        )
        # 把 Runner 的 WS 健康状态引用注入 Watchdog
        watchdog._ws_event_at = runner._last_ws_event_at

        # ---- Factor ----
        factor_aggregator = FactorAggregator(repos=repos, settings=settings)
        kline_aggregator = KlineAggregator(
            repos=repos,
            settings=settings,
            exchange="okx",
        )

        # ---- Signal（LLM-First 决策核心）----
        llm_client = LLMClient(settings=settings)
        llm_throttle = LLMThrottleManager(
            signal_repo=signal_repo,
            settings=settings,
        )
        llm_agent = LLMAgent(
            llm_client=llm_client,
            llm_throttle=llm_throttle,
            settings=settings,
        )
        email_sender = EmailSender(settings=settings)
        signal_service = SignalService(
            repos=repos,
            factor_aggregator=factor_aggregator,
            llm_agent=llm_agent,
            email_sender=email_sender,
        )

        return cls(
            settings=settings,
            db=db,
            trade_repo=trade_repo,
            kline_repo=kline_repo,
            ob_repo=ob_repo,
            deriv_repo=deriv_repo,
            signal_repo=signal_repo,
            email_repo=email_repo,
            repos=repos,
            okx_rest=okx_rest,
            ingestion_runner=runner,
            factor_aggregator=factor_aggregator,
            kline_aggregator=kline_aggregator,
            llm_agent=llm_agent,
            email_sender=email_sender,
            signal_service=signal_service,
            instrument_refresh_task=instrument_refresh_task,
        )

    @staticmethod
    async def _refresh_instruments_loop(
        okx_rest: OKXRestClient,
        symbols: List[str],
        ws_clients: List[OKXWebSocketClient],
        fallback_value: float,
    ) -> None:
        """后台周期刷新合约面值的任务体。"""
        pending: List[str] = list(symbols)
        first_pass = True
        while pending:
            if not first_pass:
                try:
                    await asyncio.sleep(_INSTRUMENT_REFRESH_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
            first_pass = False
            still_pending: List[str] = []
            for sym in pending:
                try:
                    meta = await okx_rest.fetch_instrument_meta(sym)
                except Exception:  # noqa: BLE001
                    still_pending.append(sym)
                    logger.debug(
                        "合约元数据拉取失败 %s（仍使用 ctVal=%s 占位，将在 %ds 后重试）",
                        sym,
                        fallback_value,
                        int(_INSTRUMENT_REFRESH_INTERVAL_SECONDS),
                    )
                    continue
                ct_val = float(meta["ct_val"])
                for ws in ws_clients:
                    ws.update_contract_value(sym, ct_val)
                logger.info(
                    "已加载合约元数据 %s：ctVal=%s ctValCcy=%s",
                    sym,
                    ct_val,
                    meta.get("ct_val_ccy"),
                )
            pending = still_pending
        logger.info("合约元数据已全部加载，停止后台刷新任务")

    async def shutdown(self) -> None:
        if self.instrument_refresh_task is not None and not self.instrument_refresh_task.done():
            self.instrument_refresh_task.cancel()
            try:
                await self.instrument_refresh_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self.kline_aggregator is not None:
            await self.kline_aggregator.stop()
        if self.ingestion_runner is not None:
            await self.ingestion_runner.stop()
        await self.okx_rest.close()
        await self.db.disconnect()
