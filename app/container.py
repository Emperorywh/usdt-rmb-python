"""轻量依赖注入容器。

从 :class:`Settings` 构建并装配所有长生命周期组件。容器在 FastAPI 的
lifespan 中实例化一次并挂载到 ``app.state``；路由通过
:mod:`app.api_service.deps` 拉取依赖。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import Settings
from app.data_ingestion.okx_rest import OKXRestClient
from app.data_ingestion.okx_ws import OKXWebSocketClient
from app.data_ingestion.onchain_mock import MockOnchainProvider
from app.data_ingestion.runner import IngestionRunner
from app.data_storage.database import Database
from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.factor_engine.ic_calibrator import ICCalibrator
from app.factor_engine.klines import KlineAggregator
from app.logging_config import get_logger
from app.notification.email_sender import EmailSender
from app.signal_engine.evaluator import SignalEvaluator
from app.signal_engine.lifecycle import LifecycleTracker
from app.signal_engine.llm_agent import LLMAgent
from app.signal_engine.rules import RuleEngine
from app.signal_engine.service import SignalService

logger = get_logger(__name__)

# 启动时如果第一次 instruments 拉取失败，后台 refresh 任务的重试节奏
_INSTRUMENT_REFRESH_INTERVAL_SECONDS = 5 * 60.0


@dataclass
class AppContainer:
    """全局依赖容器。

    说明：
        - ``onchain`` 字段保留 :class:`MockOnchainProvider` 实例占位，
          但**当前不再注入到 IngestionRunner**，因此不会真正写入
          ``onchain_metrics`` 表，等接入真实链上数据源后再恢复。
    """

    settings: Settings
    db: Database
    repos: Repositories
    okx_rest: OKXRestClient
    onchain: Optional[MockOnchainProvider]
    factor_aggregator: FactorAggregator
    rule_engine: RuleEngine
    llm_agent: LLMAgent
    email_sender: EmailSender
    signal_service: SignalService
    ingestion_runner: Optional[IngestionRunner] = field(default=None)
    # 启动后台周期刷新合约面值的任务句柄；shutdown 时一并取消，避免泄漏
    instrument_refresh_task: Optional[asyncio.Task] = field(default=None)
    # P0：多周期 K 线增量聚合器；enable_mtf_factors=False 时不创建（None）
    kline_aggregator: Optional[KlineAggregator] = field(default=None)
    # P2：IC 校准任务；enable_factor_weights_table=False 时不创建（None）
    ic_calibrator: Optional[ICCalibrator] = field(default=None)
    # P2：信号生命周期跟踪任务；enable_lifecycle_tracking=False 时不创建（None）
    lifecycle_tracker: Optional[LifecycleTracker] = field(default=None)
    # P3：信号评估后台任务；enable_signal_evaluation=False 时不创建（None）
    signal_evaluator: Optional[SignalEvaluator] = field(default=None)

    @classmethod
    async def create(cls, settings: Settings) -> "AppContainer":
        # ---- Storage ----
        db = Database(
            dsn=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            max_inactive_connection_lifetime=settings.db_max_inactive_connection_lifetime,
            write_max_retries=settings.db_write_max_retries,
            write_retry_backoff=settings.db_write_retry_backoff,
        )
        await db.connect()
        repos = Repositories(db)

        # ---- Ingestion clients ----
        # 显式禁用系统代理（默认 trust_env=False）并加重试，避免因代理
        # TLS 握手抖动导致 funding / OI 轮询反复抛 httpx.ConnectError。
        okx_rest = OKXRestClient(
            base_url=settings.okx_rest_url,
            timeout=settings.okx_rest_timeout,
            max_retries=settings.okx_rest_max_retries,
            retry_backoff=settings.okx_rest_retry_backoff,
            trust_env=settings.okx_rest_trust_env,
            proxy=settings.okx_rest_proxy or None,
        )
        # 链上数据源暂时不启用：保留 mock 实例占位，便于将来切换到
        # 真实 provider 后只需替换这里的实现，下方 runner 不再注入它。
        onchain: Optional[MockOnchainProvider] = MockOnchainProvider()

        # 合约面值（ctVal）启动策略：
        # ----------------------------------------------------------------
        # 之前是"启动时同步拉 instruments，最多重试 3 次"，在国内直连
        # OKX 经常每次 ConnectTimeout，导致冷启动多花 ≈120s。
        # 现在改成：先用 default_contract_value 占位让服务立刻起来，
        # 同时启动一个后台任务异步拉真值并写回 OKXWebSocketClient。
        # ctVal 默认 0.1（ETH-USDT-SWAP）即便取不到真值也不会偏离太多。
        contract_values: Dict[str, float] = {
            sym: settings.default_contract_value for sym in settings.symbols
        }

        ws_clients = [
            OKXWebSocketClient(
                ws_url=settings.okx_ws_url,
                symbols=settings.symbols,
                depth=settings.orderbook_depth,
                contract_values=contract_values,
                default_contract_value=settings.default_contract_value,
            )
        ]
        # 后台异步拉取真实 ctVal，能拿到就 hot-update 到 ws_client；
        # 拿不到就周期重试，直到所有 symbol 都拿到为止（任务自然退出）。
        instrument_refresh_task = asyncio.create_task(
            cls._refresh_instruments_loop(
                okx_rest=okx_rest,
                symbols=list(settings.symbols),
                ws_clients=ws_clients,
                fallback_value=settings.default_contract_value,
            ),
            name="instrument-refresh",
        )
        # 注意：故意不传 onchain，停用 mock 链上指标的定时写入。
        # 等接入真实链上数据源后再把 onchain=onchain 加回去。
        runner = IngestionRunner(
            settings=settings,
            repos=repos,
            ws_clients=ws_clients,
            rest_client=okx_rest,
        )

        # ---- Factor + signal ----
        factor_aggregator = FactorAggregator(repos=repos, settings=settings)
        # P2：rule_engine 现在持有 repos 用来查 factor_weights 表（带 5min 缓存）
        rule_engine = RuleEngine(settings=settings, repos=repos)
        llm_agent = LLMAgent(settings=settings, repos=repos)
        # 邮件通知发送器：当 LLM 给出明确方向时（long/short）异步给 notification_emails
        # 表里所有 enabled=TRUE 的邮箱推送一封 HTML 提醒邮件。
        # observe / 缓存命中不发；未配置 SMTP 凭据时整体降级为 no-op。
        email_sender = EmailSender(settings=settings)
        signal_service = SignalService(
            repos=repos,
            factor_aggregator=factor_aggregator,
            rule_engine=rule_engine,
            llm_agent=llm_agent,
            email_sender=email_sender,
        )

        # ---- P0：多周期 K 线增量聚合器 ----
        # 仅当开关打开时创建实例并启动；关闭时维持 None，跳过所有 K 线写入，
        # FactorAggregator 自己会回退到老的 30 分钟单一窗口路径。
        kline_aggregator: Optional[KlineAggregator] = None
        if bool(getattr(settings, "enable_mtf_factors", False)):
            kline_aggregator = KlineAggregator(
                repos=repos,
                settings=settings,
                exchange="okx",
            )

        # ---- P2：IC 校准 + 生命周期跟踪 ----
        # 两个独立开关：任意一个关闭都不会破坏 P0/P1 行为，方便灰度上线。
        ic_calibrator: Optional[ICCalibrator] = None
        if bool(getattr(settings, "enable_factor_weights_table", False)):
            # 默认对配置里的第一个 symbol 跑校准。
            # 多 symbol 部署时可以扩展为 list[ICCalibrator]，本期保持单实例。
            primary_symbol = (
                settings.symbols[0] if settings.symbols else "ETH-USDT-SWAP"
            )
            ic_calibrator = ICCalibrator(
                settings=settings, repos=repos, symbol=primary_symbol
            )

        lifecycle_tracker: Optional[LifecycleTracker] = None
        if bool(getattr(settings, "enable_lifecycle_tracking", False)):
            lifecycle_tracker = LifecycleTracker(
                settings=settings,
                repos=repos,
                symbols=list(settings.symbols),
            )

        # ---- P3：信号评估后台任务 ----
        # 仅当总开关打开时创建实例并启动；关闭时维持 None，prompt 注入路径
        # 拿不到 24h 评估摘要会自然降级为"无系统级评估数据"，行为与 P2 一致。
        signal_evaluator: Optional[SignalEvaluator] = None
        if bool(getattr(settings, "enable_signal_evaluation", False)):
            signal_evaluator = SignalEvaluator(
                settings=settings,
                repos=repos,
                symbols=list(settings.symbols),
            )

        return cls(
            settings=settings,
            db=db,
            repos=repos,
            okx_rest=okx_rest,
            onchain=onchain,
            factor_aggregator=factor_aggregator,
            rule_engine=rule_engine,
            llm_agent=llm_agent,
            email_sender=email_sender,
            signal_service=signal_service,
            ingestion_runner=runner,
            instrument_refresh_task=instrument_refresh_task,
            kline_aggregator=kline_aggregator,
            ic_calibrator=ic_calibrator,
            lifecycle_tracker=lifecycle_tracker,
            signal_evaluator=signal_evaluator,
        )

    @staticmethod
    async def _refresh_instruments_loop(
        okx_rest: OKXRestClient,
        symbols: List[str],
        ws_clients: List[OKXWebSocketClient],
        fallback_value: float,
    ) -> None:
        """
        后台周期刷新合约面值的任务体
        ---------------------------------------------------------------
        参数：
            okx_rest:       已经构造好的 REST 客户端（带熔断）
            symbols:        需要刷新的 symbol 列表
            ws_clients:     已经存在的 WS 客户端列表，成功拿到后会被热更新
            fallback_value: 拉不到真值时使用的默认 ctVal（仅用于日志对比）
        说明：
            - 成功拿到的 symbol 从待办列表里移除，所有 symbol 都拿到后任务退出。
            - 失败不打 warn（REST 客户端内部已经做了失败摘要日志）。
            - 任何 sleep 都允许被取消（容器 shutdown 时一并清理）。
        """
        pending: List[str] = list(symbols)
        # 第一次立即试一次，避免冷启动后 5 分钟才有真值
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
        # P3：先停信号评估器；它只读 signals/lifecycle 不写主表，
        # 关停顺序放在 lifecycle 之前避免最后一轮评估读到半结算状态。
        if self.signal_evaluator is not None:
            await self.signal_evaluator.stop()
        # P2：先停 lifecycle / IC 任务，避免它们在 shutdown 期间还在写表
        if self.lifecycle_tracker is not None:
            await self.lifecycle_tracker.stop()
        if self.ic_calibrator is not None:
            await self.ic_calibrator.stop()
        # P0：先停 K 线聚合器再停采集，保证关闭时不会再有"读 trades / 写 klines"
        # 的协程在飞，避免与连接池关闭竞态。
        if self.kline_aggregator is not None:
            await self.kline_aggregator.stop()
        if self.ingestion_runner is not None:
            await self.ingestion_runner.stop()
        await self.okx_rest.close()
        await self.db.disconnect()
