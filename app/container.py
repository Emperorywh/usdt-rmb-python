"""轻量依赖注入容器。

从 :class:`Settings` 构建并装配所有长生命周期组件。容器在 FastAPI 的
lifespan 中实例化一次并挂载到 ``app.state``；路由通过
:mod:`app.api_service.deps` 拉取依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from app.config import Settings
from app.data_ingestion.okx_rest import OKXRestClient
from app.data_ingestion.okx_ws import OKXWebSocketClient
from app.data_ingestion.onchain_mock import MockOnchainProvider
from app.data_ingestion.runner import IngestionRunner
from app.data_storage.database import Database
from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.logging_config import get_logger
from app.signal_engine.llm_agent import LLMAgent
from app.signal_engine.rules import RuleEngine
from app.signal_engine.service import SignalService

logger = get_logger(__name__)


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
    signal_service: SignalService
    ingestion_runner: Optional[IngestionRunner] = field(default=None)

    @classmethod
    async def create(cls, settings: Settings) -> "AppContainer":
        # ---- Storage ----
        db = Database(
            dsn=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
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

        # 启动时尝试拉一次 instruments 元数据，构造 ctVal 映射。
        # 失败时回退到配置中的 default_contract_value，让服务仍然能跑起来。
        contract_values: Dict[str, float] = {}
        for sym in settings.symbols:
            try:
                meta = await okx_rest.fetch_instrument_meta(sym)
                contract_values[sym] = float(meta["ct_val"])
                logger.info(
                    "Loaded instrument meta %s: ctVal=%s ctValCcy=%s",
                    sym,
                    meta["ct_val"],
                    meta.get("ct_val_ccy"),
                )
            except Exception:  # noqa: BLE001
                contract_values[sym] = settings.default_contract_value
                logger.warning(
                    "Falling back to default ctVal=%s for %s",
                    settings.default_contract_value,
                    sym,
                )

        ws_clients = [
            OKXWebSocketClient(
                ws_url=settings.okx_ws_url,
                symbols=settings.symbols,
                depth=settings.orderbook_depth,
                contract_values=contract_values,
                default_contract_value=settings.default_contract_value,
            )
        ]
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
        rule_engine = RuleEngine(settings=settings)
        llm_agent = LLMAgent(settings=settings)
        signal_service = SignalService(
            repos=repos,
            factor_aggregator=factor_aggregator,
            rule_engine=rule_engine,
            llm_agent=llm_agent,
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
            signal_service=signal_service,
            ingestion_runner=runner,
        )

    async def shutdown(self) -> None:
        if self.ingestion_runner is not None:
            await self.ingestion_runner.stop()
        await self.okx_rest.close()
        await self.db.disconnect()
