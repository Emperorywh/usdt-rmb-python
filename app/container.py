"""Lightweight dependency-injection container.

Builds and wires every long-lived component from `Settings`. The container is
instantiated once during FastAPI's lifespan and attached to ``app.state``;
routes pick dependencies up via :mod:`app.api_service.deps`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings
from app.data_ingestion.okx_rest import OKXRestClient
from app.data_ingestion.okx_ws import OKXWebSocketClient
from app.data_ingestion.onchain_mock import MockOnchainProvider
from app.data_ingestion.runner import IngestionRunner
from app.data_storage.database import Database
from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.signal_engine.llm_agent import LLMAgent
from app.signal_engine.rules import RuleEngine
from app.signal_engine.service import SignalService


@dataclass
class AppContainer:
    settings: Settings
    db: Database
    repos: Repositories
    okx_rest: OKXRestClient
    onchain: MockOnchainProvider
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
        okx_rest = OKXRestClient(base_url=settings.okx_rest_url)
        onchain = MockOnchainProvider()

        ws_clients = [
            OKXWebSocketClient(
                ws_url=settings.okx_ws_url,
                symbols=settings.symbols,
                depth=settings.orderbook_depth,
            )
        ]
        runner = IngestionRunner(
            settings=settings,
            repos=repos,
            ws_clients=ws_clients,
            rest_client=okx_rest,
            onchain=onchain,
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
