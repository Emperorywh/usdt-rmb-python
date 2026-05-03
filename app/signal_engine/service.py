"""Signal service: fuses rule engine + LLM agent and persists outputs."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.logging_config import get_logger
from app.signal_engine.llm_agent import LLMAgent
from app.signal_engine.rules import RuleEngine
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


class SignalService:
    def __init__(
        self,
        repos: Repositories,
        factor_aggregator: FactorAggregator,
        rule_engine: RuleEngine,
        llm_agent: LLMAgent,
    ):
        self.repos = repos
        self.factor_aggregator = factor_aggregator
        self.rule_engine = rule_engine
        self.llm_agent = llm_agent
        self._loops: Dict[str, asyncio.Task[Any]] = {}
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------
    # one-shot generation
    # ------------------------------------------------------------------
    async def generate(self, symbol: str) -> Dict[str, Any]:
        factors = await self.factor_aggregator.compute(symbol)
        rule_signal, rule_score, contributions = self.rule_engine.evaluate(factors)

        llm_signal: Optional[TradingSignal] = await self.llm_agent.analyze(
            symbol=symbol,
            factors=factors,
            rule_signal=rule_signal,
            rule_score=rule_score,
            rule_contributions=contributions,
        )

        final = llm_signal or rule_signal
        source = "rules+llm" if llm_signal is not None else "rules"

        signal_id = await self.repos.insert_signal(
            symbol=symbol,
            bias=final.bias,
            confidence=final.confidence,
            reason=final.reason,
            risk=final.risk,
            suggestion=final.suggestion,
            factors={
                "factors": factors,
                "rule_score": rule_score,
                "rule_contributions": contributions,
                "rule_signal": rule_signal.model_dump(),
            },
            source=source,
        )

        return {
            "id": signal_id,
            "symbol": symbol,
            "source": source,
            "signal": final.model_dump(),
            "rule_signal": rule_signal.model_dump(),
            "rule_score": rule_score,
            "rule_contributions": contributions,
            "factors": factors,
        }

    # ------------------------------------------------------------------
    # periodic background loop (started from FastAPI lifespan)
    # ------------------------------------------------------------------
    async def start_periodic(self, symbols: list[str], interval_seconds: int) -> None:
        for symbol in symbols:
            if symbol in self._loops:
                continue
            self._loops[symbol] = asyncio.create_task(
                self._run_loop(symbol, interval_seconds), name=f"signal-{symbol}"
            )

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._loops.values():
            task.cancel()
        for task in self._loops.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._loops.clear()

    async def _run_loop(self, symbol: str, interval: int) -> None:
        # Wait a bit to let ingestion accumulate data on first start.
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=min(interval, 15))
        except asyncio.TimeoutError:
            pass

        while not self._stopping.is_set():
            try:
                result = await self.generate(symbol)
                logger.info(
                    "Signal[%s] %s confidence=%.2f source=%s",
                    symbol,
                    result["signal"]["bias"],
                    result["signal"]["confidence"],
                    result["source"],
                )
            except Exception:
                logger.exception("Signal generation failed for %s", symbol)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
