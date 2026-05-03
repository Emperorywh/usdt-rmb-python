"""Signal service: fuses rule engine + LLM agent and persists outputs."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.logging_config import get_logger
from app.signal_engine.llm_agent import LLMAgent, LLMAnalysisResult
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

        # llm_agent.analyze 返回 LLMAnalysisResult（包装了 TradingSignal +
        # 思考模式下的 reasoning_content）；调用失败 / 未启用时返回 None。
        llm_result: Optional[LLMAnalysisResult] = await self.llm_agent.analyze(
            symbol=symbol,
            factors=factors,
            rule_signal=rule_signal,
            rule_score=rule_score,
            rule_contributions=contributions,
        )

        if llm_result is not None:
            final: TradingSignal = llm_result.signal
            reasoning_content: Optional[str] = llm_result.reasoning_content
            source = "rules+llm"
        else:
            final = rule_signal
            reasoning_content = None
            source = "rules"

        # 仅当 LLM 实际完成分析时才入库，避免在 LLM 未触发 / 调用失败时
        # 将纯规则引擎结果写入 signals 表，造成大量低价值脏数据。
        # 此时仍正常返回规则引擎的实时计算结果用于接口响应与日志观察。
        signal_id: Optional[int] = None
        if llm_result is not None:
            # signals.factors 只保存"原始因子快照 + 规则引擎打分细节"。
            # rule_signal 本身（bias/confidence/reason）可以由 rule_score 重算得到，
            # 不再冗余写入，避免 JSON 体积膨胀。
            # reasoning_content 单独落到 signals.reasoning_content 列，仅作审计，
            # 不参与下游决策、也不会被回灌进下一轮 prompt。
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
                },
                source=source,
                reasoning_content=reasoning_content,
            )
        else:
            logger.debug(
                "Skip persisting signal for %s: LLM analysis not performed (source=%s)",
                symbol,
                source,
            )

        return {
            "id": signal_id,
            "symbol": symbol,
            "source": source,
            "persisted": signal_id is not None,
            "signal": final.model_dump(),
            "rule_signal": rule_signal.model_dump(),
            "rule_score": rule_score,
            "rule_contributions": contributions,
            "factors": factors,
            # 思维链单独以独立字段返回，便于上层接口选择性透出 / 隐藏，
            # 避免污染 signal 主体；未启用思考模式或纯规则路径时为 None。
            "reasoning_content": reasoning_content,
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
