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
            # 区分两种 source：
            # - "rules+llm"        ：本次真正触发了一次 LLM API 调用
            # - "rules+llm(cache)" ：本次命中了 LLM 节流缓存（默认 15 分钟），
            #                       并未实际调用 LLM。它只用于响应/日志展示，
            #                       不会落库（参见下方 should_persist 判断）。
            source = "rules+llm(cache)" if llm_result.from_cache else "rules+llm"
        else:
            final = rule_signal
            reasoning_content = None
            source = "rules"

        # 入库策略：
        # 1) llm_result 为 None（LLM 未启用 / 调用失败 / schema 校验失败）：不入库。
        #    避免把纯规则引擎结果当成 LLM 结论写库，造成大量低价值脏数据。
        # 2) llm_result.from_cache 为 True：本次只是复用 15 分钟内的 LLM 缓存，
        #    并未真正发起新的 LLM 推理；如果继续入库，会出现每 30s 一条
        #    reason/risk/suggestion 完全相同、只有 factors 不同的伪 LLM 记录，
        #    既污染 signals 表，也会让下游分析误以为 LLM 在反复确认同一判断。
        # 3) 只有"真正发起了一次 LLM 调用并成功解析"时（即 from_cache=False），
        #    才把这条结果落库。这样入库节奏就严格对齐 LLM_MIN_INTERVAL_SECONDS
        #    （默认 15 分钟一条），与成本预算一致。
        # 不入库时仍正常返回规则引擎 / 缓存中的判断用于接口响应与日志观察。
        should_persist = llm_result is not None and not llm_result.from_cache
        signal_id: Optional[int] = None
        if should_persist:
            # signals.factors 只保存"原始因子快照 + 规则引擎打分细节"。
            # rule_signal 本身（bias/confidence/reason）可以由 rule_score 重算得到，
            # 不再冗余写入，避免 JSON 体积膨胀。
            # reasoning_content 单独落到 signals.reasoning_content 列，仅作审计，
            # 不参与下游决策、也不会被回灌进下一轮 prompt。
            # P0 升级：把结构化交易计划字段一并落库；neutral 时全部为 None / 空。
            entry_zone_payload = (
                list(final.entry_zone) if final.entry_zone is not None else None
            )
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
                entry_zone=entry_zone_payload,
                stop_loss=final.stop_loss,
                take_profit=list(final.take_profit) if final.take_profit else None,
                risk_reward_ratio=final.risk_reward_ratio,
                position_size_pct=final.position_size_pct,
                timeframe_alignment=(
                    dict(final.timeframe_alignment)
                    if final.timeframe_alignment
                    else None
                ),
                invalidation_conditions=(
                    list(final.invalidation_conditions)
                    if final.invalidation_conditions
                    else None
                ),
            )
        else:
            logger.debug(
                "跳过信号入库 %s：source=%s（llm_present=%s，from_cache=%s）",
                symbol,
                source,
                llm_result is not None,
                llm_result.from_cache if llm_result is not None else None,
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
        # ------------------------------------------------------------------
        # 冷启动 warmup：等到一个完整 factor_window 攒满再触发首轮信号
        # ------------------------------------------------------------------
        # 背景：
        #   原实现只 sleep min(interval, 15)=15s 就开跑，但 factor_window
        #   默认 1800s（5min 也不够），首轮聚合时窗口里几乎没数据：
        #   - market_structure 1 根 bar 都凑不齐 → trend=neutral；
        #   - capital_flow 只反映几十秒成交，却被 LLM 当成 5min/30min 数据；
        #   - derivatives 在 funding/OI 还没推第一帧时全为 None；
        #   规则打分常因为 capital_flow 单边贡献越过 ±0.25 阈值生成伪信号，
        #   还会把这条质量很差的判断写进 signals 表（LLM 真调用过一次）。
        #
        # 解决：
        #   把 warmup 上限提高到与因子窗口等长，最低不少于 interval。
        #   factor_window 之外还额外加一段 settle 余量，给 OKX funding-rate
        #   / open-interest 频道首推留出窗口（funding 大约 1min 推一次）。
        warmup_seconds = max(
            interval,
            int(self.factor_aggregator.settings.factor_window_seconds),
        )
        # 用配置而不是硬编码常量，方便单测把 factor_window 调小后能立刻
        # 跑出第一条信号；这里至少额外多等 60s 让 funding-rate 首帧到位。
        warmup_seconds += 60
        logger.info(
            "信号循环 %s 冷启动 warmup %ds（等待因子窗口攒满）",
            symbol,
            warmup_seconds,
        )
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=warmup_seconds)
        except asyncio.TimeoutError:
            pass

        while not self._stopping.is_set():
            try:
                result = await self.generate(symbol)
                logger.info(
                    "信号[%s] %s confidence=%.2f source=%s",
                    symbol,
                    result["signal"]["bias"],
                    result["signal"]["confidence"],
                    result["source"],
                )
            except Exception:
                logger.exception("信号生成失败 %s", symbol)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
