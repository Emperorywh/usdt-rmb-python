"""LLM Agent 门面类（重构后）。

职责：
- 编排完整的 LLM 分析流程（节流 → prompt → 调用 → 解析 → post-check）
- 持有 NarrativeRenderer 做 7 段叙事渲染
- deterministic post-check（RR 诚实性 + plan 完整性）

所有底层逻辑已拆分到：
- llm_client.py  — LLM API 调用 + 延迟链构建 + 结果解析
- llm_prompts.py — Prompt 模板纯常量
- llm_throttle.py — DB 节流 + 缓存重建 + per-symbol 锁
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.logging_config import get_logger
from app.signal_engine.llm_client import LLMClient
from app.signal_engine.llm_throttle import LLMThrottleManager
from app.signal_engine.narrative_renderer import NarrativeRenderer
from app.signal_engine.schemas import LLMAnalysisResult, TradingSignal
from app.utils import safe_float

logger = get_logger(__name__)


class LLMAgent:
    """LLM Agent 门面类。

    编排完整的分析流程：节流 → prompt → 调用 → 解析 → post-check。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        llm_throttle: LLMThrottleManager,
        settings: Settings,
    ):
        self._client = llm_client
        self._throttle = llm_throttle
        self._settings = settings
        self._renderer = NarrativeRenderer()

    @property
    def enabled(self) -> bool:
        """是否启用 LLM。"""
        return self._client.enabled

    async def analyze(
        self,
        symbol: str,
        factors: Dict[str, Any],
        atr_floor_warning: Optional[str] = None,
    ) -> Optional[LLMAnalysisResult]:
        """执行一次 LLM 分析，带按 symbol 的节流缓存。

        流程：
        1. 节流检查（double-checked locking）
        2. 构建 prompt 输入（NarrativeRenderer 渲染 + 占位符填充）
        3. LLM 调用
        4. 结果解析（兼容思考/非思考/直接 TradingSignal 三种返回）
        5. Schema 校验
        6. deterministic post-check
        7. 返回 LLMAnalysisResult
        """
        if not self._client.enabled:
            logger.info("LLM 已禁用（未配置 DEEPSEEK_API_KEY），返回 None")
            return None

        interval = self._throttle.min_interval

        # 节流检查（内含 double-checked locking）
        cached = await self._throttle.check_throttle(symbol, min_interval=interval)
        if cached is not None:
            return cached

        # ---- 真正发起 LLM 调用 ----
        try:
            logger.info("LLM 发起调用 -> %s（固定节流间隔=%ds）", symbol, interval)
            prompt_inputs = self._build_prompt_inputs(
                symbol=symbol, factors=factors, atr_floor_warning=atr_floor_warning,
            )
            _llm_call_start = time.monotonic()
            result = await self._client.invoke(prompt_inputs)
            _llm_call_latency_ms = int(
                (time.monotonic() - _llm_call_start) * 1000
            )
        except Exception:
            logger.exception("LangChain 调用失败，返回 None")
            return None

        # ---- 解析 LLM 返回 ----
        parsed: Any
        raw_message: Any = None
        parsing_error: Any = None

        if isinstance(result, dict) and ("parsed" in result or "raw" in result):
            parsed = result.get("parsed")
            raw_message = result.get("raw")
            parsing_error = result.get("parsing_error")
        elif isinstance(result, TradingSignal):
            parsed = result
        elif hasattr(result, "content"):
            raw_message = result
            content = getattr(result, "content", "")
            parsed = LLMClient.parse_json_content(content, symbol)
            if parsed is None:
                return None
        else:
            parsed = result

        if parsing_error is not None:
            logger.error("LLM 结构化解析错误 %s：%s", symbol, parsing_error)
            return None

        # ---- Schema 校验 ----
        signal: Optional[TradingSignal]
        if isinstance(parsed, TradingSignal):
            signal = parsed
        elif isinstance(parsed, dict):
            try:
                signal = TradingSignal.model_validate(parsed)
            except Exception:
                logger.exception("LLM 输出未通过 schema 校验")
                return None
        else:
            logger.error("LLM 解析结果类型异常 %s：%r", symbol, type(parsed))
            return None

        # ---- 提取元数据 ----
        reasoning_content = LLMClient.extract_reasoning(raw_message)
        in_tok, out_tok, total_tok = LLMClient.extract_token_usage(raw_message)
        logger.info(
            "LLM 调用完成 symbol=%s latency_ms=%d in_tok=%d out_tok=%d total_tok=%d "
            "thinking=%s reasoning_len=%d",
            symbol,
            _llm_call_latency_ms,
            in_tok,
            out_tok,
            total_tok,
            self._client.chain_thinking_mode,
            len(reasoning_content) if reasoning_content else 0,
        )
        if reasoning_content:
            logger.debug("已捕获 LLM 思维链 %s（%d 字符）", symbol, len(reasoning_content))
        elif self._settings.llm.deepseek_thinking_enabled:
            logger.debug(
                "思考模式已启用，但 %s 的原始消息中未找到 reasoning_content", symbol,
            )

        # ---- deterministic post-check ----
        try:
            signal, post_check_issues = self._post_check_signal(signal=signal)
            if post_check_issues:
                logger.info(
                    "LLM post-check 命中 %s：原 bias=%s 校验失败 %d 项 → 已处理",
                    symbol,
                    signal.bias,
                    len(post_check_issues),
                )
        except Exception:
            logger.exception("LLM post-check 异常 symbol=%s（信号原样透传）", symbol)

        return LLMAnalysisResult(
            signal=signal,
            reasoning_content=reasoning_content,
            from_cache=False,
        )

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------
    def _build_prompt_inputs(
        self,
        symbol: str,
        factors: Dict[str, Any],
        atr_floor_warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """渲染 HUMAN_PROMPT 占位符所需的 dict。"""
        LLMClient.validate_symbol(symbol)

        ts = factors.get("computed_at") or ""
        by_tf = factors.get("by_timeframe") or {}
        ms_15m = ((by_tf.get("15m") or {}).get("market_structure")) or {}
        last_close_v = safe_float(ms_15m.get("last_close"))
        last_close = f"{last_close_v:.2f}" if last_close_v is not None else "-"

        sections = self._renderer.render_sections(factors)

        return {
            "symbol": symbol,
            "ts": ts,
            "last_close": last_close,
            "market_state": sections["market_state"],
            "mtf_direction": sections["mtf_direction"],
            "capital_action": sections["capital_action"],
            "derivatives": sections["derivatives"],
            "key_levels": sections["key_levels"],
            "liquidity": sections["liquidity"],
            "liquidations": sections["liquidations"],
            "atr_floor_warning": atr_floor_warning or "",
        }

    # ------------------------------------------------------------------
    # deterministic post-check
    # ------------------------------------------------------------------
    _POST_CHECK_RR_TOLERANCE: float = 0.05

    @classmethod
    def _post_check_signal(
        cls,
        signal: TradingSignal,
    ) -> Tuple[TradingSignal, List[str]]:
        """对 LLM 输出做 deterministic post-check（仅 RR 自动修复 + 信息性日志）。"""
        if signal.bias == "neutral":
            return signal, []

        notes: List[str] = []

        ez = signal.entry_zone
        sl = signal.stop_loss
        tps = list(signal.take_profit or [])

        entry_mid = (float(ez[0]) + float(ez[1])) / 2.0
        sl_f = float(sl)
        tp1 = float(tps[0])
        risk = abs(entry_mid - sl_f)
        reward = abs(tp1 - entry_mid)

        rr_recalc = reward / risk
        rr_self = signal.risk_reward_ratio

        if (
            rr_self is not None
            and rr_self > 0
            and abs(rr_recalc - float(rr_self)) / float(rr_self) > cls._POST_CHECK_RR_TOLERANCE
        ):
            logger.warning(
                "LLM post-check RR 自动修复：llm_post_check_rr_autofix=1 "
                "bias=%s 自报 RR=%.3f 复算 RR=%.3f → 使用复算值",
                signal.bias, float(rr_self), rr_recalc,
            )
            object.__setattr__(signal, "risk_reward_ratio", round(rr_recalc, 4))
            notes.append(
                f"自报 RR={float(rr_self):.3f} 与复算 RR={rr_recalc:.3f} 偏差 "
                f"{abs(rr_recalc - float(rr_self)) / float(rr_self) * 100:.1f}% "
                f"> 容忍 {cls._POST_CHECK_RR_TOLERANCE * 100:.0f}%，已自动修复"
            )

        tfa = signal.timeframe_alignment or {}
        missing_tf = [tf for tf in ("5m", "15m", "1h", "4h", "1d") if tf not in tfa]
        if missing_tf:
            notes.append(
                f"timeframe_alignment 缺周期 {missing_tf}（信息性记录）"
            )

        if notes:
            logger.info(
                "LLM post-check notes（信息性）: bias=%s notes=%s",
                signal.bias, notes,
            )

        return signal, notes
