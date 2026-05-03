"""LangChain analysis agent backed by DeepSeek (OpenAI-compatible).

Uses ``ChatOpenAI`` from ``langchain-openai`` with ``base_url`` overridden to
DeepSeek's endpoint. Output is constrained to :class:`TradingSignal` via
``with_structured_output(...)`` (function-calling), guaranteeing valid JSON or
raising on malformed responses.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.logging_config import get_logger
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


SYSTEM_PROMPT = """\
You are a senior quantitative trading analyst specialised in crypto derivatives.
You are given a structured snapshot of factors derived from real-time market and
on-chain data, plus a preliminary score from a deterministic rule engine.

Output ONE concise trading-bias call as a structured JSON object that conforms
EXACTLY to the provided schema. The signal is advisory only; do not execute.

Guidelines:
- Weight the six factor groups: capital flow, order book, derivatives, on-chain,
  market structure, market participants.
- If signals conflict, prefer "neutral" with lower confidence.
- "confidence" must be in [0, 1] and reflect agreement across factors.
- "reason" must cite the strongest specific numbers from the factors.
- "risk" must list invalidation conditions (e.g. funding flips, key level break).
- "suggestion" must be specific (entry zone, invalidation, target). Mention that
  this is advisory only.
- Always answer in English unless the user explicitly requests another language.
"""

HUMAN_PROMPT = """\
Symbol: {symbol}
Timestamp: {ts}

Rule-engine pre-screen: bias={rule_bias} confidence={rule_confidence} score={rule_score}
Rule contributions: {rule_contributions}

Factors (JSON):
{factors_json}

Produce the final TradingSignal."""


class LLMAgent:
    """Wrapper around the LangChain DeepSeek chain with structured output."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._chain = None  # built lazily

    def _build_chain(self):
        # Imported lazily so unit tests can monkeypatch without an API key.
        from langchain_openai import ChatOpenAI

        if not self.settings.deepseek_api_key:
            logger.warning(
                "DEEPSEEK_API_KEY is empty - LLM analysis will be skipped at runtime"
            )

        llm = ChatOpenAI(
            model=self.settings.deepseek_model,
            api_key=self.settings.deepseek_api_key or "missing-key",
            base_url=self.settings.deepseek_base_url,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout,
            max_retries=2,
        )
        structured = llm.with_structured_output(
            TradingSignal, method="function_calling"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        return prompt | structured

    @property
    def enabled(self) -> bool:
        return bool(self.settings.deepseek_api_key)

    async def analyze(
        self,
        symbol: str,
        factors: Dict[str, Any],
        rule_signal: TradingSignal,
        rule_score: float,
        rule_contributions: Dict[str, float],
    ) -> Optional[TradingSignal]:
        """Run the LLM. Returns ``None`` on disabled agent or any failure."""

        if not self.enabled:
            logger.info("LLM disabled (no DEEPSEEK_API_KEY); using rule-engine output")
            return None

        if self._chain is None:
            try:
                self._chain = self._build_chain()
            except Exception:
                logger.exception("Failed to build LangChain chain")
                return None

        try:
            result = await self._chain.ainvoke(
                {
                    "symbol": symbol,
                    "ts": factors.get("computed_at"),
                    "rule_bias": rule_signal.bias,
                    "rule_confidence": rule_signal.confidence,
                    "rule_score": round(rule_score, 4),
                    "rule_contributions": json.dumps(rule_contributions),
                    "factors_json": json.dumps(factors, ensure_ascii=False, default=str),
                }
            )
        except Exception:
            logger.exception("LangChain invocation failed; falling back to rule engine")
            return None

        if isinstance(result, TradingSignal):
            return result
        try:
            return TradingSignal.model_validate(result)
        except Exception:
            logger.exception("LLM output failed schema validation")
            return None
