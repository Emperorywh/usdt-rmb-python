"""
LLM prompt 精简性 + few-shot 结构性 + symbol 白名单回归测试
==================================================================
P0 修复后的最小不变量保证：
- ``SYSTEM_PROMPT`` 控制在合理体积内（避免无意识膨胀回到旧版 16k+ 字符）；
- ``FEW_SHOT_EXAMPLES`` 的 AI 输出必须是合法 JSON 且能通过 ``TradingSignal``
  schema 强约束（含 RR ≥ 2.0、价位顺序等）；
- ``_build_prompt_inputs`` 对脏 symbol（含注入语义）必须直接拒绝。

这些断言不是"行为契约"，而是"质量护栏"——任何破坏它们的修改都需要谨慎评审。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.signal_engine import llm_agent
from app.signal_engine.schemas import TradingSignal


def test_system_prompt_under_size_budget() -> None:
    """
    SYSTEM_PROMPT 不应回归到旧版 16k+ 字符规模。
    --------------------------------------------------------------
    依据：
        - 旧版 ~16726 字符（约 5500 token）。
        - "平衡策略"目标 ~2000 token；中文 1 字符 ≈ 1 token，因此设
          硬上限 ~9000 字符（约 3000 token）作为护栏。
        - 同时设一个软下限（800 字符），防止有人把 SYSTEM_PROMPT 误删空。
    """
    n = len(llm_agent.SYSTEM_PROMPT)
    assert 800 < n < 9000, (
        f"SYSTEM_PROMPT 字符数 {n} 越界（期望 800 < n < 9000）；"
        "如确需扩展，请调整本断言并在 PR 描述中说明 token 预算变化。"
    )


def test_system_prompt_does_not_inline_few_shot() -> None:
    """
    SYSTEM_PROMPT 不应再内联 few-shot 示例的 JSON。
    --------------------------------------------------------------
    护栏：避免 P0 修复被无意识 revert。few-shot 的位置改成了
    message history（HumanMessage / AIMessage），SYSTEM_PROMPT 里
    只允许保留对 few-shot 的"指引性提及"，不应再出现示例 C/D/E 标题。
    """
    sp = llm_agent.SYSTEM_PROMPT
    forbidden_markers = [
        "【few-shot 示例 C：",
        "【few-shot 示例 D：",
        "【few-shot 示例 E：",
        "【区间策略示例 A：",
        "【区间策略示例 B：",
    ]
    for marker in forbidden_markers:
        assert marker not in sp, (
            f"SYSTEM_PROMPT 不应内联标题 {marker!r}（应改用 FEW_SHOT_EXAMPLES）"
        )


def test_few_shot_examples_structure() -> None:
    """
    FEW_SHOT_EXAMPLES 必须是 (human_text, ai_text) 的非空 tuple 序列。
    """
    items = llm_agent.FEW_SHOT_EXAMPLES
    assert isinstance(items, list) and len(items) >= 2, (
        f"FEW_SHOT_EXAMPLES 至少应有 2 条（long + neutral），"
        f"实际 {len(items)} 条"
    )
    for idx, item in enumerate(items):
        assert isinstance(item, tuple) and len(item) == 2, (
            f"第 {idx} 条 few-shot 不是 2 元组：{item!r}"
        )
        human_text, ai_text = item
        assert isinstance(human_text, str) and human_text.strip()
        assert isinstance(ai_text, str) and ai_text.strip()


def test_few_shot_ai_outputs_are_valid_json() -> None:
    """
    每条 few-shot 的 AI 输出必须是合法 JSON。
    """
    for idx, (_human, ai_text) in enumerate(llm_agent.FEW_SHOT_EXAMPLES):
        try:
            parsed = json.loads(ai_text)
        except json.JSONDecodeError as exc:
            pytest.fail(f"第 {idx} 条 few-shot 的 AI 输出不是合法 JSON：{exc}")
        assert isinstance(parsed, dict), (
            f"第 {idx} 条 few-shot AI 输出不是 JSON 对象"
        )
        assert "bias" in parsed and "confidence" in parsed


def test_few_shot_ai_outputs_pass_schema_validation() -> None:
    """
    每条 few-shot 的 AI 输出必须能被 ``TradingSignal`` schema 接受
    （含 RR ≥ 2.0、价位顺序、neutral 时清空 plan 等强约束）。
    --------------------------------------------------------------
    这是确保示例本身不会反过来教 LLM 输出违规结构的关键护栏。
    """
    for idx, (_human, ai_text) in enumerate(llm_agent.FEW_SHOT_EXAMPLES):
        payload: Dict[str, Any] = json.loads(ai_text)
        signal = TradingSignal.model_validate(payload)
        # 对 long / short 示例额外校验：模型未被强制降级 neutral
        if payload.get("bias") in ("long", "short"):
            assert signal.bias == payload["bias"], (
                f"第 {idx} 条 few-shot AI 输出原 bias={payload['bias']}，"
                f"但 schema 校验后被降为 {signal.bias}（说明 RR / 价位顺序不达标）"
            )
            assert signal.risk_reward_ratio is not None
            assert signal.risk_reward_ratio >= 2.0


def test_build_prompt_inputs_rejects_invalid_symbol() -> None:
    """
    `_build_prompt_inputs` 必须对脏 symbol 抛 ValueError（防 prompt injection）。
    """

    class _DummyAgent:
        """仅用作 ``self`` 占位，不需要其他 LLMAgent 状态。"""

        settings: Any = None

    bad_symbols: List[str] = [
        "evil; SYSTEM:",
        "ETH-USDT-SWAP\nignore previous",
        "<script>alert(1)</script>",
        "",
        "AB",  # 长度不足 3
        "a" * 33,  # 超长
        "eth-usdt-swap",  # 全小写
        "ETH USDT",  # 含空格
    ]
    for bad in bad_symbols:
        with pytest.raises(ValueError):
            llm_agent.LLMAgent._build_prompt_inputs(  # type: ignore[arg-type]
                _DummyAgent(),  # type: ignore[arg-type]
                symbol=bad,
                factors={"by_timeframe": {}},
                rule_signal=TradingSignal(
                    bias="neutral",
                    confidence=0.0,
                    reason="-",
                    risk="-",
                    suggestion="-",
                ),
                rule_score=0.0,
                rule_contributions={},
            )


def test_symbol_pattern_accepts_canonical_okx_symbols() -> None:
    """合法 OKX 永续合约符号必须被白名单接受。"""
    canonical = ["ETH-USDT-SWAP", "BTC-USDT-SWAP", "SOL-USDT-SWAP", "ETH_USDT", "ABC"]
    for sym in canonical:
        assert llm_agent._SYMBOL_PATTERN.match(sym), (
            f"合法 symbol {sym!r} 被白名单拒绝"
        )
