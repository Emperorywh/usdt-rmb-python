"""SignalService 决策守门员（4 道闸门）单元测试
=================================================================
对应 P3 升级：

闸 1 (_decision_atr_gate)
    atr_pct_15m 低于阈值 → 触发 atr_too_low

闸 2 (_decision_cooldown_gate)
    最近连续 N 条 sl_hit 且 latest exit_at 未过冷静期 → 触发

闸 3 (_decision_direction_stability_gate)
    方向反转 + |Δprice| < 0.6 × ATR(1h) → 触发

闸 4 (_decision_rule_conflict_gate)
    sign(rule_score) ≠ sign(LLM bias) 且历史冲突胜率 < 阈值 → 触发

辅助方法 _make_gated_neutral_signal / _force_neutral_preserving_text
    构造的 TradingSignal 必须满足 schema：
        bias=neutral 时 plan 字段全部为 None / 空。

这些测试**不依赖** DB / LLM —— 用最小 stub 替代 repos 与 settings。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.signal_engine.schemas import TradingSignal
from app.signal_engine.service import SignalService


# ======================================================================
# 通用 stub
# ======================================================================
def _settings(**overrides: Any) -> Any:
    """
    构造一个具备 P3 默认阈值的 settings 占位对象
    --------------------------------------------------------------
    使用 SimpleNamespace 风格（MagicMock + spec=None）让 getattr 可拿到默认。
    传入 overrides 可覆盖任意单项，便于"开关 / 阈值"边界测试。
    """
    s = MagicMock()
    s.enable_decision_gates = False
    s.decision_min_atr_pct_15m = 0.0025
    s.decision_direction_flip_min_price_move_atr_1h = 0.6
    s.decision_cooldown_consecutive_sl_threshold = 2
    s.decision_cooldown_minutes = 60
    s.decision_max_position_size_pct = 0.10
    s.decision_kelly_aggressiveness = 0.5
    s.decision_min_rr_ratio = 2.0
    s.decision_min_sl_distance_atr_15m = 1.5
    s.decision_min_sl_distance_pct = 0.005
    s.decision_rule_llm_conflict_window = 5
    s.decision_rule_llm_conflict_winrate_threshold = 0.4
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_service(
    settings: Optional[Any] = None,
    repos: Optional[Any] = None,
) -> SignalService:
    """
    用 stub 构造 SignalService，避免触碰 DB / LLM
    --------------------------------------------------------------
    factor_aggregator 仅暴露 .settings 属性即可（generate() 路径外不会调到 .compute）。
    """
    settings = settings or _settings()
    repos = repos or MagicMock()
    factor_aggregator = MagicMock()
    factor_aggregator.settings = settings
    rule_engine = MagicMock()
    llm_agent = MagicMock()
    return SignalService(
        repos=repos,
        factor_aggregator=factor_aggregator,
        rule_engine=rule_engine,
        llm_agent=llm_agent,
        email_sender=None,
    )


def _factors_15m(*, atr_14: float, last_close: float) -> Dict[str, Any]:
    """构造一个仅含 15m market_structure 的因子字典。"""
    return {
        "by_timeframe": {
            "15m": {
                "market_structure": {
                    "atr_14": atr_14,
                    "last_close": last_close,
                }
            }
        }
    }


def _factors_15m_1h(
    *,
    atr_15m: float = 5.0,
    atr_1h: float = 10.0,
    last_close: float = 3000.0,
) -> Dict[str, Any]:
    """构造同时含 15m + 1h market_structure 的因子字典。"""
    return {
        "by_timeframe": {
            "15m": {
                "market_structure": {
                    "atr_14": atr_15m,
                    "last_close": last_close,
                }
            },
            "1h": {
                "market_structure": {
                    "atr_14": atr_1h,
                    "last_close": last_close,
                }
            },
        }
    }


# ======================================================================
# 闸 1：ATR 过低门禁
# ======================================================================
def test_atr_gate_triggers_when_below_threshold() -> None:
    """atr_14 / last_close = 5 / 3000 = 0.00167 < 0.0025（默认阈值）→ 触发"""
    svc = _make_service()
    factors = _factors_15m(atr_14=5.0, last_close=3000.0)
    assert svc._decision_atr_gate(factors) == "atr_too_low"


def test_atr_gate_passes_above_threshold() -> None:
    """atr_14 / last_close = 10 / 3000 = 0.00333 > 0.0025 → 不触发"""
    svc = _make_service()
    factors = _factors_15m(atr_14=10.0, last_close=3000.0)
    assert svc._decision_atr_gate(factors) is None


def test_atr_gate_missing_data_does_not_trigger() -> None:
    """数据缺失（无 by_timeframe / 空 dict）→ 保守不触发，让后续路径决定"""
    svc = _make_service()
    assert svc._decision_atr_gate({}) is None
    factors = {"by_timeframe": {"15m": {"market_structure": {}}}}
    assert svc._decision_atr_gate(factors) is None


def test_atr_gate_threshold_zero_disabled() -> None:
    """阈值 ≤ 0 视为禁用本闸"""
    svc = _make_service(_settings(decision_min_atr_pct_15m=0))
    factors = _factors_15m(atr_14=0.001, last_close=3000.0)
    assert svc._decision_atr_gate(factors) is None


# ======================================================================
# 闸 2：连续止损冷静期
# ======================================================================
def test_cooldown_gate_triggers_when_recent_two_sl_hits() -> None:
    """最近 2 条都是 sl_hit 且最新 exit_at = 30 分钟前 < 60 分钟阈值 → 触发"""
    repos = MagicMock()
    now = datetime.now(timezone.utc)
    repos.fetch_recent_settled_lifecycles = AsyncMock(
        return_value=[
            {"status": "sl_hit", "exit_at": now - timedelta(minutes=30)},
            {"status": "sl_hit", "exit_at": now - timedelta(minutes=50)},
        ]
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(svc._decision_cooldown_gate("ETH-USDT-SWAP"))
    assert result == "cooldown_consecutive_sl"


def test_cooldown_gate_passes_when_outside_window() -> None:
    """连续 sl_hit 但最新 exit_at 已经 90 分钟前 > 60 分钟阈值 → 不触发"""
    repos = MagicMock()
    now = datetime.now(timezone.utc)
    repos.fetch_recent_settled_lifecycles = AsyncMock(
        return_value=[
            {"status": "sl_hit", "exit_at": now - timedelta(minutes=90)},
            {"status": "sl_hit", "exit_at": now - timedelta(minutes=120)},
        ]
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(svc._decision_cooldown_gate("ETH-USDT-SWAP"))
    assert result is None


def test_cooldown_gate_passes_with_mixed_status() -> None:
    """近 2 条不全是 sl_hit（有 tp1_hit）→ 不触发"""
    repos = MagicMock()
    now = datetime.now(timezone.utc)
    repos.fetch_recent_settled_lifecycles = AsyncMock(
        return_value=[
            {"status": "sl_hit", "exit_at": now - timedelta(minutes=10)},
            {"status": "tp1_hit", "exit_at": now - timedelta(minutes=30)},
        ]
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(svc._decision_cooldown_gate("ETH-USDT-SWAP"))
    assert result is None


def test_cooldown_gate_passes_with_insufficient_samples() -> None:
    """已结算样本数不足 N → 不触发，让 LLM 在样本积累期照常出手"""
    repos = MagicMock()
    repos.fetch_recent_settled_lifecycles = AsyncMock(
        return_value=[{"status": "sl_hit", "exit_at": datetime.now(timezone.utc)}]
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(svc._decision_cooldown_gate("ETH-USDT-SWAP"))
    assert result is None


# ======================================================================
# 闸 3：方向稳定性
# ======================================================================
def _long_signal(confidence: float = 0.7) -> TradingSignal:
    """构造一条最小可用的 long 信号"""
    return TradingSignal(
        bias="long",
        confidence=confidence,
        reason="rule signal",
        risk="r",
        suggestion="s",
        entry_zone=(2995.0, 3005.0),
        stop_loss=2950.0,
        take_profit=[3050.0, 3100.0],
        risk_reward_ratio=2.0,
        position_size_pct=0.05,
        timeframe_alignment={"5m": "long", "15m": "long", "1h": "long", "4h": "long", "1d": "long"},
        invalidation_conditions=["a", "b"],
    )


def test_direction_stability_triggers_when_flip_with_micro_move() -> None:
    """
    上一条 short，本次 long；当前价 3000 vs 上次 entry mid 3001 → |Δ|=1
    阈值 0.6 × ATR(1h)=10 = 6 → 1 < 6，触发翻转保护
    """
    repos = MagicMock()
    repos.fetch_latest_signal_judgment = AsyncMock(
        return_value={
            "bias": "short",
            "entry_zone_low": 2999.0,
            "entry_zone_high": 3003.0,
        }
    )
    svc = _make_service(repos=repos)
    factors = _factors_15m_1h(atr_1h=10.0, last_close=3000.0)
    result = asyncio.run(
        svc._decision_direction_stability_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            factors=factors,
        )
    )
    assert result == "direction_flip_micro_move"


def test_direction_stability_passes_when_price_moved_enough() -> None:
    """
    上一条 short entry mid 2900；本次 last_close 3000 → |Δ|=100
    阈值 0.6 × ATR(1h)=10 = 6 → 100 > 6，不触发
    """
    repos = MagicMock()
    repos.fetch_latest_signal_judgment = AsyncMock(
        return_value={
            "bias": "short",
            "entry_zone_low": 2895.0,
            "entry_zone_high": 2905.0,
        }
    )
    svc = _make_service(repos=repos)
    factors = _factors_15m_1h(atr_1h=10.0, last_close=3000.0)
    result = asyncio.run(
        svc._decision_direction_stability_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            factors=factors,
        )
    )
    assert result is None


def test_direction_stability_passes_when_same_direction() -> None:
    """同向（上次 long，本次 long）不触发本闸"""
    repos = MagicMock()
    repos.fetch_latest_signal_judgment = AsyncMock(
        return_value={
            "bias": "long",
            "entry_zone_low": 2999.0,
            "entry_zone_high": 3003.0,
        }
    )
    svc = _make_service(repos=repos)
    factors = _factors_15m_1h(atr_1h=10.0, last_close=3000.0)
    result = asyncio.run(
        svc._decision_direction_stability_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            factors=factors,
        )
    )
    assert result is None


def test_direction_stability_passes_no_history() -> None:
    """无历史信号 → 不触发"""
    repos = MagicMock()
    repos.fetch_latest_signal_judgment = AsyncMock(return_value=None)
    svc = _make_service(repos=repos)
    factors = _factors_15m_1h(atr_1h=10.0, last_close=3000.0)
    result = asyncio.run(
        svc._decision_direction_stability_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            factors=factors,
        )
    )
    assert result is None


# ======================================================================
# 闸 4：规则 vs LLM 冲突保护
# ======================================================================
def _conflict_history(
    *,
    n_samples: int,
    n_wins: int,
    bias: str = "long",
    rule_score: float = -0.5,
) -> List[Dict[str, Any]]:
    """
    构造 n_samples 条"反向冲突"的已结算样本：
        bias='long' 但 rule_score=-0.5
    n_wins 条 status=tp1_hit，其余 sl_hit
    """
    rows: List[Dict[str, Any]] = []
    for i in range(n_samples):
        rows.append(
            {
                "signal_id": i,
                "ts": datetime.now(timezone.utc),
                "bias": bias,
                "rule_score": rule_score,
                "lifecycle_status": "tp1_hit" if i < n_wins else "sl_hit",
                "pnl_pct": 0.02 if i < n_wins else -0.01,
            }
        )
    return rows


def test_rule_conflict_gate_triggers_with_low_winrate() -> None:
    """
    本轮 LLM=long，rule_score=-0.5 反向冲突；近 4 条同样反向冲突里 1 胜 3 负
    → 胜率 25% < 40%，触发
    """
    repos = MagicMock()
    repos.fetch_recent_signals_for_conflict_check = AsyncMock(
        return_value=_conflict_history(n_samples=4, n_wins=1)
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(
        svc._decision_rule_conflict_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            rule_score=-0.5,
        )
    )
    assert result == "rule_llm_conflict_low_winrate"


def test_rule_conflict_gate_passes_with_high_winrate() -> None:
    """胜率 75% > 40% → 不触发"""
    repos = MagicMock()
    repos.fetch_recent_signals_for_conflict_check = AsyncMock(
        return_value=_conflict_history(n_samples=4, n_wins=3)
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(
        svc._decision_rule_conflict_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            rule_score=-0.5,
        )
    )
    assert result is None


def test_rule_conflict_gate_passes_when_aligned() -> None:
    """LLM=long + rule_score=+0.4 同向 → 不触发本闸"""
    repos = MagicMock()
    repos.fetch_recent_signals_for_conflict_check = AsyncMock(return_value=[])
    svc = _make_service(repos=repos)
    result = asyncio.run(
        svc._decision_rule_conflict_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            rule_score=0.4,
        )
    )
    assert result is None


def test_rule_conflict_gate_passes_with_insufficient_samples() -> None:
    """历史冲突样本不足 max(2, window/2) → 不触发"""
    repos = MagicMock()
    repos.fetch_recent_signals_for_conflict_check = AsyncMock(
        return_value=_conflict_history(n_samples=1, n_wins=0)
    )
    svc = _make_service(repos=repos)
    result = asyncio.run(
        svc._decision_rule_conflict_gate(
            symbol="ETH-USDT-SWAP",
            llm_signal=_long_signal(),
            rule_score=-0.5,
        )
    )
    assert result is None


# ======================================================================
# 工具方法：gated neutral 信号构造
# ======================================================================
def test_make_gated_neutral_signal_clears_plan() -> None:
    """前置闸门触发 → 构造的 neutral signal 必须 plan 字段全部为空"""
    rule_signal = _long_signal()
    out = SignalService._make_gated_neutral_signal(
        rule_signal=rule_signal, gate_reason="atr_too_low"
    )
    assert out.bias == "neutral"
    assert out.entry_zone is None
    assert out.stop_loss is None
    assert out.take_profit == []
    assert out.risk_reward_ratio is None
    assert out.position_size_pct is None
    assert "atr_too_low" in out.reason
    assert out.confidence == 0.0


def test_force_neutral_preserving_text_keeps_text_clears_plan() -> None:
    """
    后置闸门触发 → bias 改 neutral、plan 清空、reason 加前缀，
    但 timeframe_alignment / invalidation_conditions 与 risk/suggestion 保留。
    """
    src = _long_signal(confidence=0.85)
    out = SignalService._force_neutral_preserving_text(
        signal=src, gate_reason="direction_flip_micro_move"
    )
    assert out.bias == "neutral"
    assert out.entry_zone is None
    assert out.take_profit == []
    assert "direction_flip_micro_move" in out.reason
    assert out.confidence <= 0.5  # 收紧
    assert out.timeframe_alignment == src.timeframe_alignment
    assert out.invalidation_conditions == src.invalidation_conditions
