"""SignalService._apply_size_override 的纯逻辑单元测试
=================================================================
验证 P3 服务端 size 覆盖公式：

    base = max(0, (conf - 0.5) × 2)
    win_rate_factor = clamp(2 × win_rate - 0.5, 0.2, 1.0) （决定性样本）
        样本不足时 win_rate_factor = 0.5
    new_size = base × win_rate_factor × kelly
    clamp 到 [0, decision_max_position_size_pct]
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import math

from app.signal_engine.schemas import TradingSignal
from app.signal_engine.service import SignalService


def _settings(
    *,
    max_size: float = 0.10,
    kelly: float = 0.5,
) -> Any:
    s = MagicMock()
    s.decision_max_position_size_pct = max_size
    s.decision_kelly_aggressiveness = kelly
    return s


def _make_service(settings: Any) -> SignalService:
    factor_aggregator = MagicMock()
    factor_aggregator.settings = settings
    return SignalService(
        repos=MagicMock(),
        factor_aggregator=factor_aggregator,
        rule_engine=MagicMock(),
        llm_agent=MagicMock(),
        email_sender=None,
    )


def _long_signal(confidence: float, position_size_pct: float = 0.20) -> TradingSignal:
    """构造一条带 plan 的 long 信号；position_size_pct 故意取 schema 上限内大值。"""
    return TradingSignal(
        bias="long",
        confidence=confidence,
        reason="test",
        risk="r",
        suggestion="s",
        entry_zone=(2995.0, 3005.0),
        stop_loss=2950.0,
        take_profit=[3050.0, 3100.0],
        risk_reward_ratio=2.0,
        position_size_pct=position_size_pct,
        timeframe_alignment={"5m": "long", "15m": "long", "1h": "long", "4h": "long", "1d": "long"},
        invalidation_conditions=["a", "b"],
    )


def _settled_rows(
    *, wins: int, losses: int, neutrals: int = 0
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _ in range(wins):
        rows.append({"status": "tp1_hit"})
    for _ in range(losses):
        rows.append({"status": "sl_hit"})
    for _ in range(neutrals):
        rows.append({"status": "expired"})
    return rows


# ----------------------------------------------------------------------
# Case 1：confidence 边界 - conf ≤ 0.5 → size 归零
# ----------------------------------------------------------------------
def test_size_override_zero_when_conf_at_threshold() -> None:
    """conf=0.5 → base=0 → new_size=0"""
    svc = _make_service(_settings())
    sig = _long_signal(confidence=0.5)
    out = svc._apply_size_override(signal=sig, recent_settled=[])
    assert out.position_size_pct == 0.0


def test_size_override_zero_when_conf_below_threshold() -> None:
    """conf=0.3 → max(0, (0.3-0.5)*2) = 0 → new_size=0"""
    svc = _make_service(_settings())
    sig = _long_signal(confidence=0.3)
    out = svc._apply_size_override(signal=sig, recent_settled=[])
    assert out.position_size_pct == 0.0


# ----------------------------------------------------------------------
# Case 2：样本不足时 win_rate_factor 默认 0.5
# ----------------------------------------------------------------------
def test_size_override_default_winrate_factor_when_no_history() -> None:
    """
    conf=0.8 → base=0.6
    无历史 → win_rate_factor=0.5
    kelly=0.5
    new_size = 0.6 × 0.5 × 0.5 = 0.15 → clamp 到 0.10
    """
    svc = _make_service(_settings(max_size=0.10))
    sig = _long_signal(confidence=0.8)
    out = svc._apply_size_override(signal=sig, recent_settled=[])
    assert math.isclose(out.position_size_pct, 0.10, abs_tol=1e-9)


# ----------------------------------------------------------------------
# Case 3：高胜率 → win_rate_factor 上限 1.0
# ----------------------------------------------------------------------
def test_size_override_with_high_winrate() -> None:
    """
    conf=0.8 → base=0.6
    历史 4 胜 0 负 → win_rate=1.0 → factor=clamp(2*1-0.5,0.2,1.0)=1.0
    new_size = 0.6 × 1.0 × 0.5 = 0.30 → clamp 到 max_size=0.10
    """
    svc = _make_service(_settings(max_size=0.10))
    sig = _long_signal(confidence=0.8)
    out = svc._apply_size_override(
        signal=sig, recent_settled=_settled_rows(wins=4, losses=0)
    )
    assert math.isclose(out.position_size_pct, 0.10, abs_tol=1e-9)


# ----------------------------------------------------------------------
# Case 4：低胜率 → win_rate_factor 下限 0.2
# ----------------------------------------------------------------------
def test_size_override_with_zero_winrate_clamped_to_floor() -> None:
    """
    conf=0.8 → base=0.6
    历史 0 胜 4 负 → win_rate=0 → factor=clamp(2*0-0.5,0.2,1.0)=0.2
    new_size = 0.6 × 0.2 × 0.5 = 0.06
    """
    svc = _make_service(_settings(max_size=0.10))
    sig = _long_signal(confidence=0.8)
    out = svc._apply_size_override(
        signal=sig, recent_settled=_settled_rows(wins=0, losses=4)
    )
    assert math.isclose(out.position_size_pct, 0.06, abs_tol=1e-6)


# ----------------------------------------------------------------------
# Case 5：max_size 边界 - 配置成 0.05 时 clamp 更严
# ----------------------------------------------------------------------
def test_size_override_respects_max_size_setting() -> None:
    """max_size=0.05 → 即使 conf=1 / win_rate=1 也最多 0.05"""
    svc = _make_service(_settings(max_size=0.05))
    sig = _long_signal(confidence=1.0)
    out = svc._apply_size_override(
        signal=sig, recent_settled=_settled_rows(wins=10, losses=0)
    )
    # base=1.0, factor=1.0, kelly=0.5 → 0.5 → clamp 0.05
    assert math.isclose(out.position_size_pct, 0.05, abs_tol=1e-9)


# ----------------------------------------------------------------------
# Case 6：kelly 因子 - 半凯利 vs 全凯利
# ----------------------------------------------------------------------
def test_size_override_full_kelly_vs_half_kelly() -> None:
    """
    conf=0.7, win_rate=0.5 → factor=clamp(2*0.5-0.5,0.2,1)=0.5
    base = 0.4
    half kelly (0.5): 0.4 × 0.5 × 0.5 = 0.10 → clamp 0.10
    full kelly (1.0): 0.4 × 0.5 × 1.0 = 0.20 → clamp 0.10
    （两者在 max=0.10 时打平，但中间值不同）
    """
    sig = _long_signal(confidence=0.7)
    settled = _settled_rows(wins=2, losses=2)
    half = _make_service(_settings(max_size=0.20, kelly=0.5))
    full = _make_service(_settings(max_size=0.20, kelly=1.0))
    out_half = half._apply_size_override(signal=sig, recent_settled=settled)
    out_full = full._apply_size_override(signal=sig, recent_settled=settled)
    assert math.isclose(out_half.position_size_pct, 0.10, abs_tol=1e-6)
    assert math.isclose(out_full.position_size_pct, 0.20, abs_tol=1e-6)


# ----------------------------------------------------------------------
# Case 7：max_size 配置非法（None / 0）→ 直接放行原 signal
# ----------------------------------------------------------------------
def test_size_override_disabled_when_max_size_zero() -> None:
    """max_size=0 视为禁用本逻辑，原 signal 不被修改"""
    svc = _make_service(_settings(max_size=0))
    sig = _long_signal(confidence=0.9, position_size_pct=0.20)
    out = svc._apply_size_override(signal=sig, recent_settled=[])
    assert out is sig
    assert math.isclose(out.position_size_pct, 0.20, abs_tol=1e-9)
