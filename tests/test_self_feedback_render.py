"""LLMAgent._render_self_feedback 的纯逻辑单元测试
=================================================================
覆盖三种成绩 case：
    1) 5 条全 expired-未触发：胜率不应假性归零，反而应通过 fill_rate 段提示
       "入场区间过窄/过远"，且不再出现"上一条仍未结算"告警；
    2) 5 条全 sl_hit（曾入场都亏）：判断质量段应给出 0% 胜率；
    3) 混合（部分 triggered + 部分未触发）：判断质量与触发率两段都应有数。

这些测试**不依赖**任何外部服务（数据库 / LLM API）—— 直接拿
LLMAgent._render_self_feedback 的 classmethod 验证渲染输出。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from app.signal_engine.llm_agent import LLMAgent


# ----------------------------------------------------------------------
# 辅助构造
# ----------------------------------------------------------------------
def _row(
    *,
    bias: str = "long",
    status: str,
    triggered: bool,
    pnl_pct: float | None = None,
    confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    构造一条 fetch_recent_settled_lifecycles 风格的 dict
    -----------------------------------------------------------------
    triggered=True  → triggered_at + triggered_price 非空，模拟"曾入场"
    triggered=False → triggered_at = None，模拟"价格从未走进 entry_zone"
    """
    base_ts = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    return {
        "signal_id": 1,
        "symbol": "ETH-USDT-SWAP",
        "bias": bias,
        "status": status,
        "entry_zone_low": 2400.0,
        "entry_zone_high": 2410.0,
        "triggered_at": base_ts if triggered else None,
        "triggered_price": 2405.0 if triggered else None,
        "pnl_pct": pnl_pct,
        "max_favorable_pct": 0.005 if triggered else None,
        "max_adverse_pct": -0.003 if triggered else None,
        "signal_ts": base_ts,
        "confidence": confidence,
        "factors": {"factors": {"regime": "ranging"}},
        "reason": "test",
    }


def _render(rows: List[Dict[str, Any]]) -> str:
    """
    驱动被测函数，返回渲染后的字符串
    """
    return LLMAgent._render_self_feedback(
        symbol="ETH-USDT-SWAP",
        recent_settled=rows,
    )


# ----------------------------------------------------------------------
# Case 1：5 条全 expired-未触发（重构前会假性 0% 胜率）
# ----------------------------------------------------------------------
def test_render_all_expired_untriggered_no_winrate_pollution() -> None:
    """
    历史版本会把这 5 条全计入分母得到"胜率 0%"并强压 confidence；
    新版本必须：
        - 判断质量段显示"无曾入场样本"，不出现 0% 字样；
        - 触发率段显示 0/5；
        - 触发率提示 reason 反思入场设计；
        - 不再出现"⚠️ 上一条信号 #X 仍未结算"段。
    """
    rows = [
        _row(status="expired", triggered=False) for _ in range(5)
    ]
    text = _render(rows)

    assert "无曾入场样本" in text
    assert "判断质量段" in text
    assert "触发率段" in text
    assert "0/5 触发入场" in text
    assert "fill_rate=0%" in text
    assert "入场区间过窄" in text
    assert "仍未结算" not in text
    assert "持仓中" not in text


# ----------------------------------------------------------------------
# Case 2：5 条全 sl_hit（曾入场全亏）
# ----------------------------------------------------------------------
def test_render_all_sl_hit_winrate_zero() -> None:
    """
    判断质量段：5 笔曾入场，5 输 0 赢，胜率 = 0%（基于 wins+losses 分母）。
    触发率段：5/5。
    """
    rows = [
        _row(status="sl_hit", triggered=True, pnl_pct=-0.01) for _ in range(5)
    ]
    text = _render(rows)

    assert "曾入场样本 = 5 / 5" in text
    assert "胜率=0%" in text
    assert "0赢 / 5输" in text
    assert "fill_rate=100%" in text
    # fill_rate 满，不应触发"过窄"提示
    assert "入场区间过窄" not in text


# ----------------------------------------------------------------------
# Case 3：混合（2 触发 + 3 未触发）
# ----------------------------------------------------------------------
def test_render_mixed_triggered_and_untriggered() -> None:
    """
    构造：
        - 1 笔 tp1_hit（赢）
        - 1 笔 sl_hit（输）
        - 3 笔 expired-未触发
    判断质量段：wins=1 losses=1，胜率 = 50%；
    触发率段：fill_rate = 2/5 = 40%；
    """
    rows = [
        _row(status="tp1_hit", triggered=True, pnl_pct=0.02),
        _row(status="sl_hit", triggered=True, pnl_pct=-0.01),
        _row(status="expired", triggered=False),
        _row(status="expired", triggered=False),
        _row(status="expired", triggered=False),
    ]
    text = _render(rows)

    assert "曾入场样本 = 2 / 5" in text
    assert "胜率=50%" in text
    assert "1赢 / 1输" in text
    assert "fill_rate=40%" in text
    # 40% 略低于 30% 阈值线之上，不应触发提示
    assert "入场区间过窄" not in text


# ----------------------------------------------------------------------
# Case 4：空样本（冷启动）
# ----------------------------------------------------------------------
def test_render_empty_recent_settled() -> None:
    """
    没有任何历史样本时，渲染必须输出"样本不足"占位段，
    且不能抛异常。
    """
    text = _render([])

    assert "样本不足" in text
    assert "判断质量段" not in text
    assert "触发率段" not in text


# ----------------------------------------------------------------------
# Case 5：曾入场后超时（expired-after-triggered）单独显示，不计入胜率分母
# ----------------------------------------------------------------------
def test_render_expired_after_triggered_excluded_from_winrate() -> None:
    """
    1 笔 tp1_hit + 2 笔 expired-after-triggered（曾入场但超时退出）。
    判断质量段必须显示：1 赢 / 0 输 / 2 笔超时不计入分母 → 胜率 100%。
    """
    rows = [
        _row(status="tp1_hit", triggered=True, pnl_pct=0.02),
        _row(status="expired", triggered=True, pnl_pct=0.001),
        _row(status="expired", triggered=True, pnl_pct=-0.002),
    ]
    text = _render(rows)

    assert "曾入场样本 = 3 / 3" in text
    assert "胜率=100%" in text
    assert "1赢 / 0输" in text
    assert "另有 2 笔曾入场但超时未到 SL/TP" in text
