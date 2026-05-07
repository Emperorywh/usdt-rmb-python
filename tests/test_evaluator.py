"""SignalEvaluator.compute_metrics 的纯逻辑单元测试
=================================================================
覆盖 P3 评估系统核心的 13 个指标：
    total_signals / triggered_count / fill_rate /
    wins / losses / expired_after_triggered / win_rate /
    avg_pnl_pct / total_pnl_pct /
    max_favorable_avg / max_adverse_avg /
    sharpe_estimated /
    direction_flip_count / direction_flip_rate /
    brier_score

测试**不依赖**任何外部服务（数据库 / LLM）—— 直接拿
SignalEvaluator.compute_metrics 的 classmethod 喂固定输入。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import math

from app.signal_engine.evaluator import SignalEvaluator


# ----------------------------------------------------------------------
# 辅助构造
# ----------------------------------------------------------------------
def _row(
    *,
    bias: str,
    confidence: Optional[float] = None,
    triggered: bool = False,
    status: Optional[str] = None,
    pnl_pct: Optional[float] = None,
    mfp: Optional[float] = None,
    mav: Optional[float] = None,
    ts_offset_seconds: int = 0,
) -> Dict[str, Any]:
    """
    构造一条 ``fetch_signals_for_evaluation`` 风格的 dict
    ----------------------------------------------------------
    所有可选字段按调用方需要给即可；缺省时模拟"该字段在 DB 中为 NULL"。
    """
    base_ts = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    return {
        "signal_id": 1,
        "ts": base_ts,
        "bias": bias,
        "confidence": confidence,
        "source": "rules+llm",
        "lifecycle_status": status,
        "triggered_at": base_ts if triggered else None,
        "triggered_price": 100.0 if triggered else None,
        "exit_price": 100.0 + (pnl_pct or 0) * 100.0 if pnl_pct is not None else None,
        "pnl_pct": pnl_pct,
        "max_favorable_pct": mfp,
        "max_adverse_pct": mav,
    }


# ----------------------------------------------------------------------
# Case 1：空输入 → 全部归零，比例字段为 None
# ----------------------------------------------------------------------
def test_compute_metrics_empty() -> None:
    """空输入下 total/triggered=0；除法分母为 0 的指标全部 None。"""
    m = SignalEvaluator.compute_metrics([])
    assert m["total_signals"] == 0
    assert m["triggered_count"] == 0
    assert m["fill_rate"] is None
    assert m["wins"] == 0
    assert m["losses"] == 0
    assert m["expired_after_triggered"] == 0
    assert m["win_rate"] is None
    assert m["avg_pnl_pct"] is None
    assert m["total_pnl_pct"] is None
    assert m["max_favorable_avg"] is None
    assert m["max_adverse_avg"] is None
    assert m["sharpe_estimated"] is None
    assert m["direction_flip_count"] == 0
    assert m["direction_flip_rate"] is None
    assert m["brier_score"] is None


# ----------------------------------------------------------------------
# Case 2：触发率 + 胜率 + PnL + Sharpe
# ----------------------------------------------------------------------
def test_compute_metrics_basic_winloss_and_sharpe() -> None:
    """
    构造 5 条信号：
        - 3 条曾入场：2 胜（tp1_hit / tp2_hit）+ 1 负（sl_hit）
        - 2 条未入场（lifecycle 未 join 上）
    期望：
        total=5, triggered=3, fill_rate=3/5
        wins=2, losses=1, win_rate=2/3
        decided 数据：pnl=[+0.02, +0.04, -0.015]
        avg_pnl ≈ 0.0150
        std ≈ 0.0285
        sharpe ≈ 0.0150 / 0.0285 ≈ 0.526
    """
    rows: List[Dict[str, Any]] = [
        _row(bias="long", confidence=0.7, triggered=True, status="tp1_hit", pnl_pct=0.02, mfp=0.025, mav=-0.005),
        _row(bias="long", confidence=0.6, triggered=True, status="tp2_hit", pnl_pct=0.04, mfp=0.045, mav=-0.002),
        _row(bias="short", confidence=0.55, triggered=True, status="sl_hit", pnl_pct=-0.015, mfp=0.003, mav=-0.018),
        _row(bias="long", confidence=0.5, triggered=False, status="expired"),
        _row(bias="long", confidence=0.5, triggered=False, status=None),
    ]
    m = SignalEvaluator.compute_metrics(rows)
    assert m["total_signals"] == 5
    assert m["triggered_count"] == 3
    assert math.isclose(m["fill_rate"], 3 / 5, abs_tol=1e-6)
    assert m["wins"] == 2
    assert m["losses"] == 1
    assert m["expired_after_triggered"] == 0  # 未入场的 expired 不计入
    assert math.isclose(m["win_rate"], 2 / 3, abs_tol=1e-6)
    # PnL 段
    assert math.isclose(m["avg_pnl_pct"], (0.02 + 0.04 - 0.015) / 3, abs_tol=1e-9)
    assert math.isclose(m["total_pnl_pct"], 0.02 + 0.04 - 0.015, abs_tol=1e-9)
    # MFP / MAP 平均
    assert math.isclose(m["max_favorable_avg"], (0.025 + 0.045 + 0.003) / 3, abs_tol=1e-9)
    assert math.isclose(m["max_adverse_avg"], (-0.005 - 0.002 - 0.018) / 3, abs_tol=1e-9)
    # Sharpe > 0（avg 为正）
    assert m["sharpe_estimated"] is not None
    assert m["sharpe_estimated"] > 0
    # 方向翻转：long, long, short, long, long → 翻转 2 次
    assert m["direction_flip_count"] == 2
    assert math.isclose(m["direction_flip_rate"], 2 / 4, abs_tol=1e-6)


# ----------------------------------------------------------------------
# Case 3：方向翻转率 - whipsaw 场景
# ----------------------------------------------------------------------
def test_direction_flip_rate_whipsaw() -> None:
    """
    long → short → long → short → long → short → long
    7 条信号，每条与前一条都翻转 → flip_count=6, flip_rate=6/6=1.0
    """
    rows = [_row(bias="long")]
    for i in range(6):
        rows.append(_row(bias="short" if i % 2 == 0 else "long"))
    m = SignalEvaluator.compute_metrics(rows)
    assert m["total_signals"] == 7
    assert m["direction_flip_count"] == 6
    assert math.isclose(m["direction_flip_rate"], 1.0, abs_tol=1e-9)


def test_direction_flip_rate_all_neutral() -> None:
    """全部 neutral → 不翻转，flip_count=0；total=1 时 rate=None。"""
    rows = [_row(bias="neutral") for _ in range(5)]
    m = SignalEvaluator.compute_metrics(rows)
    assert m["direction_flip_count"] == 0
    assert math.isclose(m["direction_flip_rate"], 0.0, abs_tol=1e-9)


# ----------------------------------------------------------------------
# Case 4：Brier score - 校准良好 vs 校准很差
# ----------------------------------------------------------------------
def test_brier_score_well_calibrated() -> None:
    """
    构造 4 条 conf=1.0 都赢 + 4 条 conf=0.0 都输 → 完全校准
    Brier = mean((1-1)^2, (1-1)^2, ..., (0-0)^2, ...) = 0
    """
    rows = []
    for _ in range(4):
        rows.append(_row(bias="long", confidence=1.0, triggered=True, status="tp1_hit", pnl_pct=0.02))
    for _ in range(4):
        rows.append(_row(bias="long", confidence=0.0, triggered=True, status="sl_hit", pnl_pct=-0.01))
    m = SignalEvaluator.compute_metrics(rows)
    assert m["brier_score"] is not None
    assert math.isclose(m["brier_score"], 0.0, abs_tol=1e-9)


def test_brier_score_worst_calibration() -> None:
    """
    构造 conf=1.0 全输 + conf=0.0 全赢 → 完全失校
    Brier = mean((1-0)^2 .., (0-1)^2 ..) = 1.0
    """
    rows = []
    for _ in range(4):
        rows.append(_row(bias="long", confidence=1.0, triggered=True, status="sl_hit", pnl_pct=-0.01))
    for _ in range(4):
        rows.append(_row(bias="long", confidence=0.0, triggered=True, status="tp1_hit", pnl_pct=0.02))
    m = SignalEvaluator.compute_metrics(rows)
    assert m["brier_score"] is not None
    assert math.isclose(m["brier_score"], 1.0, abs_tol=1e-9)


def test_brier_skips_undecided_samples() -> None:
    """
    expired_after_triggered / 未结算样本不参与 Brier 计算
    （只有 sl/tp 才有"中没中"的明确结论）。
    """
    rows = [
        _row(bias="long", confidence=0.8, triggered=True, status="tp1_hit"),
        _row(bias="long", confidence=0.8, triggered=True, status="expired"),  # 不计入
        _row(bias="long", confidence=0.8, triggered=False, status="invalidated"),  # 不计入
    ]
    m = SignalEvaluator.compute_metrics(rows)
    # 只一个样本 (0.8 - 1)^2 = 0.04
    assert m["brier_score"] is not None
    assert math.isclose(m["brier_score"], 0.04, abs_tol=1e-9)
    assert m["expired_after_triggered"] == 1


# ----------------------------------------------------------------------
# Case 5：Sharpe 防御性 - 单样本 / 全相同样本
# ----------------------------------------------------------------------
def test_sharpe_single_sample_returns_none() -> None:
    """单样本 std 无法计算 → Sharpe = None。"""
    rows = [
        _row(bias="long", confidence=0.7, triggered=True, status="tp1_hit", pnl_pct=0.02),
    ]
    m = SignalEvaluator.compute_metrics(rows)
    assert m["sharpe_estimated"] is None


def test_sharpe_zero_variance_returns_none() -> None:
    """所有样本 PnL 相同 → std=0 → Sharpe = None（除以 0 防御）。"""
    rows = [
        _row(bias="long", confidence=0.7, triggered=True, status="tp1_hit", pnl_pct=0.02),
        _row(bias="long", confidence=0.7, triggered=True, status="tp1_hit", pnl_pct=0.02),
        _row(bias="long", confidence=0.7, triggered=True, status="tp1_hit", pnl_pct=0.02),
    ]
    m = SignalEvaluator.compute_metrics(rows)
    assert m["sharpe_estimated"] is None
