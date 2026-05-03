"""衍生品因子（多周期版本）。

P0 升级要点
============
- 老接口 ``compute_derivatives_factors(funding, oi_history)`` 保留，
  在 ``enable_mtf_factors=False`` 时由老聚合器使用。
- 新增 ``compute_derivatives_per_timeframe(funding, oi_history, kline_first_ts, kline_last_ts)``：
  funding_rate 全周期共享（resampling 没意义，结算粒度 ~1min），
  oi_change_pct 用该周期 K 线的起止时间点计算。
- 新增 oi_price_relation 字段，4 象限分类：
    OI↑ Px↑ → long_build
    OI↑ Px↓ → short_build
    OI↓ Px↑ → short_cover
    OI↓ Px↓ → long_cover
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# 老接口（保留兼容）
# ----------------------------------------------------------------------
def compute_derivatives_factors(
    funding: Optional[Dict[str, Any]],
    oi_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    老版衍生品因子：funding 取最新，OI 用窗口首尾差
    -----------------------------------------------------------------
    参数：
        funding:    funding_rates 表里最近一条记录，可空
        oi_history: 按 ts 升序的 OI 样本列表
    返回：
        funding_rate / next_settlement_at / oi_first / oi_last /
        oi_change_pct / oi_samples
    说明：
        在 enable_mtf_factors=False 时由老聚合器使用。
    """
    funding_rate: Optional[float] = None
    next_settlement_at = None
    if funding:
        funding_rate = float(funding.get("funding_rate") or 0.0)
        next_settlement_at = funding.get("next_funding_ts")

    oi_first: Optional[float] = None
    oi_last: Optional[float] = None
    oi_change_pct: Optional[float] = None
    if oi_history:
        oi_first = float(oi_history[0]["oi"]) if oi_history[0].get("oi") is not None else None
        oi_last = float(oi_history[-1]["oi"]) if oi_history[-1].get("oi") is not None else None
        if oi_first and oi_last and oi_first > 0:
            oi_change_pct = round((oi_last - oi_first) / oi_first, 6)

    return {
        "funding_rate": funding_rate,
        "next_settlement_at": (
            next_settlement_at.isoformat() if next_settlement_at else None
        ),
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_change_pct": oi_change_pct,
        "oi_samples": len(oi_history),
    }


# ----------------------------------------------------------------------
# 多周期版（P0 主接口）
# ----------------------------------------------------------------------
def compute_derivatives_per_timeframe(
    funding: Optional[Dict[str, Any]],
    oi_history: List[Dict[str, Any]],
    klines: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    单个周期的衍生品因子：funding 共享 + OI 起止差 + 4 象限关系
    -----------------------------------------------------------------
    参数：
        funding:    funding_rates 最新一条记录，全周期共享
        oi_history: 整个分析窗口内的 OI 样本（按 ts 升序）
        klines:     当前周期的 K 线序列（按 ts 升序，含未封盘 bar）
    返回：
        funding_rate_now / oi_change_pct / oi_first / oi_last /
        oi_price_relation / next_settlement_at
    说明：
        - oi_change_pct = (oi_at_kline_last_ts - oi_at_kline_first_ts) / oi_at_kline_first_ts
          其中 oi_at_X 为 oi_history 中"小于等于 X"的最后一条样本
          的近似（线性插值开销不值得）。
        - oi_price_relation 同时需要 OI 与价的方向：用周期内第一根 / 最后一根
          K 线的 close 比较。
    """
    funding_rate_now: Optional[float] = None
    next_settlement_at = None
    if funding:
        funding_rate_now = float(funding.get("funding_rate") or 0.0)
        next_settlement_at = funding.get("next_funding_ts")

    if not klines or not oi_history:
        return {
            "funding_rate_now": funding_rate_now,
            "oi_change_pct": None,
            "oi_first": None,
            "oi_last": None,
            "oi_price_relation": "unknown",
            "next_settlement_at": (
                next_settlement_at.isoformat() if next_settlement_at else None
            ),
        }

    first_ts = klines[0]["ts"]
    last_ts = klines[-1]["ts"]
    oi_first = _oi_at_or_before(oi_history, first_ts)
    oi_last = _oi_at_or_before(oi_history, last_ts)
    oi_change_pct: Optional[float] = None
    if oi_first is not None and oi_last is not None and oi_first > 0:
        oi_change_pct = round((oi_last - oi_first) / oi_first, 6)

    price_first = _safe_float(klines[0].get("close"))
    price_last = _safe_float(klines[-1].get("close"))
    oi_price_relation = _classify_oi_price(
        oi_change_pct, price_last - price_first
    )

    return {
        "funding_rate_now": funding_rate_now,
        "oi_change_pct": oi_change_pct,
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_price_relation": oi_price_relation,
        "next_settlement_at": (
            next_settlement_at.isoformat() if next_settlement_at else None
        ),
    }


# ----------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------
def _oi_at_or_before(
    oi_history: List[Dict[str, Any]], target: datetime
) -> Optional[float]:
    """
    在升序 OI 序列中找出 ts <= target 的最后一条记录的 oi 值
    -----------------------------------------------------------------
    参数：
        oi_history: 按 ts 升序的 OI 样本列表
        target:     目标时间戳
    返回：
        oi 值；找不到时返回 None。
    """
    if not oi_history:
        return None
    out: Optional[float] = None
    for row in oi_history:
        ts = row.get("ts")
        if ts is None:
            continue
        if ts <= target:
            try:
                out = float(row.get("oi"))
            except (TypeError, ValueError):
                continue
        else:
            break
    return out


def _classify_oi_price(
    oi_change_pct: Optional[float], price_change: float
) -> str:
    """
    OI 与价的 4 象限分类
    -----------------------------------------------------------------
    参数：
        oi_change_pct: OI 在该周期内的相对变动
        price_change:  该周期内的价差（last - first）
    返回：
        'long_build'  : OI↑ Px↑（多头新开仓）
        'short_build' : OI↑ Px↓（空头新开仓）
        'short_cover' : OI↓ Px↑（空头止盈平仓）
        'long_cover'  : OI↓ Px↓（多头止损平仓）
        'unknown'     : 数据不足时返回
    """
    if oi_change_pct is None:
        return "unknown"
    oi_up = oi_change_pct > 0
    px_up = price_change > 0
    if oi_up and px_up:
        return "long_build"
    if oi_up and not px_up:
        return "short_build"
    if not oi_up and px_up:
        return "short_cover"
    return "long_cover"


def _safe_float(v: Any) -> float:
    """
    安全 float 转换，None 或异常时返回 0.0
    """
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
