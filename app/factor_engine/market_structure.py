"""市场结构因子（多周期版本）。

P0 升级要点
============
- 老接口 ``compute_market_structure(trades)`` 仍然保留，由老聚合器
  在 ``enable_mtf_factors=False`` 时使用。
- 新增 ``compute_market_structure_from_klines(klines)``：直接基于
  指定周期的 K 线序列（最近 N 根 bar，N=80 默认）计算：

  ===========================  =======================================================
  字段                          公式
  ===========================  =======================================================
  trend                        基于 swing HH/HL/LH/LL 判定 → uptrend/downtrend/range
  atr_14                       Wilder ATR(14)，numpy 自实现
  supports[3] / resistances[3] 由 swing low/high 中筛选与 last_close 最近的 3 个
  last_close                   最新一根的收盘价
  slope                        全部 closes 的最小二乘线性回归斜率
  ===========================  =======================================================

- 严格不引入 TA 库；ATR / pivots / 线性回归全部用 numpy 实现。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------
# 公共辅助
# ----------------------------------------------------------------------
def _resample_minute_ohlc(trades: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    把成交按 1 分钟粒度聚合成 OHLC bar
    -----------------------------------------------------------------
    参数：
        trades: 按 ts 升序排列的成交 dict 列表
    返回：
        包含 o/h/l/c/v/ts 的 bar 列表，仅供老接口使用。
    """
    if not trades:
        return []
    bars: Dict[int, Dict[str, float]] = {}
    order: List[int] = []
    for t in trades:
        ts = t["ts"]
        epoch = int(ts.timestamp() // 60)
        price = float(t["price"])
        if epoch not in bars:
            bars[epoch] = {
                "o": price, "h": price, "l": price, "c": price,
                "v": 0.0, "ts": epoch * 60,
            }
            order.append(epoch)
        bar = bars[epoch]
        bar["h"] = max(bar["h"], price)
        bar["l"] = min(bar["l"], price)
        bar["c"] = price
        bar["v"] += float(t["size"])
    return [bars[k] for k in order]


def _find_pivot_indices(
    values: np.ndarray, left: int = 2, right: int = 2, kind: str = "high"
) -> List[int]:
    """
    找出 swing pivot 的索引
    -----------------------------------------------------------------
    参数：
        values: 一维序列（high 序列或 low 序列）
        left:   左侧窗口长度
        right:  右侧窗口长度
        kind:   'high' - 找局部高点；'low' - 找局部低点
    返回：
        命中 pivot 的索引列表（升序）。
    """
    n = len(values)
    if n < left + right + 1:
        return []
    out: List[int] = []
    for i in range(left, n - right):
        window = values[i - left : i + right + 1]
        v = values[i]
        if kind == "high" and v == np.max(window):
            out.append(i)
        elif kind == "low" and v == np.min(window):
            out.append(i)
    return out


def _wilder_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> Optional[float]:
    """
    Wilder ATR 实现（numpy 自实现，禁止引入 TA 库）
    -----------------------------------------------------------------
    参数：
        highs / lows / closes: 等长一维数组
        period:                ATR 周期，默认 14
    返回：
        最新一个 ATR 值；样本不足时返回 None。
    说明：
        Wilder 平滑 = 第一个 ATR 取前 ``period`` 个 TR 的算术平均，
        之后递推 ``atr = (atr_prev * (period-1) + tr_curr) / period``。
    """
    n = len(closes)
    if n < period + 1:
        return None
    tr = np.zeros(n, dtype=float)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        h_l = highs[i] - lows[i]
        h_pc = abs(highs[i] - closes[i - 1])
        l_pc = abs(lows[i] - closes[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)
    atr = float(np.mean(tr[1 : period + 1]))
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


def _linear_regression_slope(values: np.ndarray) -> Optional[float]:
    """
    最小二乘线性回归斜率
    -----------------------------------------------------------------
    参数：
        values: 一维 numpy 数组
    返回：
        斜率 float；样本不足或全 NaN 时返回 None。
    """
    if values is None or len(values) < 3:
        return None
    x = np.arange(len(values), dtype=float)
    try:
        slope, _ = np.polyfit(x, values.astype(float), 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _classify_trend(highs: np.ndarray, lows: np.ndarray) -> str:
    """
    基于 swing HH/HL/LH/LL 判定短期趋势
    -----------------------------------------------------------------
    参数：
        highs / lows: 一维收盘价序列
    返回：
        'uptrend' / 'downtrend' / 'range'
    """
    high_idx = _find_pivot_indices(highs, kind="high")
    low_idx = _find_pivot_indices(lows, kind="low")
    if len(high_idx) < 2 or len(low_idx) < 2:
        return "range"
    hh = highs[high_idx[-1]] > highs[high_idx[-2]]
    hl = lows[low_idx[-1]] > lows[low_idx[-2]]
    lh = highs[high_idx[-1]] < highs[high_idx[-2]]
    ll = lows[low_idx[-1]] < lows[low_idx[-2]]
    if hh and hl:
        return "uptrend"
    if lh and ll:
        return "downtrend"
    return "range"


# ----------------------------------------------------------------------
# 老接口（保留兼容）
# ----------------------------------------------------------------------
def compute_market_structure(
    trades: List[Dict[str, Any]],
    levels_count: int = 3,
) -> Dict[str, Any]:
    """
    老版市场结构因子：基于 trades 重采样到 1 分钟 bar
    -----------------------------------------------------------------
    参数：
        trades:       按 ts 升序排列的成交 dict 列表
        levels_count: 支撑 / 阻力位最多保留的档位数
    返回：
        含 trend / supports / resistances / last_price / bar_count / slope 的 dict
    说明：
        在 enable_mtf_factors=False 时由老聚合器调用，作为安全回退路径。
    """
    bars = _resample_minute_ohlc(trades)
    if len(bars) < 6:
        return {
            "available": False,
            "trend": "neutral",
            "supports": [],
            "resistances": [],
            "last_price": float(bars[-1]["c"]) if bars else None,
            "bar_count": len(bars),
        }
    highs = np.array([b["h"] for b in bars], dtype=float)
    lows = np.array([b["l"] for b in bars], dtype=float)
    closes = np.array([b["c"] for b in bars], dtype=float)

    last_close = float(closes[-1])
    high_pivots = highs[_find_pivot_indices(highs, kind="high")]
    low_pivots = lows[_find_pivot_indices(lows, kind="low")]

    resistances = sorted(
        {round(float(p), 4) for p in high_pivots if p >= last_close}
    )[:levels_count]
    supports = sorted(
        {round(float(p), 4) for p in low_pivots if p <= last_close},
        reverse=True,
    )[:levels_count]

    return {
        "available": True,
        "trend": _classify_trend(highs, lows),
        "supports": supports,
        "resistances": resistances,
        "last_price": last_close,
        "bar_count": len(bars),
        "slope": round(_linear_regression_slope(closes) or 0.0, 6),
    }


# ----------------------------------------------------------------------
# 多周期版（P0 主接口）
# ----------------------------------------------------------------------
def compute_market_structure_from_klines(
    klines: List[Dict[str, Any]],
    levels_count: int = 3,
) -> Dict[str, Any]:
    """
    从某周期的最近 N 根 K 线计算市场结构因子
    -----------------------------------------------------------------
    参数：
        klines:       按 ts 升序的 K 线 dict 列表（含未封盘 bar）
        levels_count: 支撑 / 阻力位最多保留的档位数
    返回：
        含 available / trend / atr_14 / supports / resistances /
        last_close / slope / bar_count 的 dict。
    """
    if len(klines) < 8:
        last = klines[-1] if klines else None
        return {
            "available": False,
            "trend": "neutral",
            "atr_14": None,
            "supports": [],
            "resistances": [],
            "last_close": float(last["close"]) if last and last.get("close") else None,
            "slope": None,
            "bar_count": len(klines),
        }

    highs = np.array([float(b.get("high") or 0.0) for b in klines], dtype=float)
    lows = np.array([float(b.get("low") or 0.0) for b in klines], dtype=float)
    closes = np.array([float(b.get("close") or 0.0) for b in klines], dtype=float)
    last_close = float(closes[-1])

    trend = _classify_trend(highs, lows)
    atr = _wilder_atr(highs, lows, closes, period=14)
    slope = _linear_regression_slope(closes)

    high_pivots = highs[_find_pivot_indices(highs, kind="high")]
    low_pivots = lows[_find_pivot_indices(lows, kind="low")]

    # 阻力：高于 last_close 的 swing 高点中"距 last_close 最近的 levels_count 个"
    res_candidates = sorted(
        {round(float(p), 4) for p in high_pivots if p >= last_close}
    )
    resistances = res_candidates[:levels_count]
    # 支撑：低于 last_close 的 swing 低点中"距 last_close 最近的 levels_count 个"
    sup_candidates = sorted(
        {round(float(p), 4) for p in low_pivots if p <= last_close},
        reverse=True,
    )
    supports = sup_candidates[:levels_count]

    return {
        "available": True,
        "trend": trend,
        # numpy 标量必须显式转 Python float，否则 JSONB 序列化会抛
        # TypeError: Object of type float64 is not JSON serializable
        "atr_14": round(float(atr), 4) if atr is not None else None,
        "supports": supports,
        "resistances": resistances,
        "last_close": round(float(last_close), 4),
        "slope": round(float(slope), 6) if slope is not None else None,
        "bar_count": len(klines),
    }
