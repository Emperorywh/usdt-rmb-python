"""Market structure factors: support/resistance + trend (HH/HL).

We avoid an external TA library by computing pivots directly from a list of
trade prices. The implementation is intentionally simple but produces useful
context for the downstream LLM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _resample_minute_ohlc(trades: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Group trades into 1-minute OHLC bars."""
    if not trades:
        return []
    bars: Dict[int, Dict[str, float]] = {}
    order: List[int] = []
    for t in trades:
        ts = t["ts"]
        epoch = int(ts.timestamp() // 60)
        price = float(t["price"])
        if epoch not in bars:
            bars[epoch] = {"o": price, "h": price, "l": price, "c": price, "v": 0.0, "ts": epoch * 60}
            order.append(epoch)
        bar = bars[epoch]
        bar["h"] = max(bar["h"], price)
        bar["l"] = min(bar["l"], price)
        bar["c"] = price
        bar["v"] += float(t["size"])
    return [bars[k] for k in order]


def _find_pivots(values: List[float], left: int = 2, right: int = 2) -> List[int]:
    """Return indices of pivot points using a simple ``left``/``right`` window."""
    pivots: List[int] = []
    n = len(values)
    for i in range(left, n - right):
        window = values[i - left : i + right + 1]
        if values[i] == max(window) or values[i] == min(window):
            pivots.append(i)
    return pivots


def compute_market_structure(
    trades: List[Dict[str, Any]],
    levels_count: int = 3,
) -> Dict[str, Any]:
    """Estimate support/resistance and short-term trend (HH/HL or LH/LL).

    Trends:
      * ``"uptrend"`` when both highs and lows are rising (HH + HL)
      * ``"downtrend"`` when both highs and lows are falling (LH + LL)
      * ``"range"`` otherwise
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

    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]
    closes = [b["c"] for b in bars]

    high_pivots = [highs[i] for i in _find_pivots(highs)]
    low_pivots = [lows[i] for i in _find_pivots(lows)]

    last_close = closes[-1]

    resistances = sorted(
        {round(p, 4) for p in high_pivots if p >= last_close},
        reverse=False,
    )[:levels_count]
    supports = sorted(
        {round(p, 4) for p in low_pivots if p <= last_close},
        reverse=True,
    )[:levels_count]

    trend = "range"
    if len(high_pivots) >= 2 and len(low_pivots) >= 2:
        hh = high_pivots[-1] > high_pivots[-2]
        hl = low_pivots[-1] > low_pivots[-2]
        lh = high_pivots[-1] < high_pivots[-2]
        ll = low_pivots[-1] < low_pivots[-2]
        if hh and hl:
            trend = "uptrend"
        elif lh and ll:
            trend = "downtrend"

    # Simple slope of close prices (linear regression).
    slope = 0.0
    if len(closes) >= 3:
        x = np.arange(len(closes), dtype=float)
        y = np.asarray(closes, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])

    return {
        "available": True,
        "trend": trend,
        "supports": supports,
        "resistances": resistances,
        "last_price": last_close,
        "bar_count": len(bars),
        "slope": round(slope, 6),
    }
