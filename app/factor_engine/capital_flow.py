"""Capital flow factors: net_flow and CVD over a rolling window."""
from __future__ import annotations

from typing import Any, Dict, List


def compute_capital_flow(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute capital flow factors from a list of trades.

    Parameters
    ----------
    trades:
        Sequence of trade rows (dicts) ordered by ``ts`` ascending. Each item
        must contain ``price``, ``size`` and ``side`` (``"buy"`` / ``"sell"``).

    Returns
    -------
    dict with:
      * ``buy_volume`` - cumulative buy notional within window
      * ``sell_volume`` - cumulative sell notional within window
      * ``net_flow`` - buy_volume - sell_volume (positive => buyer dominance)
      * ``cvd`` - cumulative volume delta (buy_size - sell_size)
      * ``trade_count`` - number of trades observed
    """
    buy_vol = 0.0
    sell_vol = 0.0
    cvd = 0.0
    for t in trades:
        price = float(t["price"])
        size = float(t["size"])
        notional = price * size
        if t["side"] == "buy":
            buy_vol += notional
            cvd += size
        else:
            sell_vol += notional
            cvd -= size
    return {
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
        "net_flow": round(buy_vol - sell_vol, 4),
        "cvd": round(cvd, 6),
        "trade_count": len(trades),
    }
