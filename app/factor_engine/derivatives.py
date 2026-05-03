"""Derivatives factors: funding rate level + open interest change."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_derivatives_factors(
    funding: Optional[Dict[str, Any]],
    oi_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine the latest funding rate and an OI history list.

    ``oi_history`` should be ordered ascending by ``ts``. We compute the
    percentage change between the first and last sample in the window.
    """
    funding_rate: Optional[float] = None
    next_funding_ts = None
    if funding:
        funding_rate = float(funding.get("funding_rate") or 0.0)
        next_funding_ts = funding.get("next_funding_ts")

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
        "next_funding_ts": next_funding_ts.isoformat() if next_funding_ts else None,
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_change_pct": oi_change_pct,
        "oi_samples": len(oi_history),
    }
