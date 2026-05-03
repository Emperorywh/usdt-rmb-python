"""Order-book factors: bid/ask imbalance and liquidity walls."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_orderbook_factors(
    snapshot: Optional[Dict[str, Any]],
    wall_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """Derive order-book factors from a single snapshot.

    Parameters
    ----------
    snapshot:
        A repository row with ``bids`` / ``asks`` keys, where each level is a
        ``[price, size]`` pair (top of book first).
    wall_multiplier:
        A level whose size is at least ``wall_multiplier`` * mean(size) is
        flagged as a liquidity wall.
    """
    if not snapshot:
        return {
            "available": False,
            "imbalance": 0.0,
            "best_bid": None,
            "best_ask": None,
            "spread": None,
            "bid_walls": [],
            "ask_walls": [],
        }

    bids: List[List[float]] = snapshot.get("bids") or []
    asks: List[List[float]] = snapshot.get("asks") or []

    bid_qty = sum(float(b[1]) for b in bids) if bids else 0.0
    ask_qty = sum(float(a[1]) for a in asks) if asks else 0.0
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0

    best_bid = float(bids[0][0]) if bids else None
    best_ask = float(asks[0][0]) if asks else None
    spread = (best_ask - best_bid) if (best_bid and best_ask) else None

    def _walls(side: List[List[float]]) -> List[Dict[str, float]]:
        if not side:
            return []
        sizes = [float(level[1]) for level in side]
        mean = sum(sizes) / len(sizes)
        if mean <= 0:
            return []
        threshold = mean * wall_multiplier
        out = []
        for level in side:
            sz = float(level[1])
            if sz >= threshold:
                out.append({"price": float(level[0]), "size": sz})
        return out

    return {
        "available": True,
        "imbalance": round(imbalance, 6),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": round(spread, 6) if spread is not None else None,
        "bid_qty": round(bid_qty, 6),
        "ask_qty": round(ask_qty, 6),
        "bid_walls": _walls(bids),
        "ask_walls": _walls(asks),
    }
