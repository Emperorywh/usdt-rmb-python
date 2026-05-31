"""订单簿指标计算（从 factor_engine/orderbook.py 提取）。

此模块包含 data_ingestion 层所需的 orderbook 指标聚合函数，
避免采集层反向依赖因子计算层。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _walls(
    side: List[List[float]], wall_multiplier: float
) -> List[Dict[str, float]]:
    """
    从盘口某一侧筛选大墙
    -----------------------------------------------------------------
    参数：
        side:            [[price, size], ...]
        wall_multiplier: 阈值倍数（size ≥ mean × N 视为墙）
    返回：
        每档大墙的 {price, size} dict 列表。
    """
    if not side:
        return []
    sizes = [float(level[1]) for level in side]
    mean = sum(sizes) / len(sizes)
    if mean <= 0:
        return []
    threshold = mean * wall_multiplier
    out: List[Dict[str, float]] = []
    for level in side:
        sz = float(level[1])
        if sz >= threshold:
            out.append({"price": float(level[0]), "size": sz})
    return out


def compute_orderbook_metric_row(
    snapshot: Dict[str, Any],
    wall_multiplier: float = 3.0,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    把一条 orderbook 事件聚合成 orderbook_metrics 表的一行（给 runner 用）
    -----------------------------------------------------------------
    参数：
        snapshot:        WS 推送出来的 dict，含 bids / asks
        wall_multiplier: 大墙阈值倍数（默认 3.0）
        top_n:           前 N 档累计 notional（默认 5）
    返回：
        含 imbalance / bid_qty / ask_qty / top5_bid_notional /
        top5_ask_notional / bid_wall_count / ask_wall_count /
        spread_bp / mid_price 的 dict。
    说明：
        所有数值都已经经过 ctVal 换算（runner 上游 okx_ws 已乘 ctVal），
        这里只做纯数学聚合，方便 runner 直接落库。
    """
    bids: List[List[float]] = snapshot.get("bids") or []
    asks: List[List[float]] = snapshot.get("asks") or []

    bid_qty = sum(float(b[1]) for b in bids)
    ask_qty = sum(float(a[1]) for a in asks)
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0

    top_n_bids = bids[:top_n]
    top_n_asks = asks[:top_n]
    top_n_bid_notional = sum(float(p) * float(s) for p, s in (b[:2] for b in top_n_bids))
    top_n_ask_notional = sum(float(p) * float(s) for p, s in (a[:2] for a in top_n_asks))

    bid_walls = _walls(bids, wall_multiplier)
    ask_walls = _walls(asks, wall_multiplier)

    best_bid = float(bids[0][0]) if bids else None
    best_ask = float(asks[0][0]) if asks else None
    mid_price: Optional[float] = None
    spread_bp: Optional[float] = None
    if best_bid and best_ask and best_bid > 0 and best_ask > 0:
        mid_price = (best_bid + best_ask) / 2.0
        if mid_price > 0:
            spread_bp = (best_ask - best_bid) / mid_price * 10_000.0

    return {
        "imbalance": round(imbalance, 6),
        "bid_qty": round(bid_qty, 6),
        "ask_qty": round(ask_qty, 6),
        "top5_bid_notional": round(top_n_bid_notional, 4),
        "top5_ask_notional": round(top_n_ask_notional, 4),
        "bid_wall_count": len(bid_walls),
        "ask_wall_count": len(ask_walls),
        "spread_bp": round(spread_bp, 4) if spread_bp is not None else None,
        "mid_price": round(mid_price, 6) if mid_price is not None else None,
    }
