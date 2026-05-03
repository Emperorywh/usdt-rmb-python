"""订单簿因子（P1 升级：单快照 → 单快照 + 时序）。

P0 老接口 ``compute_orderbook_factors`` 仍然保留，用于：
- 把单条最新快照解算成 imbalance / 大墙 / 价差；
- ``enable_orderbook_timeseries=False`` 灰度回滚通道。

P1 新接口 ``compute_orderbook_factors_timeseries``：
- 接收 ``latest_snapshot`` + ``recent_metrics``（最近 5/15 分钟的
  ``orderbook_metrics`` 序列），输出动态盘口因子：
    imbalance_now / imbalance_slope_5m / imbalance_zscore_15m
    bid_wall_persistence_avg_s / ask_wall_persistence_avg_s
    wall_distance_pct
    liquidity_vacuum_above / liquidity_vacuum_below
    spread_bp_now
- 同时把 P0 字段（available / imbalance / best_bid / best_ask /
  spread / bid_walls / ask_walls）一并保留，避免规则引擎、Prompt
  渲染、API 响应需要为 P0 字段额外做兼容分支。

设计原则
========
- 计算全部用 Python 内置数学 + numpy，禁止引入新依赖。
- 任何窗口样本不足时，返回 None 而不是占位 0，让 LLM 与规则引擎
  能显式判断"该指标暂时不可用"。
- 单次计算 < 5ms（最坏情况下 90 行 ts + 简单回归），不会成为热路径瓶颈。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------
# 单快照接口（P0 保留 + P1 复用 latest_snapshot 解析路径）
# ----------------------------------------------------------------------
def compute_orderbook_factors(
    snapshot: Optional[Dict[str, Any]],
    wall_multiplier: float = 3.0,
) -> Dict[str, Any]:
    """
    基于一条最新订单簿快照计算静态盘口因子
    -----------------------------------------------------------------
    参数：
        snapshot:        repos 行，含 ``bids`` / ``asks``（每档 [price, size]）
        wall_multiplier: 单档 size ≥ wall_multiplier × mean(size) 视为大墙
    返回：
        含 available / imbalance / best_bid / best_ask / spread /
        bid_qty / ask_qty / bid_walls / ask_walls 的 dict
    说明：
        - 与 P0 行为完全一致；P1 时序接口会复用其计算逻辑。
        - 不依赖时序数据，单次 < 1ms。
    """
    if not snapshot:
        return _empty_orderbook()

    bids: List[List[float]] = snapshot.get("bids") or []
    asks: List[List[float]] = snapshot.get("asks") or []

    bid_qty = sum(float(b[1]) for b in bids) if bids else 0.0
    ask_qty = sum(float(a[1]) for a in asks) if asks else 0.0
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0

    best_bid = float(bids[0][0]) if bids else None
    best_ask = float(asks[0][0]) if asks else None
    spread = (best_ask - best_bid) if (best_bid and best_ask) else None

    return {
        "available": True,
        "imbalance": round(imbalance, 6),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": round(spread, 6) if spread is not None else None,
        "bid_qty": round(bid_qty, 6),
        "ask_qty": round(ask_qty, 6),
        "bid_walls": _walls(bids, wall_multiplier),
        "ask_walls": _walls(asks, wall_multiplier),
    }


# ----------------------------------------------------------------------
# P1 主接口：单快照 + 最近 metrics 序列 → 时序因子
# ----------------------------------------------------------------------
def compute_orderbook_factors_timeseries(
    latest_snapshot: Optional[Dict[str, Any]],
    recent_metrics: List[Dict[str, Any]],
    wall_multiplier: float = 3.0,
    now: Optional[datetime] = None,
    window_seconds: int = 900,
    baseline_seconds: int = 3600,
    vacuum_band_pct: float = 0.01,
    vacuum_threshold_pct: float = 0.30,
) -> Dict[str, Any]:
    """
    单快照 + 最近 N 分钟 orderbook_metrics 序列 → 时序盘口因子
    -----------------------------------------------------------------
    参数：
        latest_snapshot:      最新一条 orderbook_snapshots 行
        recent_metrics:       fetch_orderbook_metrics_since 升序结果，
                              覆盖窗口至少 max(window_seconds, baseline_seconds)
        wall_multiplier:      复用 P0 的大墙阈值
        now:                  当前 UTC 时间，None 时取系统时间
        window_seconds:       imbalance_zscore / wall_persistence 的统计窗口
        baseline_seconds:     liquidity_vacuum 的历史均值基线窗口
        vacuum_band_pct:      流动性真空判断的价格半径（默认 ±1%）
        vacuum_threshold_pct: 当前带宽内挂单 < 基线均值 × N 时视为真空
    返回：
        合并后的 dict，键含 P0 静态字段 + P1 时序字段；
        样本不足时对应 P1 字段返回 None / False，而不是抛错。
    """
    base = compute_orderbook_factors(latest_snapshot, wall_multiplier)
    base["recent_metric_count"] = len(recent_metrics or [])

    now = now or datetime.now(timezone.utc)
    metrics = list(recent_metrics or [])

    # ---- imbalance_now：直接复用静态字段，避免来回换算 ----
    imbalance_now = base.get("imbalance")
    base["imbalance_now"] = imbalance_now

    # ---- imbalance_slope_5m：最近 5 分钟 imbalance 的线性回归斜率 ----
    base["imbalance_slope_5m"] = _slope_within(
        metrics, key="imbalance", now=now, seconds=5 * 60
    )

    # ---- imbalance_zscore_15m：(now - mean_15m) / std_15m ----
    base["imbalance_zscore_15m"] = _zscore_within(
        metrics,
        key="imbalance",
        now=now,
        seconds=window_seconds,
        current=imbalance_now,
    )

    # ---- 当前快照的 spread_bp / mid_price（不依赖时序）----
    best_bid = base.get("best_bid")
    best_ask = base.get("best_ask")
    mid_price: Optional[float] = None
    spread_bp_now: Optional[float] = None
    if best_bid and best_ask and best_bid > 0 and best_ask > 0:
        mid_price = (best_bid + best_ask) / 2.0
        if mid_price > 0:
            spread_bp_now = round(
                (best_ask - best_bid) / mid_price * 10_000.0, 4
            )
    base["mid_price"] = round(mid_price, 6) if mid_price is not None else None
    base["spread_bp_now"] = spread_bp_now

    # ---- bid/ask wall persistence：当前墙在过去窗口内出现过的累计秒数均值 ----
    base["bid_wall_persistence_avg_s"] = _wall_persistence_seconds(
        walls=base.get("bid_walls") or [],
        metrics=metrics,
        now=now,
        seconds=5 * 60,
        side="bid",
    )
    base["ask_wall_persistence_avg_s"] = _wall_persistence_seconds(
        walls=base.get("ask_walls") or [],
        metrics=metrics,
        now=now,
        seconds=5 * 60,
        side="ask",
    )

    # ---- 最近一面墙的距离百分比（按方向各一个）----
    base["wall_distance_pct"] = _nearest_wall_distance_pct(
        bid_walls=base.get("bid_walls") or [],
        ask_walls=base.get("ask_walls") or [],
        mid_price=mid_price,
    )

    # ---- 流动性真空：当前价上下 ±1% 区间挂单 vs 1h 基线 ----
    vacuum_above, vacuum_below = _liquidity_vacuum(
        snapshot=latest_snapshot,
        metrics=metrics,
        now=now,
        baseline_seconds=baseline_seconds,
        mid_price=mid_price,
        band_pct=vacuum_band_pct,
        threshold_pct=vacuum_threshold_pct,
    )
    base["liquidity_vacuum_above"] = vacuum_above
    base["liquidity_vacuum_below"] = vacuum_below

    return base


# ----------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------
def _empty_orderbook() -> Dict[str, Any]:
    """
    构造一个"快照不可用"的占位返回，键集合保持稳定
    -----------------------------------------------------------------
    返回：
        所有字段安全默认值；available=False 让上层显式跳过。
    """
    return {
        "available": False,
        "imbalance": 0.0,
        "best_bid": None,
        "best_ask": None,
        "spread": None,
        "bid_qty": 0.0,
        "ask_qty": 0.0,
        "bid_walls": [],
        "ask_walls": [],
    }


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


def _slope_within(
    metrics: List[Dict[str, Any]],
    key: str,
    now: datetime,
    seconds: int,
) -> Optional[float]:
    """
    对最近 seconds 秒内 metric[key] 序列做线性回归，返回斜率
    -----------------------------------------------------------------
    参数：
        metrics: 升序的 orderbook_metrics dict 列表
        key:     字段名（如 'imbalance'）
        now:     当前 UTC 时间
        seconds: 回看秒数
    返回：
        斜率 float；样本 < 3 时返回 None。
    说明：
        x 轴用"距 now 的秒数"做尺度，斜率单位 = 单位时间（秒）的变化量；
        因为 imbalance ∈ [-1, 1] 较小，结果数量级一般在 1e-5 ~ 1e-3，
        渲染到 prompt 时按 6 位小数即可保留有效信号。
    """
    cutoff = now - timedelta(seconds=seconds)
    xs: List[float] = []
    ys: List[float] = []
    for m in metrics:
        ts = m.get("ts")
        if ts is None or ts < cutoff:
            continue
        v = m.get(key)
        if v is None:
            continue
        try:
            ys.append(float(v))
        except (TypeError, ValueError):
            continue
        xs.append((ts - cutoff).total_seconds())
    if len(xs) < 3:
        return None
    try:
        slope, _ = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)
        return round(float(slope), 8)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _zscore_within(
    metrics: List[Dict[str, Any]],
    key: str,
    now: datetime,
    seconds: int,
    current: Optional[float],
) -> Optional[float]:
    """
    计算 current 相对于过去 seconds 秒内 metric[key] 序列均值/标准差的 z-score
    -----------------------------------------------------------------
    参数：
        metrics: 升序的 orderbook_metrics dict 列表
        key:     字段名
        now:     当前 UTC 时间
        seconds: 回看秒数
        current: 当前值；None 或 std<eps 时返回 None
    返回：
        z-score float；样本 < 5 时返回 None。
    """
    if current is None:
        return None
    cutoff = now - timedelta(seconds=seconds)
    vals: List[float] = []
    for m in metrics:
        ts = m.get("ts")
        if ts is None or ts < cutoff:
            continue
        v = m.get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(vals) < 5:
        return None
    arr = np.array(vals, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < 1e-9:
        return None
    return round((float(current) - mu) / sd, 4)


def _wall_persistence_seconds(
    walls: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    now: datetime,
    seconds: int,
    side: str,
) -> Optional[float]:
    """
    估算当前所有墙在过去 seconds 秒内出现过的累计秒数均值
    -----------------------------------------------------------------
    参数：
        walls:   当前快照中识别出的大墙列表
        metrics: 升序的 orderbook_metrics dict 列表
        now:     当前 UTC 时间
        seconds: 回看秒数
        side:    'bid' / 'ask'，决定从 metric 的 bid_wall_count /
                 ask_wall_count 取计数
    返回：
        累计秒数均值（int 秒近似）；当前没墙或样本太少时返回 None。
    说明：
        我们没有逐档历史 bids/asks（那样存储成本太高），所以 wall
        的"持续时长"用一个近似指标：
            count_history = 过去窗口内有过 ≥1 面墙的样本数
            avg_count_active = 这些样本上墙数的均值
            每个样本之间的近似间隔 ≈ window / max(len, 1) 秒
        最终输出 = count_history × interval × avg_count_active / current_wall_count
        含义大致等于"当前每面墙被观测到的累计在场秒数"。这是 P1 阶段
        的近似估计；P2 接 incremental orderbook 后再换成精确版本。
    """
    if not walls:
        return None
    cutoff = now - timedelta(seconds=seconds)
    key = "bid_wall_count" if side == "bid" else "ask_wall_count"
    counts: List[int] = []
    for m in metrics:
        ts = m.get("ts")
        if ts is None or ts < cutoff:
            continue
        c = m.get(key)
        try:
            ci = int(c) if c is not None else 0
        except (TypeError, ValueError):
            ci = 0
        counts.append(ci)
    if not counts:
        return 0.0

    active = [c for c in counts if c > 0]
    if not active:
        return 0.0

    interval = float(seconds) / max(len(counts), 1)
    avg_active_count = float(np.mean(active))
    persistence = interval * len(active) * avg_active_count / max(len(walls), 1)
    return round(min(persistence, float(seconds)), 1)


def _nearest_wall_distance_pct(
    bid_walls: List[Dict[str, Any]],
    ask_walls: List[Dict[str, Any]],
    mid_price: Optional[float],
) -> Dict[str, Optional[float]]:
    """
    计算最近一面 bid 墙 / ask 墙到中间价的距离百分比
    -----------------------------------------------------------------
    参数：
        bid_walls / ask_walls: P0 _walls 输出
        mid_price:             (best_bid + best_ask) / 2
    返回：
        {bid: float|None, ask: float|None}
        bid 取最高价的 bid 墙，ask 取最低价的 ask 墙；正数表示在另一侧。
    """
    out: Dict[str, Optional[float]] = {"bid": None, "ask": None}
    if not mid_price or mid_price <= 0:
        return out
    if bid_walls:
        bw = max(bid_walls, key=lambda w: float(w.get("price") or 0.0))
        out["bid"] = round((float(bw["price"]) - mid_price) / mid_price, 6)
    if ask_walls:
        aw = min(ask_walls, key=lambda w: float(w.get("price") or 1e18))
        out["ask"] = round((float(aw["price"]) - mid_price) / mid_price, 6)
    return out


def _liquidity_vacuum(
    snapshot: Optional[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    now: datetime,
    baseline_seconds: int,
    mid_price: Optional[float],
    band_pct: float,
    threshold_pct: float,
) -> tuple[bool, bool]:
    """
    判断当前价上方 / 下方 ±band_pct 区间的挂单是否构成"流动性真空"
    -----------------------------------------------------------------
    参数：
        snapshot:         最新快照（用于上方挂单总量）
        metrics:          升序 orderbook_metrics（用于基线均值）
        now:              当前 UTC 时间
        baseline_seconds: 基线窗口（默认 1h）
        mid_price:        中间价
        band_pct:         上下扫描区间半径
        threshold_pct:    当前带宽内挂单 / 基线均值 × N 即视为真空
    返回：
        (vacuum_above, vacuum_below) 两个布尔；任意计算前提缺失时返回 (False, False)
    说明：
        基线 = 过去 baseline_seconds 内 (top5_bid_notional 均值, top5_ask_notional 均值)；
        当前带内挂单总量 = snapshot 中价格落入 [mid*(1-band), mid*(1+band)] 的 size × price 累计。
    """
    if snapshot is None or mid_price is None or mid_price <= 0:
        return False, False
    bids = snapshot.get("bids") or []
    asks = snapshot.get("asks") or []
    band_low = mid_price * (1.0 - band_pct)
    band_high = mid_price * (1.0 + band_pct)

    notional_above = sum(
        float(p) * float(s)
        for p, s in (a[:2] for a in asks)
        if mid_price < float(p) <= band_high
    )
    notional_below = sum(
        float(p) * float(s)
        for p, s in (b[:2] for b in bids)
        if band_low <= float(p) < mid_price
    )

    cutoff = now - timedelta(seconds=baseline_seconds)
    bid_notional_hist: List[float] = []
    ask_notional_hist: List[float] = []
    for m in metrics:
        ts = m.get("ts")
        if ts is None or ts < cutoff:
            continue
        bn = m.get("top5_bid_notional")
        an = m.get("top5_ask_notional")
        if bn is not None:
            try:
                bid_notional_hist.append(float(bn))
            except (TypeError, ValueError):
                pass
        if an is not None:
            try:
                ask_notional_hist.append(float(an))
            except (TypeError, ValueError):
                pass

    vacuum_above = False
    vacuum_below = False
    if ask_notional_hist:
        baseline_ask = float(np.mean(ask_notional_hist))
        if baseline_ask > 0 and notional_above < baseline_ask * threshold_pct:
            vacuum_above = True
    if bid_notional_hist:
        baseline_bid = float(np.mean(bid_notional_hist))
        if baseline_bid > 0 and notional_below < baseline_bid * threshold_pct:
            vacuum_below = True
    return vacuum_above, vacuum_below


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
