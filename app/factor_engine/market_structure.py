"""市场结构因子（多周期版本）。

基于指定周期的 K 线序列（最近 N 根 bar，N=80 默认）计算：

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


def _bollinger_width(
    closes: np.ndarray, period: int = 20, k: float = 2.0
) -> Optional[float]:
    """
    布林带宽度 = (上轨 - 下轨) / mid（最近一根 bar）
    -----------------------------------------------------------------
    参数：
        closes: 收盘价序列
        period: 回看周期（默认 20）
        k:      ±k 倍标准差（默认 2.0）
    返回：
        bb_width float；样本不足时返回 None。
    说明：
        - 用 numpy 总体标准差（与 TradingView 默认一致）。
        - 输出归一到 mid，让不同价位档的合约可直接对比。
    """
    if closes is None or len(closes) < period:
        return None
    window = closes[-period:]
    mid = float(np.mean(window))
    if mid <= 0:
        return None
    sd = float(np.std(window))
    width = (2.0 * k * sd) / mid
    return round(float(width), 6)


def _adx_14(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> Optional[float]:
    """
    Wilder ADX(14) 实现，纯 numpy 自实现
    -----------------------------------------------------------------
    参数：
        highs / lows / closes: 等长一维数组
        period:                ADX 周期（默认 14）
    返回：
        最新 ADX 值；样本不足时返回 None。
    说明：
        - 步骤：
            +DM = max(high - high_prev, 0)，且 > -DM 时保留，否则置 0
            -DM = max(low_prev - low, 0)，且 > +DM 时保留，否则置 0
            TR  = max(h-l, |h-c_prev|, |l-c_prev|)
            +DI = 100 × Wilder平滑(+DM) / Wilder平滑(TR)
            -DI = 100 × Wilder平滑(-DM) / Wilder平滑(TR)
            DX  = 100 × |+DI - -DI| / (+DI + -DI)
            ADX = Wilder平滑(DX, period)
        - 直接 RMA 替代 SMA 起始：第一次 RMA 取前 period 个 DX 的均值。
    """
    n = len(closes)
    if n < 2 * period + 1:
        return None
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    tr = np.zeros(n, dtype=float)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    def _rma(values: np.ndarray) -> np.ndarray:
        """对 values 做 Wilder RMA 平滑（返回等长数组，前 period-1 个置 NaN）"""
        out = np.full_like(values, np.nan, dtype=float)
        if len(values) < period + 1:
            return out
        seed = float(np.sum(values[1 : period + 1]))
        out[period] = seed
        for i in range(period + 1, len(values)):
            out[i] = out[i - 1] - (out[i - 1] / period) + values[i]
        return out

    sm_plus = _rma(plus_dm)
    sm_minus = _rma(minus_dm)
    sm_tr = _rma(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * np.where(sm_tr > 0, sm_plus / sm_tr, 0.0)
        minus_di = 100.0 * np.where(sm_tr > 0, sm_minus / sm_tr, 0.0)
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    valid_dx = dx[period:]
    if len(valid_dx) < period:
        return None
    adx = float(np.mean(valid_dx[:period]))
    for i in range(period, len(valid_dx)):
        adx = (adx * (period - 1) + valid_dx[i]) / period
    return round(adx, 4)


def _swept_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    lookback: int = 12,
    sweep_pct: float = 0.0005,
) -> tuple[bool, bool]:
    """
    判断最近 lookback 根 bar 内是否出现"刺破前 swing 高低点又收回"
    -----------------------------------------------------------------
    参数：
        highs / lows / closes: 一维数组
        lookback:              回看根数（默认 12）
        sweep_pct:             刺破幅度阈值（相对前 swing 极值，默认 0.05%）
    返回：
        (swept_high_recent, swept_low_recent)
    说明：
        - 前 swing 高/低 = lookback 之前的最大/最小值；
        - "刺破并收回" = 最近 lookback 内某根 bar high > prev_swing_high*(1+pct)，
          但 close <= prev_swing_high；low 反之。
        - 样本不足时返回 (False, False)。
    """
    n = len(closes)
    if n < lookback + 5:
        return False, False
    prev_h = float(np.max(highs[: n - lookback]))
    prev_l = float(np.min(lows[: n - lookback]))
    swept_high = False
    swept_low = False
    for i in range(n - lookback, n):
        if (
            not swept_high
            and prev_h > 0
            and highs[i] > prev_h * (1.0 + sweep_pct)
            and closes[i] <= prev_h
        ):
            swept_high = True
        if (
            not swept_low
            and prev_l > 0
            and lows[i] < prev_l * (1.0 - sweep_pct)
            and closes[i] >= prev_l
        ):
            swept_low = True
    return swept_high, swept_low


def _value_area(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    coverage: float = 0.70,
) -> tuple[Optional[float], Optional[float]]:
    """
    用 K 线 OHLC 近似的 70% 成交量集中区间（Value Area High / Low）
    -----------------------------------------------------------------
    参数：
        highs / lows / closes / volumes: 等长一维数组
        coverage:                         覆盖比例（默认 0.70）
    返回：
        (value_area_high, value_area_low)；样本不足时返回 (None, None)
    说明：
        - 严格 TPO/Volume Profile 需要逐笔成交，本项目仅基于 K 线 OHLC：
          把每根 bar 的 volume 均匀摊到 [low, high] 区间内的若干价格桶
          （bucket 数 ≈ 50），再从"成交量密度最高的桶"两侧扩张到 coverage。
        - 这是 Value Area 的近似实现，足够给 LLM 做"区间策略 supports/resistances"
          的辅助输入，不用于精确量化决策。
    """
    n = len(closes)
    if n < 8:
        return None, None
    px_min = float(np.min(lows))
    px_max = float(np.max(highs))
    if px_max <= px_min:
        return None, None
    bucket_count = 50
    bucket_size = (px_max - px_min) / bucket_count
    if bucket_size <= 0:
        return None, None
    profile = np.zeros(bucket_count, dtype=float)
    for i in range(n):
        h, l, v = float(highs[i]), float(lows[i]), float(volumes[i] or 0.0)
        if v <= 0 or h <= l:
            continue
        b_low = max(0, int((l - px_min) // bucket_size))
        b_high = min(bucket_count - 1, int((h - px_min) // bucket_size))
        if b_high < b_low:
            continue
        share = v / (b_high - b_low + 1)
        profile[b_low : b_high + 1] += share

    total = float(np.sum(profile))
    if total <= 0:
        return None, None
    target = total * coverage
    poc = int(np.argmax(profile))
    accumulated = float(profile[poc])
    lo = hi = poc
    while accumulated < target and (lo > 0 or hi < bucket_count - 1):
        left = float(profile[lo - 1]) if lo > 0 else -1.0
        right = float(profile[hi + 1]) if hi < bucket_count - 1 else -1.0
        if right >= left:
            hi += 1
            accumulated += max(0.0, right)
        else:
            lo -= 1
            accumulated += max(0.0, left)
    vah = px_min + (hi + 1) * bucket_size
    val = px_min + lo * bucket_size
    return round(float(vah), 4), round(float(val), 4)


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
# 多周期版（主接口）
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
            "adx_14": None,
            "bb_width": None,
            "supports": [],
            "resistances": [],
            "last_close": float(last["close"]) if last and last.get("close") else None,
            "slope": None,
            "bar_count": len(klines),
            "swept_high_recent": False,
            "swept_low_recent": False,
            "value_area_high": None,
            "value_area_low": None,
        }

    highs = np.array([float(b.get("high") or 0.0) for b in klines], dtype=float)
    lows = np.array([float(b.get("low") or 0.0) for b in klines], dtype=float)
    closes = np.array([float(b.get("close") or 0.0) for b in klines], dtype=float)
    volumes = np.array([float(b.get("volume") or 0.0) for b in klines], dtype=float)
    last_close = float(closes[-1])

    trend = _classify_trend(highs, lows)
    atr = _wilder_atr(highs, lows, closes, period=14)
    slope = _linear_regression_slope(closes)
    bb_w = _bollinger_width(closes, period=20, k=2.0)
    adx = _adx_14(highs, lows, closes, period=14)
    swept_high, swept_low = _swept_levels(
        highs, lows, closes, lookback=12, sweep_pct=0.0005
    )
    vah, val = _value_area(highs, lows, closes, volumes, coverage=0.70)

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
        "adx_14": adx,
        "bb_width": bb_w,
        "supports": supports,
        "resistances": resistances,
        "last_close": round(float(last_close), 4),
        "slope": round(float(slope), 6) if slope is not None else None,
        "bar_count": len(klines),
        "swept_high_recent": swept_high,
        "swept_low_recent": swept_low,
        "value_area_high": vah,
        "value_area_low": val,
    }
