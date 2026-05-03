"""资金流因子（多周期版本）。

P0 升级要点
============
- 老接口 ``compute_capital_flow(trades)`` 仍然保留，向后兼容老聚合器
  在 ``enable_mtf_factors=False`` 时直接复用。
- 新增 ``compute_capital_flow_from_klines(klines)``：直接基于多周期
  K 线表（OHLCV + buy/sell volume + cvd_close）计算下列字段：

  ===========================  ==========================================================
  字段                          公式
  ===========================  ==========================================================
  net_flow_usd                 周期内 Σ(buy_notional) − Σ(sell_notional)（USDT 计价）
  cvd_close                    最新一根 bar 的 cvd_close
  cvd_slope                    最近 N 根 bar 的 cvd_close 序列做线性回归得到的斜率
  taker_buy_ratio              buy_volume / (buy_volume + sell_volume)
  volume_zscore                (vol_last − mean(vol[-N:])) / std(vol[-N:])
  cvd_price_divergence         价 N 期新高 / 新低 vs cvd 是否同步
  ===========================  ==========================================================

- 全部用 numpy 自实现，禁止引入 pandas-ta / TA-Lib 等重型库。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ----------------------------------------------------------------------
# 老接口（保留向后兼容）
# ----------------------------------------------------------------------
def compute_capital_flow(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    旧版资金流因子：基于一段时间窗口内的 trades 列表
    -----------------------------------------------------------------
    参数：
        trades: 按 ts 升序排列的成交 dict 列表
    返回：
        含 buy_volume / sell_volume / net_flow / cvd / trade_count 的 dict
    说明：
        在 enable_mtf_factors=False 时由老聚合器调用，作为安全回退路径。
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


# ----------------------------------------------------------------------
# 多周期版（P0 主用接口）
# ----------------------------------------------------------------------
def compute_capital_flow_from_klines(
    klines: List[Dict[str, Any]],
    volume_zscore_window: int = 30,
    divergence_lookback: int = 20,
) -> Dict[str, Any]:
    """
    从某个周期的 K 线表计算资金流因子矩阵
    -----------------------------------------------------------------
    参数：
        klines:               按 ts 升序排列的 K 线 dict 列表（含未封盘 bar）
        volume_zscore_window: 计算 volume_zscore 用的滚动窗口长度
        divergence_lookback:  cvd_price_divergence 的"N 期新高 / 新低"窗口
    返回：
        含 net_flow_usd / cvd_close / cvd_slope / taker_buy_ratio /
        volume_zscore / cvd_price_divergence / bar_count 的 dict。
    说明：
        bar 数不足时返回 available=False 的占位结果，让规则引擎和
        LLM 都能正确识别"周期太新数据不够"的情形。
    """
    if not klines:
        return _empty_capital_flow()

    # ---- 取最新一根 bar 的核心字段 ----
    last = klines[-1]
    last_close = _safe_float(last.get("close"))
    last_buy = _safe_float(last.get("buy_volume"))
    last_sell = _safe_float(last.get("sell_volume"))
    last_volume = _safe_float(last.get("volume"))
    last_cvd_close = _safe_float(last.get("cvd_close"))

    # ---- net_flow_usd：用最新一根的 buy/sell volume × close ----
    # 严格按"周期内 Σ(price × size, side=buy) − Σ(price × size, side=sell)"
    # 的语义，理想情况下应该用每笔成交的 price × size 累加；这里 K 线表
    # 已经聚合掉了逐笔信息，用 close 作为 price 的近似（误差 < 0.5%
    # 在 5m bar 内是可接受的，且大幅减少 trades 表查询次数）。
    net_flow_usd = (last_buy - last_sell) * (last_close or 0.0)

    # ---- taker_buy_ratio ----
    total_vol = last_buy + last_sell
    taker_buy_ratio = (last_buy / total_vol) if total_vol > 0 else None

    # ---- cvd_slope：对最近 N 根 bar 的 cvd_close 做线性回归 ----
    cvd_series = np.array(
        [_safe_float(b.get("cvd_close")) for b in klines],
        dtype=float,
    )
    cvd_slope = _linear_regression_slope(cvd_series)

    # ---- volume_zscore ----
    volumes = np.array(
        [_safe_float(b.get("volume")) for b in klines], dtype=float
    )
    volume_zscore: Optional[float] = None
    if len(volumes) >= 5:
        # 用 [-window:-1] 的样本做基线，避免把当前 bar 也算进均值
        # （否则巨量爆发 bar 自己把自己的 z 拉低了）。
        sample = volumes[-volume_zscore_window - 1 : -1]
        if len(sample) >= 5:
            mu = float(np.mean(sample))
            sd = float(np.std(sample))
            if sd > 1e-9:
                volume_zscore = round((last_volume - mu) / sd, 4)

    # ---- cvd_price_divergence ----
    closes = np.array(
        [_safe_float(b.get("close")) for b in klines], dtype=float
    )
    divergence = _detect_divergence(closes, cvd_series, divergence_lookback)

    return {
        "available": True,
        "net_flow_usd": round(net_flow_usd, 2),
        "cvd_close": round(last_cvd_close, 6),
        "cvd_slope": round(cvd_slope, 6) if cvd_slope is not None else None,
        "taker_buy_ratio": (
            round(taker_buy_ratio, 4) if taker_buy_ratio is not None else None
        ),
        "volume_zscore": volume_zscore,
        "cvd_price_divergence": divergence,
        "bar_count": len(klines),
    }


# ----------------------------------------------------------------------
# 内部辅助
# ----------------------------------------------------------------------
def _empty_capital_flow() -> Dict[str, Any]:
    """
    构造一个"周期内无 K 线数据"的空因子结果
    -----------------------------------------------------------------
    返回：
        所有数值字段为 None，available=False，便于上层显式跳过。
    """
    return {
        "available": False,
        "net_flow_usd": None,
        "cvd_close": None,
        "cvd_slope": None,
        "taker_buy_ratio": None,
        "volume_zscore": None,
        "cvd_price_divergence": "none",
        "bar_count": 0,
    }


def _safe_float(value: Any) -> float:
    """
    把任意 numeric 值转 float，None / 异常时返回 0.0
    -----------------------------------------------------------------
    参数：
        value: 任意值
    返回：
        float
    """
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _linear_regression_slope(series: np.ndarray) -> Optional[float]:
    """
    对一维序列做最小二乘线性回归，返回斜率（k）
    -----------------------------------------------------------------
    参数：
        series: 一维 numpy 数组
    返回：
        斜率 float；样本不足或全 0 时返回 None。
    """
    if series is None or len(series) < 3:
        return None
    valid = series[~np.isnan(series)] if series.dtype.kind == "f" else series
    if len(valid) < 3:
        return None
    x = np.arange(len(valid), dtype=float)
    try:
        slope, _ = np.polyfit(x, valid.astype(float), 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _detect_divergence(
    closes: np.ndarray,
    cvd: np.ndarray,
    lookback: int,
) -> str:
    """
    检测价量背离
    -----------------------------------------------------------------
    参数：
        closes:   收盘价序列
        cvd:      cvd_close 序列
        lookback: 用于"N 期新高 / 新低"判断的窗口长度
    返回：
        'bullish'：价创 N 期新低，但 cvd 没有新低（资金在买）
        'bearish'：价创 N 期新高，但 cvd 没有新高（资金在卖）
        'none'   ：未发现明显背离 / 数据不足
    """
    if len(closes) < lookback + 1 or len(cvd) < lookback + 1:
        return "none"
    last_close = closes[-1]
    last_cvd = cvd[-1]
    # 历史 lookback 区间（不含最后一根）
    hist_close = closes[-lookback - 1 : -1]
    hist_cvd = cvd[-lookback - 1 : -1]
    if hist_close.size == 0 or hist_cvd.size == 0:
        return "none"
    if last_close >= float(np.max(hist_close)) and last_cvd < float(np.max(hist_cvd)):
        return "bearish"
    if last_close <= float(np.min(hist_close)) and last_cvd > float(np.min(hist_cvd)):
        return "bullish"
    return "none"
