"""衍生品因子（P1 升级：funding 分位数 + OI/价散度 + 持仓比）。

P0 老接口 ``compute_derivatives_factors`` 与 ``compute_derivatives_per_timeframe``
全部保留并继续填充原字段，仅在末尾追加 P1 新字段。

P1 新增字段（每周期 derivatives 块）：
    funding_rate_pct_rank_7d : 当前 funding 在 7 天历史中的分位数（0~1）
    funding_extreme          : 'long_squeeze_risk' / 'short_squeeze_risk' / 'neutral'
    oi_price_divergence      : 'potential_top' / 'potential_bottom' / 'none'

P1 新增字段（顶层 derivatives 维度，全周期共享）：
    account_long_short_ratio       : 散户多空账户比（反指）
    account_contract_ratio         : 合约级散户多空（反指，更精确）
    top_trader_position_ratio      : 精英持仓比（顺指）
    retail_vs_smart_divergence     : 'bearish_warning' / 'bullish_warning' / 'aligned' / 'unknown'

OI/价 散度规则
============
    OI 创新高 + 价格未创新高 → potential_top
    OI 创新低 + 价格未创新低 → potential_bottom
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
    funding_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    单个周期的衍生品因子：funding 共享 + OI 起止差 + 4 象限关系 +
    P1 funding 分位数 + OI/价 散度
    -----------------------------------------------------------------
    参数：
        funding:         funding_rates 最新一条记录，全周期共享
        oi_history:      整个分析窗口内的 OI 样本（按 ts 升序）
        klines:          当前周期的 K 线序列（按 ts 升序，含未封盘 bar）
        funding_history: 7 天 funding 历史（升序），P1 新增；为空时分位数返回 None
    返回：
        含 P0 老字段（funding_rate_now / oi_change_pct / oi_first / oi_last
        / oi_price_relation / next_settlement_at）+ P1 新字段
        （funding_rate_pct_rank_7d / funding_extreme / oi_price_divergence）
        的 dict。
    说明：
        - oi_change_pct 取 oi_history 中 "ts <= 周期端点"的最后一条样本近似。
        - oi_price_divergence 用周期内 OI 与 close 的"是否同时刷新历史极值"
          来判断潜在顶/底；与 oi_price_relation 互补：后者看方向，本字段
          看"是否衰竭"。
        - funding_rate_pct_rank_7d 没有 funding_history 时返回 None；
          有则等于 (≤ current 的样本数) / 总样本数，∈ [0, 1]。
    """
    funding_rate_now: Optional[float] = None
    next_settlement_at = None
    if funding:
        funding_rate_now = float(funding.get("funding_rate") or 0.0)
        next_settlement_at = funding.get("next_funding_ts")

    funding_pct_rank, funding_extreme = _funding_pct_rank_and_extreme(
        current=funding_rate_now,
        history=funding_history,
    )

    base_empty = {
        "funding_rate_now": funding_rate_now,
        "oi_change_pct": None,
        "oi_first": None,
        "oi_last": None,
        "oi_price_relation": "unknown",
        "oi_price_divergence": "none",
        "funding_rate_pct_rank_7d": funding_pct_rank,
        "funding_extreme": funding_extreme,
        "next_settlement_at": (
            next_settlement_at.isoformat() if next_settlement_at else None
        ),
    }

    if not klines or not oi_history:
        return base_empty

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

    oi_price_divergence = _classify_oi_price_divergence(
        oi_history=oi_history,
        klines=klines,
    )

    return {
        "funding_rate_now": funding_rate_now,
        "oi_change_pct": oi_change_pct,
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_price_relation": oi_price_relation,
        "oi_price_divergence": oi_price_divergence,
        "funding_rate_pct_rank_7d": funding_pct_rank,
        "funding_extreme": funding_extreme,
        "next_settlement_at": (
            next_settlement_at.isoformat() if next_settlement_at else None
        ),
    }


# ----------------------------------------------------------------------
# P1：持仓比因子（顶层 derivatives 维度，全周期共享）
# ----------------------------------------------------------------------
def compute_position_ratio_factors(
    latest_ratios: Dict[str, Dict[str, Any]],
    bearish_account_threshold: float = 1.5,
    bullish_account_threshold: float = 0.7,
    smart_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    把 fetch_latest_position_ratios 的结果聚合成因子层
    -----------------------------------------------------------------
    参数：
        latest_ratios:             repos.fetch_latest_position_ratios 返回
        bearish_account_threshold: 散户多空比 ≥ 此值时视为"散户狂多"
        bullish_account_threshold: 散户多空比 ≤ 此值时视为"散户狂空"
        smart_threshold:           精英多空比 ≥ 此值视为"精英偏多"
    返回：
        含 account_long_short_ratio / account_contract_ratio /
        top_trader_position_ratio / retail_vs_smart_divergence 的 dict
    说明：
        - 任意维度数据缺失时该字段保持 None；divergence 当散户或精英任一
          缺失时返回 'unknown'。
        - 反指逻辑：散户狂多 + 精英反向 → 'bearish_warning'；
                    散户狂空 + 精英反向 → 'bullish_warning'；
                    其他 → 'aligned'（不一定是看多看空，只表示无显著反向）。
        - 阈值宽容性：bearish/bullish 用对称阈值，避免边界抖动。
    """
    def _ratio(key: str) -> Optional[float]:
        row = latest_ratios.get(key)
        if not row:
            return None
        v = row.get("ratio")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    account = _ratio("account")
    account_contract = _ratio("account_contract")
    top_trader = _ratio("top_trader_position")

    # 散户优先用合约级（更精确），缺失时退化到币种级
    retail_signal = account_contract if account_contract is not None else account
    divergence = "unknown"
    if retail_signal is not None and top_trader is not None:
        if (
            retail_signal >= bearish_account_threshold
            and top_trader < smart_threshold
        ):
            divergence = "bearish_warning"
        elif (
            retail_signal <= bullish_account_threshold
            and top_trader > smart_threshold
        ):
            divergence = "bullish_warning"
        else:
            divergence = "aligned"

    return {
        "account_long_short_ratio": account,
        "account_contract_ratio": account_contract,
        "top_trader_position_ratio": top_trader,
        "retail_vs_smart_divergence": divergence,
    }


# ----------------------------------------------------------------------
# P1：funding 分位数 + 极值标签
# ----------------------------------------------------------------------
def _funding_pct_rank_and_extreme(
    current: Optional[float],
    history: Optional[List[Dict[str, Any]]],
) -> tuple[Optional[float], str]:
    """
    计算 current 在 history 中的百分位 + 极值标签
    -----------------------------------------------------------------
    参数：
        current: 当前 funding_rate
        history: 7 天 funding 历史（升序），元素 dict 含 'funding_rate'
    返回：
        (pct_rank, extreme_label)
            pct_rank ∈ [0, 1]，None 表示样本不足或 current 缺失
            extreme_label ∈ {'long_squeeze_risk', 'short_squeeze_risk', 'neutral'}
    说明：
        - 取 ≤ current 的样本数 / 总样本数（含 current 自身）作为分位。
        - rank > 0.95：funding 接近 7 天最高 → 多头拥挤，潜在 long squeeze。
        - rank < 0.05：funding 接近 7 天最低 → 空头拥挤，潜在 short squeeze。
        - 样本 < 50 时不打极值标签（噪声太大），返回 'neutral'。
    """
    if current is None or not history:
        return None, "neutral"
    vals: List[float] = []
    for row in history:
        v = row.get("funding_rate")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(vals) < 5:
        return None, "neutral"
    le_count = sum(1 for v in vals if v <= current)
    rank = round(le_count / len(vals), 4)

    if len(vals) >= 50:
        if rank > 0.95:
            return rank, "long_squeeze_risk"
        if rank < 0.05:
            return rank, "short_squeeze_risk"
    return rank, "neutral"


def _classify_oi_price_divergence(
    oi_history: List[Dict[str, Any]],
    klines: List[Dict[str, Any]],
) -> str:
    """
    判断 OI 与价是否出现"创新高/新低"散度（潜在顶/底）
    -----------------------------------------------------------------
    参数：
        oi_history: 升序 OI 样本（覆盖 K 线时间段）
        klines:     当前周期 K 线（升序）
    返回：
        'potential_top' / 'potential_bottom' / 'none'
    说明：
        - 仅看周期内的极值：
            OI_last == max(OI) AND price_last < max(price) → potential_top
            OI_last == min(OI) AND price_last > min(price) → potential_bottom
        - 极值用容忍度比较，避免数值微抖：相对差 < 0.05% 视为相等。
    """
    if not oi_history or not klines:
        return "none"
    try:
        oi_vals = [float(r.get("oi") or 0.0) for r in oi_history if r.get("oi") is not None]
        price_vals = [float(b.get("close") or 0.0) for b in klines if b.get("close") is not None]
    except (TypeError, ValueError):
        return "none"
    if not oi_vals or not price_vals:
        return "none"
    oi_last = oi_vals[-1]
    oi_max = max(oi_vals)
    oi_min = min(oi_vals)
    px_last = price_vals[-1]
    px_max = max(price_vals)
    px_min = min(price_vals)

    eps = 5e-4
    oi_at_top = oi_max > 0 and (oi_max - oi_last) / oi_max < eps
    oi_at_bot = oi_min > 0 and (oi_last - oi_min) / oi_min < eps
    px_below_top = px_max > 0 and (px_max - px_last) / px_max > eps
    px_above_bot = px_min > 0 and (px_last - px_min) / px_min > eps

    if oi_at_top and px_below_top:
        return "potential_top"
    if oi_at_bot and px_above_bot:
        return "potential_bottom"
    return "none"


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
