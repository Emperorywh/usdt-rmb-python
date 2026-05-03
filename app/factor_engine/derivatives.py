"""衍生品因子：资金费率水平 + 持仓量变动百分比。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_derivatives_factors(
    funding: Optional[Dict[str, Any]],
    oi_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    计算衍生品因子
    -------------------------------------------------------------------
    参数：
        funding:    funding_rates 表里最近一条记录，可空。
        oi_history: open_interest 表里按 ``ts`` 升序排列的样本列表。
                    会用首尾两点的差值计算 OI 变动百分比。
    返回：
        含有以下键的因子 dict：
            funding_rate:        最新资金费率（小数表示）
            next_settlement_at:  下次资金费率结算时间（ISO 字符串），
                                 是结算时间而非"下次费率值"，仅供参考
            oi_first / oi_last:  窗口首尾两个 OI 样本
            oi_change_pct:       OI 在窗口内的相对变动
            oi_samples:          窗口内的 OI 样本数
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
        # 下次资金费率结算时间（不是"下次费率值"），改名避免 LLM 误解
        "next_settlement_at": (
            next_settlement_at.isoformat() if next_settlement_at else None
        ),
        "oi_first": oi_first,
        "oi_last": oi_last,
        "oi_change_pct": oi_change_pct,
        "oi_samples": len(oi_history),
    }
