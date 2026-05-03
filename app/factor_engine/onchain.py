"""链上因子计算：薄薄一层数据转换。

说明：
    当前未启用。FactorAggregator 已经停止调用本函数，
    等接入真实链上数据源（Glassnode / Nansen / Etherscan 等）
    并恢复 ingestion 端写入后再启用。
"""
# TODO: 等真实链上数据源接入后再启用
from __future__ import annotations

from typing import Any, Dict, Optional


def compute_onchain_factors(latest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    链上因子转换函数（当前未启用）
    -------------------------------------------------------------------
    参数：
        latest: ``onchain_metrics`` 表中最近一行的 dict，可空。
    返回：
        一个固定结构的因子 dict，包含交易所净流出、巨鲸交易数、
        gas 费用、销毁速率等关键链上指标。
    """
    if not latest:
        return {
            "available": False,
            "exchange_inflow": None,
            "exchange_outflow": None,
            "net_exchange_flow": None,
            "whale_tx_count": None,
            "gas_fee_gwei": None,
            "burn_rate": None,
        }
    inflow = float(latest.get("exchange_inflow") or 0)
    outflow = float(latest.get("exchange_outflow") or 0)
    return {
        "available": True,
        "exchange_inflow": inflow,
        "exchange_outflow": outflow,
        # Positive => more coins leaving exchanges (bullish accumulation).
        "net_exchange_flow": round(outflow - inflow, 4),
        "whale_tx_count": int(latest.get("whale_tx_count") or 0),
        "gas_fee_gwei": float(latest.get("gas_fee_gwei") or 0),
        "burn_rate": float(latest.get("burn_rate") or 0),
    }
