"""On-chain factor calculation: thin transformation layer."""
from __future__ import annotations

from typing import Any, Dict, Optional


def compute_onchain_factors(latest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
