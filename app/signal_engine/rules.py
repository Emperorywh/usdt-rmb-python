"""Rule-based pre-screening engine.

Applies a small set of deterministic heuristics to the aggregated factor dict
and returns a :class:`TradingSignal` plus a numeric score in ``[-1, 1]``.

The score is also exposed to the LLM agent as additional context.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from app.config import Settings
from app.signal_engine.schemas import TradingSignal


# Thresholds are deliberately conservative; tune via Settings if needed.
NET_FLOW_USD_THRESHOLD = 50_000.0
ORDERBOOK_IMBALANCE_THRESHOLD = 0.15
OI_CHANGE_THRESHOLD = 0.005  # 0.5%
FUNDING_RATE_THRESHOLD = 0.00005  # 0.005% per 8h is roughly neutral


class RuleEngine:
    """Deterministic, fast first-pass signal generator."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate(self, factors: Dict[str, Any]) -> Tuple[TradingSignal, float, Dict[str, Any]]:
        """Return ``(signal, score, breakdown)``."""

        cap = factors.get("capital_flow", {}) or {}
        ob = factors.get("orderbook", {}) or {}
        deriv = factors.get("derivatives", {}) or {}
        struct = factors.get("market_structure", {}) or {}
        onchain = factors.get("onchain", {}) or {}

        contributions: Dict[str, float] = {}

        net_flow = float(cap.get("net_flow") or 0.0)
        if net_flow > NET_FLOW_USD_THRESHOLD:
            contributions["capital_flow"] = +1.0
        elif net_flow < -NET_FLOW_USD_THRESHOLD:
            contributions["capital_flow"] = -1.0
        else:
            contributions["capital_flow"] = 0.0

        imbalance = float(ob.get("imbalance") or 0.0)
        if imbalance > ORDERBOOK_IMBALANCE_THRESHOLD:
            contributions["orderbook"] = +1.0
        elif imbalance < -ORDERBOOK_IMBALANCE_THRESHOLD:
            contributions["orderbook"] = -1.0
        else:
            contributions["orderbook"] = 0.0

        funding = float(deriv.get("funding_rate") or 0.0)
        oi_change = deriv.get("oi_change_pct")
        oi_signal = 0.0
        if oi_change is not None:
            if oi_change > OI_CHANGE_THRESHOLD and funding > FUNDING_RATE_THRESHOLD:
                oi_signal = +1.0
            elif oi_change > OI_CHANGE_THRESHOLD and funding < -FUNDING_RATE_THRESHOLD:
                oi_signal = -1.0
            elif oi_change < -OI_CHANGE_THRESHOLD:
                oi_signal = -0.5
        contributions["derivatives"] = oi_signal

        trend = struct.get("trend") or "range"
        contributions["market_structure"] = (
            +1.0 if trend == "uptrend" else -1.0 if trend == "downtrend" else 0.0
        )

        net_ex_flow = onchain.get("net_exchange_flow") or 0.0
        if net_ex_flow > 0:
            contributions["onchain"] = +0.5
        elif net_ex_flow < 0:
            contributions["onchain"] = -0.5
        else:
            contributions["onchain"] = 0.0

        weights = {
            "capital_flow": 0.30,
            "orderbook": 0.15,
            "derivatives": 0.20,
            "market_structure": 0.25,
            "onchain": 0.10,
        }
        score = sum(contributions[k] * weights[k] for k in weights)
        score = max(min(score, 1.0), -1.0)

        if score >= 0.25:
            bias = "long"
        elif score <= -0.25:
            bias = "short"
        else:
            bias = "neutral"

        confidence = round(min(abs(score), 1.0), 3)

        reason_parts = [
            f"net_flow={net_flow:.0f}",
            f"ob_imbalance={imbalance:+.3f}",
            f"funding={funding:+.6f}",
            f"oi_change={oi_change if oi_change is None else f'{oi_change:+.4f}'}",
            f"trend={trend}",
            f"net_exchange_flow={net_ex_flow:+.2f}",
        ]
        reason = "Rule-engine: " + ", ".join(reason_parts)

        risk_bits = []
        if abs(net_flow) < NET_FLOW_USD_THRESHOLD:
            risk_bits.append("low capital-flow conviction")
        if trend == "range":
            risk_bits.append("no clear trend")
        if not risk_bits:
            risk_bits.append("watch funding flips and large opposite walls")
        risk = "; ".join(risk_bits)

        if bias == "long":
            sup = struct.get("supports") or []
            res = struct.get("resistances") or []
            entry = struct.get("last_price")
            stop = sup[0] if sup else None
            target = res[0] if res else None
            suggestion = (
                f"Consider scaling in long around {entry}. "
                f"Stop near support {stop}. Target near resistance {target}."
            )
        elif bias == "short":
            sup = struct.get("supports") or []
            res = struct.get("resistances") or []
            entry = struct.get("last_price")
            stop = res[0] if res else None
            target = sup[0] if sup else None
            suggestion = (
                f"Consider scaling in short around {entry}. "
                f"Stop near resistance {stop}. Target near support {target}."
            )
        else:
            suggestion = "Stay flat; wait for clearer alignment across factors."

        signal = TradingSignal(
            bias=bias,
            confidence=confidence,
            reason=reason,
            risk=risk,
            suggestion=suggestion,
        )
        return signal, score, contributions
