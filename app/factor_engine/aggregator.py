"""Factor aggregator: pull latest data from PG and compute every factor group."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.factor_engine.capital_flow import compute_capital_flow
from app.factor_engine.derivatives import compute_derivatives_factors
from app.factor_engine.market_structure import compute_market_structure
from app.factor_engine.onchain import compute_onchain_factors
from app.factor_engine.orderbook import compute_orderbook_factors
from app.logging_config import get_logger

logger = get_logger(__name__)


class FactorAggregator:
    """Aggregates the six factor groups into a single dict."""

    def __init__(self, repos: Repositories, settings: Settings):
        self.repos = repos
        self.settings = settings

    async def compute(self, symbol: str) -> Dict[str, Any]:
        window = self.settings.factor_window_seconds
        since = datetime.now(timezone.utc) - timedelta(seconds=window)

        trades = await self.repos.fetch_recent_trades(symbol, since=since, limit=20000)
        orderbook = await self.repos.fetch_latest_orderbook(symbol)
        funding = await self.repos.fetch_latest_funding(symbol)
        oi_history = await self.repos.fetch_recent_oi(symbol, since=since, limit=2000)
        onchain_latest = await self.repos.fetch_latest_onchain()

        capital = compute_capital_flow(trades)
        ob = compute_orderbook_factors(
            orderbook, wall_multiplier=self.settings.liquidity_wall_multiplier
        )
        deriv = compute_derivatives_factors(funding, oi_history)
        struct = compute_market_structure(trades)
        onchain = compute_onchain_factors(onchain_latest)

        return {
            "symbol": symbol,
            "window_seconds": window,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "capital_flow": capital,
            "orderbook": ob,
            "derivatives": deriv,
            "market_structure": struct,
            "onchain": onchain,
            "participants": {
                # Crude proxy: high whale_tx + bullish exchange outflow => smart money long
                "whale_active": (onchain.get("whale_tx_count") or 0) >= 10,
                "smart_money_bias": (
                    "long"
                    if (onchain.get("net_exchange_flow") or 0) > 0
                    else "short"
                    if (onchain.get("net_exchange_flow") or 0) < 0
                    else "neutral"
                ),
            },
        }
