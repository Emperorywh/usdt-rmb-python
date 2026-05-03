"""Mock on-chain provider.

Generates plausible numbers so the rest of the pipeline can exercise its
codepaths without external dependencies. Replace with a real provider
(Glassnode / Nansen / Etherscan) by implementing :class:`OnchainProvider`.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict

from app.data_ingestion.base import OnchainProvider


class MockOnchainProvider(OnchainProvider):
    name = "mock"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def fetch_metrics(self) -> Dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc),
            # Net flow on exchanges (ETH units).
            "exchange_inflow": round(self._rng.uniform(0, 5000), 4),
            "exchange_outflow": round(self._rng.uniform(0, 5000), 4),
            "whale_tx_count": self._rng.randint(0, 50),
            "gas_fee_gwei": round(self._rng.uniform(5, 80), 2),
            # EIP-1559 burn (ETH per minute, mocked).
            "burn_rate": round(self._rng.uniform(0, 12), 4),
        }
