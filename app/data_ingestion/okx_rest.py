"""OKX V5 public REST client.

Used as a fallback / supplementary source for funding rate and open interest
data, plus a stub for liquidation orders (only available with auth in OKX V5,
so we just return [] when unauthorised).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from app.data_ingestion.base import ExchangeRestClient
from app.logging_config import get_logger

logger = get_logger(__name__)


def _ms_to_dt(ms: str | int | None) -> Optional[datetime]:
    if not ms:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


class OKXRestClient(ExchangeRestClient):
    name = "okx"

    def __init__(self, base_url: str = "https://www.okx.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        resp = await self._client.get(
            "/api/v5/public/funding-rate", params={"instId": symbol}
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != "0" or not body.get("data"):
            raise RuntimeError(f"OKX funding-rate error: {body}")
        item = body["data"][0]
        return {
            "exchange": self.name,
            "symbol": symbol,
            "ts": _ms_to_dt(item.get("ts")) or datetime.now(timezone.utc),
            "funding_rate": float(item["fundingRate"]),
            "next_funding_ts": _ms_to_dt(item.get("nextFundingTime")),
        }

    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        # instType inferred from symbol: -SWAP / -FUTURES style.
        inst_type = "SWAP" if symbol.endswith("-SWAP") else "FUTURES"
        resp = await self._client.get(
            "/api/v5/public/open-interest",
            params={"instType": inst_type, "instId": symbol},
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != "0" or not body.get("data"):
            raise RuntimeError(f"OKX open-interest error: {body}")
        item = body["data"][0]
        return {
            "exchange": self.name,
            "symbol": symbol,
            "ts": _ms_to_dt(item.get("ts")) or datetime.now(timezone.utc),
            "oi": float(item.get("oi") or 0),
            "oi_ccy": float(item.get("oiCcy") or 0) or None,
        }

    async def fetch_liquidation_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Public liquidation history.

        Note: OKX exposes ``/api/v5/public/liquidation-orders`` but it requires
        an ``instType`` and is rate-limited. We attempt a best-effort fetch and
        return an empty list on failure to keep the platform running.
        """
        inst_type = "SWAP" if symbol.endswith("-SWAP") else "FUTURES"
        try:
            resp = await self._client.get(
                "/api/v5/public/liquidation-orders",
                params={"instType": inst_type, "instId": symbol, "state": "filled"},
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != "0":
                return []
            return body.get("data") or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("liquidation-orders unavailable: %s", exc)
            return []
