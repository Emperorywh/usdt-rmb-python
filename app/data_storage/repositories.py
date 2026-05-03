"""Repository functions: write/read helpers for every table.

Kept intentionally lean (no ORM) for high-frequency writes. All public
methods are coroutines and accept plain Python dicts/sequences.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from app.data_storage.database import Database
from app.logging_config import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_dec(v: Any) -> Optional[Decimal]:
    """Coerce numerics into ``Decimal`` for asyncpg NUMERIC columns."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))


class Repositories:
    """Aggregates all table-level repository methods on one object."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # trades
    # ------------------------------------------------------------------
    async def insert_trades(self, rows: Sequence[Dict[str, Any]]) -> int:
        """Bulk insert trades. Duplicate (exchange, symbol, trade_id) are skipped."""
        if not rows:
            return 0
        records = [
            (
                r["exchange"],
                r["symbol"],
                r["ts"],
                _to_dec(r["price"]),
                _to_dec(r["size"]),
                r["side"],
                r.get("trade_id"),
            )
            for r in rows
        ]
        async with self.db.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO trades (exchange, symbol, ts, price, size, side, trade_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (exchange, symbol, trade_id) DO NOTHING
                """,
                records,
            )
        return len(records)

    async def fetch_recent_trades(
        self, symbol: str, since: datetime, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT exchange, symbol, ts, price, size, side, trade_id
                FROM trades
                WHERE symbol = $1 AND ts >= $2
                ORDER BY ts ASC
                LIMIT $3
                """,
                symbol,
                since,
                limit,
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # orderbook
    # ------------------------------------------------------------------
    async def insert_orderbook(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        bids: List[List[float]],
        asks: List[List[float]],
    ) -> None:
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orderbook_snapshots (exchange, symbol, ts, bids, asks)
                VALUES ($1, $2, $3, $4, $5)
                """,
                exchange,
                symbol,
                ts,
                bids,
                asks,
            )

    async def fetch_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT exchange, symbol, ts, bids, asks
                FROM orderbook_snapshots
                WHERE symbol = $1
                ORDER BY ts DESC
                LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # funding rate
    # ------------------------------------------------------------------
    async def insert_funding_rate(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        funding_rate: float,
        next_funding_ts: Optional[datetime] = None,
    ) -> None:
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO funding_rates (exchange, symbol, ts, funding_rate, next_funding_ts)
                VALUES ($1, $2, $3, $4, $5)
                """,
                exchange,
                symbol,
                ts,
                _to_dec(funding_rate),
                next_funding_ts,
            )

    async def fetch_latest_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT exchange, symbol, ts, funding_rate, next_funding_ts
                FROM funding_rates
                WHERE symbol = $1
                ORDER BY ts DESC LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # open interest
    # ------------------------------------------------------------------
    async def insert_open_interest(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        oi: float,
        oi_ccy: Optional[float] = None,
    ) -> None:
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO open_interest (exchange, symbol, ts, oi, oi_ccy)
                VALUES ($1, $2, $3, $4, $5)
                """,
                exchange,
                symbol,
                ts,
                _to_dec(oi),
                _to_dec(oi_ccy),
            )

    async def fetch_recent_oi(
        self, symbol: str, since: datetime, limit: int = 500
    ) -> List[Dict[str, Any]]:
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, oi, oi_ccy
                FROM open_interest
                WHERE symbol = $1 AND ts >= $2
                ORDER BY ts ASC LIMIT $3
                """,
                symbol,
                since,
                limit,
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # onchain
    # ------------------------------------------------------------------
    async def insert_onchain(self, row: Dict[str, Any]) -> None:
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO onchain_metrics
                    (ts, exchange_inflow, exchange_outflow, whale_tx_count, gas_fee_gwei, burn_rate)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                row.get("ts", _utcnow()),
                _to_dec(row.get("exchange_inflow")),
                _to_dec(row.get("exchange_outflow")),
                row.get("whale_tx_count"),
                _to_dec(row.get("gas_fee_gwei")),
                _to_dec(row.get("burn_rate")),
            )

    async def fetch_latest_onchain(self) -> Optional[Dict[str, Any]]:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT ts, exchange_inflow, exchange_outflow, whale_tx_count,
                       gas_fee_gwei, burn_rate
                FROM onchain_metrics
                ORDER BY ts DESC LIMIT 1
                """
            )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    async def insert_signal(
        self,
        symbol: str,
        bias: str,
        confidence: float,
        reason: str,
        risk: str,
        suggestion: str,
        factors: Dict[str, Any],
        source: str = "rules+llm",
        ts: Optional[datetime] = None,
    ) -> int:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals
                    (ts, symbol, bias, confidence, reason, risk, suggestion, factors, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                RETURNING id
                """,
                ts or _utcnow(),
                symbol,
                bias,
                _to_dec(confidence),
                reason,
                risk,
                suggestion,
                factors,
                source,
            )
        return int(row["id"])

    async def fetch_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, ts, symbol, bias, confidence, reason, risk, suggestion,
                       factors, source
                FROM signals
                WHERE symbol = $1
                ORDER BY ts DESC LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None
