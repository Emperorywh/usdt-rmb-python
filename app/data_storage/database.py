"""asyncpg connection pool wrapper."""
from __future__ import annotations

import json
from typing import Optional

import asyncpg

from app.logging_config import get_logger

logger = get_logger(__name__)


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register JSON codecs so ``dict``/``list`` are auto-encoded for JSONB."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


class Database:
    """Thin wrapper around an asyncpg pool with connection-level JSON codecs."""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10):
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        logger.info("Connecting to PostgreSQL pool=%d-%d", self._min_size, self._max_size)
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=_init_connection,
            command_timeout=30,
        )
        logger.info("PostgreSQL pool ready")

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialised. Call connect() first.")
        return self._pool

    def acquire(self):
        return self.pool.acquire()
