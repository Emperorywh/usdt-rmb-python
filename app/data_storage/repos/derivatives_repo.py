"""Derivatives repository: funding_rates, open_interest, liquidations."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from app.logging_config import get_logger

from .base import BaseRepo
from .helpers import parse_delete_count, to_dec

logger = get_logger(__name__)


class DerivativesRepo(BaseRepo):
    """funding_rates / open_interest / liquidations 三张表的仓储。"""

    # ------------------------------------------------------------------
    # funding_rates
    # ------------------------------------------------------------------
    async def insert_funding_rate(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        funding_rate: float,
        next_funding_ts: Optional[datetime] = None,
    ) -> None:
        """
        写入一条资金费率记录
        ---------------------------------------------------------------
        说明：
            - 与 (exchange, symbol, ts) 完全相同的记录视为重复，
              依赖唯一约束 + ON CONFLICT DO NOTHING 抑制重复入库。
            - 这样 WS 与 REST 两路即便并发写入也不会产生脏数据。
        参数：
            exchange:        交易所标识，如 'okx'
            symbol:          合约代码，如 'ETH-USDT-SWAP'
            ts:              资金费率对应的时间戳（来自交易所）
            funding_rate:    资金费率（小数表示，例如 0.0001 表示 0.01%）
            next_funding_ts: 下一次资金费率结算时间，可空
        """

        async def _do() -> None:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO funding_rates (exchange, symbol, ts, funding_rate, next_funding_ts)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (exchange, symbol, ts) DO NOTHING
                    """,
                    exchange,
                    symbol,
                    ts,
                    to_dec(funding_rate),
                    next_funding_ts,
                )

        await self._db.run_with_retry(_do, op_name="insert_funding_rate")

    async def fetch_latest_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self._db.acquire() as conn:
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

    async def fetch_funding_rates_since(
        self, symbol: str, since: datetime, limit: int = 20000
    ) -> List[Dict[str, Any]]:
        """
        读取 [since, now] 内的资金费率历史（升序）
        --------------------------------------------------------------
        参数：
            symbol: 合约代码
            since:  起始 UTC 时间（含）
            limit:  防御性上限；7 天 × WS+REST ≈ 几百到几千行
        返回：
            按 ts 升序的 dict 列表，键含 ts / funding_rate
        说明：
            - 用于 derivatives 因子的 funding_rate_pct_rank_7d 分位数计算。
            - 走 idx_funding_symbol_ts，单次扫描 < 30ms。
        """
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, funding_rate
                FROM funding_rates
                WHERE symbol = $1 AND ts >= $2
                ORDER BY ts ASC
                LIMIT $3
                """,
                symbol,
                since,
                limit,
            )
        return [dict(r) for r in rows]

    async def delete_funding_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 funding_rates 行
        """
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM funding_rates WHERE ts < $1",
                cutoff,
            )
        return parse_delete_count(result)

    # ------------------------------------------------------------------
    # open_interest
    # ------------------------------------------------------------------
    async def insert_open_interest(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        oi: float,
        oi_ccy: Optional[float] = None,
    ) -> None:
        """
        写入一条持仓量（Open Interest）记录
        ---------------------------------------------------------------
        说明：
            - 与 (exchange, symbol, ts) 完全相同的记录视为重复，
              依赖唯一约束 + ON CONFLICT DO NOTHING 抑制重复入库。
            - OI 不会每秒变化，REST 60 秒拉一次完全够用。
        参数：
            exchange: 交易所标识，如 'okx'
            symbol:   合约代码，如 'ETH-USDT-SWAP'
            ts:       OI 数据的时间戳
            oi:       持仓量（合约张数）
            oi_ccy:   持仓量按基础币种折算（ETH），可空
        """

        async def _do() -> None:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO open_interest (exchange, symbol, ts, oi, oi_ccy)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (exchange, symbol, ts) DO NOTHING
                    """,
                    exchange,
                    symbol,
                    ts,
                    to_dec(oi),
                    to_dec(oi_ccy),
                )

        await self._db.run_with_retry(_do, op_name="insert_open_interest")

    async def fetch_recent_oi(
        self, symbol: str, since: datetime, limit: int = 500
    ) -> List[Dict[str, Any]]:
        async with self._db.acquire() as conn:
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

    async def delete_oi_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 open_interest 行
        """
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM open_interest WHERE ts < $1",
                cutoff,
            )
        return parse_delete_count(result)

    # ------------------------------------------------------------------
    # liquidations
    # ------------------------------------------------------------------
    async def insert_liquidations(self, rows: Sequence[Dict[str, Any]]) -> int:
        """
        批量写入爆仓记录
        --------------------------------------------------------------
        参数：
            rows: 每条字段需含
                exchange / symbol / ts / side(long|short) / price / size / notional
        说明：
            - 使用 (exchange, symbol, ts, side, price, size) 复合唯一键，
              ON CONFLICT DO NOTHING 抑制 WS 重连时的重复推送。
            - 高频写入优先保证幂等，单笔不入库不重要，永远不能写脏。
        返回：
            尝试入库的行数（不代表实际新增数）。
        """
        if not rows:
            return 0
        records = [
            (
                r["exchange"],
                r["symbol"],
                r["ts"],
                r["side"],
                to_dec(r["price"]),
                to_dec(r["size"]),
                to_dec(r.get("notional")),
            )
            for r in rows
        ]

        async def _do() -> None:
            async with self._db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO liquidations
                        (exchange, symbol, ts, side, price, size, notional)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (exchange, symbol, ts, side, price, size) DO NOTHING
                    """,
                    records,
                )

        await self._db.run_with_retry(_do, op_name="insert_liquidations")
        return len(records)

    async def fetch_liquidations_since(
        self, symbol: str, since: datetime
    ) -> List[Dict[str, Any]]:
        """
        读取最近一段时间内的爆仓事件（用于因子层滚动窗口聚合）
        --------------------------------------------------------------
        参数：
            symbol: 合约代码
            since:  起始 UTC 时间（含）
        返回：
            按 ts 升序排列的爆仓事件 dict 列表。
        """
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, side, price, size, notional
                FROM liquidations
                WHERE symbol = $1 AND ts >= $2
                ORDER BY ts ASC
                """,
                symbol,
                since,
            )
        return [dict(r) for r in rows]
