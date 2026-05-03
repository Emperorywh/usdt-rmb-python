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
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO funding_rates (exchange, symbol, ts, funding_rate, next_funding_ts)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (exchange, symbol, ts) DO NOTHING
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
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO open_interest (exchange, symbol, ts, oi, oi_ccy)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (exchange, symbol, ts) DO NOTHING
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
        reasoning_content: Optional[str] = None,
    ) -> int:
        """
        写入一条信号记录
        ---------------------------------------------------------------
        参数：
            symbol             ：合约代码
            bias               ：方向偏置（long / short / neutral）
            confidence         ：置信度，[0, 1] 区间
            reason             ：判断依据（中文文本）
            risk               ：失效条件（中文文本）
            suggestion         ：操作建议（中文文本）
            factors            ：因子快照 + 规则引擎打分细节，落到 JSONB
            source             ：来源标识（rules / rules+llm）
            ts                 ：信号时间，缺省取当前 UTC 时间
            reasoning_content  ：DeepSeek 思考模式下的思维链原文，仅用于
                                 事后审计。未启用思考模式或纯规则引擎时为 None。
        返回：
            新插入行的自增 id
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals
                    (ts, symbol, bias, confidence, reason, risk, suggestion,
                     factors, source, reasoning_content)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
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
                reasoning_content,
            )
        return int(row["id"])

    async def fetch_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        读取指定 symbol 最近一条信号
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            最新一条 signals 行的字典；不存在则返回 None。
            返回字段中包含 reasoning_content（思维链审计原文，可为 None）。
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, ts, symbol, bias, confidence, reason, risk, suggestion,
                       factors, source, reasoning_content
                FROM signals
                WHERE symbol = $1
                ORDER BY ts DESC LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # 数据保留 / 清理
    # ------------------------------------------------------------------
    # 设计说明：
    # - 高频写入的 trades / orderbook_snapshots 不做清理会让磁盘线性增长，
    #   实测 ETH-USDT-SWAP 单 symbol 一年可吃掉 60+ GB。
    # - 但信号引擎只读最近 factor_window_seconds 内的数据，再老的纯属占地，
    #   因此用按时间窗口的 DELETE 把表稳定在一个有限规模。
    # - 这里只负责"按 ts 删除"，不负责 VACUUM —— 依赖 PG 的 autovacuum
    #   渐进回收死元组，避免引入显式锁表的风险。
    # - 所有删除方法返回被删除的行数，方便上层做日志 / 监控。
    async def delete_trades_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 trades 行
        --------------------------------------------------------------
        参数：
            cutoff: 截止时间（含时区的 UTC datetime），早于该时间的成交全部删除
        返回：
            被删除的行数
        """
        async with self.db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM trades WHERE ts < $1",
                cutoff,
            )
        return _parse_delete_count(result)

    async def delete_orderbook_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 orderbook_snapshots 行
        --------------------------------------------------------------
        参数：
            cutoff: 截止时间（含时区的 UTC datetime）
        返回：
            被删除的行数
        """
        async with self.db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM orderbook_snapshots WHERE ts < $1",
                cutoff,
            )
        return _parse_delete_count(result)

    async def delete_signals_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 signals 行
        --------------------------------------------------------------
        参数：
            cutoff: 截止时间（含时区的 UTC datetime）
        返回：
            被删除的行数
        说明：
            signals 体积小但条数多（30 秒一条），默认保留 30 天。
            如果需要长期审计可在配置里把保留期调大或设为 0（永不清理）。
        """
        async with self.db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM signals WHERE ts < $1",
                cutoff,
            )
        return _parse_delete_count(result)


def _parse_delete_count(status: str) -> int:
    """
    从 asyncpg ``execute`` 返回的命令状态字符串中解析受影响行数
    --------------------------------------------------------------
    参数：
        status: 形如 ``"DELETE 123"`` 的命令完成状态串
    返回：
        删除的行数；解析失败时返回 0（仅用于日志，不影响业务正确性）
    说明：
        asyncpg 的 ``Connection.execute`` 不像 ``fetch`` 那样直接给出
        rowcount，需要从 PG 协议返回的 CommandComplete 字符串里抠数字。
    """
    if not status:
        return 0
    parts = status.strip().split()
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0
