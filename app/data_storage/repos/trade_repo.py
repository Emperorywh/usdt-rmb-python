"""Trade table repository: insert / query / cleanup for trades."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from app.logging_config import get_logger

from .base import BaseRepo
from .helpers import to_dec

logger = get_logger(__name__)


class TradeRepo(BaseRepo):
    """trades 表仓储：批量写入、查询、按时间清理。"""

    async def insert_trades(self, rows: Sequence[Dict[str, Any]]) -> int:
        """
        批量写入成交记录
        ---------------------------------------------------------------
        说明：
            - (exchange, symbol, trade_id) 唯一约束 + ON CONFLICT DO NOTHING
              保证幂等：连接被静默断开时整批数据可放心重试，不会产生重复。
            - 走 db.run_with_retry，瞬时连接错误（WinError 121 等）会自动
              退避重试，避免一批 50 条成交因一次"僵尸连接"全部丢失。
        """
        if not rows:
            return 0
        records = [
            (
                r["exchange"],
                r["symbol"],
                r["ts"],
                to_dec(r["price"]),
                to_dec(r["size"]),
                r["side"],
                r.get("trade_id"),
            )
            for r in rows
        ]

        async def _do() -> None:
            async with self._db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO trades (exchange, symbol, ts, price, size, side, trade_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (exchange, symbol, trade_id) DO NOTHING
                    """,
                    records,
                )

        await self._db.run_with_retry(_do, op_name="insert_trades")
        return len(records)

    async def fetch_recent_trades(
        self, symbol: str, since: datetime, limit: int = 5000
    ) -> List[Dict[str, Any]]:
        async with self._db.acquire() as conn:
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

    async def aggregate_trades_in_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        在数据库端把 [start, end) 范围内的成交聚合成一行 OHLC + 量能
        --------------------------------------------------------------
        参数：
            symbol: 合约代码
            start:  起始 UTC 时间（含）
            end:    截止 UTC 时间（不含）
        返回：
            None：窗口内无任何成交；
            否则返回 dict，键固定为：
                trade_count : int   - 成交笔数
                open / high / low / close : Decimal - OHLC（NUMERIC 精度）
                volume      : Decimal - 总量
                buy_volume  : Decimal - 主动买量
                sell_volume : Decimal - 主动卖量
                cvd_delta   : Decimal - 本窗口 CVD 增量（buy - sell）
        设计：
            - 与原先"把成交全部拉回 Python 再做循环累加"等价，但只产生
              一次网络往返且只传 1 行，不再随窗口长度线性放大。
            - high/low/sum/count 走单次 ``idx_trades_symbol_ts`` 区间扫描；
              open/close 用两个 LIMIT 1 子查询，单 row 索引点查，O(log N)。
            - 窗口为空时聚合函数返回 NULL，count = 0；调用方按"无成交"
              处理。
        """
        sql = """
            SELECT
                COUNT(*)::INTEGER                                              AS trade_count,
                MAX(price)                                                     AS high,
                MIN(price)                                                     AS low,
                SUM(size)                                                      AS volume,
                SUM(CASE WHEN side = 'buy'  THEN size ELSE 0    END)           AS buy_volume,
                SUM(CASE WHEN side = 'sell' THEN size ELSE 0    END)           AS sell_volume,
                SUM(CASE WHEN side = 'buy'  THEN size ELSE -size END)          AS cvd_delta,
                (
                    SELECT price FROM trades
                    WHERE symbol = $1 AND ts >= $2 AND ts < $3
                    ORDER BY ts ASC
                    LIMIT 1
                )                                                              AS open,
                (
                    SELECT price FROM trades
                    WHERE symbol = $1 AND ts >= $2 AND ts < $3
                    ORDER BY ts DESC
                    LIMIT 1
                )                                                              AS close
            FROM trades
            WHERE symbol = $1 AND ts >= $2 AND ts < $3
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(sql, symbol, start, end)
        if row is None:
            return None
        trade_count = int(row["trade_count"] or 0)
        if trade_count == 0:
            return None
        return dict(row)

    async def delete_trades_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 trades 行
        --------------------------------------------------------------
        参数：
            cutoff: 截止时间（含时区的 UTC datetime），早于该时间的成交全部删除
        返回：
            被删除的行数
        """
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM trades WHERE ts < $1",
                cutoff,
            )
        from .helpers import parse_delete_count

        return parse_delete_count(result)
