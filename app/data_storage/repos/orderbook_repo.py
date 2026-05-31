"""Orderbook repository: snapshots + metrics tables."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

from .base import BaseRepo
from .helpers import parse_delete_count, to_dec

logger = get_logger(__name__)


class OrderbookRepo(BaseRepo):
    """orderbook_snapshots + orderbook_metrics 两张表的仓储。"""

    # ------------------------------------------------------------------
    # orderbook_snapshots
    # ------------------------------------------------------------------
    async def insert_orderbook(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        bids: List[List[float]],
        asks: List[List[float]],
    ) -> None:
        """
        写入一条订单簿快照
        ---------------------------------------------------------------
        说明：
            - 信号引擎只读最新一条快照；快照表自身没有唯一约束，
              即便瞬时错误重试导致"重复一行"也只会让最近一秒的快照
              多出一份，对策略无影响。
            - 走 db.run_with_retry：避免连接被断开时一条快照写失败
              拖累整个 _dispatch 协程。
        """

        async def _do() -> None:
            async with self._db.acquire() as conn:
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

        await self._db.run_with_retry(_do, op_name="insert_orderbook")

    async def fetch_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        async with self._db.acquire() as conn:
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

    async def delete_orderbook_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 orderbook_snapshots 行
        --------------------------------------------------------------
        参数：
            cutoff: 截止时间（含时区的 UTC datetime）
        返回：
            被删除的行数
        """
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM orderbook_snapshots WHERE ts < $1",
                cutoff,
            )
        return parse_delete_count(result)

    # ------------------------------------------------------------------
    # orderbook_metrics
    # ------------------------------------------------------------------
    async def insert_orderbook_metric(
        self,
        exchange: str,
        symbol: str,
        ts: datetime,
        metric: Dict[str, Any],
    ) -> None:
        """
        写入一条订单簿时序指标
        --------------------------------------------------------------
        参数：
            exchange : 交易所标识，如 'okx'
            symbol   : 合约代码
            ts       : 指标时间戳（来自 WS 推送）
            metric   : 已经聚合好的指标 dict，键见 SQL VALUES 列表
        说明：
            - UNIQUE(exchange, symbol, ts) + ON CONFLICT DO NOTHING：
              10s 节流粒度下 ts 天然去重；偶发重复写入直接吞掉。
            - 走 db.run_with_retry，瞬时连接错误自动指数退避。
            - 单行 < 100 字节，比写 orderbook_snapshots（~2KB JSONB）轻得多。
        """

        async def _do() -> None:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO orderbook_metrics
                        (exchange, symbol, ts, imbalance,
                         bid_qty, ask_qty,
                         top5_bid_notional, top5_ask_notional,
                         bid_wall_count, ask_wall_count,
                         spread_bp, mid_price)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (exchange, symbol, ts) DO NOTHING
                    """,
                    exchange,
                    symbol,
                    ts,
                    to_dec(metric.get("imbalance")),
                    to_dec(metric.get("bid_qty")),
                    to_dec(metric.get("ask_qty")),
                    to_dec(metric.get("top5_bid_notional")),
                    to_dec(metric.get("top5_ask_notional")),
                    int(metric.get("bid_wall_count") or 0),
                    int(metric.get("ask_wall_count") or 0),
                    to_dec(metric.get("spread_bp")),
                    to_dec(metric.get("mid_price")),
                )

        await self._db.run_with_retry(_do, op_name="insert_orderbook_metric")

    async def fetch_orderbook_metrics_since(
        self,
        symbol: str,
        since: datetime,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        读取 [since, now] 内的订单簿时序指标（按 ts 升序返回）
        --------------------------------------------------------------
        参数：
            symbol : 合约代码
            since  : 起始 UTC 时间（含）
            limit  : 防御性上限，避免长时间历史误传爆查询
        返回：
            按 ts 升序的 dict 列表；字段与 insert_orderbook_metric 对齐。
        说明：
            - 走 idx_orderbook_metrics_symbol_ts (symbol, ts DESC) 索引；
              15 分钟窗口下 ~90 行，1 小时基线 ~360 行，单次扫描 < 5ms。
        """
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, imbalance, bid_qty, ask_qty,
                       top5_bid_notional, top5_ask_notional,
                       bid_wall_count, ask_wall_count,
                       spread_bp, mid_price
                FROM orderbook_metrics
                WHERE symbol = $1 AND ts >= $2
                ORDER BY ts ASC
                LIMIT $3
                """,
                symbol,
                since,
                limit,
            )
        return [dict(r) for r in rows]

    async def delete_orderbook_metrics_older_than(self, cutoff: datetime) -> int:
        """
        删除 ts < cutoff 的 orderbook_metrics 行
        --------------------------------------------------------------
        参数：
            cutoff: 截止时间（UTC datetime）
        返回：
            被删除的行数
        说明：
            与 orderbook_snapshots 的清理逻辑一致；P1 升级在
            retention 任务里追加该表，沿用 orderbook 保留时长。
        """
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM orderbook_metrics WHERE ts < $1",
                cutoff,
            )
        return parse_delete_count(result)
