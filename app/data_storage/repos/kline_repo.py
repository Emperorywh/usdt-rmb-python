"""Kline repository: 6 张同构 K 线表的 upsert / query."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

from .base import BaseRepo
from .helpers import to_dec

logger = get_logger(__name__)

# 安全白名单：所有 SQL 拼表名前都必须先在这里校验，避免 SQL 注入。
_KLINE_TABLES = {
    "1m": "klines_1m",
    "5m": "klines_5m",
    "15m": "klines_15m",
    "1h": "klines_1h",
    "4h": "klines_4h",
    "1d": "klines_1d",
}


def _kline_table(timeframe: str) -> str:
    """
    把周期标签解析成对应的表名
    --------------------------------------------------------------
    参数：
        timeframe: '1m' / '5m' / '15m' / '1h' / '4h' / '1d'
    返回：
        真实表名字符串。
    异常：
        ValueError - 周期不在白名单时抛出，防止注入。
    """
    table = _KLINE_TABLES.get(timeframe)
    if table is None:
        raise ValueError(f"未知 K 线周期: {timeframe}")
    return table


class KlineRepo(BaseRepo):
    """klines_{1m,5m,15m,1h,4h,1d} 六张同构表的仓储。"""

    async def upsert_kline(
        self,
        timeframe: str,
        exchange: str,
        symbol: str,
        ts: datetime,
        ohlc: Dict[str, Any],
        closed: bool,
    ) -> None:
        """
        增量写入或更新一根 K 线
        --------------------------------------------------------------
        参数：
            timeframe: 周期标签（同 _KLINE_TABLES）
            exchange:  交易所标识，如 'okx'
            symbol:    合约代码
            ts:        周期开始时间（已对齐到周期边界，UTC）
            ohlc:      包含 open/high/low/close/volume/buy_volume/
                       sell_volume/cvd_close/trade_count 的 dict
            closed:    True - 该 bar 已封盘；False - 当前活跃 bar，可继续滚动
        说明：
            ON CONFLICT DO UPDATE 让"未封盘 bar"持续被新成交滚动覆盖；
            一旦置 closed=TRUE，后续不应再写同一根（aggregator 自身约束）。
            性能：一次 UPSERT 走 (symbol, ts) 唯一索引，单次 < 10ms。
        """
        table = _kline_table(timeframe)

        async def _do() -> None:
            async with self._db.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {table}
                        (exchange, symbol, ts, open, high, low, close,
                         volume, buy_volume, sell_volume, cvd_close,
                         trade_count, closed)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (exchange, symbol, ts) DO UPDATE SET
                        high        = GREATEST({table}.high, EXCLUDED.high),
                        low         = LEAST({table}.low, EXCLUDED.low),
                        close       = EXCLUDED.close,
                        volume      = EXCLUDED.volume,
                        buy_volume  = EXCLUDED.buy_volume,
                        sell_volume = EXCLUDED.sell_volume,
                        cvd_close   = EXCLUDED.cvd_close,
                        trade_count = EXCLUDED.trade_count,
                        closed      = EXCLUDED.closed
                    """,
                    exchange,
                    symbol,
                    ts,
                    to_dec(ohlc.get("open")),
                    to_dec(ohlc.get("high")),
                    to_dec(ohlc.get("low")),
                    to_dec(ohlc.get("close")),
                    to_dec(ohlc.get("volume")),
                    to_dec(ohlc.get("buy_volume")),
                    to_dec(ohlc.get("sell_volume")),
                    to_dec(ohlc.get("cvd_close")),
                    int(ohlc.get("trade_count") or 0),
                    bool(closed),
                )

        await self._db.run_with_retry(_do, op_name=f"upsert_kline[{timeframe}]")

    async def fetch_recent_klines(
        self,
        timeframe: str,
        symbol: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        读取指定周期最近 N 根 K 线（升序返回）
        --------------------------------------------------------------
        参数：
            timeframe: 周期标签
            symbol:    合约代码
            limit:     最多返回的 bar 数量（含未封盘 bar）
        返回：
            按 ts 升序的 dict 列表，键包含 ts/open/high/low/close/
            volume/buy_volume/sell_volume/cvd_close/trade_count/closed。
        """
        table = _kline_table(timeframe)
        async with self._db.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT ts, open, high, low, close, volume,
                       buy_volume, sell_volume, cvd_close,
                       trade_count, closed
                FROM {table}
                WHERE symbol = $1
                ORDER BY ts DESC
                LIMIT $2
                """,
                symbol,
                limit,
            )
        return list(reversed([dict(r) for r in rows]))

    async def fetch_latest_kline(
        self, timeframe: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        读取指定周期的最新一根 K 线（封盘或未封盘均算）
        --------------------------------------------------------------
        参数：
            timeframe: 周期标签
            symbol:    合约代码
        返回：
            最新一根 K 线的 dict，找不到则返回 None。
        """
        table = _kline_table(timeframe)
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT ts, open, high, low, close, volume,
                       buy_volume, sell_volume, cvd_close,
                       trade_count, closed
                FROM {table}
                WHERE symbol = $1
                ORDER BY ts DESC LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None
