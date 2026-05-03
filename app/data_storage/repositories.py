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
                _to_dec(r["price"]),
                _to_dec(r["size"]),
                r["side"],
                r.get("trade_id"),
            )
            for r in rows
        ]

        async def _do() -> None:
            async with self.db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO trades (exchange, symbol, ts, price, size, side, trade_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (exchange, symbol, trade_id) DO NOTHING
                    """,
                    records,
                )

        await self.db.run_with_retry(_do, op_name="insert_trades")
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

        await self.db.run_with_retry(_do, op_name="insert_orderbook")

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

        async def _do() -> None:
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

        await self.db.run_with_retry(_do, op_name="insert_funding_rate")

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

        async def _do() -> None:
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

        await self.db.run_with_retry(_do, op_name="insert_open_interest")

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
    # liquidations（P0 新增）
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
                _to_dec(r["price"]),
                _to_dec(r["size"]),
                _to_dec(r.get("notional")),
            )
            for r in rows
        ]

        async def _do() -> None:
            async with self.db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO liquidations
                        (exchange, symbol, ts, side, price, size, notional)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (exchange, symbol, ts, side, price, size) DO NOTHING
                    """,
                    records,
                )

        await self.db.run_with_retry(_do, op_name="insert_liquidations")
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
        async with self.db.acquire() as conn:
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

    # ------------------------------------------------------------------
    # K 线（P0 新增 - 6 张同构表）
    # ------------------------------------------------------------------
    # 安全白名单：所有 SQL 拼表名前都必须先在这里校验，避免 SQL 注入。
    _KLINE_TABLES = {
        "1m": "klines_1m",
        "5m": "klines_5m",
        "15m": "klines_15m",
        "1h": "klines_1h",
        "4h": "klines_4h",
        "1d": "klines_1d",
    }

    @classmethod
    def _kline_table(cls, timeframe: str) -> str:
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
        table = cls._KLINE_TABLES.get(timeframe)
        if table is None:
            raise ValueError(f"未知 K 线周期: {timeframe}")
        return table

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
        table = self._kline_table(timeframe)

        async def _do() -> None:
            async with self.db.acquire() as conn:
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
                    _to_dec(ohlc.get("open")),
                    _to_dec(ohlc.get("high")),
                    _to_dec(ohlc.get("low")),
                    _to_dec(ohlc.get("close")),
                    _to_dec(ohlc.get("volume")),
                    _to_dec(ohlc.get("buy_volume")),
                    _to_dec(ohlc.get("sell_volume")),
                    _to_dec(ohlc.get("cvd_close")),
                    int(ohlc.get("trade_count") or 0),
                    bool(closed),
                )

        await self.db.run_with_retry(_do, op_name=f"upsert_kline[{timeframe}]")

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
        table = self._kline_table(timeframe)
        async with self.db.acquire() as conn:
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
        table = self._kline_table(timeframe)
        async with self.db.acquire() as conn:
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

    # ------------------------------------------------------------------
    # orderbook_metrics（P1 新增 - 订单簿时序）
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
            async with self.db.acquire() as conn:
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
                    _to_dec(metric.get("imbalance")),
                    _to_dec(metric.get("bid_qty")),
                    _to_dec(metric.get("ask_qty")),
                    _to_dec(metric.get("top5_bid_notional")),
                    _to_dec(metric.get("top5_ask_notional")),
                    int(metric.get("bid_wall_count") or 0),
                    int(metric.get("ask_wall_count") or 0),
                    _to_dec(metric.get("spread_bp")),
                    _to_dec(metric.get("mid_price")),
                )

        await self.db.run_with_retry(_do, op_name="insert_orderbook_metric")

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
        async with self.db.acquire() as conn:
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
        async with self.db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM orderbook_metrics WHERE ts < $1",
                cutoff,
            )
        return _parse_delete_count(result)

    # ------------------------------------------------------------------
    # position_ratios（P1 新增 - 散户/精英多空比）
    # ------------------------------------------------------------------
    async def insert_position_ratios(self, rows: Sequence[Dict[str, Any]]) -> int:
        """
        批量写入持仓比（散户多空 / 精英持仓比）
        --------------------------------------------------------------
        参数：
            rows: 每条字段需包含
                exchange / symbol / ts / ratio_type /
                long_ratio / short_ratio / ratio
        返回：
            尝试入库的行数
        说明：
            - 复合唯一键 (exchange, symbol, ts, ratio_type) +
              ON CONFLICT DO NOTHING；同一周期 + 同一维度只保留一条。
            - 拉取失败 → 返回 0 行；本方法对空 rows 也安全（早返）。
        """
        if not rows:
            return 0
        records = [
            (
                r["exchange"],
                r["symbol"],
                r["ts"],
                r["ratio_type"],
                _to_dec(r.get("long_ratio")),
                _to_dec(r.get("short_ratio")),
                _to_dec(r.get("ratio")),
            )
            for r in rows
        ]

        async def _do() -> None:
            async with self.db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO position_ratios
                        (exchange, symbol, ts, ratio_type,
                         long_ratio, short_ratio, ratio)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (exchange, symbol, ts, ratio_type) DO NOTHING
                    """,
                    records,
                )

        await self.db.run_with_retry(_do, op_name="insert_position_ratios")
        return len(records)

    async def fetch_latest_position_ratios(
        self, symbol: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        读取每个 ratio_type 维度下"最新一条"持仓比
        --------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            {ratio_type: {ts, long_ratio, short_ratio, ratio}}
            找不到时该 ratio_type 不会出现在返回字典里。
        说明：
            - 用 DISTINCT ON 让 PG 在索引上"每组取首条"，单次 < 5ms。
            - 因子层依赖最新值即可，不再读历史序列。
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (ratio_type)
                       ratio_type, ts, long_ratio, short_ratio, ratio
                FROM position_ratios
                WHERE symbol = $1
                ORDER BY ratio_type, ts DESC
                """,
                symbol,
            )
        return {r["ratio_type"]: dict(r) for r in rows}

    # ------------------------------------------------------------------
    # funding_rates 历史读取（P1 新增 - 7 天分位数）
    # ------------------------------------------------------------------
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
        async with self.db.acquire() as conn:
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
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(sql, symbol, start, end)
        if row is None:
            return None
        trade_count = int(row["trade_count"] or 0)
        if trade_count == 0:
            return None
        return dict(row)

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
        entry_zone: Optional[List[float]] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[List[float]] = None,
        risk_reward_ratio: Optional[float] = None,
        position_size_pct: Optional[float] = None,
        timeframe_alignment: Optional[Dict[str, Any]] = None,
        invalidation_conditions: Optional[List[str]] = None,
    ) -> int:
        """
        写入一条信号记录（含 P0 结构化交易计划字段）
        ---------------------------------------------------------------
        参数：
            symbol                  ：合约代码
            bias                    ：方向偏置（long / short / neutral）
            confidence              ：置信度，[0, 1] 区间
            reason / risk           ：判断依据 / 失效条件（中文文本）
            suggestion              ：操作建议（中文文本，保留兼容）
            factors                 ：因子快照 + 规则引擎打分细节
            source                  ：来源标识（rules / rules+llm）
            ts                      ：信号时间，缺省取当前 UTC
            reasoning_content       ：DeepSeek 思维链审计原文
            entry_zone              ：可执行入场区间 [low, high]
            stop_loss               ：止损价
            take_profit             ：止盈位列表（≥2 档）
            risk_reward_ratio       ：tp1 vs sl 的盈亏比
            position_size_pct       ：建议仓位占比 [0, 0.25]
            timeframe_alignment     ：5 个周期方向投票 {"5m":"long",...}
            invalidation_conditions ：量化失效条件列表
        返回：
            新插入行的自增 id
        说明：
            P0 升级：新增 7 个结构化字段使用 JSONB / NUMERIC 列。
            空值（neutral 时）保持 NULL，避免污染下游聚合统计。
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals
                    (ts, symbol, bias, confidence, reason, risk, suggestion,
                     factors, source, reasoning_content,
                     entry_zone, stop_loss, take_profit, risk_reward_ratio,
                     position_size_pct, timeframe_alignment,
                     invalidation_conditions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10,
                        $11::jsonb, $12, $13::jsonb, $14, $15, $16::jsonb, $17::jsonb)
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
                entry_zone,
                _to_dec(stop_loss),
                take_profit,
                _to_dec(risk_reward_ratio),
                _to_dec(position_size_pct),
                timeframe_alignment,
                invalidation_conditions,
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

    async def fetch_latest_signal_full(
        self, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        读取指定 symbol 最近一条信号的"全字段视图"（含 P0 结构化交易计划 + lifecycle）
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            dict（不存在则返回 None），字段包含：
                - signals 表全部列（id / ts / bias / confidence / reason / risk /
                  suggestion / factors / source / reasoning_content /
                  entry_zone / stop_loss / take_profit / risk_reward_ratio /
                  position_size_pct / timeframe_alignment / invalidation_conditions）
                - 通过 LEFT JOIN signal_lifecycle 拼上的实战结果列：
                  lifecycle_status / triggered_at / triggered_price /
                  exit_at / exit_price / pnl_pct / max_favorable_pct /
                  max_adverse_pct / lifecycle_expires_at / lifecycle_updated_at
        说明：
            - 专供前端"分析卡片 / 详情页"使用，一次拿齐所有可视化字段，
              避免前端串行多次调用 /signal + /signals/{id}/attribution + lifecycle stats。
            - LEFT JOIN：未启用 lifecycle_tracking 或 lifecycle 行尚未生成时，
              JOIN 出来的列会全为 NULL，前端按 None 处理即可。
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    s.id, s.ts, s.symbol, s.bias, s.confidence,
                    s.reason, s.risk, s.suggestion,
                    s.factors, s.source, s.reasoning_content,
                    s.entry_zone, s.stop_loss, s.take_profit,
                    s.risk_reward_ratio, s.position_size_pct,
                    s.timeframe_alignment, s.invalidation_conditions,
                    sl.status        AS lifecycle_status,
                    sl.triggered_at  AS lifecycle_triggered_at,
                    sl.triggered_price,
                    sl.exit_at       AS lifecycle_exit_at,
                    sl.exit_price,
                    sl.pnl_pct,
                    sl.max_favorable_pct,
                    sl.max_adverse_pct,
                    sl.expires_at    AS lifecycle_expires_at,
                    sl.updated_at    AS lifecycle_updated_at
                FROM signals s
                LEFT JOIN signal_lifecycle sl ON sl.signal_id = s.id
                WHERE s.symbol = $1
                ORDER BY s.ts DESC
                LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None

    async def fetch_recent_signals_full(
        self,
        symbol: str,
        limit: int = 20,
        bias: Optional[str] = None,
        source_like: Optional[str] = None,
        only_persisted: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        读取指定 symbol 最近 N 条信号的"全字段视图"（含 lifecycle）
        ---------------------------------------------------------------
        参数：
            symbol         : 合约代码
            limit          : 返回条数上限（防御性 1~200）
            bias           : 可选过滤，'long' / 'short' / 'neutral'
            source_like    : 可选 ILIKE 模式（如 '%llm%' 只看 LLM 路径）
            only_persisted : 默认 True；仅返回真正落库的信号（即排除掉
                             根本没插入 signals 表的纯规则缓存路径——但因为
                             那种信号本来就没入库，这里 "True" 是 No-op，
                             保留参数语义便于将来扩展过滤策略）。
        返回：
            按 ts 倒序的 dict 列表，字段同 fetch_latest_signal_full。
        说明：
            - 专供前端"历史列表 / 时间线"使用；体积比 LIMIT N 个独立查询小得多。
            - factors 列虽是 JSONB，但每条体积可控（< 50KB）；建议 limit ≤ 50。
        """
        safe_limit = max(1, min(int(limit), 200))
        clauses = ["s.symbol = $1"]
        params: List[Any] = [symbol]
        if bias in ("long", "short", "neutral"):
            params.append(bias)
            clauses.append(f"s.bias = ${len(params)}")
        if source_like:
            params.append(source_like)
            clauses.append(f"s.source ILIKE ${len(params)}")
        params.append(safe_limit)
        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT
                s.id, s.ts, s.symbol, s.bias, s.confidence,
                s.reason, s.risk, s.suggestion,
                s.factors, s.source, s.reasoning_content,
                s.entry_zone, s.stop_loss, s.take_profit,
                s.risk_reward_ratio, s.position_size_pct,
                s.timeframe_alignment, s.invalidation_conditions,
                sl.status        AS lifecycle_status,
                sl.triggered_at  AS lifecycle_triggered_at,
                sl.triggered_price,
                sl.exit_at       AS lifecycle_exit_at,
                sl.exit_price,
                sl.pnl_pct,
                sl.max_favorable_pct,
                sl.max_adverse_pct,
                sl.expires_at    AS lifecycle_expires_at,
                sl.updated_at    AS lifecycle_updated_at
            FROM signals s
            LEFT JOIN signal_lifecycle sl ON sl.signal_id = s.id
            WHERE {where_sql}
            ORDER BY s.ts DESC
            LIMIT ${len(params)}
        """
        # only_persisted 当前为 No-op 占位，避免未使用变量告警
        _ = only_persisted
        async with self.db.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]

    async def fetch_signal_full_by_id(
        self, signal_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        按 id 读取单条信号的"全字段视图"（含 lifecycle）
        ---------------------------------------------------------------
        参数：
            signal_id: signals.id
        返回：
            字段同 fetch_latest_signal_full；找不到返回 None。
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    s.id, s.ts, s.symbol, s.bias, s.confidence,
                    s.reason, s.risk, s.suggestion,
                    s.factors, s.source, s.reasoning_content,
                    s.entry_zone, s.stop_loss, s.take_profit,
                    s.risk_reward_ratio, s.position_size_pct,
                    s.timeframe_alignment, s.invalidation_conditions,
                    sl.status        AS lifecycle_status,
                    sl.triggered_at  AS lifecycle_triggered_at,
                    sl.triggered_price,
                    sl.exit_at       AS lifecycle_exit_at,
                    sl.exit_price,
                    sl.pnl_pct,
                    sl.max_favorable_pct,
                    sl.max_adverse_pct,
                    sl.expires_at    AS lifecycle_expires_at,
                    sl.updated_at    AS lifecycle_updated_at
                FROM signals s
                LEFT JOIN signal_lifecycle sl ON sl.signal_id = s.id
                WHERE s.id = $1
                """,
                signal_id,
            )
        return dict(row) if row else None

    async def fetch_latest_signal_judgment(
        self, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        轻量读取指定 symbol 最近一条信号的"判断字段"
        ---------------------------------------------------------------
        说明：
            - 与 ``fetch_latest_signal`` 的区别：**故意不读取 factors 列**。
              factors 是 JSONB，单条几十 KB；LLM 节流判定每 30 秒会查一次，
              没必要把因子快照也拉过来反序列化，浪费带宽与 CPU。
            - 该方法专供 LLMAgent 的"DB 节流"使用：
                1) 拿 ts 判断距上次 LLM 分析是否 ≥ min_interval；
                2) 命中节流时用 bias / confidence / reason / risk / suggestion
                   重建 TradingSignal，连同 reasoning_content 一起返回给上层。
            - 走的是 idx_signals_symbol_ts (symbol, ts DESC) 索引，单点查询，
              成本远低于 LLM 调用，不会成为热路径瓶颈。
        参数：
            symbol: 合约代码
        返回：
            最新一条 signals 行的判断字段字典；不存在则返回 None。
            字段：ts / bias / confidence / reason / risk / suggestion /
                  reasoning_content。
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT ts, bias, confidence, reason, risk, suggestion,
                       reasoning_content
                FROM signals
                WHERE symbol = $1
                ORDER BY ts DESC
                LIMIT 1
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

    # ------------------------------------------------------------------
    # P2：因子权重表（factor_weights）
    # ------------------------------------------------------------------
    # 设计要点：
    #   - 同一 (regime, timeframe) 下所有 factor_name 的 weight 总和 = 1.0
    #     由 IC 校准任务在 UPSERT 之前归一好；这里只负责"原子地批量写入"，
    #     不在 SQL 层校验总和（PG 没有便宜的"组内求和约束"，引入触发器
    #     反而会拖慢热路径）。
    #   - 读侧使用 (regime, timeframe) 的索引一次取出整组权重，
    #     交给 RuleEngine 按 (factor_group, factor_name) 拼回内存字典。
    async def upsert_factor_weights(self, rows: Sequence[Dict[str, Any]]) -> int:
        """
        批量 UPSERT 因子权重
        --------------------------------------------------------------
        参数：
            rows: 每条字段需含
                regime / timeframe / factor_group / factor_name / weight
                可选：ic_30d / ic_90d / sample_count
        返回：
            尝试入库的行数（实际 UPSERT 数）。
        说明：
            UNIQUE(regime, timeframe, factor_group, factor_name) +
            ON CONFLICT DO UPDATE：每次校准任务直接覆盖 weight + IC 指标
            + 样本数 + updated_at；没有跑校准的组合保留旧值（基线种子）。
        """
        if not rows:
            return 0
        records = [
            (
                r["regime"],
                r["timeframe"],
                r["factor_group"],
                r["factor_name"],
                _to_dec(r["weight"]),
                _to_dec(r.get("ic_30d")),
                _to_dec(r.get("ic_90d")),
                int(r["sample_count"]) if r.get("sample_count") is not None else None,
                r.get("updated_at") or _utcnow(),
            )
            for r in rows
        ]

        async def _do() -> None:
            async with self.db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO factor_weights
                        (regime, timeframe, factor_group, factor_name,
                         weight, ic_30d, ic_90d, sample_count, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (regime, timeframe, factor_group, factor_name)
                    DO UPDATE SET
                        weight       = EXCLUDED.weight,
                        ic_30d       = EXCLUDED.ic_30d,
                        ic_90d       = EXCLUDED.ic_90d,
                        sample_count = EXCLUDED.sample_count,
                        updated_at   = EXCLUDED.updated_at
                    """,
                    records,
                )

        await self.db.run_with_retry(_do, op_name="upsert_factor_weights")
        return len(records)

    async def fetch_all_factor_weights(self) -> List[Dict[str, Any]]:
        """
        一次性读出整张 factor_weights 表
        --------------------------------------------------------------
        说明：
            - RuleEngine 启动 / 周期刷新缓存时调用一次即可；表规模上限
              为 (regime 数 × timeframe 数 × factor 数)，量级 < 数百行，
              全表扫描成本远低于按 (regime, timeframe) 多次查询的累计开销。
        返回：
            含 regime / timeframe / factor_group / factor_name / weight /
            ic_30d / ic_90d / sample_count / updated_at 的 dict 列表。
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT regime, timeframe, factor_group, factor_name,
                       weight, ic_30d, ic_90d, sample_count, updated_at
                FROM factor_weights
                """,
            )
        return [dict(r) for r in rows]

    async def fetch_factor_weights_by_regime(
        self, regime: str
    ) -> List[Dict[str, Any]]:
        """
        按 regime 读取权重（含 overall 兜底），归因 API 用
        --------------------------------------------------------------
        参数：
            regime: 'trending_up' / 'trending_down' / 'ranging' / 'breakout'
                    / 'breakdown' / 'transitional' / 'overall'
        返回：
            dict 列表（仅匹配 regime 的行；'overall' 留给上层合并）。
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT regime, timeframe, factor_group, factor_name,
                       weight, ic_30d, ic_90d, sample_count, updated_at
                FROM factor_weights
                WHERE regime = $1
                ORDER BY timeframe, factor_group, factor_name
                """,
                regime,
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # P2：信号生命周期表（signal_lifecycle）
    # ------------------------------------------------------------------
    async def insert_signal_lifecycle(
        self,
        signal_id: int,
        symbol: str,
        bias: str,
        entry_zone: Optional[List[float]],
        stop_loss: Optional[float],
        take_profit: Optional[List[float]],
        expires_at: datetime,
    ) -> None:
        """
        信号入库后立刻挂一条 pending 的 lifecycle 行
        --------------------------------------------------------------
        参数：
            signal_id   ：刚 INSERT 进 signals 表的自增 id（FK 主键）
            symbol      ：合约代码
            bias        ：long / short / neutral；neutral 时可直接置 expired
            entry_zone  ：[low, high]（可空，长度 2）
            stop_loss   ：止损价（可空）
            take_profit ：止盈位列表（可空，原样写入 JSONB）
            expires_at  ：超时强制结算的 UTC 时间（一般 = 信号 ts + 24h）
        说明：
            - neutral 信号也会落 lifecycle，但 status 直接置 expired，
              方便统一统计"近 N 条已结算判断"。
            - long/short 缺关键价位的旧 / 异常信号在 lifecycle 任务首轮扫到时
              会被置 expired，不强行回填。
        """
        ez_low: Optional[Decimal] = None
        ez_high: Optional[Decimal] = None
        if entry_zone and len(entry_zone) == 2:
            try:
                ez_low = _to_dec(min(float(entry_zone[0]), float(entry_zone[1])))
                ez_high = _to_dec(max(float(entry_zone[0]), float(entry_zone[1])))
            except (TypeError, ValueError):
                ez_low = ez_high = None

        initial_status = "expired" if bias == "neutral" else "pending"

        async def _do() -> None:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO signal_lifecycle
                        (signal_id, symbol, bias,
                         entry_zone_low, entry_zone_high, stop_loss, take_profit,
                         status, expires_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, NOW())
                    ON CONFLICT (signal_id) DO NOTHING
                    """,
                    signal_id,
                    symbol,
                    bias,
                    ez_low,
                    ez_high,
                    _to_dec(stop_loss),
                    take_profit,
                    initial_status,
                    expires_at,
                )

        await self.db.run_with_retry(_do, op_name="insert_signal_lifecycle")

    async def fetch_open_signal_lifecycles(
        self, symbol: Optional[str] = None, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        拉取所有 status ∈ {pending, triggered} 的生命周期行
        --------------------------------------------------------------
        参数：
            symbol: 可选过滤；None 时拉全表（多 symbol 部署时配合任务调度用）
            limit:  防御性上限
        返回：
            dict 列表，含 signal_id / symbol / bias / entry_zone_* / stop_loss /
            take_profit / status / triggered_at / triggered_price /
            max_favorable_pct / max_adverse_pct / expires_at / created_at。
        说明：
            走偏序索引 idx_signal_lifecycle_open_expires，扫描成本 O(未结算行数)。
        """
        async with self.db.acquire() as conn:
            if symbol is None:
                rows = await conn.fetch(
                    """
                    SELECT signal_id, symbol, bias,
                           entry_zone_low, entry_zone_high, stop_loss, take_profit,
                           status, triggered_at, triggered_price,
                           max_favorable_pct, max_adverse_pct,
                           expires_at, created_at
                    FROM signal_lifecycle
                    WHERE status IN ('pending', 'triggered')
                    ORDER BY expires_at ASC
                    LIMIT $1
                    """,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT signal_id, symbol, bias,
                           entry_zone_low, entry_zone_high, stop_loss, take_profit,
                           status, triggered_at, triggered_price,
                           max_favorable_pct, max_adverse_pct,
                           expires_at, created_at
                    FROM signal_lifecycle
                    WHERE status IN ('pending', 'triggered') AND symbol = $1
                    ORDER BY expires_at ASC
                    LIMIT $2
                    """,
                    symbol,
                    limit,
                )
        return [dict(r) for r in rows]

    async def update_signal_lifecycle(
        self,
        signal_id: int,
        *,
        status: Optional[str] = None,
        triggered_at: Optional[datetime] = None,
        triggered_price: Optional[float] = None,
        exit_at: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        max_favorable_pct: Optional[float] = None,
        max_adverse_pct: Optional[float] = None,
    ) -> None:
        """
        增量更新一条 lifecycle 行
        --------------------------------------------------------------
        说明：
            - 只动传入的字段（None 表示不更新），最终 updated_at 必更新；
            - 用 COALESCE($n, 当前列) 让 SQL 一段就能跑完，避免 N 个 UPDATE。
        """
        async def _do() -> None:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE signal_lifecycle SET
                        status            = COALESCE($2, status),
                        triggered_at      = COALESCE($3, triggered_at),
                        triggered_price   = COALESCE($4, triggered_price),
                        exit_at           = COALESCE($5, exit_at),
                        exit_price        = COALESCE($6, exit_price),
                        pnl_pct           = COALESCE($7, pnl_pct),
                        max_favorable_pct = COALESCE($8, max_favorable_pct),
                        max_adverse_pct   = COALESCE($9, max_adverse_pct),
                        updated_at        = NOW()
                    WHERE signal_id = $1
                    """,
                    signal_id,
                    status,
                    triggered_at,
                    _to_dec(triggered_price),
                    exit_at,
                    _to_dec(exit_price),
                    _to_dec(pnl_pct),
                    _to_dec(max_favorable_pct),
                    _to_dec(max_adverse_pct),
                )

        await self.db.run_with_retry(_do, op_name="update_signal_lifecycle")

    async def fetch_recent_settled_lifecycles(
        self,
        symbol: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        读取指定 symbol 最近 N 条已结算的 lifecycle 行（含 signals 表的元数据）
        --------------------------------------------------------------
        参数：
            symbol : 合约代码
            limit  : 最近 N 条（默认 5，给 LLM 自我反馈用）
        返回：
            dict 列表，按 updated_at 倒序。每条额外带 ts / regime（来自 signals.factors）
            / confidence / reason，便于 prompt 构造紧凑表格。
        说明：
            - 已结算 = status ∈ {sl_hit, tp1_hit, tp2_hit, expired, invalidated}。
            - 与 signals JOIN 一次拿齐，避免 LLM 路径多打一轮 RTT。
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sl.signal_id, sl.symbol, sl.bias, sl.status,
                    sl.entry_zone_low, sl.entry_zone_high,
                    sl.stop_loss, sl.take_profit,
                    sl.triggered_at, sl.triggered_price,
                    sl.exit_at, sl.exit_price, sl.pnl_pct,
                    sl.max_favorable_pct, sl.max_adverse_pct,
                    sl.created_at, sl.updated_at, sl.expires_at,
                    s.ts AS signal_ts,
                    s.confidence,
                    s.reason,
                    s.factors
                FROM signal_lifecycle sl
                JOIN signals s ON s.id = sl.signal_id
                WHERE sl.symbol = $1
                  AND sl.status IN ('sl_hit', 'tp1_hit', 'tp2_hit',
                                    'expired', 'invalidated')
                ORDER BY sl.updated_at DESC
                LIMIT $2
                """,
                symbol,
                limit,
            )
        return [dict(r) for r in rows]

    async def fetch_open_signal_lifecycle_for_symbol(
        self, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        读取指定 symbol 最近一条 status ∈ {pending, triggered} 的 lifecycle
        --------------------------------------------------------------
        说明：
            LLM 自我反馈的 SYSTEM_PROMPT 约束需要它："如果上一条信号还在
            triggered，新判断方向不能与其相反"。
        """
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT signal_id, symbol, bias, status,
                       entry_zone_low, entry_zone_high,
                       stop_loss, take_profit,
                       triggered_at, triggered_price,
                       max_favorable_pct, max_adverse_pct,
                       created_at, expires_at
                FROM signal_lifecycle
                WHERE symbol = $1
                  AND status IN ('pending', 'triggered')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None

    async def fetch_lifecycle_stats(
        self,
        symbol: str,
        since: datetime,
    ) -> Dict[str, Any]:
        """
        近 N 天信号胜率 / 平均 RR / 平均 PnL / 各 regime 命中率（API 用）
        --------------------------------------------------------------
        参数：
            symbol : 合约代码
            since  : 起始 UTC 时间（含），按 lifecycle.created_at 过滤
        返回：
            {
              "total": ..,
              "win": ..,
              "loss": ..,
              "expired": ..,
              "win_rate": float,
              "avg_pnl_pct": float,
              "avg_rr": float,
              "by_regime": {regime: {...}, ...}
            }
        说明：
            - win = tp1_hit / tp2_hit；loss = sl_hit；expired/invalidated 单算。
            - regime 来自 signals.factors->>'regime'；新格式直接挂在根；
              老格式 / 字段缺失时归到 'unknown'。
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sl.status, sl.pnl_pct,
                       COALESCE(s.factors->'factors'->>'regime',
                                s.factors->>'regime',
                                'unknown') AS regime,
                       s.risk_reward_ratio
                FROM signal_lifecycle sl
                JOIN signals s ON s.id = sl.signal_id
                WHERE sl.symbol = $1
                  AND sl.created_at >= $2
                  AND sl.status IN ('sl_hit', 'tp1_hit', 'tp2_hit',
                                    'expired', 'invalidated')
                """,
                symbol,
                since,
            )

        total = len(rows)
        win = sum(1 for r in rows if r["status"] in ("tp1_hit", "tp2_hit"))
        loss = sum(1 for r in rows if r["status"] == "sl_hit")
        expired_cnt = sum(1 for r in rows if r["status"] == "expired")
        pnl_values = [float(r["pnl_pct"]) for r in rows if r["pnl_pct"] is not None]
        rr_values = [
            float(r["risk_reward_ratio"])
            for r in rows
            if r["risk_reward_ratio"] is not None
        ]
        win_rate = (win / total) if total else 0.0
        avg_pnl = (sum(pnl_values) / len(pnl_values)) if pnl_values else 0.0
        avg_rr = (sum(rr_values) / len(rr_values)) if rr_values else 0.0

        by_regime: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            reg = r["regime"] or "unknown"
            bucket = by_regime.setdefault(
                reg, {"total": 0, "win": 0, "loss": 0, "expired": 0, "pnl_sum": 0.0}
            )
            bucket["total"] += 1
            if r["status"] in ("tp1_hit", "tp2_hit"):
                bucket["win"] += 1
            elif r["status"] == "sl_hit":
                bucket["loss"] += 1
            elif r["status"] == "expired":
                bucket["expired"] += 1
            if r["pnl_pct"] is not None:
                bucket["pnl_sum"] += float(r["pnl_pct"])

        for bucket in by_regime.values():
            bucket["win_rate"] = (
                bucket["win"] / bucket["total"] if bucket["total"] else 0.0
            )
            bucket["avg_pnl_pct"] = (
                bucket["pnl_sum"] / bucket["total"] if bucket["total"] else 0.0
            )

        return {
            "total": total,
            "win": win,
            "loss": loss,
            "expired": expired_cnt,
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl,
            "avg_rr": avg_rr,
            "by_regime": by_regime,
        }

    # ------------------------------------------------------------------
    # P2：mark price + 历史信号查询（IC 校准 / lifecycle 用）
    # ------------------------------------------------------------------
    async def fetch_latest_kline_close_within(
        self,
        timeframe: str,
        symbol: str,
        max_age_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """
        读取指定周期最新一根 K 线的 close + ts，并校验"年龄 ≤ max_age_seconds"
        --------------------------------------------------------------
        参数：
            timeframe       : '1m' / '5m' / ... 白名单内
            symbol          : 合约代码
            max_age_seconds : 最大允许的"距 now 的秒数"；超过则视为陈旧
        返回：
            {ts, close} 或 None（K 线缺失 / 陈旧）
        说明：
            生命周期任务专用——K 线断流时（max_age_seconds 一般取 60s），
            本方法返回 None，调用方据此跳过本轮判断，避免误结算。
        """
        latest = await self.fetch_latest_kline(timeframe=timeframe, symbol=symbol)
        if not latest:
            return None
        ts = latest.get("ts")
        if ts is None:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > max_age_seconds:
            return None
        return {"ts": ts, "close": latest.get("close")}

    async def fetch_kline_close_at(
        self,
        timeframe: str,
        symbol: str,
        target_ts: datetime,
    ) -> Optional[float]:
        """
        取在 target_ts 之后最早一根 K 线的 close
        --------------------------------------------------------------
        参数：
            timeframe : '1m' / '5m' / '1h' / '4h'
            symbol    : 合约代码
            target_ts : 目标时间（IC 校准里 = signal.ts + Δh）
        返回：
            最近未来 K 线的 close（float）；找不到时返回 None。
        说明：
            - "之后最早"而非"之前最近"是为了 IC 计算的因果性：
              forward_return = (px_future - px_now) / px_now，px_future
              必须严格在 px_now 之后；早于 target_ts 的 close 会引入未来函数。
            - 走 idx_klines_<tf>_symbol_ts (symbol, ts DESC) 索引，
              ts >= target_ts 用 ASC 第一条命中，单次查询 < 5ms。
        """
        table = self._kline_table(timeframe)
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT close
                FROM {table}
                WHERE symbol = $1 AND ts >= $2
                ORDER BY ts ASC
                LIMIT 1
                """,
                symbol,
                target_ts,
            )
        if not row or row["close"] is None:
            return None
        return float(row["close"])

    async def fetch_signals_with_factors_since(
        self,
        symbol: str,
        since: datetime,
        source_like: str = "%llm%",
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        IC 校准任务用：拉取最近一段时间内 source 含 'llm' 的 signals 行
        --------------------------------------------------------------
        参数：
            symbol      : 合约代码
            since       : 起始 UTC 时间（含），通常 = now - 90d
            source_like : ILIKE 模式，默认 '%llm%' 匹配 'rules+llm' / 'rules+llm(cache)'
            limit       : 防御性上限；30s 一条 × 30 天 ≈ 86400，不会触顶
        返回：
            dict 列表，含 id / ts / bias / confidence / factors（JSONB 反序列化为 dict）
            / source。
        说明：
            - factors 列体积大，IC 任务只跑一次/天，单次扫描 < 1s 可接受。
            - 之所以用 ILIKE 而不是 = 'rules+llm'：是为了兼容历史 / 未来的 source
              变体（例如 'rules+llm(cache)'、'rules+llm(shadow)' 等）。
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, ts, bias, confidence, source, factors
                FROM signals
                WHERE symbol = $1
                  AND ts >= $2
                  AND source ILIKE $3
                ORDER BY ts ASC
                LIMIT $4
                """,
                symbol,
                since,
                source_like,
                limit,
            )
        return [dict(r) for r in rows]

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
