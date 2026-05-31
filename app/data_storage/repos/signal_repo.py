"""Signal repository: signals table insert / query / cleanup."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

from .base import BaseRepo
from .helpers import parse_delete_count, to_dec, utcnow

logger = get_logger(__name__)


class SignalRepo(BaseRepo):
    """signals 表仓储：写入、查询、按时间清理。"""

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
        async with self._db.acquire() as conn:
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
                ts or utcnow(),
                symbol,
                bias,
                to_dec(confidence),
                reason,
                risk,
                suggestion,
                factors,
                source,
                reasoning_content,
                entry_zone,
                to_dec(stop_loss),
                take_profit,
                to_dec(risk_reward_ratio),
                to_dec(position_size_pct),
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
        async with self._db.acquire() as conn:
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
        读取指定 symbol 最近一条信号的"全字段视图"（结构化交易计划）
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            dict（不存在则返回 None），字段为 signals 表全部列：
                id / ts / bias / confidence / reason / risk / suggestion /
                factors / source / reasoning_content / entry_zone /
                stop_loss / take_profit / risk_reward_ratio /
                position_size_pct / timeframe_alignment /
                invalidation_conditions
        说明：
            - 专供前端"分析卡片 / 详情页"使用，一次拿齐所有可视化字段。
            - LLM-First 重构后 signal_lifecycle 表已被整体删除；前端如需
              "信号实战结果"列请走业务侧自己跟踪。
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, ts, symbol, bias, confidence,
                    reason, risk, suggestion,
                    factors, source, reasoning_content,
                    entry_zone, stop_loss, take_profit,
                    risk_reward_ratio, position_size_pct,
                    timeframe_alignment, invalidation_conditions
                FROM signals
                WHERE symbol = $1
                ORDER BY ts DESC
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
        读取指定 symbol 最近 N 条信号的"全字段视图"（列表场景，不含思维链全文）
        ---------------------------------------------------------------
        参数：
            symbol         : 合约代码
            limit          : 返回条数上限（防御性 1~200）
            bias           : 可选过滤，'long' / 'short' / 'neutral'
            source_like    : 可选 ILIKE 模式（如 '%llm%' 只看 LLM 路径）
            only_persisted : 默认 True，No-op 占位，保留参数语义。
        返回：
            按 ts 倒序的 dict 列表。

        与 ``fetch_latest_signal_full`` 的差异（关键性能修复）：
            列表视图永远不会展示思维链全文（前端详情页才走
            ``fetch_signal_full_by_id`` 单条拉取），但
            ``reasoning_content`` 单条可达数十 KB。一次拉 100 条会让单次
            网络传输 5 MB+，叠加 asyncpg 默认 30s ``command_timeout`` 极易
            超时；超时后 pool release 还会触发
            ``cannot switch to state X; another operation in progress``，
            连带后续写库连接全部坏掉。

            因此本方法**不再 SELECT reasoning_content 原文**，而是用 SQL
            端的 ``length()`` / ``IS NOT NULL`` 派生出
            ``reasoning_total_chars`` 与 ``reasoning_available``，
            ``serialize_signal_full`` 在 ``include_reasoning=False``
            场景下兼容这两个派生列；详情页（include_reasoning=True）
            仍走 ``fetch_signal_full_by_id`` 单条拉取。
        """
        safe_limit = max(1, min(int(limit), 200))
        clauses = ["symbol = $1"]
        params: List[Any] = [symbol]
        if bias in ("long", "short", "neutral"):
            params.append(bias)
            clauses.append(f"bias = ${len(params)}")
        if source_like:
            params.append(source_like)
            clauses.append(f"source ILIKE ${len(params)}")
        params.append(safe_limit)
        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT
                id, ts, symbol, bias, confidence,
                reason, risk, suggestion,
                factors, source,
                (reasoning_content IS NOT NULL)        AS reasoning_available,
                COALESCE(length(reasoning_content), 0) AS reasoning_total_chars,
                entry_zone, stop_loss, take_profit,
                risk_reward_ratio, position_size_pct,
                timeframe_alignment, invalidation_conditions
            FROM signals
            WHERE {where_sql}
            ORDER BY ts DESC
            LIMIT ${len(params)}
        """
        _ = only_persisted
        # 显式给 60s 超时：列表查询要拉 100 条 factors JSONB（每条几十 KB），
        # 比单点写库更耗时，pool 默认 30s 不够用，宽松一点避免误判超时。
        async with self._db.acquire() as conn:
            rows = await conn.fetch(sql, *params, timeout=60.0)
        return [dict(r) for r in rows]

    async def fetch_signal_full_by_id(
        self, signal_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        按 id 读取单条信号的"全字段视图"
        ---------------------------------------------------------------
        参数：
            signal_id: signals.id
        返回：
            字段同 fetch_latest_signal_full；找不到返回 None。
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id, ts, symbol, bias, confidence,
                    reason, risk, suggestion,
                    factors, source, reasoning_content,
                    entry_zone, stop_loss, take_profit,
                    risk_reward_ratio, position_size_pct,
                    timeframe_alignment, invalidation_conditions
                FROM signals
                WHERE id = $1
                """,
                signal_id,
            )
        return dict(row) if row else None

    async def fetch_latest_signal_judgment(
        self, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        轻量读取指定 symbol 最近一条信号的"判断字段 + 结构化交易计划"
        ---------------------------------------------------------------
        说明：
            - 与 ``fetch_latest_signal`` 的区别：**故意不读取 factors 列**。
              factors 是 JSONB，单条几十 KB；LLM 节流判定每 30 秒会查一次，
              没必要把因子快照也拉过来反序列化，浪费带宽与 CPU。
            - 该方法专供 LLMAgent 的"DB 节流"使用：
                1) 拿 ts 判断距上次 LLM 分析是否 ≥ min_interval；
                2) 命中节流时用 bias / confidence / reason / risk / suggestion
                   + 结构化交易计划（entry_zone / SL / TP / RR / 仓位 / 多周期
                   投票 / 失效条件）完整重建 TradingSignal，连同
                   reasoning_content 一起返回给上层。
            - 走的是 idx_signals_symbol_ts (symbol, ts DESC) 索引，单点查询，
              成本远低于 LLM 调用，不会成为热路径瓶颈。

        P0 Quant 修复 #3：以前只 SELECT 5 个文本字段，导致缓存命中重建出来
        的 TradingSignal 在 bias=long 时 entry_zone/SL/TP 全空、被迫绕过
        model_validator 强约束，前端会拿到一条"我是 long、可是没有任何
        入场计划"的脏信号。这里把 P0 升级新增的 7 个结构化列也拉出来。
        参数：
            symbol: 合约代码
        返回：
            最新一条 signals 行的判断字段字典；不存在则返回 None。
            字段：ts / bias / confidence / reason / risk / suggestion /
                  reasoning_content / entry_zone(JSONB) / stop_loss(NUMERIC) /
                  take_profit(JSONB) / risk_reward_ratio(NUMERIC) /
                  position_size_pct(NUMERIC) / timeframe_alignment(JSONB) /
                  invalidation_conditions(JSONB)。
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT ts, bias, confidence, reason, risk, suggestion,
                       reasoning_content,
                       entry_zone, stop_loss, take_profit,
                       risk_reward_ratio, position_size_pct,
                       timeframe_alignment, invalidation_conditions
                FROM signals
                WHERE symbol = $1
                ORDER BY ts DESC
                LIMIT 1
                """,
                symbol,
            )
        return dict(row) if row else None

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
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM signals WHERE ts < $1",
                cutoff,
            )
        return parse_delete_count(result)
