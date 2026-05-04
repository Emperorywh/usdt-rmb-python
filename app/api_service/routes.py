"""HTTP routes（P0/P1/P2 共用）。

P2 新增接口：
* GET  /signals/{signal_id}/attribution   信号因子贡献度归因
* GET  /factors/weights/current           当前生效的权重表（按 regime 过滤）
* GET  /signals/lifecycle/stats           近 N 天信号胜率 / 平均 RR / 各 regime 命中率
* POST /admin/calibrate-ic                手动触发一次 IC 校准（带 token）
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.api_service.analysis_view import serialize_signal_full
from app.api_service.deps import (
    get_container,
    get_factor_aggregator,
    get_signal_service,
)
from app.container import AppContainer
from app.factor_engine.aggregator import FactorAggregator
from app.signal_engine.service import SignalService

router = APIRouter()


# 简单的邮箱格式校验：避免引入 email-validator 这一可选依赖。
# pydantic 的 EmailStr 需要 email-validator 才能工作，作为兜底。
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _validate_email_str(email: str) -> str:
    """
    轻量校验邮箱格式，去掉首尾空格并统一小写
    --------------------------------------------------------------
    参数：
        email: 原始邮箱字符串
    返回：
        清洗后的邮箱
    异常：
        HTTPException(400) - 格式不合法
    """
    cleaned = (email or "").strip()
    if not cleaned or not _EMAIL_RE.match(cleaned):
        raise HTTPException(status_code=400, detail=f"邮箱格式不合法: {email!r}")
    return cleaned.lower()


class NotificationEmailCreate(BaseModel):
    """
    新增邮件收件人入参
    --------------------------------------------------------------
    字段：
        email   : 收件邮箱（必填，UNIQUE）
        name    : 备注名（可空）
        enabled : 是否启用（默认 True）
    """

    email: str = Field(..., description="收件邮箱")
    name: Optional[str] = Field(default=None, max_length=128, description="备注名")
    enabled: bool = Field(default=True, description="是否启用")


class NotificationEmailUpdate(BaseModel):
    """
    更新邮件收件人入参
    --------------------------------------------------------------
    所有字段都是可空的（PATCH 语义）；显式置空 name 请传空字符串以外的标记，
    本接口暂不支持把 name 重置回 NULL。
    """

    email: Optional[str] = Field(default=None, description="收件邮箱")
    name: Optional[str] = Field(default=None, max_length=128, description="备注名")
    enabled: Optional[bool] = Field(default=None, description="是否启用")


class TestEmailRequest(BaseModel):
    """
    /emails/test 入参：发送一封测试邮件到指定地址
    """

    email: str = Field(..., description="测试邮箱地址")


@router.get("/health", tags=["meta"])
async def health(container: AppContainer = Depends(get_container)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "symbols": container.settings.symbols,
        "exchanges": container.settings.exchanges,
        "llm_enabled": container.llm_agent.enabled,
    }


@router.get("/healthz", tags=["meta"])
async def healthz(container: AppContainer = Depends(get_container)) -> Dict[str, Any]:
    """
    采集通道详细健康度
    -------------------------------------------------------------------
    返回字段：
        - status:  'ok' / 'degraded'，是否所有 WS 频道都在新鲜窗口内
        - ws:      {symbol: {kind: {age_seconds, last_event_at}}}
                   每个 (symbol, kind) 的 WS 推送年龄；kind ∈
                   {trade, orderbook, ticker, funding_rate, open_interest}
        - rest:    {op_name: {state, consecutive_failures,
                              cooldown_remaining, last_error,
                              last_success_at, success_count, failure_count}}
                   每个 REST endpoint 的熔断状态
    用途：
        - 运维监控 / 仪表盘判断"WS 是否在推" / "REST 是否被熔断"
        - 信号引擎可读取本接口在数据陈旧时主动降级
    """
    ws_snapshot = (
        container.ingestion_runner.ws_health_snapshot()
        if container.ingestion_runner is not None
        else {}
    )
    rest_snapshot = container.okx_rest.health_snapshot()

    # status 判定：只要有任意 (symbol, funding_rate/open_interest) 通道
    # 在 staleness 阈值之上即视为 degraded；trade/orderbook 静默 60s 也算异常。
    degraded = False
    for symbol_view in ws_snapshot.values():
        for kind, info in symbol_view.items():
            age = float(info.get("age_seconds") or 0.0)
            if kind == "funding_rate" and age > 5 * 60:
                degraded = True
            elif kind == "open_interest" and age > 60:
                degraded = True
            elif kind in ("trade", "orderbook") and age > 60:
                degraded = True

    return {
        "status": "degraded" if degraded else "ok",
        "ws": ws_snapshot,
        "rest": rest_snapshot,
    }


def _resolve_symbol(symbol: Optional[str], container: AppContainer) -> str:
    if symbol:
        return symbol
    if container.settings.symbols:
        return container.settings.symbols[0]
    raise HTTPException(status_code=400, detail="No symbol configured")


@router.get("/factors", tags=["factors"])
async def get_factors(
    symbol: Optional[str] = Query(default=None),
    container: AppContainer = Depends(get_container),
    aggregator: FactorAggregator = Depends(get_factor_aggregator),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    return await aggregator.compute(sym)


@router.get("/signal", tags=["signal"])
async def get_signal(
    symbol: Optional[str] = Query(default=None),
    include_reasoning: bool = Query(
        default=False,
        description="是否返回思考模式下的 reasoning_content 原文（可能很长）",
    ),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    row = await container.repos.fetch_latest_signal(sym)
    if not row:
        raise HTTPException(status_code=404, detail="No signal yet, try /signal/refresh")
    payload: Dict[str, Any] = {
        "timestamp": row["ts"].isoformat(),
        "symbol": row["symbol"],
        "source": row["source"],
        "signal": {
            "bias": row["bias"],
            "confidence": float(row["confidence"]),
            "reason": row["reason"],
            "risk": row["risk"],
            "suggestion": row["suggestion"],
        },
        "factors": row["factors"],
        # 仅暴露"是否存在思维链"作为元信息；具体内容只在显式请求时返回，
        # 避免默认响应被一两万字的思维链撑爆带宽与日志。
        "reasoning_available": bool(row.get("reasoning_content")),
    }
    if include_reasoning:
        payload["reasoning_content"] = row.get("reasoning_content")
    return payload


@router.post("/signal/refresh", tags=["signal"])
async def refresh_signal(
    symbol: Optional[str] = Query(default=None),
    include_reasoning: bool = Query(
        default=False,
        description="是否返回本次 LLM 的 reasoning_content 原文（可能很长）",
    ),
    container: AppContainer = Depends(get_container),
    signal_service: SignalService = Depends(get_signal_service),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    result = await signal_service.generate(sym)
    payload: Dict[str, Any] = {
        "timestamp": result["factors"]["computed_at"],
        "symbol": sym,
        "source": result["source"],
        "signal": result["signal"],
        "rule_signal": result["rule_signal"],
        "rule_score": result["rule_score"],
        # 注意：persisted=False 时 reasoning_content 不会进入 DB（纯规则引擎路径）
        "persisted": result["persisted"],
        "reasoning_available": result["reasoning_content"] is not None,
    }
    if include_reasoning:
        payload["reasoning_content"] = result["reasoning_content"]
    return payload


# ======================================================================
# 前端综合分析视图（最近一次 / 历史列表 / 单条详情）
# ----------------------------------------------------------------------
# 设计目标：
#   1) 一次拿齐前端"分析卡片 / 详情页"需要的所有字段，避免串行多次调用
#      /signal + /signals/{id}/attribution + lifecycle stats。
#   2) 把所有 Decimal / datetime / 枚举字段都做成对前端友好的形态：
#      - Decimal → float、datetime → ISO8601；
#      - bias / source / lifecycle_status 都补一份中文 label + Tag color；
#      - take_profit 自动算好"距入场点的收益百分比"，前端直接渲染；
#      - rule_contributions 已经按贡献度绝对值排好序并截断 top N。
#   3) 默认带 factors_snapshot（可关），不带 reasoning_content 全文（可显式打开），
#      避免默认响应被一两万字思维链撑爆。
# ======================================================================
@router.get("/analysis/latest", tags=["analysis"])
async def get_latest_analysis(
    symbol: Optional[str] = Query(default=None, description="合约代码；缺省取 settings.symbols[0]"),
    include_factors: bool = Query(
        default=True,
        description="是否携带完整 factors 快照；前端列表页可置 False 节省带宽",
    ),
    include_reasoning: bool = Query(
        default=False,
        description="是否携带 LLM 思维链全文（reasoning_content，可能很长）",
    ),
    top_contributions: int = Query(
        default=10, ge=1, le=50, description="规则引擎归因 top N 因子",
    ),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    返回最新一次综合分析（前端友好视图）
    -------------------------------------------------------------------
    包含：信号判断 / 结构化交易计划 / 多周期共振 / regime / 流动性地图 /
    规则引擎归因 / 信号生命周期实战表现 等。
    """
    sym = _resolve_symbol(symbol, container)
    row = await container.repos.fetch_latest_signal_full(sym)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=(
                "暂无分析结果，请稍后再试或调用 POST /signal/refresh 主动触发"
            ),
        )
    return serialize_signal_full(
        row,
        include_factors_snapshot=bool(include_factors),
        include_reasoning=bool(include_reasoning),
        rule_contributions_top_n=int(top_contributions),
    )


@router.get("/analysis/history", tags=["analysis"])
async def get_analysis_history(
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100, description="返回条数（1~100）"),
    bias: Optional[str] = Query(
        default=None,
        description="按 bias 过滤：long / short / neutral",
    ),
    source: Optional[str] = Query(
        default=None,
        description="source ILIKE 模式（如 '%llm%' 只看 LLM 路径）",
    ),
    include_factors: bool = Query(
        default=False,
        description="列表场景默认不返回 factors 快照，节省带宽；详情页再单独取",
    ),
    top_contributions: int = Query(default=5, ge=1, le=50),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    返回最近 N 条综合分析（按时间倒序，前端时间线 / 历史列表用）
    -------------------------------------------------------------------
    返回：
        {
            "symbol": ..., "count": N,
            "items": [<同 /analysis/latest 的视图>, ...]
        }
    """
    if bias is not None and bias not in ("long", "short", "neutral"):
        raise HTTPException(
            status_code=400, detail="bias 必须是 long / short / neutral"
        )
    sym = _resolve_symbol(symbol, container)
    rows = await container.repos.fetch_recent_signals_full(
        symbol=sym,
        limit=int(limit),
        bias=bias,
        source_like=source,
    )
    items = [
        serialize_signal_full(
            r,
            include_factors_snapshot=bool(include_factors),
            include_reasoning=False,
            rule_contributions_top_n=int(top_contributions),
        )
        for r in rows
    ]
    return {
        "symbol": sym,
        "count": len(items),
        "filters": {"bias": bias, "source_like": source, "limit": int(limit)},
        "items": items,
    }


@router.get("/analysis/{signal_id}", tags=["analysis"])
async def get_analysis_by_id(
    signal_id: int,
    include_factors: bool = Query(default=True),
    include_reasoning: bool = Query(default=False),
    top_contributions: int = Query(default=20, ge=1, le=100),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    按 signals.id 读取单条综合分析详情
    -------------------------------------------------------------------
    与 /analysis/latest 字段一致；详情页通常打开 include_reasoning=true 看思维链全文。
    """
    row = await container.repos.fetch_signal_full_by_id(signal_id)
    if not row:
        raise HTTPException(status_code=404, detail="signal_id not found")
    return serialize_signal_full(
        row,
        include_factors_snapshot=bool(include_factors),
        include_reasoning=bool(include_reasoning),
        rule_contributions_top_n=int(top_contributions),
    )


# ======================================================================
# P2：归因 / 权重 / 生命周期统计
# ======================================================================
@router.get("/signals/{signal_id}/attribution", tags=["signal", "p2"])
async def get_signal_attribution(
    signal_id: int,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    返回某条信号的因子贡献度分解（基于规则引擎 contributions + LLM reasoning_content 摘要）
    -------------------------------------------------------------------
    返回字段：
        signal_id        : 入参
        symbol / ts / bias / confidence / source
        rule_score       : 规则引擎打分
        rule_contributions : 原子因子粒度的贡献度（来自 signals.factors.rule_contributions）
        regime           : 当时的 market regime（来自 signals.factors.factors.regime）
        weights_snapshot : 该 regime 下当前生效的权重（用于"当时打分用了哪些权重"溯源）
        reasoning_excerpt: LLM 思维链前 800 字符（仅审计用，全文走 /signal?include_reasoning）
    说明：
        signals.factors 是 JSONB，里面的 rule_contributions 由 service 层在
        持久化时写入；P2 升级后 contributions 已经是"tf:group.factor_name"粒度。
    """
    async with container.db.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, ts, symbol, bias, confidence, factors, source,
                   reasoning_content
            FROM signals
            WHERE id = $1
            """,
            signal_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="signal_id not found")

    factors_blob = row["factors"] or {}
    inner_factors = (
        factors_blob.get("factors")
        if isinstance(factors_blob, dict) and "factors" in factors_blob
        else factors_blob
    )
    rule_contribs = (
        factors_blob.get("rule_contributions")
        if isinstance(factors_blob, dict)
        else None
    )
    rule_score = (
        factors_blob.get("rule_score")
        if isinstance(factors_blob, dict)
        else None
    )
    regime = (
        inner_factors.get("regime")
        if isinstance(inner_factors, dict)
        else None
    ) or "overall"

    # 当前权重表快照（按 regime 取，附带 overall 兜底）
    weights_snapshot: List[Dict[str, Any]] = []
    try:
        weights_rows = await container.repos.fetch_factor_weights_by_regime(regime)
        weights_snapshot = [
            {
                "timeframe": r["timeframe"],
                "factor_group": r["factor_group"],
                "factor_name": r["factor_name"],
                "weight": float(r["weight"]),
                "ic_30d": float(r["ic_30d"]) if r["ic_30d"] is not None else None,
                "ic_90d": float(r["ic_90d"]) if r["ic_90d"] is not None else None,
                "sample_count": r["sample_count"],
            }
            for r in weights_rows
        ]
    except Exception:
        weights_snapshot = []

    reasoning = row.get("reasoning_content") or ""
    excerpt = reasoning[:800] if isinstance(reasoning, str) else ""

    return {
        "signal_id": row["id"],
        "ts": row["ts"].isoformat(),
        "symbol": row["symbol"],
        "bias": row["bias"],
        "confidence": float(row["confidence"]),
        "source": row["source"],
        "regime": regime,
        "rule_score": rule_score,
        "rule_contributions": rule_contribs or {},
        "weights_snapshot": weights_snapshot,
        "reasoning_excerpt": excerpt,
        "reasoning_total_chars": len(reasoning) if isinstance(reasoning, str) else 0,
    }


@router.get("/factors/weights/current", tags=["factors", "p2"])
async def get_current_factor_weights(
    regime: Optional[str] = Query(
        default=None,
        description="按 regime 过滤；不传则返回全表",
    ),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    返回当前生效的因子权重（按 regime 维度过滤；不传 regime 则返回全表）
    -------------------------------------------------------------------
    用途：
        前端展示 "当前 regime=trending_up 下，net_flow_usd 在 5m 上权重 0.18"
        这种归因细节；同时给运维 / 量化研究员一个"快速查看 IC 校准结果"的入口。
    """
    if regime:
        rows = await container.repos.fetch_factor_weights_by_regime(regime)
    else:
        rows = await container.repos.fetch_all_factor_weights()
    return {
        "count": len(rows),
        "regime_filter": regime,
        "shadow_mode": bool(
            getattr(container.settings, "ic_calibrator_shadow_mode", True)
        ),
        "weights": [
            {
                "regime": r["regime"],
                "timeframe": r["timeframe"],
                "factor_group": r["factor_group"],
                "factor_name": r["factor_name"],
                "weight": float(r["weight"]),
                "ic_30d": float(r["ic_30d"]) if r["ic_30d"] is not None else None,
                "ic_90d": float(r["ic_90d"]) if r["ic_90d"] is not None else None,
                "sample_count": r["sample_count"],
                "updated_at": (
                    r["updated_at"].isoformat()
                    if r.get("updated_at") is not None
                    else None
                ),
            }
            for r in rows
        ],
    }


@router.get("/signals/lifecycle/stats", tags=["signal", "p2"])
async def get_lifecycle_stats(
    symbol: Optional[str] = Query(default=None),
    days: int = Query(default=7, ge=1, le=180),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    近 N 天信号胜率 / 平均 RR / 平均 PnL / 各 regime 下的命中率
    -------------------------------------------------------------------
    参数：
        symbol: 合约代码；缺省取 settings.symbols[0]
        days  : 统计窗口（天），上限 180 天，避免误传巨大窗口拖垮 DB
    返回：
        {
          "symbol": ...,
          "since": ...,
          "total": ..,
          "win_rate": float,
          "avg_pnl_pct": float,
          "avg_rr": float,
          "by_regime": {regime: {...}}
        }
    """
    sym = _resolve_symbol(symbol, container)
    since = datetime.now(timezone.utc) - timedelta(days=int(days))
    stats = await container.repos.fetch_lifecycle_stats(symbol=sym, since=since)
    return {
        "symbol": sym,
        "since": since.isoformat(),
        "days": days,
        **stats,
    }


@router.post("/admin/calibrate-ic", tags=["admin", "p2"])
async def admin_calibrate_ic(
    container: AppContainer = Depends(get_container),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    """
    手动立刻触发一次 IC 校准（不等待下一个周期）
    -------------------------------------------------------------------
    Header：
        X-Admin-Token : 必须等于 settings.ic_calibrator_admin_token；
                         token 留空时本接口直接 403（避免误暴露重计算入口）。
    返回：
        校准报告摘要（与 logs/ic_calibration_*.json 对齐）。
    说明：
        - 任务内部带 asyncio.Lock，与 cron 周期任务串行，不会并发跑两轮；
        - container.ic_calibrator 为 None 时（开关关闭）返回 503。
    """
    expected = (container.settings.ic_calibrator_admin_token or "").strip()
    if not expected:
        raise HTTPException(
            status_code=403,
            detail="admin calibrate disabled: ic_calibrator_admin_token is empty",
        )
    if (x_admin_token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="invalid admin token")

    if container.ic_calibrator is None:
        raise HTTPException(
            status_code=503, detail="IC calibrator is disabled at startup"
        )
    report = await container.ic_calibrator.run_once(triggered_by="admin")
    return {
        "ran_at": report.ran_at.isoformat(),
        "finished_at": report.finished_at.isoformat(),
        "skipped": report.skipped,
        "skipped_reason": report.skipped_reason,
        "total_signals_30d": report.total_signals_30d,
        "total_signals_90d": report.total_signals_90d,
        "total_records_30d": report.total_records_30d,
        "groups_updated": report.groups_updated,
        "groups_skipped_low_sample": report.groups_skipped_low_sample,
        "weights_written": report.weights_written,
    }


# ======================================================================
# 邮件通知收件人 CRUD（notification_emails）
# ----------------------------------------------------------------------
# 设计原则：
#   - 路径 /emails 而非 /admin/emails，方便前端配置页直接调用；
#   - 所有写操作都做"邮箱格式预校验 + 唯一约束错误友好转换"；
#   - 输出统一走 _serialize_notification_email_row，前端拿到的字段固定。
# ======================================================================
def _serialize_notification_email_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 notification_emails 表行序列化为前端友好的 dict
    --------------------------------------------------------------
    参数：
        row : repos 返回的 dict（包含 datetime / bool 等原始字段）
    返回：
        统一格式的 dict（datetime 统一 ISO8601 字符串）
    """
    created_at = row.get("created_at")
    updated_at = row.get("updated_at")
    return {
        "id": row.get("id"),
        "email": row.get("email"),
        "name": row.get("name"),
        "enabled": bool(row.get("enabled")),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else None,
    }


@router.get("/emails", tags=["notification"])
async def list_notification_emails(
    only_enabled: bool = Query(default=False, description="仅返回启用的收件人"),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    列出所有邮件通知收件人
    --------------------------------------------------------------
    参数：
        only_enabled : True 时仅返回 enabled=TRUE 的行；默认 False 看全部
    返回：
        {"count": N, "items": [...]}
    """
    rows = await container.repos.list_notification_emails(only_enabled=only_enabled)
    items = [_serialize_notification_email_row(r) for r in rows]
    return {"count": len(items), "items": items}


@router.post("/emails", tags=["notification"], status_code=201)
async def create_notification_email(
    payload: NotificationEmailCreate,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    新增一条邮件通知收件人
    --------------------------------------------------------------
    入参（JSON Body）：
        email   : 收件邮箱（必填）
        name    : 备注名（可空）
        enabled : 是否启用（默认 True）
    返回：
        新建行的完整 dict
    错误：
        400 - 邮箱格式非法
        409 - 邮箱已存在
    """
    email = _validate_email_str(payload.email)
    try:
        row = await container.repos.insert_notification_email(
            email=email,
            name=(payload.name or None),
            enabled=bool(payload.enabled),
        )
    except Exception as exc:  # noqa: BLE001
        # asyncpg 的 UniqueViolationError 路径
        msg = str(exc)
        if "notification_emails_unique" in msg or "duplicate key" in msg.lower():
            raise HTTPException(
                status_code=409, detail=f"邮箱已存在：{email}"
            ) from exc
        raise HTTPException(status_code=500, detail=f"新增邮箱失败：{msg}") from exc
    return _serialize_notification_email_row(row)


@router.get("/emails/{email_id}", tags=["notification"])
async def get_notification_email(
    email_id: int,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    按 id 查询单条邮件通知收件人详情
    --------------------------------------------------------------
    错误：
        404 - id 不存在
    """
    row = await container.repos.fetch_notification_email_by_id(email_id)
    if not row:
        raise HTTPException(status_code=404, detail="未找到对应邮箱")
    return _serialize_notification_email_row(row)


@router.put("/emails/{email_id}", tags=["notification"])
async def update_notification_email(
    email_id: int,
    payload: NotificationEmailUpdate,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    更新一条邮件通知收件人（PATCH 语义：未提供的字段保持不变）
    --------------------------------------------------------------
    错误：
        400 - 邮箱格式非法
        404 - id 不存在
        409 - 修改后的邮箱与他人冲突
    """
    new_email: Optional[str] = None
    if payload.email is not None:
        new_email = _validate_email_str(payload.email)
    try:
        row = await container.repos.update_notification_email(
            email_id,
            email=new_email,
            name=payload.name,
            enabled=payload.enabled,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "notification_emails_unique" in msg or "duplicate key" in msg.lower():
            raise HTTPException(
                status_code=409, detail=f"邮箱与他人冲突：{new_email}"
            ) from exc
        raise HTTPException(status_code=500, detail=f"更新邮箱失败：{msg}") from exc
    if not row:
        raise HTTPException(status_code=404, detail="未找到对应邮箱")
    return _serialize_notification_email_row(row)


@router.delete("/emails/{email_id}", tags=["notification"])
async def delete_notification_email(
    email_id: int,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    删除一条邮件通知收件人
    --------------------------------------------------------------
    错误：
        404 - id 不存在
    """
    ok = await container.repos.delete_notification_email(email_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到对应邮箱")
    return {"ok": True, "id": email_id}


@router.post("/emails/test", tags=["notification"])
async def send_test_notification_email(
    payload: TestEmailRequest,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    发送一封测试邮件（用于校验 SMTP 配置是否正确）
    --------------------------------------------------------------
    入参：
        email : 测试收件邮箱
    错误：
        400 - 邮箱格式非法
        503 - 邮件通知未启用 / SMTP 凭据缺失
        500 - SMTP 实际发送失败
    """
    email = _validate_email_str(payload.email)
    sender = container.email_sender
    if sender is None or not sender.enabled:
        raise HTTPException(
            status_code=503,
            detail="邮件通知未启用或 SMTP 凭据未配置（检查 ENABLE_EMAIL_NOTIFICATION / SMTP_USER / SMTP_PASSWORD）",
        )
    try:
        await sender.send_test_email(email)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"测试邮件发送失败：{exc}"
        ) from exc
    return {"ok": True, "to": email}
