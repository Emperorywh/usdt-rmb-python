"""HTTP routes（P0/P1/P2 共用）。

P2 新增接口：
* GET  /signals/{signal_id}/attribution   信号因子贡献度归因
* GET  /factors/weights/current           当前生效的权重表（按 regime 过滤）
* GET  /signals/lifecycle/stats           近 N 天信号胜率 / 平均 RR / 各 regime 命中率
* POST /admin/calibrate-ic                手动触发一次 IC 校准（带 token）
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.api_service.deps import (
    get_container,
    get_factor_aggregator,
    get_signal_service,
)
from app.container import AppContainer
from app.factor_engine.aggregator import FactorAggregator
from app.signal_engine.service import SignalService

router = APIRouter()


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
