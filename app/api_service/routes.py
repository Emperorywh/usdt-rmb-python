"""HTTP routes."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

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
