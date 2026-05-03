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
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    row = await container.repos.fetch_latest_signal(sym)
    if not row:
        raise HTTPException(status_code=404, detail="No signal yet, try /signal/refresh")
    return {
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
    }


@router.post("/signal/refresh", tags=["signal"])
async def refresh_signal(
    symbol: Optional[str] = Query(default=None),
    container: AppContainer = Depends(get_container),
    signal_service: SignalService = Depends(get_signal_service),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    result = await signal_service.generate(sym)
    return {
        "timestamp": result["factors"]["computed_at"],
        "symbol": sym,
        "source": result["source"],
        "signal": result["signal"],
        "rule_signal": result["rule_signal"],
        "rule_score": result["rule_score"],
    }
