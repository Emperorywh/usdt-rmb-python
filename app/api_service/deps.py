"""FastAPI dependencies that pull objects out of the AppContainer."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from app.container import AppContainer
from app.factor_engine.aggregator import FactorAggregator
from app.signal_engine.service import SignalService


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return container


def get_signal_service(
    container: AppContainer = Depends(get_container),
) -> SignalService:
    return container.signal_service


def get_factor_aggregator(
    container: AppContainer = Depends(get_container),
) -> FactorAggregator:
    return container.factor_aggregator
