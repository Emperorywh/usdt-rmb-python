"""Pydantic schema for the trading signal output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TradingSignal(BaseModel):
    """Strict JSON schema returned by both rule engine and LLM agent."""

    bias: Literal["long", "short", "neutral"] = Field(
        description="Directional bias for the next short-term horizon."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score in [0, 1] for the chosen bias.",
    )
    reason: str = Field(
        description="Concise reasoning that cites the strongest factors.",
    )
    risk: str = Field(
        description="Primary risks / invalidation conditions to watch.",
    )
    suggestion: str = Field(
        description="Actionable trading suggestion (entry, stop, size). Advisory only.",
    )

    model_config = {"extra": "forbid"}
