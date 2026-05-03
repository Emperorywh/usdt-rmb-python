"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/eth_analysis"
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # OKX
    okx_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    okx_rest_url: str = "https://www.okx.com"
    symbols: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["ETH-USDT-SWAP"]
    )
    exchanges: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["okx"]
    )

    # DeepSeek / LangChain
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_timeout: int = 30

    # Signal / factor params
    signal_interval_seconds: int = 30
    factor_window_seconds: int = 300
    orderbook_depth: int = 5
    liquidity_wall_multiplier: float = 3.0
    rest_poll_interval_seconds: int = 60

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("symbols", "exchanges", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
