"""应用配置（LLM-First 架构，子配置类分组）。

配置按功能域拆为子配置类，主 ``Settings`` 组合这些子类。
所有子配置类设 ``env_prefix=""`` ，使 ``.env`` 顶层变量直接映射。

访问方式：
    settings.db.database_url
    settings.ingestion.symbols
    settings.llm.deepseek_api_key
    ...

向后兼容：主 Settings 也暴露顶层字段别名，方便迁移期混用。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class DatabaseSettings(BaseSettings):
    """数据库连接池相关配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/eth_analysis"
    )
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_pool_acquire_timeout: float = 30.0
    # Windows / 跨网络环境下，OS 经常会在长空闲后悄悄断掉 TCP 半连接，
    # 把空闲存活时间压到 60s，让连接池主动回收并新建，避开僵尸连接。
    db_max_inactive_connection_lifetime: float = 60.0
    db_write_max_retries: int = 2
    db_write_retry_backoff: float = 0.2


class IngestionSettings(BaseSettings):
    """数据采集相关配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    okx_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    okx_rest_url: str = "https://www.okx.com"
    symbols: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["ETH-USDT-SWAP"]
    )
    exchanges: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["okx"]
    )
    # OKX REST 网络行为
    okx_rest_timeout: float = 10.0
    okx_rest_max_retries: int = 3
    okx_rest_retry_backoff: float = 0.8
    okx_rest_trust_env: bool = False
    okx_rest_proxy: str = ""
    # WS / Watchdog / 熔断
    ws_ping_interval_seconds: float = 25.0
    ws_stale_funding_seconds: float = 300.0
    ws_stale_oi_seconds: float = 60.0
    watchdog_tick_seconds: float = 15.0
    watchdog_grace_seconds: float = 30.0
    breaker_base_cooldown_seconds: float = 60.0
    breaker_max_cooldown_seconds: float = 900.0
    trade_flush_interval_seconds: float = 1.0
    liquidation_flush_interval_seconds: float = 1.0
    # 订单簿写入节流
    orderbook_depth: int = 5
    orderbook_min_interval_seconds: float = 5.0
    orderbook_metrics_min_interval_seconds: float = 10.0
    orderbook_metrics_window_seconds: int = 900
    orderbook_metrics_baseline_seconds: int = 3600
    # 合约面值
    default_contract_value: float = 0.1
    rest_poll_interval_seconds: int = 60

    @field_validator("symbols", "exchanges", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class FactorSettings(BaseSettings):
    """因子引擎相关配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    factor_window_seconds: int = 1800
    mtf_lookback_bars: int = 80
    enable_mtf_factors: bool = True
    liquidity_wall_multiplier: float = 3.0
    # K 线聚合 tick 频率
    kline_tick_seconds_1m: float = 1.0
    kline_tick_seconds_5m: float = 1.0
    kline_tick_seconds_15m: float = 10.0
    kline_tick_seconds_1h: float = 10.0
    kline_tick_seconds_4h: float = 60.0
    kline_tick_seconds_1d: float = 60.0
    # 多周期因子参数
    mtf_volume_zscore_window: int = 30
    mtf_divergence_lookback: int = 20
    # 爆仓因子参数
    liquidation_windows_minutes: List[int] = Field(default_factory=lambda: [5, 15, 60])
    liquidation_cascade_multiplier: float = 5.0
    # Funding 分位数窗口
    funding_pct_rank_window_seconds: int = 7 * 86_400
    # 市场状态判定
    regime_adx_trending_threshold: float = 25.0
    regime_adx_ranging_threshold: float = 18.0
    # 流动性地图
    liquidity_round_level_step_usd: float = 50.0
    liquidity_max_levels_per_side: int = 5
    # 订单簿指标窗口
    orderbook_metrics_window_seconds: int = 900
    orderbook_metrics_baseline_seconds: int = 3600

    @field_validator("liquidation_windows_minutes", mode="before")
    @classmethod
    def _split_int_csv(cls, v):
        """允许从环境变量传入逗号分隔的整数串（如 "5,15,60"）"""
        if isinstance(v, str):
            return [int(item.strip()) for item in v.split(",") if item.strip()]
        return v


class LLMSettings(BaseSettings):
    """LLM 相关配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_thinking_enabled: bool = True
    deepseek_reasoning_effort: str = "high"
    llm_temperature: float = 0.2
    llm_timeout: int = 300
    llm_min_interval_seconds: int = 1800


class RetentionSettings(BaseSettings):
    """数据保留策略配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    retention_trades_seconds: int = 86_400
    retention_orderbook_seconds: int = 86_400
    retention_signals_seconds: int = 30 * 86_400
    retention_funding_seconds: int = 90 * 86_400
    retention_oi_seconds: int = 90 * 86_400
    retention_run_interval_seconds: int = 600


class EmailSettings(BaseSettings):
    """邮件通知相关配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    enable_email_notification: bool = False
    resend_api_key: str = ""
    resend_from: str = "ETH 量化交易系统 <noreply@your-domain.com>"
    resend_timeout: float = 20.0


class SignalSettings(BaseSettings):
    """信号风控相关配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    decision_min_atr_pct_15m: float = 0.0025
    signal_interval_seconds: int = 30
    account_risk_budget_pct: float = 0.01


class Settings(BaseSettings):
    """应用全局配置（组合子配置）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 子配置
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    factor: FactorSettings = Field(default_factory=FactorSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    signal: SignalSettings = Field(default_factory=SignalSettings)

    # 应用级配置（顶层字段，不嵌套）
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
