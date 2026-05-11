"""应用配置（LLM-First 架构）。

核心设计：
=========
经过 LLM-First 重构后，本配置文件只剩 **2 个 enable_* 开关**：

* ``enable_mtf_factors``        ：多周期因子矩阵（永远开，保留字段是为了配置
                                  快速回滚到老的单层因子聚合器以排障）
* ``enable_email_notification`` ：邮件通知主开关

以下 P2 / P3 / IC / lifecycle / decision-gate 相关字段全部已删除：

- enable_factor_weights_table / enable_lifecycle_tracking /
  enable_llm_self_feedback
- enable_decision_gates / decision_* （4 道闸门 + size 覆盖 + 缓存漂移）
- enable_signal_evaluation / signal_evaluation_*
- enable_llm_first_* 8 个灰度 flag（重构后不再需要灰度，全部行为成为常态）
- ic_calibrator_* / lifecycle_* / rule_* 阈值
- llm_min_interval_seconds_min/max / llm_first_min_interval_* 自适应节流

唯一保留下来的"服务端干预"是 ``decision_min_atr_pct_15m``——ATR floor，
极端波动率不可交易的数学底线，由 ``service.py`` 单条 if 实现。
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class Settings(BaseSettings):
    """全局配置（pydantic-settings 自动从 .env / 环境变量加载）"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==================================================================
    # 数据库
    # ==================================================================
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/eth_analysis"
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    # 连接池空闲连接最大存活时间（秒）
    # ----------------------------------------------------------------
    # Windows / 跨网络环境下，OS 经常会在长空闲后悄悄断掉 TCP 半连接，
    # 但连接池仍以为这条连接活着。下次 executemany 才会触发
    # WinError 121 + ConnectionDoesNotExistError，整批数据丢失。
    # 把空闲存活时间压到 60s，让连接池主动回收并新建，避开僵尸连接。
    # asyncpg 默认 300s，对高频写入场景偏大。
    db_max_inactive_connection_lifetime: float = 60.0
    # 数据库写操作遇到瞬时连接错误时的最大重试次数（不含首次执行）
    db_write_max_retries: int = 2
    db_write_retry_backoff: float = 0.2

    # ==================================================================
    # OKX 行情
    # ==================================================================
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
    # 是否让 httpx 读取系统代理/证书环境变量（HTTP_PROXY 等）。
    # 默认 False：实测系统代理在 TLS 握手时经常失败，直连 OKX 更稳。
    okx_rest_trust_env: bool = False
    # 显式代理 URL（留空表示不使用），优先级高于 trust_env
    # 例：http://127.0.0.1:7890 或 http://user:pass@host:port
    okx_rest_proxy: str = ""

    # ==================================================================
    # DeepSeek / LangChain
    # ==================================================================
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    # 默认使用 DeepSeek-V4-Pro，支持思考模式
    deepseek_model: str = "deepseek-v4-pro"
    # 思考模式：先输出思维链（reasoning_content），再给最终回答。
    # 注意：思考模式下 temperature/top_p/presence_penalty/frequency_penalty
    # 不会生效（不报错但被忽略）。
    deepseek_thinking_enabled: bool = True
    # 思考强度（仅在 deepseek_thinking_enabled=True 时生效），可选 high / max。
    deepseek_reasoning_effort: str = "high"
    # LLM temperature（仅在思考模式关闭时生效）
    llm_temperature: float = 0.2
    # LLM 单次调用超时（秒）。思考模式推理较慢，建议放大到 120s 起。
    llm_timeout: int = 300
    # LLM 调用最小间隔（秒）。在该窗口内同 symbol 复用上一次 LLM 返回，
    # 避免高频付费调用。设为 0 可调试期实时调用。
    llm_min_interval_seconds: int = 900

    # ==================================================================
    # 信号 / 因子参数
    # ==================================================================
    signal_interval_seconds: int = 30
    # 因子窗口：默认 30 分钟。窗口太短会导致 market_structure 凑不齐 6 根
    # 1 分钟 bar，oi_change_pct 也只能取到 1-2 个样本。
    factor_window_seconds: int = 1800
    orderbook_depth: int = 5
    liquidity_wall_multiplier: float = 3.0
    rest_poll_interval_seconds: int = 60

    # ATR 极端风控底线：15m ATR 占当前价比例（atr_14 / last_close）
    # 低于该值（默认 0.25%）视为窄幅震荡 / 无可交易波动率，直接跳过
    # LLM 调用并输出 neutral。这是 LLM-First 架构下**唯一**保留的
    # 服务端"干预"，理由是"任何 SL 都是高频陷阱"——数学上不可交易。
    # 设为 0 / 负数即可完全关闭该底线（不推荐）。
    decision_min_atr_pct_15m: float = 0.0025

    # ==================================================================
    # 多周期 / 爆仓 / 结构化交易计划
    # ==================================================================
    # 多周期 K 线增量任务每个周期的轮询节奏（秒）
    kline_tick_seconds_1m: float = 1.0
    kline_tick_seconds_5m: float = 1.0
    kline_tick_seconds_15m: float = 10.0
    kline_tick_seconds_1h: float = 10.0
    kline_tick_seconds_4h: float = 60.0
    kline_tick_seconds_1d: float = 60.0
    # market_structure 每个周期回看的 K 线根数（HH/HL pivot + ATR）
    mtf_lookback_bars: int = 80
    # capital_flow.volume_zscore 用的滚动均值长度
    mtf_volume_zscore_window: int = 30
    # cvd_price_divergence 用的 "N 期新高" 长度
    mtf_divergence_lookback: int = 20
    # 爆仓滚动窗口分钟数（cascade_signal 用）
    liquidation_windows_minutes: List[int] = Field(default_factory=lambda: [5, 15, 60])
    # cascade_signal 阈值：当前 1m 爆仓 size > 过去 1h 均值 × N 即视为级联
    liquidation_cascade_multiplier: float = 5.0
    # 风险预算占比（用于 LLM 计算 position_size_pct 的提示）
    account_risk_budget_pct: float = 0.01

    # 订单簿写入节流：每个 symbol 至少间隔 N 秒落库一次
    orderbook_min_interval_seconds: float = 5.0
    # orderbook_metrics 时序快照写入节流（秒）
    orderbook_metrics_min_interval_seconds: float = 10.0
    # orderbook_metrics 因子读取窗口 / 历史回看长度
    orderbook_metrics_window_seconds: int = 900
    orderbook_metrics_baseline_seconds: int = 3600

    # 持仓比 REST 拉取节奏（秒），默认 5 分钟
    position_ratios_poll_interval_seconds: int = 300
    # 持仓比拉取的回看周期（OKX 接口的 period 参数）
    position_ratios_period: str = "5m"
    # 持仓比拉取失败的连续容忍次数
    position_ratios_max_consecutive_errors: int = 5

    # funding 分位数计算的回看时长（秒），默认 7 天
    funding_pct_rank_window_seconds: int = 7 * 86_400

    # regime 判定阈值
    regime_adx_trending_threshold: float = 25.0
    regime_adx_ranging_threshold: float = 18.0

    # 流动性地图：整数关口的格子大小（USD），按 50 美元一档对 ETH 合理
    liquidity_round_level_step_usd: float = 50.0
    # 每个方向最多保留 N 档
    liquidity_max_levels_per_side: int = 5
    # 合约面值（ctVal）默认值；启动时若能从 OKX instruments 接口拿到则覆盖
    default_contract_value: float = 0.1

    # ==================================================================
    # 数据保留策略（防止硬盘被高频行情撑爆）
    # ==================================================================
    # trades 表保留时长（秒），默认 24 小时
    retention_trades_seconds: int = 86_400
    # orderbook_snapshots 表保留时长（秒），默认 24 小时
    retention_orderbook_seconds: int = 86_400
    # signals 表保留时长（秒），默认 30 天；设为 0 表示永不清理
    retention_signals_seconds: int = 30 * 86_400
    # 后台清理任务的运行周期（秒），默认 600 秒（10 分钟）一次；
    # 设为 0 可彻底关闭清理任务（仅当外部已有清理脚本时使用）。
    retention_run_interval_seconds: int = 600

    # ==================================================================
    # 邮件通知（Resend HTTP API）
    # ==================================================================
    # 当 LLM 输出明确方向（bias=long / short）且本次为真正的 LLM 调用
    # （from_cache=False）时，向 notification_emails 表里所有 enabled=true
    # 的邮箱发送一封 HTML 提醒。observe / neutral 不发邮件。
    enable_email_notification: bool = False
    # Resend API Key（re_ 开头），留空时整个邮件通知链路降级为 no-op
    resend_api_key: str = ""
    # 发件人地址，可填以下两种格式之一：
    #   - "noreply@your-domain.com"
    #   - "ETH 量化交易系统 <noreply@your-domain.com>"
    # 必须使用 Resend 控制台已 DNS 验证过的域名，否则 API 会返回
    # 403 invalid_from_address。
    resend_from: str = "ETH 量化交易系统 <noreply@your-domain.com>"
    # Resend HTTP 调用超时（秒）；SDK 当前不直接支持，保留为占位
    resend_timeout: float = 20.0

    # ==================================================================
    # API
    # ==================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("symbols", "exchanges", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("liquidation_windows_minutes", mode="before")
    @classmethod
    def _split_int_csv(cls, v):
        """允许从环境变量传入逗号分隔的整数串（如 "5,15,60"）"""
        if isinstance(v, str):
            return [int(item.strip()) for item in v.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
