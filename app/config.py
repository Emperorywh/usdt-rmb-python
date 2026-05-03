"""应用配置：从环境变量 / .env 文件加载。"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class Settings(BaseSettings):
    """
    全局配置
    -------------------------------------------------------------------
    通过 pydantic-settings 从 .env / 环境变量加载，所有阈值、URL、
    Key 都集中在这里，业务代码不再硬编码。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 数据库 =====
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/eth_analysis"
    )
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10

    # ===== OKX 行情 =====
    okx_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    okx_rest_url: str = "https://www.okx.com"
    symbols: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["ETH-USDT-SWAP"]
    )
    exchanges: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["okx"]
    )

    # ===== OKX REST 网络行为 =====
    # 单次 REST 请求超时（秒）
    okx_rest_timeout: float = 10.0
    # 最大重试次数，实际总请求数 = 1 + max_retries
    okx_rest_max_retries: int = 3
    # 重试指数退避基数（秒）：第 n 次重试等待 backoff * 2^n 秒
    okx_rest_retry_backoff: float = 0.8
    # 是否让 httpx 读取系统代理/证书环境变量（HTTP_PROXY 等）
    # 默认 False：实测系统代理在 TLS 握手时经常失败，直连 OKX 更稳
    okx_rest_trust_env: bool = False
    # 显式代理 URL（留空表示不使用），优先级高于 trust_env
    # 例：http://127.0.0.1:7890 或 http://user:pass@host:port
    okx_rest_proxy: str = ""

    # ===== DeepSeek / LangChain =====
    # DeepSeek API Key（OpenAI 兼容协议）
    deepseek_api_key: str = ""
    # DeepSeek API Base URL
    deepseek_base_url: str = "https://api.deepseek.com"
    # 模型名：默认使用 DeepSeek-V4-Pro，支持思考模式（thinking）
    deepseek_model: str = "deepseek-v4-pro"
    # 思考模式开关。
    # ----------------------------------------------------------------
    # 开启后，模型会先输出一段思维链（reasoning_content），再给出最终
    # 回答（content），可以显著提高复杂判断的准确率，但会增加延迟与
    # token 消耗。本项目的判断需要综合 4 类因子，开启思考模式收益较高。
    # 注意：思考模式下 temperature/top_p/presence_penalty/frequency_penalty
    # 不会生效（不报错但被忽略）。
    deepseek_thinking_enabled: bool = True
    # 思考强度。
    # ----------------------------------------------------------------
    # 仅在 deepseek_thinking_enabled=True 时生效，可选 high / max。
    # （low、medium 会被服务端映射为 high；xhigh 会被映射为 max）
    deepseek_reasoning_effort: str = "high"
    # LLM temperature（仅在思考模式关闭时生效）
    llm_temperature: float = 0.2
    # LLM 单次调用超时（秒）。思考模式推理较慢，建议放大到 120s 起。
    llm_timeout: int = 120
    # LLM 调用最小间隔（秒）。在该窗口内对同一 symbol 的请求会直接
    # 复用上一次 LLM 返回的 TradingSignal，避免高频付费调用。
    # 默认 900 秒（15 分钟），与"市场中短期波动节奏"基本匹配；
    # 若希望调试期实时调用，可设为 0。
    llm_min_interval_seconds: int = 900

    # ===== 信号 / 因子参数 =====
    signal_interval_seconds: int = 30
    # 因子窗口：默认 30 分钟。窗口太短会导致 market_structure 凑不齐
    # 6 根 1 分钟 bar，oi_change_pct 也只能取到 1-2 个样本。
    factor_window_seconds: int = 1800
    orderbook_depth: int = 5
    liquidity_wall_multiplier: float = 3.0
    rest_poll_interval_seconds: int = 60

    # ===== 规则引擎阈值 =====
    # 资金净流入绝对值（USD），超过此值才认为有方向性
    rule_net_flow_usd_threshold: float = 50_000.0
    # 盘口失衡绝对值阈值（[-1, 1] 区间）
    rule_orderbook_imbalance_threshold: float = 0.15
    # 持仓量变动百分比阈值（0.005 = 0.5%）
    rule_oi_change_threshold: float = 0.005
    # 资金费率阈值（0.00005 ≈ 0.005%/8h，过此值认为偏离中性）
    rule_funding_rate_threshold: float = 0.00005

    # 订单簿写入节流：每个 symbol 至少间隔 N 秒落库一次
    # 既保证信号引擎能拿到较新的盘口（N 秒以内），又把存储压力降到合理水平
    orderbook_min_interval_seconds: float = 5.0
    # 合约面值（ctVal）默认值 - 启动时若能从 OKX instruments 接口拿到则覆盖
    # ETH-USDT-SWAP 在 OKX 上 1 张 = 0.1 ETH（可被 instruments 元数据覆盖）
    default_contract_value: float = 0.1

    # ===== 数据保留策略（防止硬盘被高频行情撑爆）=====
    # trades 表保留时长（秒），默认 24 小时。
    # 信号引擎只查最近 factor_window_seconds 范围内的数据，再老的纯属占地方；
    # 保留 24h 主要是为了短期复盘 / 排障，不需要可改小到 2 小时（7200）。
    retention_trades_seconds: int = 86_400
    # orderbook_snapshots 表保留时长（秒），默认 24 小时。
    # 系统只取最新一条做盘口失衡，旧快照同样不会被读到。
    retention_orderbook_seconds: int = 86_400
    # signals 表保留时长（秒），默认 30 天。
    # 信号有审计 / 复盘价值，建议保留更久；设为 0 表示永不清理。
    retention_signals_seconds: int = 30 * 86_400
    # 后台清理任务的运行周期（秒），默认 600 秒（10 分钟）一次。
    # 设为 0 可彻底关闭清理任务（仅当外部已有清理脚本时使用）。
    retention_run_interval_seconds: int = 600

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
