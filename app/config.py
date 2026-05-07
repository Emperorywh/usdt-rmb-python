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
    # 连接池空闲连接最大存活时间（秒）
    # ----------------------------------------------------------------
    # Windows / 跨网络环境下，OS 经常会在长空闲后悄悄断掉 TCP 半连接，
    # 但连接池仍以为这条连接活着。下次 executemany 才会触发
    # WinError 121 + ConnectionDoesNotExistError，整批数据丢失。
    # 把空闲存活时间压到 60s，让连接池主动回收并新建，避开僵尸连接。
    # asyncpg 默认 300s，对高频写入场景偏大。
    db_max_inactive_connection_lifetime: float = 60.0
    # 数据库写操作遇到瞬时连接错误时的最大重试次数（不含首次执行）
    # ----------------------------------------------------------------
    # 仅对幂等写入生效：所有走 ON CONFLICT DO NOTHING / UPDATE 的批量
    # 写入即便部分行已写入再重试也不会产生脏数据。设为 0 可彻底关闭。
    db_write_max_retries: int = 2
    # 写操作首次重试前的等待秒数（之后按指数退避翻倍）
    db_write_retry_backoff: float = 0.2

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

    # ===== P0 多周期 / 爆仓 / 结构化交易计划 =====
    # 主功能开关：True 走多周期因子矩阵 + 结构化 TradingSignal；
    #             False 一键回退到老的 30 分钟单层因子聚合器，
    #             用于灰度回滚（老 LLM prompt + 老 schema 自动兼容）。
    enable_mtf_factors: bool = True
    # 多周期 K 线增量任务每个周期的轮询节奏（秒）
    # 5m 桶分辨率 60s 已足够，做 1s/10s/60s 三档区分主要为了降总 CPU。
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
    # cvd_price_divergence 用的"N 期新高"长度
    mtf_divergence_lookback: int = 20
    # 爆仓滚动窗口分钟数（cascade_signal 用）
    liquidation_windows_minutes: List[int] = Field(default_factory=lambda: [5, 15, 60])
    # cascade_signal 阈值：当前 1m 爆仓 size > 过去 1h 均值 × N 即视为级联
    liquidation_cascade_multiplier: float = 5.0
    # 风险预算占比（用于 LLM 计算 position_size_pct 的提示，本身不强制注入）
    account_risk_budget_pct: float = 0.01

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

    # ===== P1 升级：订单簿时序 / 持仓比 / regime / 自适应节流 =====
    # 4 个独立 enable_* 开关，便于灰度上线；任一关闭都会回退到 P0 行为。
    # ----------------------------------------------------------------
    # 订单簿时序快照：False 时 runner 不写 orderbook_metrics，
    # orderbook 因子层退化为"仅最新一条快照"的 P0 行为。
    enable_orderbook_timeseries: bool = True
    # OKX 散户多空比 / 精英持仓比：False 时 runner 不再启动对应轮询，
    # derivatives 因子层 account_long_short_ratio 等字段保持 None。
    enable_position_ratios: bool = True
    # 市场状态检测：False 时因子矩阵不挂载 regime 字段（None），
    # LLM Prompt 中 regime 段固定输出 "regime: -"。
    enable_regime: bool = True
    # 自适应 LLM 节流：False 时 compute_min_interval 直接返回 P0 的
    # llm_min_interval_seconds（默认 900s）。
    enable_adaptive_throttle: bool = True

    # orderbook_metrics 写入节流（秒）。比 orderbook_snapshots 的 5s
    # 更长一档，因为时序因子只需要相对稀疏的样本来回归斜率与 zscore。
    orderbook_metrics_min_interval_seconds: float = 10.0
    # orderbook_metrics 因子读取窗口 / 历史回看长度
    orderbook_metrics_window_seconds: int = 900  # 15 分钟
    orderbook_metrics_baseline_seconds: int = 3600  # 1 小时（vacuum 比基准）

    # 持仓比 REST 拉取节奏（秒），默认 5 分钟。OKX rubik 接口本身就是
    # 5m 粒度，再短意义不大。
    position_ratios_poll_interval_seconds: int = 300
    # 持仓比拉取的回看周期（OKX 接口的 period 参数）
    position_ratios_period: str = "5m"
    # 持仓比拉取失败的连续容忍次数；超过仍只 warn 不抛
    position_ratios_max_consecutive_errors: int = 5

    # funding 分位数计算的回看时长（秒），默认 7 天
    funding_pct_rank_window_seconds: int = 7 * 86_400

    # 自适应 LLM 节流上下限（秒），与 P0 的 llm_min_interval_seconds 配合
    llm_min_interval_seconds_min: int = 900   # 10 分钟，暴动行情下限
    llm_min_interval_seconds_max: int = 1800  # 30 分钟，平静期上限

    # regime 判定阈值；公开成 setting 是为了未来可调，不要在代码里硬编码。
    regime_adx_trending_threshold: float = 25.0
    regime_adx_ranging_threshold: float = 18.0

    # 流动性地图：整数关口的格子大小（USD）；按 50 美元一档对 ETH 合理
    liquidity_round_level_step_usd: float = 50.0
    # 流动性地图每个方向最多保留 N 档
    liquidity_max_levels_per_side: int = 5
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

    # ===== P2 升级：IC 自适应权重 + 信号生命周期 + LLM 自我反馈 =====
    # 三个独立 enable_* 开关，便于灰度上线。任一关闭都不会破坏 P0/P1 行为。
    # ----------------------------------------------------------------
    # 因子权重表：True 时 RuleEngine 走 factor_weights 表 + 缓存；
    #             False 时退回 P1 的硬编码权重（capital_flow / orderbook /
    #             derivatives / market_structure 四类等权）。
    enable_factor_weights_table: bool = True
    # 信号生命周期跟踪：True 时 service 写入 signal_lifecycle、后台跑跟踪任务；
    #                    False 时不写表也不跑跟踪，对 P0/P1 信号生成完全透明。
    enable_lifecycle_tracking: bool = True
    # LLM 自我反馈：True 时 analyze() 在 prompt 开头注入近 5 条已结算成绩单
    #              + 应用"近 5 条 ≥3 次失败必须降低 confidence"等强约束。
    #              False 时回退到 P1 prompt（不注入历史 PnL）。
    enable_llm_self_feedback: bool = True

    # IC 校准任务（影子模式）：
    # True 时 IC 任务计算结果只写表与日志，但 RuleEngine 不读 factor_weights，
    # 仍走 P1 兜底权重。两周 shadow 期对比新旧权重质量后再切到 False。
    ic_calibrator_shadow_mode: bool = True
    # IC 任务运行周期（秒），默认每天 02:00 UTC 跑一次（86400s）；
    # 为了避免引入 cron / APScheduler，启动后简单 sleep 86400s 循环跑。
    ic_calibrator_interval_seconds: int = 86_400
    # IC 校准任务每天首次启动相对 lifespan 的延迟（秒）：
    # 默认延后 30 秒，避免与冷启动 warmup / 数据回灌竞争 IO。
    ic_calibrator_first_delay_seconds: int = 30
    # IC 校准最小总样本数（30 天内全 LLM 信号）：< 该值整轮跳过校准，
    # 避免冷启动期权重崩坏。
    ic_calibrator_min_total_samples: int = 30
    # 单个 (regime, timeframe) 组合的最小样本数：< 该值不更新该组合，
    # 保留基线权重。
    ic_calibrator_min_group_samples: int = 10
    # IC 校准任务的 admin 触发 token，用于 POST /admin/calibrate-ic。
    # 留空则禁用该接口（避免误暴露重计算入口）。
    ic_calibrator_admin_token: str = ""
    # IC 校准任务输出日志目录（相对项目根 / 绝对路径均可）
    ic_calibrator_log_dir: str = "logs"

    # 生命周期跟踪任务运行节奏（秒）。30s 与 signal_interval 对齐，
    # 不会在新信号入库后等过久才进入"pending → triggered"。
    lifecycle_tick_seconds: int = 30
    # 信号默认有效期（**分钟**）：超过后强制按当前价 exit + 置 expired。
    # ----------------------------------------------------------------
    # 设计意图：
    #   原值 24 小时（lifecycle_default_ttl_hours=24）与 LLM 节流的 15–30 分钟
    #   生成节奏严重错配——24 小时内会有 48–96 条 lifecycle 行同时未结算，
    #   绝大多数 pending 信号的 entry_zone 在生成几分钟后就已经"过期"
    #   （价格走开后 entry_zone 不会随之漂移），最后只能等 24 小时后批量 expired，
    #   导致"近 5 次成绩单"长期被 expired 占满，胜率假性归零。
    # 90 分钟取舍：
    #   - 上限 = ~3 个 LLM 信号生成周期（默认 30 min × 3 = 90 min），
    #     给 entry_zone 一个合理的"等待预算"；
    #   - 配合 service.py 在新信号生成时 supersede 同 symbol 旧 pending 行
    #     （置 invalidated），让"未触发就作废"成为常态而非异常。
    # 兼容：保留 lifecycle_default_ttl_hours 字段读取，仅当显式配置为非默认值
    #       时才仍走小时通道；优先级 minutes > hours。
    lifecycle_default_ttl_minutes: int = 90
    # （deprecated）信号默认有效期（小时）—— 保留供历史 .env / 配置覆盖；
    # 新代码请使用 lifecycle_default_ttl_minutes。
    lifecycle_default_ttl_hours: int = 24
    # mark price 数据源的最大年龄（秒）：超过此值跳过本轮，避免 K 线断流误结算。
    lifecycle_mark_price_max_age_seconds: float = 60.0

    # LLM 自我反馈：注入到 prompt 的"成绩单"条数
    llm_feedback_recent_n: int = 5

    # ===== 邮件通知（Resend HTTP API）=====
    # ----------------------------------------------------------------
    # 当 LLM 输出明确方向（bias=long / short）且本次为真正的 LLM 调用
    # （from_cache=False）时，向 notification_emails 表里所有 enabled=true
    # 的邮箱发送一封 HTML 格式的提醒。observe / neutral 不发邮件，避免噪声。
    #
    # 实现：调用 Resend 的 HTTPS REST API（resend.Emails.send），不再走
    # 传统 SMTP 25/465/587 端口，规避了国内云主机出口 SMTP 端口被封禁
    # 以及 STARTTLS 抖动等问题；阻塞调用通过 asyncio.to_thread 丢到默认
    # executor 上执行，不会拖慢事件循环。
    #
    # 主开关：ENABLE_EMAIL_NOTIFICATION=false 时整个邮件通知链路降级
    # （即使 RESEND_API_KEY 已配置也不会发送，方便本地开发屏蔽）。
    enable_email_notification: bool = False
    # Resend API Key（在 https://resend.com/api-keys 申请，re_ 开头）。
    # 留空时整个邮件通知链路降级为 no-op。
    resend_api_key: str = ""
    # 发件人地址，可填以下两种格式之一：
    #   - "noreply@your-domain.com"                           （仅邮箱）
    #   - "ETH 量化交易系统 <noreply@your-domain.com>"          （带显示名）
    # 必须使用 Resend 控制台已 DNS 验证过的域名，否则 API 会返回
    # 403 invalid_from_address。
    # 注意：``onboarding@resend.dev`` 是 Resend 官方测试地址，
    #       只能发到账号注册邮箱本人，发给其它收件人会被直接拒绝，
    #       不适合生产 / 多收件人场景。
    resend_from: str = "ETH 量化交易系统 <noreply@your-domain.com>"
    # Resend HTTP 调用超时（秒）。SDK 默认无超时，这里加一道保险
    # 防止异常情况下后台任务卡死。当前未真正传给 SDK，仅作语义占位
    # （SDK 暂不暴露 timeout 参数，靠业务层 to_thread + asyncio 的整体
    # 调度兜底）。保留字段是为将来 SDK 升级后零成本接入。
    resend_timeout: float = 20.0

    # ===== P3 升级：决策层守门员 + 离线评估系统 =====
    # ----------------------------------------------------------------
    # 设计动机：
    #   实盘观察发现 LLM 在窄幅震荡市里"叙事质量很好但方向预测接近随机"，
    #   连续 4 条信号在 90 分钟内 long → short → short → long 来回翻转，
    #   把"信号质量"完全交给 LLM 是反量化的。
    #   P3 升级把"是否值得让 LLM 出手"以及"出手后是否值得真的下到 25%
    #   仓位"两件事**前置到服务端硬规则**，LLM 仍然负责方向 + 结构 +
    #   叙事，但不再决定"出不出"和"押多大"。
    # 总开关：
    #   enable_decision_gates=False 时所有 4 道闸门 + size 覆盖全部跳过，
    #   行为退回 P2（与本轮升级前完全一致），便于灰度回滚。
    enable_decision_gates: bool = True
    # 闸 1（ATR 门禁）：15m ATR 占当前价比例（atr_14 / last_close）
    # 低于该值视为窄幅震荡，跳过 LLM 调用直接输出 neutral。
    # 0.0025 = 0.25%。ETH 永续 1m 随机噪声大约 0.05–0.15%，
    # 当 15m ATR 跌到 0.25% 以下意味着"任何 SL 都是高频陷阱"。
    decision_min_atr_pct_15m: float = 0.0025
    # 闸 3（方向稳定性）：当前 LLM 输出方向与最近 1 条信号相反时，
    # 必须满足"价格变动 ≥ 该倍数 × ATR(1h)"才允许翻转，否则降级 neutral。
    # 0.6 是经验取舍：太大容易错过真实反转；太小起不到去噪效果。
    decision_direction_flip_min_price_move_atr_1h: float = 0.6
    # 闸 2（冷静期）：连续 N 次 sl_hit 触发冷静期。
    decision_cooldown_consecutive_sl_threshold: int = 2
    # 冷静期时长（分钟）。期间所有方向输出强制降为 neutral。
    decision_cooldown_minutes: int = 60
    # size 覆盖：服务端最终 clamp 上限（覆盖 schema 里 0.25 的硬上限）。
    # 即便 LLM 给到 0.25、conf=0.9，最大也只能下 0.10。
    decision_max_position_size_pct: float = 0.10
    # size 覆盖：凯利激进度（0.5 = 半凯利，业界常用保守值）。
    decision_kelly_aggressiveness: float = 0.5
    # 决策层最低 RR 业务门槛（schema 里仍保留 1.5 作为最后安全网）。
    # 2.0 取舍依据：胜率 < 50% 时 RR=1.5 即负期望（p × R - (1-p) < 0
    # 在 R=1.5 下需要 p > 0.4），抬到 2.0 让 p > 1/3 即可正期望。
    decision_min_rr_ratio: float = 2.0
    # SL 最小距离（× ATR(15m)）。1.5 × ATR(15m) 足以避开多数插针扫损。
    decision_min_sl_distance_atr_15m: float = 1.5
    # SL 最小距离绝对下限（占入场价比例）。0.5% 是 ETH 永续的经验下限。
    decision_min_sl_distance_pct: float = 0.005
    # 闸 4（规则 vs LLM 冲突保护）：回溯近 N 条已结算信号判定胜率。
    decision_rule_llm_conflict_window: int = 5
    # 闸 4：胜率阈值。LLM 与规则引擎反向且历史胜率低于此值时降级 neutral。
    decision_rule_llm_conflict_winrate_threshold: float = 0.4
    # 离线评估系统总开关：False 时不启动 SignalEvaluator 后台任务，
    # prompt 中评估段也降级为"无数据"。
    enable_signal_evaluation: bool = True
    # 评估任务运行节奏（秒）。5 分钟一轮足够：indicator 是分钟级聚合，
    # 不需要每 30 秒重算。
    signal_evaluation_interval_seconds: int = 300
    # 评估窗口（分钟列表）。1h / 6h / 24h 三档：1h 反映短期波段，
    # 6h 反映半日态势，24h 反映系统级稳定性（喂回 LLM prompt 用 24h）。
    signal_evaluation_windows_minutes: Annotated[List[int], NoDecode] = Field(
        default_factory=lambda: [60, 360, 1440]
    )

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

    @field_validator("liquidation_windows_minutes", mode="before")
    @classmethod
    def _split_int_csv(cls, v):
        """
        允许从环境变量传入逗号分隔的整数串（如 "5,15,60"）
        ----------------------------------------------------------
        参数：
            v: 原始值，可能是字符串、列表或 None
        返回：
            int 列表
        """
        if isinstance(v, str):
            return [int(item.strip()) for item in v.split(",") if item.strip()]
        return v

    @field_validator("signal_evaluation_windows_minutes", mode="before")
    @classmethod
    def _split_eval_windows_csv(cls, v):
        """
        允许从环境变量传入逗号分隔的整数串（如 "60,360,1440"）
        ----------------------------------------------------------
        参数：
            v: 原始值，可能是字符串、列表或 None
        返回：
            int 列表
        说明：
            与 liquidation_windows_minutes 共用同一套语义，但单独写一个 validator
            是为了避免"两个字段共用同一 validator 时 Pydantic 的兼容性问题"。
        """
        if isinstance(v, str):
            return [int(item.strip()) for item in v.split(",") if item.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
