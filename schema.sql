-- ============================================================
-- ETH Trading Analysis Platform - PostgreSQL Schema
-- 在阿里云 PostgreSQL 上执行：psql -h <host> -U <user> -d <db> -f schema.sql
-- ============================================================

-- 1) 逐笔成交
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    price           NUMERIC(24, 10) NOT NULL,
    size            NUMERIC(30, 12) NOT NULL,
    side            TEXT        NOT NULL CHECK (side IN ('buy', 'sell')),
    trade_id        TEXT,
    CONSTRAINT trades_unique UNIQUE (exchange, symbol, trade_id)
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades (ts DESC);

-- 2) 订单簿快照
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    bids            JSONB       NOT NULL,
    asks            JSONB       NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_ts ON orderbook_snapshots (symbol, ts DESC);

-- 3) 资金费率
--    UNIQUE(exchange, symbol, ts) 用来抑制 WS / REST 双路重复写入
CREATE TABLE IF NOT EXISTS funding_rates (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    funding_rate    NUMERIC(24, 12),
    next_funding_ts TIMESTAMPTZ,
    CONSTRAINT funding_rates_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts ON funding_rates (symbol, ts DESC);

-- 4) 持仓量
--    UNIQUE(exchange, symbol, ts) 用来抑制 WS / REST 双路重复写入
CREATE TABLE IF NOT EXISTS open_interest (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    oi              NUMERIC(30, 10),
    oi_ccy          NUMERIC(30, 10),
    CONSTRAINT open_interest_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_oi_symbol_ts ON open_interest (symbol, ts DESC);

-- 5) 链上指标（预留，先用 mock）
CREATE TABLE IF NOT EXISTS onchain_metrics (
    id                  BIGSERIAL PRIMARY KEY,
    ts                  TIMESTAMPTZ NOT NULL,
    exchange_inflow     NUMERIC(30, 10),
    exchange_outflow    NUMERIC(30, 10),
    whale_tx_count      INTEGER,
    gas_fee_gwei        NUMERIC(20, 6),
    burn_rate           NUMERIC(30, 10)
);
CREATE INDEX IF NOT EXISTS idx_onchain_ts ON onchain_metrics (ts DESC);

-- 6) 信号输出
--   reasoning_content：DeepSeek 思考模式下模型先输出的"思维链"原文，
--   仅用于审计 / 复盘，不参与下游决策。普通模式下为 NULL。
CREATE TABLE IF NOT EXISTS signals (
    id                 BIGSERIAL PRIMARY KEY,
    ts                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol             TEXT        NOT NULL,
    bias               TEXT        NOT NULL CHECK (bias IN ('long', 'short', 'neutral')),
    confidence         NUMERIC(6, 4) NOT NULL,
    reason             TEXT,
    risk               TEXT,
    suggestion         TEXT,
    factors            JSONB,
    source             TEXT        NOT NULL DEFAULT 'rules+llm',
    reasoning_content  TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals (symbol, ts DESC);

-- 兼容已有部署：旧库里 signals 表可能没有 reasoning_content 列，
-- 这里用 ADD COLUMN IF NOT EXISTS 做幂等迁移，不会影响新库。
ALTER TABLE signals ADD COLUMN IF NOT EXISTS reasoning_content TEXT;

-- ============================================================
-- P0 升级：爆仓 + 多周期 K 线 + signals 结构化字段
-- ============================================================
-- 设计取舍：
--   1) K 线选择"每个周期一张表"而不是"单表带 timeframe 列"，
--      原因详见 README / 交付说明：
--        - 每个周期的写入热路径完全独立，索引体积更小，
--          (symbol, ts DESC) 单索引即可，不需要复合 (symbol,timeframe,ts)；
--        - 高周期表 (1d / 4h) 行数极少，单独查询时计划器更准确；
--        - 物化视图方案被否：写入侧需要持续 INSERT ... ON CONFLICT
--          DO UPDATE 让"未封盘 bar"实时滚动，PG 物化视图不支持
--          增量更新（CONCURRENTLY 也只能整表刷新），与本场景不匹配。
--      因此采用"应用层按周期增量任务 + 6 张物理表"的方案。
--   2) 表结构 6 份完全一致，便于 repos 用同一段模板代码批量生成。
--   3) UNIQUE(exchange, symbol, ts) 让 ON CONFLICT DO UPDATE 既能
--      持续刷新当前未封盘 bar，也能在重启 / 任务并发时保持幂等。

-- 7) 爆仓
--   说明：
--     - side = 'long' 表示一个多头仓位被强平（OKX 推送的 side=sell）
--     - side = 'short' 表示一个空头仓位被强平（OKX 推送的 side=buy）
--     - notional = price * size（已乘 ctVal 换算成基础币种）
--     - UNIQUE 用 (exchange, symbol, ts, side, price, size) 兜底，
--       因为 OKX liquidation-orders 不带稳定的全局 ID，同一时间戳上
--       多笔不同价格 / 数量的强平也要全部入库。
CREATE TABLE IF NOT EXISTS liquidations (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    side            TEXT        NOT NULL CHECK (side IN ('long', 'short')),
    price           NUMERIC(24, 10) NOT NULL,
    size            NUMERIC(30, 12) NOT NULL,
    notional        NUMERIC(30, 10),
    CONSTRAINT liquidations_unique UNIQUE (exchange, symbol, ts, side, price, size)
);
CREATE INDEX IF NOT EXISTS idx_liquidations_symbol_ts ON liquidations (symbol, ts DESC);

-- 8) 多周期 K 线（5 个周期 + 1m 内部用，共 6 张表，结构完全一致）
--   ts：周期开始时间，对齐到周期边界（UTC）
--   closed：封盘标记。最新一根永远 closed=FALSE，跨边界后置 TRUE 并新建下一根。
--   cvd_close：周期收盘时刻的累计 CVD（buy_size - sell_size），用于跨周期算 cvd_slope。
--   buy_volume / sell_volume：基础币种数量（已乘 ctVal）。
CREATE TABLE IF NOT EXISTS klines_1m (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(24, 10),
    high            NUMERIC(24, 10),
    low             NUMERIC(24, 10),
    close           NUMERIC(24, 10),
    volume          NUMERIC(30, 12),
    buy_volume      NUMERIC(30, 12),
    sell_volume     NUMERIC(30, 12),
    cvd_close       NUMERIC(30, 12),
    trade_count     INTEGER,
    closed          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT klines_1m_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_klines_1m_symbol_ts ON klines_1m (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS klines_5m (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(24, 10),
    high            NUMERIC(24, 10),
    low             NUMERIC(24, 10),
    close           NUMERIC(24, 10),
    volume          NUMERIC(30, 12),
    buy_volume      NUMERIC(30, 12),
    sell_volume     NUMERIC(30, 12),
    cvd_close       NUMERIC(30, 12),
    trade_count     INTEGER,
    closed          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT klines_5m_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_klines_5m_symbol_ts ON klines_5m (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS klines_15m (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(24, 10),
    high            NUMERIC(24, 10),
    low             NUMERIC(24, 10),
    close           NUMERIC(24, 10),
    volume          NUMERIC(30, 12),
    buy_volume      NUMERIC(30, 12),
    sell_volume     NUMERIC(30, 12),
    cvd_close       NUMERIC(30, 12),
    trade_count     INTEGER,
    closed          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT klines_15m_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_klines_15m_symbol_ts ON klines_15m (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS klines_1h (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(24, 10),
    high            NUMERIC(24, 10),
    low             NUMERIC(24, 10),
    close           NUMERIC(24, 10),
    volume          NUMERIC(30, 12),
    buy_volume      NUMERIC(30, 12),
    sell_volume     NUMERIC(30, 12),
    cvd_close       NUMERIC(30, 12),
    trade_count     INTEGER,
    closed          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT klines_1h_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_klines_1h_symbol_ts ON klines_1h (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS klines_4h (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(24, 10),
    high            NUMERIC(24, 10),
    low             NUMERIC(24, 10),
    close           NUMERIC(24, 10),
    volume          NUMERIC(30, 12),
    buy_volume      NUMERIC(30, 12),
    sell_volume     NUMERIC(30, 12),
    cvd_close       NUMERIC(30, 12),
    trade_count     INTEGER,
    closed          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT klines_4h_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_klines_4h_symbol_ts ON klines_4h (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS klines_1d (
    id              BIGSERIAL PRIMARY KEY,
    exchange        TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    open            NUMERIC(24, 10),
    high            NUMERIC(24, 10),
    low             NUMERIC(24, 10),
    close           NUMERIC(24, 10),
    volume          NUMERIC(30, 12),
    buy_volume      NUMERIC(30, 12),
    sell_volume     NUMERIC(30, 12),
    cvd_close       NUMERIC(30, 12),
    trade_count     INTEGER,
    closed          BOOLEAN     NOT NULL DEFAULT FALSE,
    CONSTRAINT klines_1d_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_klines_1d_symbol_ts ON klines_1d (symbol, ts DESC);

-- 9) signals 表追加 P0 结构化交易计划字段
--   全部用 JSONB / NUMERIC，便于在不破坏 v0 schema 的前提下灰度上线。
--   ALTER 必须 IF NOT EXISTS，老库 / 新库通用。
ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_zone               JSONB;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS stop_loss                NUMERIC(24, 10);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS take_profit              JSONB;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS risk_reward_ratio        NUMERIC(10, 4);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS position_size_pct        NUMERIC(8, 6);
ALTER TABLE signals ADD COLUMN IF NOT EXISTS timeframe_alignment      JSONB;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS invalidation_conditions  JSONB;

-- ============================================================
-- P1 升级：订单簿时序 + 主力/散户持仓比
-- ============================================================
-- 设计取舍：
--   1) 订单簿时序：原 orderbook_snapshots 仅保留"最新一条快照"语义，
--      P1 需要"过去 5/15 分钟的盘口动态"。直接把 bids/asks JSONB 全
--      存进时序表会让单 symbol 一天写入 8000+ JSONB 行（10s 节流），
--      JSONB 体积大、读时反序列化慢，索引也不友好。
--      所以新增 orderbook_metrics：只存"已经聚合好的标量指标"
--      （imbalance / wall 计数 / spread 等），单行 < 100 字节，
--      读 5/15 分钟序列只需 ~30/90 行，时序回放成本极低。
--   2) 持仓比：OKX rubik 接口本身就已经按周期聚合（5m/15m/1h/1d），
--      没必要再做二次聚合，只需把每次 REST 拉取的几条历史落库即可。
--      ratio_type 用 CHECK 约束区分 4 种维度，方便未来加新维度。
--   3) 两张表都加 UNIQUE 抑制重复写入，符合 P0 既定的"WS/REST 双路
--      幂等"风格；orderbook_metrics 用 (exchange, symbol, ts) 即可，
--      因为 10s 节流粒度下 ts 已经是天然去重键。

-- 10) 订单簿时序指标
--   说明：
--     - 与 orderbook_snapshots 完全解耦：snapshots 是"原始 bids/asks"快照，
--       供回放与离线复盘；orderbook_metrics 是"已聚合标量"，供因子层时序分析。
--     - imbalance / spread_bp 是开窗口统计（5m / 15m / 1h）的基础列。
--     - wall 计数 / notional 用于检测"大墙撤单"和"流动性真空"。
CREATE TABLE IF NOT EXISTS orderbook_metrics (
    id                   BIGSERIAL PRIMARY KEY,
    exchange             TEXT        NOT NULL,
    symbol               TEXT        NOT NULL,
    ts                   TIMESTAMPTZ NOT NULL,
    imbalance            NUMERIC(10, 6),
    bid_qty              NUMERIC(30, 10),
    ask_qty              NUMERIC(30, 10),
    top5_bid_notional    NUMERIC(30, 4),
    top5_ask_notional    NUMERIC(30, 4),
    bid_wall_count       INTEGER,
    ask_wall_count       INTEGER,
    spread_bp            NUMERIC(12, 4),
    mid_price            NUMERIC(24, 10),
    CONSTRAINT orderbook_metrics_unique UNIQUE (exchange, symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_orderbook_metrics_symbol_ts
    ON orderbook_metrics (symbol, ts DESC);

-- 11) 主力 / 散户多空 / 持仓比
--   ratio_type：
--     'account'             - 散户层多空账户比（按 ccy）
--     'account_contract'    - 散户层多空账户比（按 instId，更精确）
--     'top_trader_account'  - 精英账户多空比（按账户数）
--     'top_trader_position' - 精英账户多空比（按持仓量，更顺指）
--   long_ratio / short_ratio：百分比形式（0~1），来自 OKX 原始
--   ratio：long / short 比值（OKX 部分接口直接提供）
--   设计取舍：
--     - 不为每种 ratio_type 单独建表：4 种维度 schema 完全一致，
--       单表 + ratio_type 列让读侧可以一次拉齐多维度，省一次 JOIN。
--     - UNIQUE 含 ratio_type，避免不同维度互相挤占同一时间戳。
CREATE TABLE IF NOT EXISTS position_ratios (
    id           BIGSERIAL PRIMARY KEY,
    exchange     TEXT        NOT NULL,
    symbol       TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    ratio_type   TEXT        NOT NULL CHECK (
        ratio_type IN (
            'account', 'account_contract',
            'top_trader_account', 'top_trader_position'
        )
    ),
    long_ratio   NUMERIC(10, 6),
    short_ratio  NUMERIC(10, 6),
    ratio        NUMERIC(12, 6),
    CONSTRAINT position_ratios_unique
        UNIQUE (exchange, symbol, ts, ratio_type)
);
CREATE INDEX IF NOT EXISTS idx_position_ratios_symbol_ts
    ON position_ratios (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_position_ratios_symbol_type_ts
    ON position_ratios (symbol, ratio_type, ts DESC);

-- ============================================================
-- P2 升级：因子权重表（IC 自适应） + 信号生命周期表（自我反馈）
-- ============================================================
-- 设计取舍（详见 P2 升级说明文档）：
--   1) 因子权重表 factor_weights 替代代码里硬编码的"按 regime 切换"权重，
--      支持 IC 自动校准任务后台覆盖；同一 (regime, timeframe) 下所有
--      factor_name 的 weight 总和归一为 1.0。
--   2) 信号生命周期表 signal_lifecycle 跟踪每条 signals 行从 pending →
--      triggered → sl_hit / tp1_hit / tp2_hit / expired 的状态机，
--      给 LLM "自我反馈"提供数据基础：每次 analyze() 时回查最近 5 条已结算
--      记录注入到 prompt，让模型看到自己上一轮判断的真实 PnL。
--   3) 两张表都使用 IF NOT EXISTS，旧库/新库幂等；signal_lifecycle 通过 FK
--      ON DELETE CASCADE 与 signals 同生共死，避免 retention 任务删 signals
--      后留下孤儿 lifecycle 行。

-- 12) 因子权重表（替代规则引擎里的硬编码权重）
--   说明：
--     - 同一 (regime, timeframe) 下所有 factor_name 的 weight 总和应 = 1.0
--       （由 IC 校准任务在 UPSERT 时归一），规则引擎读到后直接累加；
--       归一不依赖 DB 约束，避免高频 UPSERT 时 DEFERRABLE 检查的开销。
--     - factor_group 与 factor_name 都是冗余但便于人类阅读 / 归因 API 使用：
--       factor_group ∈ {capital_flow, orderbook, derivatives, market_structure, liquidity}
--     - ic_30d / ic_90d / sample_count 是观测窗口指标，仅用于审计与归因展示。
--     - regime='overall' / timeframe='overall' 作为兜底维度：当具体
--       (regime, timeframe) 没有该 factor 时，规则引擎用它作为后备权重。
CREATE TABLE IF NOT EXISTS factor_weights (
    id              BIGSERIAL PRIMARY KEY,
    regime          TEXT        NOT NULL,
    timeframe       TEXT        NOT NULL,
    factor_group    TEXT        NOT NULL,
    factor_name     TEXT        NOT NULL,
    weight          NUMERIC(10, 6) NOT NULL,
    ic_30d          NUMERIC(10, 6),
    ic_90d          NUMERIC(10, 6),
    sample_count    INTEGER,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT factor_weights_unique
        UNIQUE (regime, timeframe, factor_group, factor_name)
);
CREATE INDEX IF NOT EXISTS idx_factor_weights_regime_tf
    ON factor_weights (regime, timeframe);

-- 13) 信号生命周期跟踪表（pending → triggered → sl_hit / tp_hit / expired）
--   设计要点：
--     - 主键直接复用 signals.id：1:1 关系，删除 signals 时级联删除 lifecycle，
--       不需要在 signals 表加任何反向引用列。
--     - status 用 CHECK 列举枚举值，避免拼写漂移；增加新状态时只需要改这里。
--     - take_profit 用 JSONB 而非 NUMERIC[]：与 signals.take_profit 列对齐，
--       方便整段拷贝；此外历史 signals 行的 take_profit 可能为空数组 / null，
--       JSONB 更宽容。
--     - max_favorable_pct / max_adverse_pct 用 NUMERIC(10,6)：百分比形式，
--       例如 0.012345 表示 +1.2345%，统计聚合不需要再除一遍。
--     - expires_at 列单独建偏序索引：lifecycle 任务每 30 秒扫"未结算且即将
--       过期"的行，加 WHERE 过滤的偏序索引可以把扫描成本降到 O(快到期数量)。
CREATE TABLE IF NOT EXISTS signal_lifecycle (
    signal_id           BIGINT      PRIMARY KEY
                        REFERENCES signals(id) ON DELETE CASCADE,
    symbol              TEXT        NOT NULL,
    bias                TEXT        NOT NULL CHECK (bias IN ('long', 'short', 'neutral')),
    entry_zone_low      NUMERIC(24, 10),
    entry_zone_high     NUMERIC(24, 10),
    stop_loss           NUMERIC(24, 10),
    take_profit         JSONB,
    status              TEXT        NOT NULL CHECK (status IN (
        'pending', 'triggered', 'sl_hit', 'tp1_hit', 'tp2_hit',
        'expired', 'invalidated'
    )),
    triggered_at        TIMESTAMPTZ,
    triggered_price     NUMERIC(24, 10),
    exit_at             TIMESTAMPTZ,
    exit_price          NUMERIC(24, 10),
    pnl_pct             NUMERIC(12, 6),
    max_favorable_pct   NUMERIC(12, 6),
    max_adverse_pct     NUMERIC(12, 6),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_symbol_status
    ON signal_lifecycle (symbol, status);
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_symbol_updated
    ON signal_lifecycle (symbol, updated_at DESC);
-- 偏序索引：只覆盖未结算行，配合 lifecycle 任务"扫到期 + 触发"双查询路径
CREATE INDEX IF NOT EXISTS idx_signal_lifecycle_open_expires
    ON signal_lifecycle (expires_at)
    WHERE status IN ('pending', 'triggered');

-- ============================================================
-- 邮件通知收件人表（notification_emails）
-- ============================================================
-- 设计要点：
--   1) 当 LLM 输出明确方向（bias=long / short）且为本轮真实调用时，
--      系统会向"该表中所有 enabled=TRUE"的邮箱推送一封 HTML 提醒。
--   2) email 列做 UNIQUE：同一个邮箱不允许重复录入，避免给同一收件人
--      重复发邮件；CRUD 接口在 INSERT 冲突时会返回 409。
--   3) enabled 用布尔列做"软启停"，避免删除后失去历史；管理员可以临时
--      关闭某个邮箱而不丢配置。
--   4) name 仅作管理员备注用（如"老板邮箱"/"风控小组"），可空。
CREATE TABLE IF NOT EXISTS notification_emails (
    id          BIGSERIAL    PRIMARY KEY,
    email       TEXT         NOT NULL,
    name        TEXT,
    enabled     BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT notification_emails_unique UNIQUE (email)
);
CREATE INDEX IF NOT EXISTS idx_notification_emails_enabled
    ON notification_emails (enabled);

-- ============================================================
-- P3 升级：信号评估表（signal_evaluation）
-- ============================================================
-- 设计要点：
--   1) 离线评估系统每隔 SIGNAL_EVALUATION_INTERVAL_SECONDS 跑一轮，
--      对每个 (symbol, window_minutes) 组合写入一行；窗口默认 60/360/1440 三档。
--   2) 不做"幂等 unique"——历次评估都新增一行，便于看指标随时间漂移。
--   3) 主用消费场景：LLM prompt 注入 24h（window_minutes=1440）最新一行，
--      把胜率 / 翻转率 / Sharpe / Brier 等指标喂给 LLM 做自适应放慢决策。
--   4) 字段语义：
--        total_signals           : 窗口内 LLM 入库信号总数
--        triggered_count         : 价格曾走进 entry_zone 的样本数
--        fill_rate               : triggered_count / total_signals
--        wins / losses           : 仅"曾入场"样本里的 tp_hit / sl_hit
--        expired_after_triggered : 曾入场但超时未触发 SL/TP 的笔数
--        win_rate                : wins / (wins + losses)
--        avg_pnl_pct             : 曾入场样本的平均 PnL（百分比形式）
--        total_pnl_pct           : 曾入场样本的累计 PnL
--        max_favorable_avg       : 曾入场样本的"最大有利波动"平均值
--        max_adverse_avg         : 曾入场样本的"最大不利波动"平均值
--        sharpe_estimated        : avg_pnl / std(pnl)（无年化，仅做窗口间比较）
--        direction_flip_count    : 窗口内相邻信号方向翻转次数
--        direction_flip_rate     : direction_flip_count / max(total_signals - 1, 1)
--        brier_score             : mean((conf - actual_outcome)^2)，越低越校准
CREATE TABLE IF NOT EXISTS signal_evaluation (
    id                       BIGSERIAL PRIMARY KEY,
    symbol                   TEXT        NOT NULL,
    window_minutes           INT         NOT NULL,
    evaluated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_signals            INT         NOT NULL,
    triggered_count          INT         NOT NULL,
    fill_rate                NUMERIC(6, 4),
    wins                     INT,
    losses                   INT,
    expired_after_triggered  INT,
    win_rate                 NUMERIC(6, 4),
    avg_pnl_pct              NUMERIC(10, 6),
    total_pnl_pct            NUMERIC(10, 6),
    max_favorable_avg        NUMERIC(10, 6),
    max_adverse_avg          NUMERIC(10, 6),
    sharpe_estimated         NUMERIC(8, 4),
    direction_flip_count     INT,
    direction_flip_rate      NUMERIC(6, 4),
    brier_score              NUMERIC(8, 4)
);
CREATE INDEX IF NOT EXISTS idx_signal_evaluation_lookup
    ON signal_evaluation (symbol, window_minutes, evaluated_at DESC);
