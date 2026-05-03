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
