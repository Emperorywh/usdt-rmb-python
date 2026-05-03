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
