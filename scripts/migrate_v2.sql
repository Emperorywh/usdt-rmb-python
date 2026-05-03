-- ============================================================
-- 迁移脚本 v2：清理 funding_rates / open_interest 重复数据并加唯一键
-- ------------------------------------------------------------
-- 用途：
--   早期版本 funding_rates / open_interest 没有唯一约束，
--   WebSocket 与 REST 两路同时写入产生大量重复。
--   本脚本：
--     1) 通过 (exchange, symbol, ts) 分组只保留 id 最小的一行；
--     2) 给两表追加 UNIQUE 约束，防止后续再次写入重复。
--
-- 使用方法：
--   psql -h <host> -U <user> -d eth_analysis -f scripts/migrate_v2.sql
--
-- 注意：
--   - 在生产库执行前请先做备份。
--   - 脚本是幂等的：约束已存在时会跳过 ALTER。
-- ============================================================

BEGIN;

-- ---------- funding_rates 去重 ----------
DELETE FROM funding_rates fr
USING funding_rates dup
WHERE fr.id > dup.id
  AND fr.exchange = dup.exchange
  AND fr.symbol   = dup.symbol
  AND fr.ts       = dup.ts;

-- ---------- open_interest 去重 ----------
DELETE FROM open_interest oi
USING open_interest dup
WHERE oi.id > dup.id
  AND oi.exchange = dup.exchange
  AND oi.symbol   = dup.symbol
  AND oi.ts       = dup.ts;

-- ---------- 追加唯一约束（已存在时跳过） ----------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'funding_rates_unique'
    ) THEN
        ALTER TABLE funding_rates
            ADD CONSTRAINT funding_rates_unique UNIQUE (exchange, symbol, ts);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'open_interest_unique'
    ) THEN
        ALTER TABLE open_interest
            ADD CONSTRAINT open_interest_unique UNIQUE (exchange, symbol, ts);
    END IF;
END
$$;

COMMIT;
