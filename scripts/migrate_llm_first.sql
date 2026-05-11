-- ============================================================
-- 迁移脚本：LLM-First 重构 · 删除 3 张废弃表
-- ------------------------------------------------------------
-- 用途：
--   LLM-First 重构后，以下 3 张表与对应模块整体下线，本脚本一键 DROP：
--     - factor_weights      （RuleEngine + ICCalibrator 已删除）
--     - signal_lifecycle    （LifecycleTracker 已删除）
--     - signal_evaluation   （SignalEvaluator 已删除）
--
-- 使用方法：
--   psql -h <host> -U <user> -d eth_analysis -f scripts/migrate_llm_first.sql
--
-- 注意：
--   - **执行前请先做完整备份**：3 张表中的历史数据将永久丢失，
--     若需保留用作离线分析请先 \copy 出 CSV。
--   - signal_lifecycle 通过 FK ON DELETE CASCADE 引用 signals.id，
--     DROP TABLE 不会影响 signals 本表数据。
--   - 脚本幂等：表不存在时 DROP TABLE IF EXISTS 自动跳过。
--   - 老库 signals.source 列默认值仍是 'rules+llm'；本脚本顺手把
--     默认值改成 'llm'，不会回填历史行。
-- ============================================================

BEGIN;

DROP TABLE IF EXISTS signal_evaluation CASCADE;
DROP TABLE IF EXISTS signal_lifecycle CASCADE;
DROP TABLE IF EXISTS factor_weights CASCADE;

-- signals.source 默认值更新（不回填历史行）
ALTER TABLE signals ALTER COLUMN source SET DEFAULT 'llm';

COMMIT;
