# SPEC_REFACTOR_2026.md

> **ETH/USDT 实时交易分析平台 — 系统重构规格说明**
> 版本: 1.0
> 日期: 2026-05-31
> 状态: 待执行
> 覆盖范围: Phase 1（债务清理）+ Phase 2（架构拆分）+ Phase 3（性能优化）
> 不包含: Phase 4（可观测性 & 运维），将单独规划

---

## 目录

- [1. 重构总览](#1-重构总览)
- [2. 核心决策记录](#2-核心决策记录)
- [3. 目标架构](#3-目标架构)
- [4. Phase 1: 债务清理](#4-phase-1-债务清理)
- [5. Phase 2: 架构拆分](#5-phase-2-架构拆分)
- [6. Phase 3: 性能优化](#6-phase-3-性能优化)
- [7. 文件变更清单](#7-文件变更清单)
- [8. 风险与缓解](#8-风险与缓解)
- [9. 验证检查清单](#9-验证检查清单)
- [10. 不在范围内](#10-不在范围内)

---

## 1. 重构总览

### 1.1 动机

当前代码库 ~8,500 行，存在以下核心问题：
- **repositories.py (1328行)**：全局依赖热点，13 张表的全部 CRUD 集中在一个类
- **runner.py (780行)**：6 种后台任务的启停、缓冲、节流逻辑交织
- **llm_agent.py (1136行)**：prompt 组装、LLM 调用、schema 校验、节流逻辑未分层
- **零测试覆盖**：无回归安全网，重构风险高
- **性能瓶颈**：因子聚合 9 次串行 DB 查询，延迟 200-500ms
- **代码卫生**：7 个文件共 8 处重复的 `_safe_float`/`_to_float`（含 llm_agent.py 2 处）、4 个残留老接口、未启用的链上模块

### 1.2 执行策略

```
Phase 1 (清理)  →  Phase 2 (架构拆分)  →  Phase 3 (性能优化)
  ~1 周                ~2-3 周                 ~2 周
```

- **测试策略**：直接重构 + 手动验证，不单独补测试
- **部署模式**：维持单体进程，不做分布式拆分

### 1.3 预期成果

| 指标 | 重构前 | 重构后目标 |
|------|--------|------------|
| 最大单文件行数 | 1328 行 (repositories.py) | < 400 行 |
| 串行 DB 查询次数 (因子聚合) | 9 次 | 2-3 次并行组 |
| 因子聚合延迟 | 200-500ms | < 150ms |
| `_safe_float` 重复实现数 | 8 处 (7 文件) | 1 处 |
| 老接口/死代码 | 6 处 | 0 处 |
| 跨层反向依赖 | 1 处 (runner→orderbook) | 0 处 |
| 配置项可调性 | 部分硬编码魔法值 | 全部通过 .env 可配 |

---

## 2. 核心决策记录

### 2.1 架构决策

| 编号 | 决策 | 选择 | 理由 |
|------|------|------|------|
| ADR-01 | Repo 拆分策略 | 按表领域拆分为独立类 | 最清晰的依赖边界，每个调用方只依赖需要的 repo |
| ADR-02 | Repo 拆分方式 | Big Bang 一次性切换 | 最干净的切换方式，无遗留代码 |
| ADR-03 | Runner 拆分粒度 | 每个职责一个 Worker 类 | 最彻底的解耦，每个 worker 可独立理解、测试、调试 |
| ADR-04 | Worker 通信方式 | 数据直写 DB | Worker 直接依赖 repo 写数据，无事件总线。~~EventBus 推迟到 Phase 4~~ |
| ADR-05 | Worker Task 管理 | Worker 自管理 (start/stop) | 每个 Worker 拥有自己的 asyncio.Task，Runner 编排启停 |
| ADR-06 | ~~EventBus 实现~~ | ~~自写轻量 EventBus~~ | ~~零外部依赖，满足监控事件广播需求~~ **推迟到 Phase 4** |
| ADR-07 | DI Container 设计 | Container 持有各独立 Repo | 依赖精确性最高，类型系统帮助发现遗漏 |
| ADR-08 | Container 初始化 | 保持单一大 create() 方法 | 所有初始化逻辑在一处可见，顺序依赖清晰 |
| ADR-09 | 部署模式 | 维持单体进程 | 不引入分布式复杂度 |
| ADR-10 | 配置组织 | 拆为子配置类 | Settings ~50 字段 ~238 行，需要分组管理 |

### 2.2 工程决策

| 编号 | 决策 | 选择 | 理由 |
|------|------|------|------|
| EDR-01 | 老接口清理时机 | 先删再拆 | 减少拆分时需分析的代码量 |
| ADR-02 | safe_float 策略 | 新建 app/utils.py | 统一行为，一处修改全局生效 |
| EDR-03 | structlog 处理 | 从 requirements.txt 移除 | 未使用，减少依赖体积 |
| EDR-04 | 链上模块 | 直接删除 | 从未启用，无参考价值保留必要 |
| EDR-05 | API 版本前缀 | 不加 | 内部使用，无外部消费者 |
| EDR-06 | DB Schema 变更 | 直接改 schema.sql + 手动 ALTER | 单实例部署，无多环境管理需求 |
| EDR-07 | LLM Agent 拆分 | 拆为 Client + Prompt + Throttle | 职责分层，prompt 可独立迭代 |
| EDR-08 | NarrativeRenderer | 保持原样 | 内聚性高，不增加重构范围 |
| EDR-09 | 因子并行化范围 | 并行化 + DB 端计算下推 | 中等投入获得最大延迟改善 |
| EDR-10 | 魔法值处理 | 全部提升到 Settings | 消除所有硬编码，运维无需改代码 |

---

## 3. 目标架构

### 3.1 目标目录结构

```
app/
├── __init__.py
├── main.py                          # FastAPI 入口 (不变)
├── config.py                        # 拆为子配置类 (DatabaseSettings, IngestionSettings, ...)
├── container.py                     # DI 容器 (持有各独立 repo + worker)
├── logging_config.py                # 日志配置 (不变)
├── utils.py                         # [NEW] 公共工具 (safe_float 等)
│
├── data_ingestion/                  # 数据采集层
│   ├── __init__.py
│   ├── base.py                      # 抽象基类 (不变)
│   ├── okx_ws.py                    # OKX WebSocket (不变)
│   ├── okx_rest.py                  # OKX REST (不变)
│   ├── runner.py                    # 精简编排器 (~100 行)
│   └── workers/                     # [NEW] 独立 Worker
│       ├── __init__.py
│       ├── trade_buffer.py          # Trade 批量缓冲与 flush
│       ├── orderbook_writer.py      # Orderbook 写入 + 指标计算
│       ├── rest_watchdog.py         # REST 兜底拉取
│       └── retention.py             # 数据保留清理
│
├── data_storage/                    # 数据存储层
│   ├── __init__.py
│   ├── database.py                  # asyncpg 连接池 (不变)
│   ├── orderbook_metrics.py         # [NEW] 从 factor_engine 移入
│   └── repos/                       # [NEW] 按领域拆分的 Repo
│       ├── __init__.py              # 导出所有 repo
│       ├── base.py                  # BaseRepo 基类
│       ├── trade_repo.py            # trades 表
│       ├── kline_repo.py            # klines_{1m..1d} 表
│       ├── orderbook_repo.py        # orderbook_snapshots + orderbook_metrics
│       ├── derivatives_repo.py      # funding_rates + open_interest + liquidations
│       ├── signal_repo.py           # signals 表
│       └── email_repo.py            # notification_emails 表
│
├── factor_engine/                   # 因子引擎 (核心计算逻辑不变)
│   ├── __init__.py
│   ├── aggregator.py               # 因子聚合器 (查询并行化改造)
│   ├── klines.py                   # K 线增量聚合 (不变)
│   ├── capital_flow.py             # 资金流因子 (删除老接口)
│   ├── orderbook.py                # 订单簿因子 (删除老接口 + 移出 compute_orderbook_metric_row)
│   ├── derivatives.py              # 衍生品因子 (删除老接口)
│   ├── market_structure.py         # 市场结构因子 (删除老接口)
│   ├── regime.py                   # 市场状态判定 (不变)
│   └── liquidity.py               # 流动性地图 (不变)
│
├── signal_engine/                   # 信号引擎
│   ├── __init__.py
│   ├── llm_client.py               # [NEW] LLM 调用 + 延迟链构建 + 结果解析 (~250 行)
│   ├── llm_prompts.py              # [NEW] Prompt 模板（纯常量）(~150 行)
│   ├── llm_throttle.py             # [NEW] DB 节流 + 缓存重建 + 并发锁 (~180 行)
│   ├── llm_agent.py                # 精简门面 (~200 行：analyze 编排 + post-check)
│   ├── schemas.py                  # TradingSignal pydantic (不变)
│   ├── narrative_renderer.py       # 叙事渲染 (不变)
│   └── service.py                  # 信号服务 (不变)
│
├── api_service/                     # API 层 (不变)
│   ├── __init__.py
│   ├── routes.py
│   ├── deps.py
│   └── analysis_view.py
│
└── notification/                    # 通知层 (不变)
    ├── __init__.py
    └── email_sender.py
```

### 3.2 删除文件

| 文件 | 原因 |
|------|------|
| `app/data_ingestion/onchain_mock.py` | 链上模块从未启用，直接删除 |
| `app/factor_engine/onchain.py` | 链上模块从未启用，直接删除 |

### 3.3 重命名/移动

| 原路径 | 目标路径 | 原因 |
|--------|----------|------|
| `app/data_storage/repositories.py` | 拆分到 `app/data_storage/repos/*.py` | 按领域拆分 |
| `app/data_ingestion/runner.py` 中的逻辑 | 拆分到 `app/data_ingestion/workers/*.py` | 按职责拆分 |
| `factor_engine/orderbook.py:compute_orderbook_metric_row` | `app/data_storage/orderbook_metrics.py` | 消除跨层反向依赖 |

### 3.4 目标依赖关系

```
main → container → {repos/*, workers/*, services}
api_service → {repos/*, services}
signal_engine → {repos/*, factor_engine, notification}
factor_engine → repos/*
data_ingestion → {repos/*, workers/*}
workers → repos/*
notification → signal_engine (schemas)

无循环依赖 ✓
无跨层反向依赖 ✓ (runner → factor_engine 的反向依赖已消除)
```

---

## 4. Phase 1: 债务清理

**目标**：消除死代码和重复代码，为架构拆分建立干净基础
**预计工时**：~1 周
**风险等级**：低

### 4.1 任务清单

#### P1-01: 新建 `app/utils.py` 统一 `_safe_float`

**当前问题**：7 个文件共 8 处各自实现了行为不一致的 float 安全转换

| # | 文件 | 函数名 | 行为 |
|---|------|--------|------|
| 1 | `factor_engine/capital_flow.py:174` | `_safe_float` | **不处理** NaN/Inf，失败返回 0.0 |
| 2 | `factor_engine/derivatives.py:389` | `_safe_float` | **不处理** NaN/Inf，失败返回 0.0 |
| 3 | `factor_engine/liquidity.py:222` | `_to_float` | 处理 None/NaN/Inf，失败返回 None |
| 4 | `signal_engine/narrative_renderer.py:44` | `_to_float` | 处理 None/NaN/Inf，失败返回 None |
| 5 | `signal_engine/llm_agent.py:61` | `_to_float_safe` (模块级) | 处理 None，失败返回 None |
| 6 | `signal_engine/llm_agent.py:511` | `_to_float` (嵌套于 `_collect_cached_plan_kwargs`) | 处理 None，失败返回 None |
| 7 | `signal_engine/service.py:243` | `_safe_float` | 处理 None，失败返回 None |
| 8 | `api_service/analysis_view.py:61` | `_to_float` | 处理 None/Decimal，失败返回 None |

> **注意**：`notification/email_sender.py` 中的 `_fmt_price` 是格式化函数（`f"{float(v):.{digits}f}"`），不是安全转换工具，**不在替换范围内**。`_fmt_price` 内部改用 `safe_float` 即可，但函数签名和行为保持不变。

**目标实现**：

```python
# app/utils.py
from __future__ import annotations
from typing import Optional
import math

def safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """统一的 float 安全转换。

    处理 None / str / NaN / Inf 等所有边界情况。

    Args:
        value: 任意输入值
        default: 转换失败时返回的默认值 (默认 None)

    Returns:
        float 或 default
    """
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result
```

**变更范围**：
1. 创建 `app/utils.py`
2. 逐文件替换上表 8 处 `_safe_float` / `_to_float` / `_to_float_safe` → `from app.utils import safe_float`
3. 删除各文件中的旧实现
4. 统一行为：所有调用点都处理 NaN/Inf/None/str
5. **特别处理 `capital_flow.py` 和 `derivatives.py`**：原 `_safe_float` 失败时返回 `0.0` 而非 `None`，且**不检查 NaN/Inf**。替换为 `safe_float(v, default=0.0)` 后会新增 NaN/Inf 拒绝能力（原实现允许 `float("nan")` 原样传递）。这是**行为增强**而非纯等价替换，需评估因子计算中 NaN 出现的可能性及影响
6. `email_sender.py` 的 `_fmt_price` 内部 `float(v)` 改为 `safe_float(v)`，降低异常风险

#### P1-02: 删除因子引擎老接口

**当前问题**：4 个模块残留了未使用的老入口函数
- `capital_flow.py:33` — `compute_capital_flow(trades)` + `_resample_minute_ohlc`
- `orderbook.py:37` — `compute_orderbook_factors()`
- `derivatives.py:31` — `compute_derivatives_factors()`
- `market_structure.py:378` — `compute_market_structure(trades)`

**操作**：
1. 全局搜索确认这些函数无调用方 (grep 函数名)
2. 删除函数定义及其专属辅助函数
3. 保留各模块中标注为 "从 klines 计算" 的新接口（如 `compute_capital_flow_from_klines`）

#### P1-03: 从 `requirements.txt` 移除 structlog

**操作**：
1. 删除 `requirements.txt` 中的 `structlog>=24.4.0` 行
2. 确认 `app/logging_config.py` 无 structlog import（当前已确认使用 stdlib）
3. 确认无其他文件 import structlog

#### P1-04: 删除链上模块

**操作**：
1. 删除 `app/data_ingestion/onchain_mock.py`
2. 删除 `app/factor_engine/onchain.py`
3. **container.py 清理**：
   - 删除 `MockOnchainProvider` 的 import（L20）
   - 删除 `onchain: Optional[MockOnchainProvider]` 字段（L51）
   - 删除 `onchain=onchain` 构造参数（L90 实例化 + L159 传参）
4. **runner.py 清理**：
   - 删除 `OnchainProvider` 的 import（L32）
   - 删除构造函数 `onchain` 参数（L51）和 `self.onchain = onchain` 赋值（L74）
   - 删除整个 `_run_onchain_poller` 方法（L753-779，当前已不被 `start()` 调用但方法体仍引用已删除的 onchain 模块）
5. **repositories.py 清理**：
   - 删除 `insert_onchain` 方法（L268-283）
   - 删除 `fetch_latest_onchain` 方法（L284-294）
   - 这两个方法依赖已删除的 `onchain_metrics` 表，不删会导致死代码残留
6. **base.py 清理**：
   - 删除 `OnchainProvider` 抽象基类（当前唯一实现者 `MockOnchainProvider` 已删除，无其他实现）
7. 全局验证：`grep -r "onchain" app/` 确认无遗漏引用

   > **注意**：经核实，`aggregator.py` 中**不存在** onchain 相关引用（`compute_onchain_factors` 无匹配），无需处理。

#### P1-05: 提取 `compute_orderbook_metric_row` 到 data_storage

**当前问题**：`data_ingestion/runner.py` 导入 `factor_engine/orderbook.py:compute_orderbook_metric_row`，形成采集层→计算层的反向依赖

**操作**：
1. 在 `app/data_storage/` 下新建 `orderbook_metrics.py`
2. 将 `compute_orderbook_metric_row` 函数从 `factor_engine/orderbook.py` 移入
3. 更新 `runner.py` 的 import：`from app.data_storage.orderbook_metrics import compute_orderbook_metric_row`
4. 确认 `factor_engine/orderbook.py` 中不再导出此函数

#### P1-06: 魔法值提升到 Settings

**当前问题**：部分参数硬编码在代码中，无法通过 .env 调整

经核实，**以下参数已在 `config.py` 中通过 Settings 配置**，无需重复提升：

| 已有 Settings 字段 | config.py 行号 | runner.py 使用方式 |
|-------------------|---------------|-------------------|
| `orderbook_min_interval_seconds` (默认 5.0) | L154 | `self.settings.orderbook_min_interval_seconds` |
| `orderbook_metrics_min_interval_seconds` (默认 10.0) | L156 | `getattr(self.settings, "orderbook_metrics_min_interval_seconds", 0.0)` |
| `retention_run_interval_seconds` (默认 600) | L193 | `self.settings.retention_run_interval_seconds` |

**真正需要提升的硬编码参数清单**（经核实确实硬编码，无 Settings 字段）：

| # | 当前位置 | 当前代码 | 新 Settings 字段名 | 默认值 | 说明 |
|---|----------|---------|-------------------|--------|------|
| 1 | `runner.py:444` | `_WS_STALE_FUNDING_SECONDS = 5 * 60.0` | `ws_stale_funding_seconds` | 300.0 | 类级别常量 |
| 2 | `runner.py:445` | `_WS_STALE_OI_SECONDS = 60.0` | `ws_stale_oi_seconds` | 60.0 | 类级别常量 |
| 3 | `runner.py:447` | `_WATCHDOG_TICK_SECONDS = 15.0` | `watchdog_tick_seconds` | 15.0 | 类级别常量 |
| 4 | `runner.py:449` | `_WATCHDOG_GRACE_SECONDS = 30.0` | `watchdog_grace_seconds` | 30.0 | 类级别常量 |
| 5 | `okx_ws.py:196` | `await asyncio.sleep(25)` | `ws_ping_interval_seconds` | 25.0 | 内联魔法数 |
| 6 | `okx_rest.py:53` | `_BREAKER_BASE_COOLDOWN = 60.0` | `breaker_base_cooldown_seconds` | 60.0 | 模块级常量 |
| 7 | `okx_rest.py:54` | `_BREAKER_MAX_COOLDOWN = 15 * 60.0` | `breaker_max_cooldown_seconds` | 900.0 | 模块级常量 |
| 8 | `runner.py` `__init__` 参数 | 构造函数默认值 `1.0` 秒 | `trade_flush_interval_seconds` | 1.0 | 已参数化但默认值硬编码在构造函数签名中 |
| 9 | `runner.py` `_run_liquidation_flusher` 内 | 内联 `1.0` 秒 | `liquidation_flush_interval_seconds` | 1.0 | 非命名常量 |

> **不需要提升**：`klines.py:46` 的 `TIMEFRAME_SECONDS` 映射是算法常量，保持硬编码。

**操作**：
1. 在 `config.py` 现有 `Settings` 类中添加上表 9 个新字段及默认值
2. 各模块改为从 `settings.xxx` 读取，替换硬编码常量/内联值
3. 更新 `.env.example` 添加所有新字段及默认值

> **注意**：此任务与 Phase 2 的 config 拆分有交叉。实际执行时先在现有 Settings 类中添加新字段，Phase 2 再统一拆分为子配置类。

#### P1-07: 数据库 Schema 防御性修改

**操作**：
1. **修改 `schema.sql`**：
   - `orderbook_snapshots`: 添加 `UNIQUE(exchange, symbol, ts)` 约束
   - `signals`: 扩展 bias 约束以包含 `observe`（**需先删除已有约束**，见下方 ALTER）
   - `signals.entry_zone`: 添加 `CHECK(jsonb_typeof(entry_zone) = 'array')`
   - `signals.take_profit`: 添加 `CHECK(jsonb_typeof(take_profit) = 'array')`
   - `orderbook_snapshots.bids/asks`: 添加 `CHECK(jsonb_typeof(bids) = 'array')`

2. **添加保留策略**：
   - `signals` 表的 `RETENTION_SIGNALS_SECONDS` 已有
   - 新增 `RETENTION_FUNDING_SECONDS` (默认 90 天)
   - 新增 `RETENTION_OI_SECONDS` (默认 90 天)
   - 在 `runner.py` (重构后为 `retention.py` worker) 的清理逻辑中增加 funding/OI 清理

3. **手动 ALTER**（一次性执行）：

   > **⚠️ signals 表的 bias 约束已有旧版本** `CHECK (bias IN ('long','short','neutral'))`
   > （schema.sql 约束名通常为 `signals_bias_check`，执行前先查 `pg_constraint` 确认实际名称）。
   > 不能直接 `ADD CONSTRAINT`——会与旧约束并存导致逻辑冲突。必须先 DROP 旧约束再 ADD 新约束。

   ```sql
   -- 第 0 步：确认旧约束实际名称
   SELECT conname FROM pg_constraint
     WHERE conrelid = 'signals'::regclass AND contype = 'c';

   -- 第 1 步：orderbook_snapshots 唯一约束（该表当前无 UNIQUE）
   ALTER TABLE orderbook_snapshots
     ADD CONSTRAINT uq_ob_snapshot UNIQUE (exchange, symbol, ts);

   -- 第 2 步：signals.bias — 先删旧约束，再建新约束（扩展 observe）
   ALTER TABLE signals DROP CONSTRAINT IF EXISTS signals_bias_check;
   ALTER TABLE signals
     ADD CONSTRAINT ck_signal_bias CHECK (bias IN ('long','short','neutral','observe'));

   -- 第 3 步：signals JSONB 数组约束
   ALTER TABLE signals
     ADD CONSTRAINT ck_entry_zone CHECK (jsonb_typeof(entry_zone) = 'array'),
     ADD CONSTRAINT ck_take_profit CHECK (jsonb_typeof(take_profit) = 'array');

   -- 第 4 步：orderbook_snapshots JSONB 数组约束
   ALTER TABLE orderbook_snapshots
     ADD CONSTRAINT ck_bids CHECK (jsonb_typeof(bids) = 'array'),
     ADD CONSTRAINT ck_asks CHECK (jsonb_typeof(asks) = 'array');
   ```

### 4.2 Phase 1 完成标准

- [ ] `app/utils.py` 存在且 `safe_float` 被所有需要 float 转换的模块使用
- [ ] 7 文件 8 处重复的 `_safe_float/_to_float/_to_float_safe` 实现已删除（含 `liquidity.py`）
- [ ] 4 个因子模块的老接口已删除，仅保留 "from_klines" 新接口
- [ ] `requirements.txt` 中无 structlog
- [ ] `onchain_mock.py` 和 `onchain.py` 已删除
- [ ] `container.py` 中无 `onchain` 字段和 `MockOnchainProvider` 引用
- [ ] `runner.py` 中无 `onchain` 参数、`OnchainProvider` import、`_run_onchain_poller` 方法
- [ ] `repositories.py` 中无 `insert_onchain` / `fetch_latest_onchain` 方法
- [ ] `base.py` 中无 `OnchainProvider` 抽象基类
- [ ] `grep -r "onchain" app/` 返回 0 结果
- [ ] `compute_orderbook_metric_row` 在 `data_storage/orderbook_metrics.py` 中
- [ ] `runner.py` 不再 import `factor_engine.orderbook`
- [ ] 所有硬编码魔法值已提升到 Settings
- [ ] `.env.example` 已更新所有新字段
- [ ] `schema.sql` 已更新约束
- [ ] 手动 ALTER 已在生产库执行
- [ ] `uvicorn app.main:app` 正常启动，日志无异常
- [ ] `/health` 和 `/healthz` 端点正常返回

---

## 5. Phase 2: 架构拆分

**目标**：降低单文件复杂度，建立清晰的模块边界
**预计工时**：~2-3 周
**风险等级**：中-高
**前置条件**：Phase 1 全部完成

### 5.1 Config 拆分为子配置类

#### P2-01: 拆分 `app/config.py`

**当前**：单一 `Settings` 类，~50 字段，~238 行
**目标**：按功能域拆为子配置类，主 `Settings` 组合这些子类

```python
# app/config.py

class DatabaseSettings(BaseSettings):
    """数据库连接池相关配置。"""
    database_url: str = "postgresql://postgres:postgres@localhost:5432/eth_analysis"
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_max_inactive_connection_lifetime: float = 60.0
    db_pool_acquire_timeout: float = 10.0
    db_write_max_retries: int = 2
    db_write_retry_backoff: float = 0.2


class IngestionSettings(BaseSettings):
    """数据采集相关配置。"""
    okx_ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    okx_rest_url: str = "https://www.okx.com"
    okx_rest_timeout: float = 10.0
    okx_rest_max_retries: int = 3
    okx_rest_retry_backoff: float = 0.8
    okx_rest_trust_env: bool = False
    okx_rest_proxy: str = ""
    symbols: List[str] = Field(default_factory=lambda: ["ETH-USDT-SWAP"])
    exchanges: List[str] = Field(default_factory=lambda: ["okx"])
    orderbook_depth: int = 5
    default_contract_value: float = 0.1
    rest_poll_interval_seconds: int = 60
    # Phase 1 新增：原硬编码的常量
    ws_ping_interval_seconds: float = 25.0
    ws_stale_funding_seconds: float = 300.0
    ws_stale_oi_seconds: float = 60.0
    watchdog_tick_seconds: float = 15.0
    watchdog_grace_seconds: float = 30.0
    breaker_base_cooldown_seconds: float = 60.0
    breaker_max_cooldown_seconds: float = 900.0
    trade_flush_interval_seconds: float = 1.0
    liquidation_flush_interval_seconds: float = 1.0
    # 以下字段已在现有 Settings 中存在，拆分后归入此类
    orderbook_min_interval_seconds: float = 5.0
    orderbook_metrics_min_interval_seconds: float = 10.0
    orderbook_metrics_window_seconds: int = 900
    orderbook_metrics_baseline_seconds: int = 3600
    retention_run_interval_seconds: int = 600


class FactorSettings(BaseSettings):
    """因子引擎相关配置。"""
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
    # 清算因子参数
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


class LLMSettings(BaseSettings):
    """LLM 相关配置。"""
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_thinking_enabled: bool = True
    deepseek_reasoning_effort: str = "high"
    llm_temperature: float = 0.2
    llm_timeout: float = 300.0
    llm_min_interval_seconds: float = 1800.0


class RetentionSettings(BaseSettings):
    """数据保留策略配置。"""
    retention_trades_seconds: int = 86400
    retention_orderbook_seconds: int = 86400
    retention_signals_seconds: int = 2592000
    retention_funding_seconds: int = 7776000    # [NEW] 90 天
    retention_oi_seconds: int = 7776000         # [NEW] 90 天


class EmailSettings(BaseSettings):
    """邮件通知相关配置。"""
    enable_email_notification: bool = False
    resend_api_key: str = ""
    resend_from: str = ""
    resend_timeout: float = 20.0


class SignalSettings(BaseSettings):
    """信号风控相关配置。"""
    decision_min_atr_pct_15m: float = 0.0025
    signal_interval_seconds: int = 30
    account_risk_budget_pct: float = 0.01


class Settings(BaseSettings):
    """应用全局配置（组合子配置）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
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
```

**变更范围**：
1. 拆分 `Settings` 为子配置类
2. 更新所有调用方：`settings.database_url` → `settings.db.database_url`，`settings.symbols` → `settings.ingestion.symbols` 等
3. 更新 `.env.example` 添加分组注释
4. 更新 `get_settings()` 和所有 `@lru_cache` 使用处

> **技术说明**：子配置类需设 `env_prefix=""`，使 `.env` 顶层变量直接映射到子类字段。主 `Settings` 保持 `extra="ignore"`，子类不认识的变量自动跳过。调用方需从 `settings.database_url` 改为 `settings.db.database_url`，全局搜索替换即可。

### 5.2 Repositories 拆分

#### P2-02: 拆分 `repositories.py` 为领域 Repo

**当前**：`Repositories` 类 1328 行，覆盖 13 张表
**目标**：7 个独立 Repo 类，每个 < 200 行

**Repo 拆分映射**：

| 新 Repo 类 | 文件 | 管理的表 | 方法数 (估计) |
|-----------|------|---------|-------------|
| `TradeRepo` | `repos/trade_repo.py` | `trades` | 3 (insert_batch, fetch_for_klines, fetch_aggregate) |
| `KlineRepo` | `repos/kline_repo.py` | `klines_{1m,5m,15m,1h,4h,1d}` | 3×6=18 (upsert/fetch_recent/fetch_latest per tf) |
| `OrderbookRepo` | `repos/orderbook_repo.py` | `orderbook_snapshots`, `orderbook_metrics` | 5 |
| `DerivativesRepo` | `repos/derivatives_repo.py` | `funding_rates`, `open_interest`, `liquidations` | 10 |
| `SignalRepo` | `repos/signal_repo.py` | `signals` | 6 |
| `EmailRepo` | `repos/email_repo.py` | `notification_emails` | 5 |

**BaseRepo 设计**：

```python
# app/data_storage/repos/base.py
from __future__ import annotations
from app.data_storage.database import Database
from app.logging_config import get_logger

class BaseRepo:
    """所有 Repo 的基类，提供 DB 连接池访问。"""

    def __init__(self, db: Database):
        self._db = db

    @property
    def pool(self):
        """快捷访问连接池。"""
        return self._db.pool
```

**执行步骤 (Big Bang)**：
1. 创建 `app/data_storage/repos/` 目录结构
2. 编写 `base.py` (BaseRepo)
3. 从 `repositories.py` 逐表提取方法到对应 repo 文件
4. 更新 `repos/__init__.py` 导出所有 repo
5. **一次性更新所有调用方**：
   - `container.py`: 创建各 repo 实例
   - `runner.py` (及后续的 workers): 依赖具体 repo
   - `aggregator.py`: 依赖具体 repo
   - `klines.py`: 依赖 KlineRepo
   - `llm_agent.py`: 依赖 SignalRepo
   - `service.py`: 依赖 SignalRepo + EmailRepo
   - `routes.py`: 依赖 SignalRepo + EmailRepo
6. 删除旧 `repositories.py`
7. 更新 `deps.py` 中的依赖注入

#### P2-03: 更新 `container.py`

**Container 改造**：

```python
@dataclass
class AppContainer:
    """全局依赖容器（重构后）。"""

    settings: Settings
    db: Database
    # Repos
    trade_repo: TradeRepo
    kline_repo: KlineRepo
    ob_repo: OrderbookRepo
    deriv_repo: DerivativesRepo
    signal_repo: SignalRepo
    email_repo: EmailRepo
    # Ingestion
    okx_rest: OKXRestClient
    ingestion_runner: IngestionRunner  # 精简编排器
    # Factor
    factor_aggregator: FactorAggregator
    kline_aggregator: KlineAggregator
    # Signal
    llm_client: LLMClient
    llm_throttle: LLMThrottleManager
    llm_agent: LLMAgent  # 精简门面
    email_sender: EmailSender
    signal_service: SignalService
    # Background
    instrument_refresh_task: Optional[asyncio.Task] = field(default=None)

    @classmethod
    async def create(cls, settings: Settings) -> "AppContainer":
        # 1. Storage
        db = Database(...)
        await db.connect()

        trade_repo = TradeRepo(db)
        kline_repo = KlineRepo(db)
        ob_repo = OrderbookRepo(db)
        deriv_repo = DerivativesRepo(db)
        signal_repo = SignalRepo(db)
        email_repo = EmailRepo(db)

        # 2. Ingestion
        okx_rest = OKXRestClient(...)
        ws_clients = [...]
        runner = IngestionRunner(
            settings=settings,
            trade_repo=trade_repo,
            ob_repo=ob_repo,
            deriv_repo=deriv_repo,
            ws_clients=ws_clients,
            rest_client=okx_rest,
        )

        # 3. Factor
        factor_aggregator = FactorAggregator(
            kline_repo=kline_repo,
            ob_repo=ob_repo,
            deriv_repo=deriv_repo,
            settings=settings,
        )
        kline_aggregator = KlineAggregator(
            kline_repo=kline_repo,
            trade_repo=trade_repo,
            settings=settings,
        )

        # 4. Signal
        llm_client = LLMClient(settings=settings)
        llm_throttle = LLMThrottleManager(signal_repo=signal_repo, settings=settings)
        llm_agent = LLMAgent(
            llm_client=llm_client,
            llm_throttle=llm_throttle,
            settings=settings,
        )
        email_sender = EmailSender(settings=settings)
        signal_service = SignalService(
            signal_repo=signal_repo,
            email_repo=email_repo,
            factor_aggregator=factor_aggregator,
            llm_agent=llm_agent,
            email_sender=email_sender,
            settings=settings,
        )

        return cls(...)
```

### 5.3 Runner 拆分为 Workers

#### P2-04: 拆分 `runner.py` 为独立 Worker

**当前**：`IngestionRunner` 780 行，管理 6 种后台任务（含 1 个已禁用的 onchain poller）
**目标**：精简编排器 (~100 行) + 5 个独立 Worker

**Worker 接口设计**：

```python
# app/data_ingestion/workers/base.py (隐含在接口中)
from abc import ABC, abstractmethod

class BaseWorker(ABC):
    """Worker 基类，定义统一的生命周期接口。"""

    def __init__(self):
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    async def run(self) -> None:
        """Worker 主循环。由 start() 创建的 Task 调用。"""
        ...

    def start(self) -> None:
        """启动 Worker 后台 Task。"""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name=f"worker-{self.__class__.__name__}")

    async def stop(self) -> None:
        """停止 Worker 后台 Task。"""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def is_healthy(self) -> bool:
        """供 Runner 检查健康状态。"""
        return self.is_running
```

**各 Worker 职责**：

| Worker | 文件 | 从 runner.py 提取的逻辑 | 依赖的 Repo |
|--------|------|----------------------|------------|
| `TradeBufferWorker` | `trade_buffer.py` | `_trade_buffer` 管理 + `_flush_trades` 循环 | `TradeRepo` |
| `OrderbookWriter` | `orderbook_writer.py` | orderbook 事件节流写入 + `compute_orderbook_metric_row` 调用 | `OrderbookRepo`, `orderbook_metrics.py` |
| `RestWatchdog` | `rest_watchdog.py` | funding/OI REST 兜底拉取 + 熔断判断 | `DerivativesRepo` |
| `RetentionCleaner` | `retention.py` | 数据保留清理（trades/orderbook/signals/funding/OI） | `TradeRepo`, `OrderbookRepo`, `SignalRepo`, `DerivativesRepo` |

**编排器 (Runner) 精简为**：

```python
# app/data_ingestion/runner.py (重构后)
class IngestionRunner:
    """数据采集编排器（重构后）。

    仅负责：
    1. WS 事件分发 → 各 Worker
    2. Worker 生命周期管理 (start/stop)
    """

    def __init__(
        self,
        settings: Settings,
        ws_clients: List[OKXWebSocketClient],
        rest_client: OKXRestClient,
        trade_buffer: TradeBufferWorker,
        ob_writer: OrderbookWriter,
        watchdog: RestWatchdog,
        retention: RetentionCleaner,
    ):
        self._ws_clients = ws_clients
        self._workers = [trade_buffer, ob_writer, watchdog, retention]
        self._trade_buffer = trade_buffer
        self._ob_writer = ob_writer
        ...

    async def start(self) -> None:
        # 启动所有 WS 连接
        for ws in self._ws_clients:
            ws.on_event = self._dispatch_event
            await ws.connect()
        # 启动所有 Worker
        for w in self._workers:
            w.start()
        # 启动 WS 消费循环
        ...

    async def stop(self) -> None:
        # 停止所有 Worker (反序)
        for w in reversed(self._workers):
            await w.stop()

    def _dispatch_event(self, event: dict) -> None:
        """将 WS 事件分发到对应 Worker。"""
        channel = event.get("channel")
        if channel in ("trades", "liquidations"):
            self._trade_buffer.enqueue(event)
        elif channel in ("books5", "tickers"):
            self._ob_writer.enqueue(event)
        elif channel in ("funding-rate", "open-interest"):
            # 直接写 deriv_repo，无需缓冲
            ...
```

### 5.4 EventBus 实现（推迟到 Phase 4）

> **推迟理由**：当前 EventBus 没有实际消费者。所有 Worker 之间唯一的通信方式是"各写各的表"，不存在需要跨 Worker 状态广播的场景。EventBus 增加 ~50 行代码 + 抽象复杂度，但 Phase 2 完成后没有实际收益。Worker 的启停和健康检查由 Runner 直接管理即可。
>
> **推迟到 Phase 4（可观测性 & 运维）**，与 Prometheus metrics 一起做：当有真实的监控消费者（如 Prometheus exporter、告警推送）时再引入 EventBus。
>
> ~~#### P2-05: 实现轻量 EventBus~~ （已推迟）

> 以下设计保留供 Phase 4 参考，Phase 2 不实施。
>
> **预定义事件类型（Phase 4 参考）**：
>
> | 事件名 | 触发者 | 数据 | 订阅者 |
> |--------|--------|------|--------|
> | `ws_connected` | Runner | `{symbol}` | (日志/监控) |
> | `ws_disconnected` | Runner | `{symbol, reason}` | RestWatchdog (触发兜底) |
> | `buffer_flush_failed` | TradeBufferWorker | `{error, buffer_size}` | Runner (告警) |
> | `breaker_open` | RestWatchdog | `{endpoint, cooldown}` | Runner (日志) |
> | `breaker_close` | RestWatchdog | `{endpoint}` | Runner (日志) |
> | `retention_complete` | RetentionCleaner | `{table, deleted_count}` | (日志) |
> | `worker_unhealthy` | Worker | `{worker_name, reason}` | Runner (告警) |

### 5.5 LLM Agent 拆分

#### P2-06: 拆分 `llm_agent.py` 为三层

**当前**：`LLMAgent` 1137 行，混合了 prompt 组装、LLM 调用、schema 校验、节流逻辑
**目标**：3 个独立模块 + 1 个精简门面

**函数/方法搬迁映射（核心——必须按此表执行，无遗漏）**：

| 原位置（`llm_agent.py`） | 目标模块 | 说明 |
|--------------------------|---------|------|
| `_build_deepseek_chat_openai_class()` (L87-128) | `llm_client.py` | 思考模式 ChatOpenAI 子类 |
| `_SYMBOL_PATTERN` (L58) | `llm_client.py` | symbol 白名单正则 |
| `_to_float_safe()` (L61-73) | `app/utils.py` (Phase 1 已统一) | float 安全转换 |
| `SYSTEM_PROMPT` (L162-213) | `llm_prompts.py` | 系统提示词常量 |
| `FEW_SHOT_EXAMPLES` (L223-284) | `llm_prompts.py` | Few-shot 示例 |
| `THINKING_OUTPUT_INSTRUCTIONS` (L291-300) | `llm_prompts.py` | 思考模式格式指令 |
| `HUMAN_PROMPT` (L311-338) | `llm_prompts.py` | 人设 prompt 模板 |
| `LLMAnalysisResult` (L131-148) | `llm_agent.py` (保留) | 结果包装 dataclass |
| `LLMAgent.__init__` (L365-374) | `llm_agent.py` (保留) | 门面构造 |
| `LLMAgent._build_chain` (L376-470) | `llm_client.py` | 延迟构建 LangChain 链（思考/非思考两条路径） |
| `LLMAgent.enabled` (L472-473) | `llm_client.py` | API Key 检查 |
| `LLMAgent.min_interval` (L477-486) | `llm_throttle.py` | 节流间隔读取 |
| `LLMAgent._get_lock` (L488-493) | `llm_throttle.py` | per-symbol 并发锁 |
| `LLMAgent._collect_cached_plan_kwargs` (L497-562) | `llm_throttle.py` | 缓存重建：从 DB 行重建 TradingSignal 构造参数 |
| `LLMAgent._load_recent_judgment` (L564-649) | `llm_throttle.py` | 节流核心：DB 查询 + 缓存命中判断 + signal 重建 |
| `LLMAgent._build_prompt_inputs` (L654-699) | `llm_agent.py` (保留) | 叙事渲染 + prompt 占位符填充 |
| `LLMAgent._extract_token_usage` (L701-750) | `llm_client.py` | 从 AIMessage 提取 token 统计 |
| `LLMAgent._extract_reasoning` (L752-786) | `llm_client.py` | 从 AIMessage 提取思维链原文 |
| `LLMAgent._parse_json_content` (L788-835) | `llm_client.py` | JSON 容错解析（去代码围栏） |
| `LLMAgent._POST_CHECK_RR_TOLERANCE` (L848) | `llm_agent.py` (保留) | post-check 容忍度常量 |
| `LLMAgent._post_check_signal` (L851-945) | `llm_agent.py` (保留) | deterministic post-check |
| `LLMAgent._force_neutral_for_post_check` (L947-986) | `llm_agent.py` (保留) | 降级 neutral 辅助 |
| `LLMAgent.analyze` (L988-1137) | `llm_agent.py` (保留) | 完整分析流程编排 |

**`llm_client.py` (~250 行)**：LLM API 调用 + 结果解析

```python
class LLMClient:
    """封装 DeepSeek LLM API 调用。

    职责：
    - 延迟构建 LangChain 调用链（思考/非思考两条路径）
    - 调用 LLM 并返回原始结果
    - 从结果中提取 reasoning_content / token_usage
    - JSON 容错解析
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._chain = None
        self._chain_thinking_mode: bool = False
        self._renderer = NarrativeRenderer()

    def _build_chain(self):
        """延迟构建 LangChain 调用链（首次 invoke 时触发）。
        包含思考模式 / 非思考模式两条路径的分支。"""
        ...

    async def invoke(self, prompt_inputs: dict) -> Any:
        """调用 LLM 并返回原始结果。"""
        if self._chain is None:
            self._chain = self._build_chain()
        return await self._chain.ainvoke(prompt_inputs)

    @staticmethod
    def extract_reasoning(raw_message) -> Optional[str]:
        """从 AIMessage 中提取 DeepSeek 思维链原文。"""
        ...

    @staticmethod
    def extract_token_usage(raw_message) -> Tuple[int, int, int]:
        """从 AIMessage 中提取 (input, output, total) token 统计。"""
        ...

    @staticmethod
    def parse_json_content(content: Any, symbol: str) -> Optional[dict]:
        """从 LLM content 文本中抽出 JSON 对象（去代码围栏）。"""
        ...
```

**`llm_prompts.py` (~150 行)**：Prompt 模板和组装（纯常量，零逻辑）

```python
# 纯字符串常量模块，从 llm_agent.py 原样搬出，内容零修改

SYSTEM_PROMPT = """\
你是一位资深加密永续合约 desk trader ...
"""
# （完整内容从 llm_agent.py:162-213 原样复制）

FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
    (human_text, ai_text),  # 从 llm_agent.py:223-283 原样复制
]

THINKING_OUTPUT_INSTRUCTIONS = """\
【输出格式硬约束】 ...
{format_instructions}
"""
# （完整内容从 llm_agent.py:291-300 原样复制）

HUMAN_PROMPT = """\
合约: {symbol}    时间: {ts}    最新价: {last_close}
...
"""
# （完整内容从 llm_agent.py:311-338 原样复制）
```

> **关键约束**：`llm_prompts.py` 是纯文本搬家，**内容零修改**。Prompt 文本的任何变化都会导致 LLM 行为漂移，这是 Phase 2 中风险最高的变更。验证标准：重构前后 `PromptBuilder.build()` 返回的消息列表文本完全相同（可用 hash 对比）。

**`llm_throttle.py` (~180 行)**：节流 + 缓存重建

```python
class LLMThrottleManager:
    """LLM 调用节流管理。

    职责：
    - per-symbol 并发锁（防同进程竞态）
    - DB 节流：查 signals 表判断是否在窗口内
    - 缓存命中时从 DB 行重建完整 TradingSignal（含结构化交易计划）
    """

    def __init__(self, signal_repo: SignalRepo, settings: Settings):
        self._signal_repo = signal_repo
        self._interval = settings.llm.llm_min_interval_seconds
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        ...

    async def check_throttle(
        self, symbol: str, min_interval: Optional[int] = None
    ) -> Optional[LLMAnalysisResult]:
        """检查节流。

        返回：
            None - 未命中缓存，应发起 LLM 调用
            LLMAnalysisResult(from_cache=True) - 命中缓存，直接复用
        步骤：
            1. 快速路径：锁外查 DB（避免抢锁）
            2. 慢速路径：拿锁后 double-checked locking 再查一次
            3. 命中时调用 _rebuild_cached_signal 从 DB 行重建 TradingSignal
        """

    @staticmethod
    def _collect_cached_plan_kwargs(row: dict) -> dict:
        """把 signals 表行的结构化交易计划列规整成 TradingSignal 构造 kwargs。
        从 llm_agent.py:497-562 原样搬出。"""
        ...

    async def _rebuild_cached_signal(self, symbol: str, row: dict) -> Optional[LLMAnalysisResult]:
        """从 DB 行重建完整 LLMAnalysisResult（含 TradingSignal + reasoning_content）。"""
        ...
```

**`llm_agent.py` (~200 行)**：精简门面（分析流程编排）

```python
class LLMAgent:
    """LLM Agent 门面类（重构后）。

    职责：
    - 编排完整的 LLM 分析流程（节流 → prompt → 调用 → 解析 → post-check）
    - 持有 NarrativeRenderer 做 7 段叙事渲染
    - deterministic post-check（RR 诚实性 + plan 完整性）
    """

    def __init__(self, llm_client: LLMClient, llm_throttle: LLMThrottleManager, settings: Settings):
        self._client = llm_client
        self._throttle = llm_throttle
        self._settings = settings
        self._renderer = NarrativeRenderer()

    async def analyze(self, symbol: str, factors: dict) -> Optional[LLMAnalysisResult]:
        """完整的 LLM 分析流程。

        步骤：
        1. 节流检查（double-checked locking）
        2. 构建 prompt 输入（NarrativeRenderer 渲染 + 占位符填充）
        3. LLM 调用（LLMClient.invoke）
        4. 结果解析（兼容思考/非思考/直接 TradingSignal 三种返回）
        5. Schema 校验（TradingSignal.model_validate）
        6. deterministic post-check（RR 诚实性 + plan 完整性）
        7. 返回 LLMAnalysisResult
        """

    def _build_prompt_inputs(self, symbol: str, factors: dict) -> dict:
        """渲染 HUMAN_PROMPT 占位符所需的 dict。从原 llm_agent.py:654-699 搬出。"""
        ...

    @classmethod
    def _post_check_signal(cls, signal: TradingSignal) -> Tuple[TradingSignal, List[str]]:
        """deterministic post-check。从原 llm_agent.py:851-945 搬出。"""
        ...

    @staticmethod
    def _force_neutral_for_post_check(signal, issues) -> TradingSignal:
        """降级 neutral。从原 llm_agent.py:948-986 搬出。"""
        ...
```

### 5.6 Phase 2 完成标准

- [ ] `app/config.py` 包含子配置类 (DatabaseSettings, IngestionSettings, ...)；`.env` 零改动验证通过
- [ ] 所有调用方使用 `settings.db.xxx`、`settings.ingestion.xxx` 等新路径
- [ ] `app/data_storage/repos/` 目录包含 7 个 repo 文件
- [ ] 旧 `repositories.py` 已删除
- [ ] `container.py` 持有各独立 repo 实例，不再引用 `onchain` / `MockOnchainProvider`
- [ ] ~~`app/data_storage/event_bus.py` 存在并可工作~~ （已推迟到 Phase 4）
- [ ] `app/data_ingestion/workers/` 目录包含 5 个 worker 文件
- [ ] `runner.py` 精简为 ~100 行编排器
- [ ] `app/signal_engine/llm_client.py` 存在，包含延迟链构建 + 结果解析
- [ ] `app/signal_engine/llm_prompts.py` 存在，prompt 文本与重构前 hash 一致
- [ ] `app/signal_engine/llm_throttle.py` 存在，包含 DB 节流 + 缓存重建
- [ ] `llm_agent.py` 精简为 ~200 行门面，仅保留 analyze 编排 + post-check
- [ ] 所有 Worker 实现了 start()/stop() 生命周期
- [ ] `uvicorn app.main:app` 正常启动
- [ ] `/health`、`/healthz` 正常
- [ ] `/signal/refresh` 能成功触发 LLM 调用并产生信号
- [ ] `/factors` 能正常返回因子数据
- [ ] 所有 API 端点 (`/analysis/*`, `/emails/*`) 正常工作
- [ ] 邮件通知（如已配置）正常发送

---

## 6. Phase 3: 性能优化

**目标**：降低因子计算延迟，改善 DB 查询效率
**预计工时**：~2 周
**风险等级**：中
**前置条件**：Phase 2 全部完成

### 6.1 因子聚合器查询并行化

#### P3-01: 并行化 `_compute_mtf` DB 查询

**当前**：`aggregator.py:131-254`，9 次串行 await，延迟 200-500ms
**目标**：2-3 组并行查询，延迟 < 150ms

**并行化方案**：

```python
async def _compute_mtf(self, symbol: str) -> dict:
    """重构后的因子矩阵计算。"""

    # ---- 第 1 组：共享数据并行拉取 (6 个查询 → 1 次并行) ----
    # 注意：当前代码中 ob_metrics / funding_history
    # 各有独立的 try/except 降级为空。并行化时必须用 return_exceptions=True
    # 保证单个查询失败不阻塞其他查询，然后在结果侧做降级处理。
    (
        orderbook_raw,       # Union[dict, Exception] — 核心数据，失败需 raise
        funding_raw,         # Union[dict, Exception] — 同上
        oi_history_raw,      # Union[List[dict], Exception] — 同上
        liquidations_raw,    # Union[List[dict], Exception] — 同上
        ob_metrics_raw,      # Union[List[dict], Exception] — 可降级
        funding_history_raw, # Union[List[dict], Exception] — 可降级
    ) = await asyncio.gather(
        self._ob_repo.fetch_latest_orderbook(symbol),
        self._deriv_repo.fetch_latest_funding(symbol),
        self._deriv_repo.fetch_recent_oi(symbol, hours=24),
        self._deriv_repo.fetch_liquidations_since(symbol, hours=2),
        self._ob_repo.fetch_orderbook_metrics_since(symbol, hours=1),
        self._deriv_repo.fetch_funding_rates_since(symbol, days=7),
        return_exceptions=True,  # 关键：单个查询失败不阻塞其他查询
    )

    # 核心数据：失败必须显式 raise，还原"串行版直接抛异常"的行为
    if isinstance(orderbook_raw, Exception):
        raise orderbook_raw
    if isinstance(funding_raw, Exception):
        raise funding_raw
    if isinstance(oi_history_raw, Exception):
        raise oi_history_raw
    if isinstance(liquidations_raw, Exception):
        raise liquidations_raw

    orderbook = orderbook_raw
    funding = funding_raw
    oi_history = oi_history_raw
    liquidations = liquidations_raw

    # 可降级数据：Exception → 空值，与当前逐个 try/except 行为一致
    recent_orderbook_metrics = (
        ob_metrics_raw if not isinstance(ob_metrics_raw, Exception) else []
    )
    if isinstance(ob_metrics_raw, Exception):
        logger.warning("拉取 orderbook_metrics 失败，退化为单快照模式", exc_info=ob_metrics_raw)
    funding_history = (
        funding_history_raw if not isinstance(funding_history_raw, Exception) else []
    )
    if isinstance(funding_history_raw, Exception):
        logger.warning("拉取 funding_history 失败", exc_info=funding_history_raw)

    # ---- 第 2 组：5 周期 K 线并行拉取 (5 个查询 → 1 次并行) ----
    # K 线查询是因子计算的核心数据源，失败不应降级，直接抛异常让上层感知
    klines_map = await asyncio.gather(*[
        self._kline_repo.fetch_recent_klines(tf, symbol, lookback)
        for tf in MTF_TIMEFRAMES
    ])

    # ---- 第 3 组：CPU 密集型因子计算 (无需并行，纯 CPU) ----
    # 与当前代码完全一致，此处省略
    ...
```

> **容错策略说明**：
> - 第 1 组全部 6 个查询使用 `return_exceptions=True` 以避免单个失败阻塞其他查询。
> - 前 4 个查询（orderbook / funding / OI / liquidations）是核心数据，gather 后逐个 `isinstance(x, Exception)` 检查并 **显式 raise**——还原串行版"直接抛异常"的行为。
> - 后 2 个查询（ob_metrics / funding_history）当前代码中已有独立 try/except 降级为空，并行化后用 `isinstance(x, Exception)` 判断后降级为空值，行为等价。
> - 第 2 组 K 线查询是因子计算的核心依赖，不使用 `return_exceptions=True`，失败直接抛。
> - **验证标准**：并行化前后 `/factors` 端点对同一时刻的数据返回完全一致的 JSON。

### 6.2 Funding 分位数计算下推到 DB

#### P3-02: Python 端 `bisect` 优化 funding 分位数计算

**当前**：每次因子计算拉取 7 天 funding 全量历史（~数千行）到 Python 端，用线性遍历或 `scipy.stats.percentileofscore` 计算分位数
**目标**：在已排序数据上用 `bisect` 做 O(log n) 二分查找，无需额外 DB 查询

**方案选择理由**：
- PostgreSQL 的 `percent_rank()` 是窗口函数，不能直接以标量值调用（"给我这个值在窗口中的百分位"需要子查询 hack），SQL 实现复杂且不可移植
- 7 天 funding 数据已经在 `_compute_mtf` 中拉到 Python 端（P3-01 第 1 组并行查询之一），无需额外 DB 请求
- 数据量 ~数百到数千行，在 Python 端排序一次 + bisect 查找，耗时 < 1ms，远小于 DB 网络往返

**实现**：

```python
# app/factor_engine/derivatives.py（在 compute_derivatives_per_timeframe 内）

import bisect
from typing import List, Optional

def compute_funding_percent_rank(
    sorted_history: List[float],
    current_rate: float,
) -> Optional[float]:
    """在已排序的 funding_rate 历史上计算 current_rate 的百分位排名。

    Args:
        sorted_history: 按 funding_rate 升序排列的历史值列表
        current_rate: 当前 funding_rate

    Returns:
        百分位排名 [0.0, 1.0]；历史为空时返回 None
    """
    if not sorted_history:
        return None
    n = len(sorted_history)
    pos = bisect.bisect_left(sorted_history, current_rate)
    # percent_rank 语义：(rank - 1) / (n - 1)
    if n == 1:
        return 1.0 if current_rate >= sorted_history[0] else 0.0
    return pos / (n - 1)
```

**变更范围**：
1. 在 `derivatives.py` 中添加 `compute_funding_percent_rank` 函数
2. 在 `_compute_mtf` 中，`funding_history` 拿到后提取 `funding_rate` 列并排序（一次性，O(n log n)）
3. 各周期 `compute_derivatives_per_timeframe` 调用时传入排序后的列表
4. 替换当前线性遍历 / `scipy` 调用为 `bisect` 查找

### 6.3 添加 Funding/OI 保留策略

#### P3-03: RetentionCleaner 增加 funding/OI 清理

**操作**：
1. 在 `RetentionSettings` 中添加 `retention_funding_seconds` 和 `retention_oi_seconds`
2. `RetentionCleaner.run()` 增加对 `funding_rates` 和 `open_interest` 表的 DELETE 逻辑
3. 默认保留 90 天

```python
# app/data_ingestion/workers/retention.py (新增逻辑)
async def _clean_derivatives(self) -> None:
    """清理 funding_rates 和 open_interest 过期数据。"""
    cutoff_funding = datetime.now(timezone.utc) - timedelta(seconds=self._settings.retention.retention_funding_seconds)
    cutoff_oi = datetime.now(timezone.utc) - timedelta(seconds=self._settings.retention.retention_oi_seconds)

    deleted_funding = await self._deriv_repo.delete_funding_before(cutoff_funding)
    deleted_oi = await self._deriv_repo.delete_oi_before(cutoff_oi)

    if deleted_funding or deleted_oi:
        logger.info("保留清理: funding=%d OI=%d", deleted_funding, deleted_oi)
```

### 6.4 Phase 3 完成标准

- [ ] `_compute_mtf` 中共享数据查询使用 `asyncio.gather(return_exceptions=True)` 并行
- [ ] 并行化后 ob_metrics / funding_history 的降级行为与串行版一致
- [ ] `_compute_mtf` 中 5 周期 K 线查询使用 `asyncio.gather` 并行
- [ ] funding 分位数计算使用 Python `bisect` 二分查找，无额外 DB 查询
- [ ] `RetentionCleaner` 包含 funding/OI 清理逻辑
- [ ] `DerivativesRepo` 包含 `delete_funding_before` 和 `delete_oi_before` 方法
- [ ] `/factors` 端点响应时间 < 150ms (本地测量)
- [ ] **并行化前后 `/factors` 返回值 JSON 完全一致**（hash 对比验证）
- [ ] 信号生成端到端延迟可感知改善

---

## 7. 文件变更清单

### 7.1 新建文件

| 文件路径 | Phase | 行数(估计) | 说明 |
|----------|-------|-----------|------|
| `app/utils.py` | P1 | ~30 | safe_float 等公共工具 |
| `app/data_storage/orderbook_metrics.py` | P1 | ~40 | 从 factor_engine 移入 |
| `app/data_storage/repos/__init__.py` | P2 | ~20 | 导出所有 repo |
| `app/data_storage/repos/base.py` | P2 | ~15 | BaseRepo 基类 |
| `app/data_storage/repos/trade_repo.py` | P2 | ~120 | trades 表 Repo |
| `app/data_storage/repos/kline_repo.py` | P2 | ~250 | klines 表 Repo (6 周期) |
| `app/data_storage/repos/orderbook_repo.py` | P2 | ~150 | orderbook 表 Repo |
| `app/data_storage/repos/derivatives_repo.py` | P2 | ~250 | derivatives 表 Repo |
| `app/data_storage/repos/signal_repo.py` | P2 | ~180 | signals 表 Repo |
| `app/data_storage/repos/email_repo.py` | P2 | ~100 | emails 表 Repo |
| ~~`app/data_storage/event_bus.py`~~ | ~~P2~~ | ~~~50~~ | ~~轻量事件总线~~ **推迟到 Phase 4** |
| `app/data_ingestion/workers/__init__.py` | P2 | ~10 | 导出所有 worker |
| `app/data_ingestion/workers/trade_buffer.py` | P2 | ~100 | Trade 缓冲 Worker |
| `app/data_ingestion/workers/orderbook_writer.py` | P2 | ~120 | Orderbook 写入 Worker |
| `app/data_ingestion/workers/rest_watchdog.py` | P2 | ~150 | REST 兜底 Worker |
| `app/data_ingestion/workers/retention.py` | P2 | ~100 | 保留清理 Worker |
| `app/signal_engine/llm_client.py` | P2 | ~250 | LLM 调用封装 + 延迟链构建 + 结果解析 |
| `app/signal_engine/llm_prompts.py` | P2 | ~150 | Prompt 模板（纯常量，零逻辑） |
| `app/signal_engine/llm_throttle.py` | P2 | ~180 | DB 节流 + 缓存重建 + per-symbol 锁 |

### 7.2 大幅修改文件

| 文件路径 | Phase | 变更说明 |
|----------|-------|---------|
| `app/config.py` | P1/P2 | 添加 ~20 新配置项 + 拆为子配置类（需先验证 .env 兼容性）。当前 Settings 有 60+ 字段 239 行，拆分后所有字段均需有归属 |
| `app/container.py` | P1/P2 | P1: 删除 onchain 相关；P2: 持有独立 repo + worker |
| `app/data_ingestion/runner.py` | P1/P2 | P1: 删除 onchain poller + OnchainProvider 引用；P2: 精简为 ~100 行编排器。当前 780 行 |
| `app/data_storage/repositories.py` | P1 | 删除 `insert_onchain` / `fetch_latest_onchain` 方法（P2 整体拆分）。当前 1328 行 |
| `app/data_ingestion/base.py` | P1 | 删除 `OnchainProvider` 抽象基类 |
| `app/signal_engine/llm_agent.py` | P2 | 精简为 ~200 行门面（analyze 编排 + post-check + prompt 构建） |
| `app/factor_engine/aggregator.py` | P2/P3 | 依赖具体 repo + 查询并行化 + bisect 分位数 |
| `app/factor_engine/capital_flow.py` | P1 | 删除老接口 + 使用 safe_float |
| `app/factor_engine/orderbook.py` | P1 | 删除老接口 + 移出 compute_orderbook_metric_row |
| `app/factor_engine/derivatives.py` | P1/P3 | P1: 删除老接口 + 使用 safe_float；P3: 新增 bisect 分位数函数 |
| `app/factor_engine/market_structure.py` | P1 | 删除老接口 |
| `app/factor_engine/klines.py` | P2 | 依赖 KlineRepo |
| `app/signal_engine/service.py` | P1/P2 | 依赖具体 repo + 使用 safe_float |
| `app/api_service/routes.py` | P2 | 依赖具体 repo |
| `app/api_service/analysis_view.py` | P1 | 使用 safe_float |
| `app/api_service/deps.py` | P2 | 更新依赖注入 |
| `app/notification/email_sender.py` | P1 | 使用 safe_float |
| `schema.sql` | P1 | 添加约束 |
| `.env.example` | P1 | 添加新配置项 |

### 7.3 删除文件

| 文件路径 | Phase | 原因 |
|----------|-------|------|
| `app/data_ingestion/onchain_mock.py` | P1 | 链上模块未启用 |
| `app/factor_engine/onchain.py` | P1 | 链上模块未启用 |
| `app/data_storage/repositories.py` | P2 | 已拆分到 repos/ |

---

## 8. 风险与缓解

### 8.1 高风险项

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Big Bang Repo 切换引入遗漏调用 | 高 | 中 | 全局 grep `container.repos` 和 `Repositories` 确认无遗漏；启动后检查所有 API 端点 |
| Config 子配置拆分 .env 映射异常 | 中 | 高 | 子配置类设 `env_prefix=""` 确保正确映射；启动后验证所有配置项读取正常 |
| Worker 拆分导致 WS 事件分发逻辑断裂 | 中 | 高 | Runner 的事件分发逻辑是核心路径，拆分时优先保证 WS → Worker 的数据通路完整性 |
| LLM Agent 拆分遗漏方法搬迁 | 中 | 高 | 严格按 P2-06 的搬迁映射表逐条执行；遗漏任一辅助方法（如 `_parse_json_content`、`_extract_reasoning`）会导致 LLM 路径不可用 |
| 链上模块删除遗漏引用 | 中 | 中 | P1-04 操作清单已扩展到 11 步；全局 `grep -r "onchain" app/` 做最终验证 |

### 8.2 中风险项

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| LLM Agent 拆分后 prompt 行为变化 | 中 | 高 | Prompt 模板提取为纯字符串常量（`llm_prompts.py`），**内容零修改**；用 hash 对比重构前后输出 |
| DB Schema 约束导致写入失败 | 低 | 中 | 启动后检查日志，确认约束无冲突 |
| 因子并行化容错与串行版不一致 | 低 | 中 | 第 1 组 gather 用 `return_exceptions=True` + 逐项 `isinstance(x, Exception)` 降级，严格对齐当前逐个 `try/except` 的行为 |
| KlineRepo 方法数量膨胀 (6 周期 × 3 方法) | 确定 | 低 | 使用参数化方法：`fetch_recent(tf, symbol, n)` 而非每个周期一个方法 |

### 8.3 无测试覆盖的额外风险

> **明确风险**：本次重构不补充测试，依赖手动验证和代码审查。
>
> **缓解措施**：
> 1. 每个 Phase 完成后执行完整的端到端手动验证
> 2. 重构前记录当前所有 API 端点的正常响应作为基准
> 3. 启动后观察日志 15 分钟，确认无异常
> 4. 特别关注：因子计算结果数值一致性（重构前后 `/factors` 对比）

---

## 9. 验证检查清单

### 9.1 每个 Phase 完成后的通用检查

```
□ uvicorn app.main:app 启动无报错
□ /health 返回 {status: "ok"}
□ /healthz 返回 WS/REST 通道健康度
□ /factors 返回完整因子数据
□ /signal 返回最新信号
□ /signal/refresh 能触发新的 LLM 调用
□ /analysis/latest 返回前端友好视图
□ /analysis/history 返回分页历史
□ /emails CRUD 正常
□ 日志中无 ERROR 级别异常
□ 观察运行 15 分钟无异常退出
```

### 9.2 Phase 1 特有检查

```
□ grep -r "structlog" app/ 无结果
□ grep -r "onchain" app/ 无结果（包括 onchain_mock / onchain.py / OnchainProvider / insert_onchain）
□ grep -r "factor_engine.orderbook" app/data_ingestion/ 无结果
□ grep -r "_safe_float\|_to_float\|_to_float_safe" app/ 仅在 utils.py 中出现定义
□ 所有 API 端点返回值与重构前一致
```

### 9.3 Phase 2 特有检查

```
□ container.repos 不存在 (已拆为 container.trade_repo 等)
□ container.onchain 不存在
□ runner.py 行数 < 150
□ llm_agent.py 行数 < 250（门面 + post-check + prompt 构建）
□ llm_client.py 包含 _build_chain / _extract_reasoning / _parse_json_content
□ llm_throttle.py 包含 _load_recent_judgment / _collect_cached_plan_kwargs
□ llm_prompts.py 的 prompt 文本 hash 与重构前一致
□ grep -r "Repositories" app/ 无结果 (旧类已删除)
□ settings.db.database_url / settings.ingestion.symbols 等新路径可正常读取
□ WS 推送的数据正常写入 DB (检查 trades/orderbook 表有新数据)
□ 信号正常产生和入库
□ LLM 缓存命中（from_cache=True）时返回的 TradingSignal 包含完整交易计划字段
```

### 9.4 Phase 3 特有检查

```
□ /factors 响应时间 < 150ms (使用 time curl 测量)
□ 并行化前后 /factors 返回值 JSON hash 一致
□ funding 分位数计算结果与重构前数值一致（bisect vs 原算法对比）
□ 保留清理正常执行 (检查 funding/OI 表有过期数据被删除)
```

---

## 10. 不在范围内

以下内容明确 **不在本次重构范围内**：

| 项目 | 原因 |
|------|------|
| Phase 4（可观测性 & 运维） | 单独规划，包括 Prometheus metrics、API 认证、LLM token 追踪 |
| 补充测试用例 | 本次选择手动验证策略 |
| API 版本前缀 (`/v1/`) | 内部使用，无外部兼容需求 |
| NarrativeRenderer 重构 | 内聚性高，不增加重构范围 |
| 多进程/微服务拆分 | 维持单体进程 |
| ORM 迁移 (Alembic/SQLAlchemy) | 手写 SQL 在高频写入场景下性能更好 |
| 前端开发 | 当前仅有 API 层 |
| 新增交易所支持 (如 Binance) | 仅关注 OKX 数据源 |
| 链上数据接入 | 直接删除 mock，未来单独规划 |

> **⚠️ 顺带修正**：当前 `CLAUDE.md` 声称项目使用 `structlog`（"Logging: `structlog` via `get_logger(__name__)`"），但实际 `logging_config.py` 使用 Python stdlib `logging`。建议在 P1-03 移除 structlog 时同步修正 `CLAUDE.md` 的日志描述。

---

## 附录 A: 预估工时汇总

| Phase | 工时 | 说明 |
|-------|------|------|
| Phase 1: 债务清理 | ~5 天 | safe_float + 老接口删除 + 链上删除 + 魔法值提升 + Schema 修改 |
| Phase 2: 架构拆分 | ~12 天 | Config 拆分(2天) + Repo 拆分(3天) + Runner 拆分(3天) + LLM 拆分(2天) + 集成调试(2天) |
| Phase 3: 性能优化 | ~7 天 | 查询并行化(3天) + DB 计算下推(2天) + 保留策略(1天) + 集成验证(1天) |
| **合计** | **~24 天** | 约 5 周工作量 |
