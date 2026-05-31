# PROJECT_KNOWLEDGE_MAP.md

> 生成时间: 2026-05-31  
> 分析工具: Claude Code (Principal Software Architect)  
> 项目: ETH/USDT 实时交易分析平台  

---

## 1. 文件统计

| 类别 | 数量 | 说明 |
|------|------|------|
| Python 业务源码 | 34 | `app/` 目录下所有 `.py` |
| 脚本 | 2 | `scripts/` |
| 测试 | 1 | `tests/conftest.py`（无实际测试用例） |
| SQL Schema | 1 | `schema.sql` |
| 配置文件 | 4 | `.env.example`, `pytest.ini`, `requirements.txt`, `CLAUDE.md` |
| 文档 | 1 | `README.md` |
| **总计** | **43** | — |

---

## 2. 目录覆盖

| 目录 | 文件数 | 已阅读 | 理解程度 |
|------|--------|--------|----------|
| `app/` | 5 | 5/5 | **High** |
| `app/data_ingestion/` | 5 | 5/5 | **High** |
| `app/data_storage/` | 3 | 3/3 | **High** |
| `app/factor_engine/` | 9 | 9/9 | **High** |
| `app/signal_engine/` | 5 | 5/5 | **High** |
| `app/api_service/` | 4 | 4/4 | **High** |
| `app/notification/` | 2 | 2/2 | **High** |
| `scripts/` | 2 | 2/2 | **High** |
| `tests/` | 1 | 1/1 | **Medium**（仅 conftest，无实际测试） |
| 根目录配置/文档 | 5 | 5/5 | **High** |

**总体理解程度: High (95%+)**

---

## 3. 核心模块清单

### 3.1 入口与编排

| 模块 | 文件 | 职责 |
|------|------|------|
| FastAPI 入口 | `app/main.py` | lifespan 启停编排：Container → Ingestion → KlineAgg → SignalLoop |
| 全局配置 | `app/config.py` | pydantic-settings，60+ 配置项，@lru_cache 单例 |
| DI 容器 | `app/container.py` | AppContainer dataclass，装配所有长生命周期组件 |
| 日志 | `app/logging_config.py` | stdlib logging 统一格式，抑制第三方库噪声 |

### 3.2 数据采集层 (`app/data_ingestion/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| 抽象基类 | `base.py` | ExchangeWebSocketClient / ExchangeRestClient / OnchainProvider |
| OKX WebSocket | `okx_ws.py` | 6 频道订阅(trades/books5/tickers/funding/OI/liquidation)，ctVal 换算，自动重连 |
| OKX REST | `okx_rest.py` | funding/OI/instruments 拉取 + 持仓比(rubik)，per-endpoint 熔断器 |
| 采集编排器 | `runner.py` | WS 消费 + trade/liq 批量 flusher + REST watchdog + 持仓比轮询 + 数据保留清理 |
| 链上 Mock | `onchain_mock.py` | 随机数生成，当前未启用 |

### 3.3 数据存储层 (`app/data_storage/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据库 | `database.py` | asyncpg 连接池封装 + 瞬时错误重试(run_with_retry) |
| 仓储 | `repositories.py` | 1300+ 行，覆盖 13 张表的全部 CRUD + 聚合查询 |

### 3.4 因子引擎 (`app/factor_engine/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| 聚合器 | `aggregator.py` | 多周期矩阵编排：5 周期 × 4 因子类 + mtf_alignment + liquidations + regime + liquidity |
| K线聚合 | `klines.py` | 6 周期增量聚合器，PG 端聚合 SQL，cvd 跨 bar 累积 |
| 资金流 | `capital_flow.py` | net_flow / CVD / taker_ratio / volume_zscore / 价量背离 |
| 订单簿 | `orderbook.py` | 静态因子 + P1 时序因子(imbalance_slope/zscore/wall_persistence/vacuum) |
| 衍生品 | `derivatives.py` | funding 分位数 + OI/价散度 + 持仓比因子(retail_vs_smart_divergence) |
| 市场结构 | `market_structure.py` | HH/HL 趋势 / ATR / ADX / BB_width / Value Area / swept levels |
| 市场状态 | `regime.py` | 6 种 regime 判定(trending_up/down/ranging/breakout/breakdown/transitional) |
| 流动性地图 | `liquidity.py` | swing 高低点 + 整数关口 + 价值区 → 双向止损池 |
| 链上因子 | `onchain.py` | 薄转换层，当前未启用 |

### 3.5 信号引擎 (`app/signal_engine/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| LLM Agent | `llm_agent.py` | DeepSeek(LangChain) 调用封装，DB 节流，thinking mode reasoning 透传，post-check |
| 信号 Schema | `schemas.py` | TradingSignal pydantic model，model_validator 价位顺序 + 数学自洽 |
| 叙事渲染 | `narrative_renderer.py` | 7 段 desk trader 叙事，强制因果解读 + 具体价位 + ATR 倍数 |
| 信号服务 | `service.py` | ATR floor 风控 → LLM 调用 → 持久化 → 邮件通知 fire-and-forget |

### 3.6 API 层 (`app/api_service/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| 路由 | `routes.py` | 14 个端点：health / factors / signal / analysis / emails CRUD |
| 依赖注入 | `deps.py` | FastAPI Depends 从 app.state.container 拉取组件 |
| 分析视图 | `analysis_view.py` | DB row → 前端友好 JSON（中文标签/颜色/时间格式化/交易计划序列化） |

### 3.7 通知层 (`app/notification/`)

| 模块 | 文件 | 职责 |
|------|------|------|
| 邮件发送 | `email_sender.py` | Resend HTTP API，HTML+纯文本模板，asyncio.to_thread 非阻塞 |

---

## 4. 数据库表覆盖

| 表名 | 用途 | Repository 方法数 | 已理解 |
|------|------|-------------------|--------|
| `trades` | 逐笔成交 | 3 (insert/fetch/aggregate) | ✅ High |
| `orderbook_snapshots` | 盘口快照 | 2 (insert/fetch_latest) | ✅ High |
| `orderbook_metrics` | 盘口时序指标 | 3 (insert/fetch/delete) | ✅ High |
| `funding_rates` | 资金费率 | 3 (insert/fetch/fetch_history) | ✅ High |
| `open_interest` | 持仓量 | 2 (insert/fetch_recent) | ✅ High |
| `liquidations` | 爆仓事件 | 2 (insert_batch/fetch_since) | ✅ High |
| `klines_{1m,5m,15m,1h,4h,1d}` | 多周期K线 | 3 (upsert/fetch_recent/fetch_latest) × 6 | ✅ High |
| `signals` | LLM 信号输出 | 6 (insert/fetch_latest/fetch_full/fetch_history/fetch_by_id/fetch_judgment) | ✅ High |
| `notification_emails` | 邮件收件人 | 5 (insert/list/fetch/update/delete) | ✅ High |
| `onchain_metrics` | 链上指标(预留) | 2 (insert/fetch_latest) | ✅ Medium (未启用) |

---

## 5. 未阅读 / 信息不足区域

| 区域 | 说明 |
|------|------|
| `tests/test_*.py` | 项目目前没有任何实际测试用例，仅有 `conftest.py` |
| `.env` | 包含真实密钥，未阅读（也不应阅读） |
| `.idea/` | IDE 配置，与业务无关 |

---

## 6. 架构总结

```
数据采集层                    因子引擎                      信号引擎
┌──────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│ OKX WS       │─────▶│ KlineAggregator(6TF) │─────▶│ NarrativeRenderer│
│ OKX REST     │      │ FactorAggregator     │      │ LLMAgent         │
│ IngestionRun │      │  ├ capital_flow       │      │ SignalService    │
│              │      │  ├ orderbook          │      │  ├ ATR floor     │
│              │      │  ├ derivatives        │      │  ├ DB throttle   │
│              │      │  ├ market_structure   │      │  └ Email notify │
│              │      │  ├ regime             │      │ TradingSignal    │
│              │      │  ├ liquidity          │      │   (pydantic)     │
│              │      │  └ onchain(disabled)  │      └──────────────────┘
└──────┬───────┘      └──────────┬───────────┘              │
       │                         │                           │
       ▼                         ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PostgreSQL (13 tables)          │  FastAPI (14 endpoints)       │
│  asyncpg + run_with_retry       │  analysis_view serializer     │
└──────────────────────────────────────────────────────────────────┘
```
