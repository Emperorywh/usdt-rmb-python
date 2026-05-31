# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ETH/USDT real-time trading analysis platform. Collects market data from OKX (WebSocket + REST), stores in PostgreSQL, computes multi-timeframe factors, and uses DeepSeek LLM (via LangChain) to produce structured trading signals. Advisory only — no automated trading.

**Architecture: LLM-First.** All directional decisions come from the LLM. There is no rule engine. The only server-side intervention is an ATR floor check that skips LLM calls when volatility is mathematically untradeable (ATR/price < 0.25%).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (API + data ingestion + signal loop)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run a single test file
pytest tests/test_example.py

# Initialize database schema (run once on PostgreSQL)
psql -h <host> -U <user> -d eth_analysis -f schema.sql

# Standalone ingestion (without API)
python -m scripts.run_ingestion
```

## Architecture

```
OKX WS/REST → IngestionRunner → PostgreSQL
                                        ↓
                               KlineAggregator (6 timeframes: 1m/5m/15m/1h/4h/1d)
                                        ↓
                               FactorAggregator (multi-timeframe)
                                        ↓
                               NarrativeRenderer (7-section desk trader narrative)
                                        ↓
                               LLMAgent (DeepSeek via LangChain)
                                        ↓
                               SignalService → signals table + email (Resend)
                                        ↓
                               FastAPI (/signal, /factors, /analysis, /emails)
```

### Key Modules

- **`app/main.py`** — FastAPI entry point. Lifespan creates `AppContainer`, starts ingestion + K-line aggregator + signal loop.
- **`app/config.py`** — `Settings` (pydantic-settings). All config from `.env`, no hardcoded magic numbers. `get_settings()` is `@lru_cache`.
- **`app/container.py`** — `AppContainer` dataclass DI container. Wires DB pool, WS/REST clients, factor aggregator, LLM agent, email sender, signal service. Created once in lifespan, accessed via `Depends(get_container)`. On startup, launches a background task to fetch real `ctVal` from OKX instruments API (falls back to `default_contract_value`).
- **`app/data_ingestion/`** — `OKXWebSocketClient`, `OKXRestClient` (with circuit breaker), `IngestionRunner` orchestrator. Runner manages WS consumers, trade/liquidation batch flushers, REST stale-watchdog, position ratios poller, and retention cleaner.
- **`app/data_storage/`** — `Database` (asyncpg pool wrapper with transient-error retry on idempotent writes) and `Repositories` (all DB access). All writes use `ON CONFLICT DO NOTHING/UPDATE` for idempotency.
- **`app/factor_engine/`** — Multi-timeframe factor matrix across 5 periods (5m/15m/1h/4h/1d). Modules: `capital_flow`, `orderbook`, `derivatives`, `market_structure`, `klines` (KlineAggregator for incremental bar building), `regime`, `liquidity`, `onchain` (disabled). `FactorAggregator.compute()` produces the full factor dict consumed by LLM.
- **`app/signal_engine/`** — `LLMAgent` (DeepSeek via LangChain, DB-based throttle with `LLM_MIN_INTERVAL_SECONDS`), `TradingSignal` schema (pydantic with strict price-order validation), `SignalService` (LLM-First signal generation + persistence + email dispatch), `NarrativeRenderer` (7-section desk trader narrative for prompt).
- **`app/api_service/`** — FastAPI routes (`routes.py`), dependency injection (`deps.py`), analysis view serializer (`analysis_view.py`).
- **`app/notification/`** — `EmailSender` (Resend HTTP API, async via `asyncio.to_thread`). Sends HTML alerts when LLM outputs long/short.

### Data Flow Details

- **Ingestion**: WS pushes trade/orderbook/funding/OI/liquidation events. Trades and liquidations are buffered and batch-flushed every 1s. Orderbook writes are throttled per-symbol (5s for snapshots, 10s for metrics). REST watchdog only fires when WS channels go stale.
- **K-lines**: `KlineAggregator` reads raw trades and builds incremental OHLCV bars across 6 timeframes (1m/5m/15m/1h/4h/1d) with `ON CONFLICT DO UPDATE` for live rolling bars. Each timeframe has its own tick interval (1s for 1m/5m, 10s for 15m/1h, 60s for 4h/1d).
- **Factors**: `FactorAggregator.compute()` reads klines + funding + OI + orderbook + liquidations + position ratios from DB, computes per-timeframe factors, then adds `mtf_alignment` (trend resonance), `regime` detection, `liquidity` map, and liquidation summary.
- **Signals**: `SignalService.generate()` checks ATR floor, calls `LLMAgent.analyze()` which throttles via `signals` table timestamps, runs LLM, validates output with `TradingSignal` schema + post-check (RR honesty), then persists. On cold start, the signal loop waits for warmup (5m bars + funding + orderbook) before running.
- **Throttling**: LLM calls are throttled per-symbol using the latest `signals.ts` in PostgreSQL as the single source of truth. Default interval: 1800s (30 min). Cache hits return `from_cache=True` and are not re-persisted.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + LLM availability |
| GET | `/healthz` | Detailed WS/REST channel freshness |
| GET | `/factors` | Real-time factor snapshot |
| GET | `/signal` | Latest persisted signal |
| POST | `/signal/refresh` | Generate a new signal (sync) |
| GET | `/analysis/latest` | Latest analysis (front-end friendly view) |
| GET | `/analysis/history` | Recent analyses (paginated, filterable by bias/source) |
| GET | `/analysis/{id}` | Single analysis by signal ID |
| GET/POST/PUT/DELETE | `/emails[/{id}]` | Notification recipient CRUD |
| POST | `/emails/test` | Send a test email |

## Configuration

All via `.env` (see `.env.example`). Key groups:

| Group | Key variables |
|-------|--------------|
| Database | `DATABASE_URL`, `DB_POOL_MIN/MAX_SIZE`, `DB_MAX_INACTIVE_CONNECTION_LIFETIME` |
| OKX | `OKX_WS_URL`, `OKX_REST_URL`, `SYMBOLS` (comma-separated), `OKX_REST_PROXY`, `OKX_REST_TRUST_ENV` |
| DeepSeek LLM | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` (default: `deepseek-v4-pro`), `DEEPSEEK_THINKING_ENABLED`, `DEEPSEEK_REASONING_EFFORT`, `LLM_TIMEOUT`, `LLM_MIN_INTERVAL_SECONDS` |
| Factors | `FACTOR_WINDOW_SECONDS` (≥1200), `MTF_LOOKBACK_BARS`, `DEFAULT_CONTRACT_VALUE` |
| Email | `ENABLE_EMAIL_NOTIFICATION`, `RESEND_API_KEY`, `RESEND_FROM` |
| ATR Floor | `DECISION_MIN_ATR_PCT_15M` (set 0 to disable) |
| Retention | `RETENTION_TRADES_SECONDS`, `RETENTION_ORDERBOOK_SECONDS`, `RETENTION_SIGNALS_SECONDS` |

## Database

PostgreSQL with asyncpg. Schema in `schema.sql`. Key tables:

- `trades`, `orderbook_snapshots`, `orderbook_metrics` — high-frequency raw data (retained 24h)
- `klines_{1m,5m,15m,1h,4h,1d}` — per-timeframe OHLCV bars (one table per period for index efficiency)
- `funding_rates`, `open_interest`, `liquidations` — derivatives data
- `signals` — LLM outputs with structured trading plan fields (retained 30 days)
- `notification_emails` — email recipients CRUD

## Testing

- `pytest` with `pytest-asyncio` in strict mode (`asyncio_mode = strict`)
- `conftest.py` adds project root to `sys.path` for `from app.xxx import ...`
- Tests live in `tests/`

## Code Conventions

- **Language**: All comments, docstrings, log messages, and LLM output (reason/risk/suggestion) are in **Simplified Chinese**. Variable names and code identifiers are in English.
- **Async everywhere**: asyncpg, httpx, websockets, LangChain `ainvoke`. No blocking calls on the event loop.
- **Logging**: Python stdlib `logging` via `get_logger(__name__)`. Configured in `app/logging_config.py`.
- **No setup.py/pyproject.toml**: The project runs as a flat `app/` package. Dependencies are in `requirements.txt` only.
- **DI pattern**: `AppContainer` dataclass created in lifespan. Routes access services via `Depends(get_container)` / `Depends(get_signal_service)`.
- **Idempotent writes**: All INSERT statements use `ON CONFLICT DO NOTHING` or `ON CONFLICT DO UPDATE`.
- **Throttling**: Write-side throttling uses `time.monotonic()` per-symbol to avoid DB write storms from high-frequency WS pushes.

## LLM Integration Notes

- **Model**: Default `deepseek-v4-pro` with thinking mode enabled. The `reasoning_content` (chain-of-thought) is captured via a custom `ChatOpenAI` subclass that patches `_create_chat_result`. Stored in `signals.reasoning_content` for audit only.
- **Prompt structure**: System prompt (desk trader persona) → 1 few-shot example (reject trade) → Human prompt (7-section narrative from `NarrativeRenderer`).
- **Narrative sections**: market_state, mtf_direction, capital_action, derivatives, key_levels, liquidity, liquidations. Each section uses causal interpretation language, not raw metric listing.
- **Output validation**: `TradingSignal` pydantic model enforces price order (long: sl < entry < tp, short: reversed), minimum 2 take-profit levels, RR > 0. Post-check verifies RR self-report accuracy within 5%. No business-floor overrides (e.g., no "RR < 2.0 → force neutral") — direction is 100% LLM-driven.
- **No rule engine**: The old rule engine (`signal_engine/rules.py`) was removed in the LLM-First refactor. Direction is 100% LLM-driven.
