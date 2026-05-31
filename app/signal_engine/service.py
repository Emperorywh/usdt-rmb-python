"""信号生成服务（LLM-Native 决策核心）。

设计原则
========
LLM 100% 拥有方向判断权；本服务只做 3 件事：

1. **数据准备**：调用 FactorAggregator 拿多周期因子矩阵；
2. **ATR 软提示**：当 15m ATR 占比低于 ``decision_min_atr_pct_15m``
   （默认 0.25%）时，将 ATR 偏低警告注入 prompt（而非硬跳过 LLM），
   让 LLM 自行判断是否影响方向决策；
3. **持久化 + 邮件通知**：仅当 LLM 真正发起调用并落库时（``from_cache=False``）
   才向 ``notification_emails`` 表里所有启用的邮箱推送 HTML 提醒。

**不做** 的事情（与重构前的 P3 路径相比）：
- 不做 4 道决策闸门（ATR-too-low 已搬到入口 1 行 if，不叫"闸门"了）；
- 不做服务端 ``position_size_pct`` 覆盖（半凯利 / 历史胜率因子）；
- 不写信号生命周期表；
- 不读历史成绩单注入 prompt（LLM 每轮独立判断，不被旧 PnL 反向影响）；
- 不调用规则引擎，方向判断 100% 来自 LLM。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.logging_config import get_logger
from app.notification.email_sender import EmailSender
from app.utils import safe_float
from app.signal_engine.llm_agent import LLMAgent
from app.signal_engine.schemas import LLMAnalysisResult, TradingSignal

logger = get_logger(__name__)


class SignalService:
    """
    LLM-First 信号生成 + 持久化 + 邮件通知
    """

    def __init__(
        self,
        repos: Repositories,
        factor_aggregator: FactorAggregator,
        llm_agent: LLMAgent,
        email_sender: Optional[EmailSender] = None,
    ):
        self.repos = repos
        self.factor_aggregator = factor_aggregator
        self.llm_agent = llm_agent
        # 邮件通知发送器：当 LLM 给出明确方向（long/short）且本轮为真实 LLM 调用
        # （from_cache=False）时调用；未注入或未启用时整个邮件链路降级为 no-op。
        self.email_sender = email_sender
        self._loops: Dict[str, asyncio.Task[Any]] = {}
        # 后台邮件发送任务集合：仅作"防 GC"强引用，避免 asyncio.create_task
        # 创建的任务在 loop 还没调度前被垃圾回收掉。任务结束后从集合中移除。
        self._email_tasks: set[asyncio.Task[Any]] = set()
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------
    # one-shot generation
    # ------------------------------------------------------------------
    async def generate(self, symbol: str) -> Dict[str, Any]:
        """
        生成一次信号并落库（如果触发了真实 LLM 调用）
        --------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            一个 dict，包含本轮的 signal / source / factors / 是否落库 /
            reasoning_content 等供 API 接口透出。
        流程：
            1) factors = factor_aggregator.compute(symbol)
            2) ATR floor 软提示：15m ATR 占比 < threshold → 注入警告到 prompt，
               不再硬跳过 LLM；
            3) llm_agent.analyze(symbol, factors, atr_floor_warning=...)
               - 返回 None：LLM 未启用 / 调用失败 → 不入库；
               - 返回 from_cache=True：节流命中 → 不入库（避免重复行）；
               - 返回 from_cache=False：真实 LLM 调用 → 入库 + 触发邮件。
        """
        factors = await self.factor_aggregator.compute(symbol)

        # ----------------------------------------------------------------
        # ATR floor 软提示：低波动时注入警告到 prompt，不再硬跳过 LLM
        # ----------------------------------------------------------------
        atr_floor_warning = self._atr_floor_check(factors)
        if atr_floor_warning is not None:
            logger.debug(
                "ATR floor warning 注入 prompt：symbol=%s", symbol,
            )

        # llm_agent.analyze 返回 LLMAnalysisResult（包装了 TradingSignal +
        # 思考模式下的 reasoning_content）；调用失败 / 未启用时返回 None。
        llm_result: Optional[LLMAnalysisResult] = await self.llm_agent.analyze(
            symbol=symbol, factors=factors, atr_floor_warning=atr_floor_warning,
        )

        if llm_result is None:
            # LLM 未启用 / 调用失败 / schema 校验失败：不入库
            logger.debug("LLM 返回 None，跳过入库 %s", symbol)
            return {
                "id": None,
                "symbol": symbol,
                "source": "llm_unavailable",
                "persisted": False,
                "signal": None,
                "factors": factors,
                "reasoning_content": None,
            }

        final = llm_result.signal
        reasoning_content = llm_result.reasoning_content
        # 区分两种 source：
        # - "llm"        ：本次真正触发了一次 LLM API 调用
        # - "llm(cache)" ：本次命中了 LLM 节流缓存，并未实际调用 LLM
        source = "llm(cache)" if llm_result.from_cache else "llm"

        # 入库策略：
        # - llm_result.from_cache=True：只是复用节流窗口内的 LLM 缓存，并未
        #   真正发起新调用；如果继续入库，会出现每轮一条 reason/risk/suggestion
        #   完全相同、只有 factors 不同的伪 LLM 记录，既污染 signals 表，也会
        #   让下游分析误以为 LLM 在反复确认同一判断。
        # - 只有 from_cache=False 时才把这条结果落库。这样入库节奏就严格对齐
        #   LLM 真实调用节奏。
        should_persist = not llm_result.from_cache
        signal_id: Optional[int] = None
        if should_persist:
            entry_zone_payload = (
                list(final.entry_zone) if final.entry_zone is not None else None
            )
            signal_id = await self.repos.insert_signal(
                symbol=symbol,
                bias=final.bias,
                confidence=final.confidence,
                reason=final.reason,
                risk=final.risk,
                suggestion=final.suggestion,
                # factors 列只保存"原始因子快照"。reasoning_content 单独落到
                # signals.reasoning_content 列仅作审计，不参与下游决策。
                factors={"factors": factors},
                source=source,
                reasoning_content=reasoning_content,
                entry_zone=entry_zone_payload,
                stop_loss=final.stop_loss,
                take_profit=list(final.take_profit) if final.take_profit else None,
                risk_reward_ratio=final.risk_reward_ratio,
                position_size_pct=final.position_size_pct,
                timeframe_alignment=(
                    dict(final.timeframe_alignment)
                    if final.timeframe_alignment
                    else None
                ),
                invalidation_conditions=(
                    list(final.invalidation_conditions)
                    if final.invalidation_conditions
                    else None
                ),
            )
        else:
            logger.debug(
                "跳过信号入库 %s：from_cache=True（节流命中，不重复入库）", symbol,
            )

        # ----------------------------------------------------------------
        # 邮件提醒：仅在以下三个条件**同时**满足时触发
        #   1) 本轮真正发起了 LLM 调用并落库（should_persist=True）
        #   2) LLM 输出方向明确（bias ∈ {long, short}）—— 观望不发
        #   3) email_sender 已注入且启用（Resend API Key / 发件人地址齐备）
        # 走 asyncio.create_task 后台发送，绝不阻塞信号生成主路径；任意失败
        # 都不会向上抛错，仅打 warning 日志。
        # ----------------------------------------------------------------
        if (
            should_persist
            and final.bias in ("long", "short")
            and self.email_sender is not None
            and self.email_sender.enabled
        ):
            task = asyncio.create_task(
                self._dispatch_signal_email(
                    symbol=symbol,
                    signal=final,
                    factors=factors,
                    signal_id=signal_id,
                ),
                name=f"signal-email-{symbol}-{signal_id}",
            )
            # 防止任务在 loop 还没调度前被 GC 回收：保持强引用直到任务完成。
            self._email_tasks.add(task)
            task.add_done_callback(self._email_tasks.discard)

        return {
            "id": signal_id,
            "symbol": symbol,
            "source": source,
            "persisted": signal_id is not None,
            "signal": final.model_dump(),
            "factors": factors,
            # 思维链单独以独立字段返回，便于上层接口选择性透出 / 隐藏，
            # 避免污染 signal 主体；未启用思考模式或纯规则路径时为 None。
            "reasoning_content": reasoning_content,
        }

    # ==================================================================
    # ATR floor（唯一的服务端"极端风控"，不叫"闸门"）
    # ==================================================================

    def _atr_floor_check(self, factors: Dict[str, Any]) -> Optional[str]:
        """
        ATR floor 极端风控：判断 15m ATR 占比是否低于阈值
        --------------------------------------------------------------
        参数：
            factors: FactorAggregator.compute 输出
        返回：
            触发时返回格式化的中文警告字符串，未触发返回 None。
        说明：
            atr_pct_15m = atr_14 / last_close。当此比例低于
            ``decision_min_atr_pct_15m``（默认 0.0025 = 0.25%）时，
            返回 ATR 偏低警告注入 prompt（不再硬跳过 LLM）。
            数据缺失时（atr_14 / last_close 任一为 None）保守起见**不**触发。
            把 ``decision_min_atr_pct_15m`` 设为 0 或负数可完全关闭该底线。
        """
        settings = self.factor_aggregator.settings
        threshold = safe_float(settings.signal.decision_min_atr_pct_15m)
        if threshold is None or threshold <= 0:
            return None
        if not isinstance(factors, dict):
            return None
        by_tf = factors.get("by_timeframe") or {}
        ms_15m = ((by_tf.get("15m") or {}).get("market_structure")) or {}
        atr_14 = safe_float(ms_15m.get("atr_14"))
        last_close = safe_float(ms_15m.get("last_close"))
        if atr_14 is None or last_close is None or last_close <= 0:
            return None
        atr_pct = atr_14 / last_close
        if atr_pct < threshold:
            logger.debug(
                "ATR floor 触发：atr_pct_15m=%.5f < threshold=%.5f",
                atr_pct, threshold,
            )
            return (
                f"⚠️ 当前 ATR(15m) 占比极低（{atr_pct * 100:.3f}%），"
                "波动率处于不可交易区间。请考虑这是否影响你的方向判断。"
            )
        return None

    # ==================================================================
    # 邮件提醒（fire-and-forget）
    # ==================================================================
    async def _dispatch_signal_email(
        self,
        *,
        symbol: str,
        signal: TradingSignal,
        factors: Dict[str, Any],
        signal_id: Optional[int],
    ) -> None:
        """
        异步给所有已启用收件人发送一封"明确方向"的交易信号 HTML 邮件
        ---------------------------------------------------------------
        参数：
            symbol     ：合约代码
            signal     ：完整 TradingSignal（已通过 schema 强约束）
            factors    ：本轮因子聚合 dict（用于 HTML 摘要 regime / current_price）
            signal_id  ：signals.id，用于在邮件页脚展示与未来排错
        说明：
            - 本方法在 generate() 的 fire-and-forget 任务里跑，**不允许**抛异常；
              任何 DB / Resend HTTP 错误都吞掉只打日志，避免拖累信号循环。
            - 收件人列表实时从 notification_emails 表里查（only_enabled=True），
              支持运维通过 API 临时禁用某个收件人即时生效。
            - 表为空 / EmailSender 未启用 / signal.bias=neutral 时直接返回，
              EmailSender 内部还会再做一次防御性校验。
        """
        if self.email_sender is None or not self.email_sender.enabled:
            return
        if signal.bias not in ("long", "short"):
            return
        try:
            recipients_rows = await self.repos.list_notification_emails(
                only_enabled=True
            )
        except Exception:
            logger.warning(
                "查询 notification_emails 失败，跳过本次邮件提醒（symbol=%s）",
                symbol,
                exc_info=True,
            )
            return

        recipients: List[str] = [
            str(r["email"]).strip()
            for r in recipients_rows
            if r.get("email") and str(r["email"]).strip()
        ]
        if not recipients:
            logger.info(
                "notification_emails 中没有任何启用的收件人，跳过本次邮件提醒（symbol=%s）",
                symbol,
            )
            return

        try:
            stats = await self.email_sender.send_signal_alert(
                recipients=recipients,
                symbol=symbol,
                signal=signal,
                factors=factors,
                signal_id=signal_id,
            )
            logger.info(
                "信号邮件提醒派发完成 symbol=%s bias=%s 收件人=%d 已发=%d 失败=%d",
                symbol,
                signal.bias,
                len(recipients),
                stats.get("sent", 0),
                stats.get("failed", 0),
            )
        except Exception:
            logger.warning(
                "信号邮件派发异常（symbol=%s，bias=%s）",
                symbol,
                signal.bias,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # periodic background loop (started from FastAPI lifespan)
    # ------------------------------------------------------------------
    async def start_periodic(self, symbols: list[str], interval_seconds: int) -> None:
        for symbol in symbols:
            if symbol in self._loops:
                continue
            self._loops[symbol] = asyncio.create_task(
                self._run_loop(symbol, interval_seconds), name=f"signal-{symbol}"
            )

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._loops.values():
            task.cancel()
        for task in self._loops.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._loops.clear()
        # 等待所有挂起的邮件任务收尾，避免 shutdown 时报
        # "Task was destroyed but it is pending"。Resend SDK 的 HTTP 请求
        # 无显式 timeout，业务侧一般在秒级返回；最多等个位数秒级别即可回收。
        pending_emails = [t for t in self._email_tasks if not t.done()]
        for t in pending_emails:
            try:
                await t
            except Exception:  # noqa: BLE001
                pass
        self._email_tasks.clear()

    async def _wait_until_ready(self, symbol: str, hard_timeout: float) -> None:
        """
        探测式 warmup：等到关键表三件套全部就位再返回
        -----------------------------------------------------------------
        参数：
            symbol:       合约代码
            hard_timeout: 兜底等待秒数；DB 完全空表时最多只等这么久
        判定"就绪"的条件（三者同时满足）：
            - 5m K 线根数 ≥ max(6, mtf_lookback_bars // 4)
              （6 根是 market_structure pivot 的下限；lookback // 4 在
              默认 80 时给到 20 根，已经够 capital_flow / divergence
              滚动窗口算出有意义的值。）
            - 最新 funding_rate 不为空（多数情况下 OKX WS 订阅后 ≤ 60s
              内会推出第一帧；REST 兜底也是分钟级）
            - 最新 orderbook_snapshot 不为空（WS 接入后秒级即可拿到）
        """
        settings = self.factor_aggregator.settings
        min_bars = max(6, int(settings.factor.mtf_lookback_bars) // 4)
        check_interval = 5.0
        started_at = time.monotonic()
        deadline = started_at + max(0.0, float(hard_timeout))

        logger.info(
            "信号循环 %s 启动探测式 warmup（min_5m_bars=%d，hard_timeout=%ds）",
            symbol,
            min_bars,
            int(hard_timeout),
        )

        last_status: Dict[str, Any] = {
            "klines_5m": 0,
            "has_funding": False,
            "has_orderbook": False,
        }

        while not self._stopping.is_set():
            try:
                klines_5m = await self.repos.fetch_recent_klines(
                    timeframe="5m", symbol=symbol, limit=min_bars
                )
                funding = await self.repos.fetch_latest_funding(symbol)
                orderbook = await self.repos.fetch_latest_orderbook(symbol)
            except Exception:
                logger.warning(
                    "信号循环 %s warmup 探测失败，稍后重试",
                    symbol,
                    exc_info=True,
                )
                klines_5m, funding, orderbook = [], None, None

            last_status = {
                "klines_5m": len(klines_5m),
                "has_funding": funding is not None,
                "has_orderbook": orderbook is not None,
            }

            if (
                last_status["klines_5m"] >= min_bars
                and last_status["has_funding"]
                and last_status["has_orderbook"]
            ):
                elapsed = time.monotonic() - started_at
                logger.info(
                    "信号循环 %s 已就绪立即开跑（耗时 %.1fs，5m bars=%d，"
                    "funding=ok，orderbook=ok）",
                    symbol,
                    elapsed,
                    last_status["klines_5m"],
                )
                return

            if time.monotonic() >= deadline:
                missing = []
                if last_status["klines_5m"] < min_bars:
                    missing.append(
                        f"5m_bars={last_status['klines_5m']}/{min_bars}"
                    )
                if not last_status["has_funding"]:
                    missing.append("funding=missing")
                if not last_status["has_orderbook"]:
                    missing.append("orderbook=missing")
                logger.warning(
                    "信号循环 %s 超时兜底开跑（hard_timeout=%ds，未就绪项: %s）；"
                    "首轮信号可能因数据不足被 LLM 跳过",
                    symbol,
                    int(hard_timeout),
                    ", ".join(missing) if missing else "none",
                )
                return

            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=check_interval
                )
            except asyncio.TimeoutError:
                pass

    async def _run_loop(self, symbol: str, interval: int) -> None:
        # 冷启动 warmup：探测式等到关键三件套（5m 线 / funding / orderbook）就位再开跑
        hard_timeout = (
            max(interval, int(self.factor_aggregator.settings.factor.factor_window_seconds))
            + 60
        )

        # warmup 阶段任何未捕获的异常都会让整个 signal-{symbol} 任务挂掉，
        # 而 asyncio.create_task 创建的任务一旦被 self._loops 强引用，异常就
        # 会被静默吞掉（既不会触发 "Task exception was never retrieved"，也
        # 进不到下面 generate() 的 try/except）。这里统一兜一层 except，
        # 让这种"启动即静默死亡"的事故至少能在日志里留下证据。
        try:
            await self._wait_until_ready(symbol, hard_timeout=hard_timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "信号循环 %s 冷启动 warmup 阶段抛异常，跳过 warmup 直接进入主循环",
                symbol,
            )

        while not self._stopping.is_set():
            try:
                result = await self.generate(symbol)
                signal_payload = result.get("signal") or {}
                source = result["source"]
                # 日志级别决策：
                # - source=="llm(cache)"：命中 LLM 节流缓存，每轮都会出现，
                #   按 INFO 打会刷屏，走 DEBUG。
                # - 其它情况（真实 LLM 调用 / LLM 不可用 等）走 INFO。
                if source == "llm(cache)":
                    log_method = logger.debug
                else:
                    log_method = logger.info
                log_method(
                    "信号[%s] %s confidence=%.2f source=%s",
                    symbol,
                    signal_payload.get("bias", "n/a"),
                    float(signal_payload.get("confidence") or 0.0),
                    source,
                )
            except Exception:
                logger.exception("信号生成失败 %s", symbol)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
