"""信号生成服务（LLM-Native 决策核心）。

设计原则
========
LLM 100% 拥有方向判断权；本服务只做 3 件事：

1. **数据准备**：调用 FactorAggregator 拿多周期因子矩阵；
2. **极端风控底线**：当 15m ATR 占比低于 ``decision_min_atr_pct_15m``
   （默认 0.25%）时跳过 LLM 调用直接 neutral —— 这是**唯一**的服务端
   "干预"，理由是"数学上不可交易"（任何 SL 都是高频陷阱）；
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
from app.signal_engine.llm_agent import LLMAgent, LLMAnalysisResult
from app.signal_engine.schemas import TradingSignal

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
        # 上一轮 ATR floor 状态（按 symbol 维度记忆），用于"边沿触发"日志：
        # - None：上一轮未触发（或还没跑过）
        # - "atr_too_low" 等 reason：上一轮已触发
        # 仅在状态变化（进入 / 切换 reason / 退出）时打 INFO，持续命中走 DEBUG，
        # 避免长时间低波动行情下反复刷屏。
        self._last_atr_floor_reason: Dict[str, Optional[str]] = {}
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
            2) ATR floor 极端风控：15m ATR 占比 < threshold → 跳过 LLM，
               构造一条 neutral 占位信号（不入库）；
            3) llm_agent.analyze(symbol, factors)
               - 返回 None：LLM 未启用 / 调用失败 → 不入库；
               - 返回 from_cache=True：节流命中 → 不入库（避免重复行）；
               - 返回 from_cache=False：真实 LLM 调用 → 入库 + 触发邮件。
        """
        factors = await self.factor_aggregator.compute(symbol)

        # ----------------------------------------------------------------
        # 唯一的服务端"干预"：ATR floor
        # ----------------------------------------------------------------
        # 当 15m ATR 占比低于阈值时，市场处于无可交易波动状态——任何 SL 都
        # 是高频陷阱。这是数学上的不可交易性，不是主观规则；阈值（默认
        # 0.25%）由 settings.decision_min_atr_pct_15m 控制，置 0 / 负数
        # 即可完全关闭该底线（不推荐）。
        atr_floor_reason = self._atr_floor_check(factors)
        # 边沿触发日志：与上一轮相同（持续触发 / 持续未触发）时不打 INFO，
        # 仅在状态变化（进入 / 切换 reason / 退出）那一轮打 INFO。
        # 持续命中时降级到 DEBUG，开 DEBUG 日志仍可追踪每轮的拦截记录。
        prev_atr_floor_reason = self._last_atr_floor_reason.get(symbol)
        self._last_atr_floor_reason[symbol] = atr_floor_reason
        if atr_floor_reason is not None:
            is_atr_state_change = prev_atr_floor_reason != atr_floor_reason
            log_method = logger.info if is_atr_state_change else logger.debug
            log_method(
                "ATR floor 触发：symbol=%s 跳过 LLM 调用（reason=%s）",
                symbol, atr_floor_reason,
            )
            final = self._make_atr_floor_neutral_signal(atr_floor_reason)
            return {
                "id": None,
                "symbol": symbol,
                "source": f"atr_floor:{atr_floor_reason}",
                "persisted": False,
                "signal": final.model_dump(),
                "factors": factors,
                "reasoning_content": None,
            }
        if prev_atr_floor_reason is not None:
            # 退出边沿：上一轮还在 ATR floor 拦截，这一轮已恢复正常波动率。
            # 打一条 INFO 让运维 / 日志读者清楚"观望状态已解除"。
            logger.info(
                "ATR floor 解除：symbol=%s 波动率恢复正常（prev_reason=%s）",
                symbol, prev_atr_floor_reason,
            )

        # llm_agent.analyze 返回 LLMAnalysisResult（包装了 TradingSignal +
        # 思考模式下的 reasoning_content）；调用失败 / 未启用时返回 None。
        llm_result: Optional[LLMAnalysisResult] = await self.llm_agent.analyze(
            symbol=symbol, factors=factors,
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
    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        """把任意输入转 float，失败 / NaN / Inf 返回 None"""
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f

    def _atr_floor_check(self, factors: Dict[str, Any]) -> Optional[str]:
        """
        ATR floor 极端风控：判断 15m ATR 占比是否低于阈值
        --------------------------------------------------------------
        参数：
            factors: FactorAggregator.compute 输出
        返回：
            触发时返回 "atr_too_low"，未触发返回 None。
        说明：
            atr_pct_15m = atr_14 / last_close。当此比例低于
            ``decision_min_atr_pct_15m``（默认 0.0025 = 0.25%）时，
            视为"数学上不可交易"（任何 SL 都是高频陷阱），跳过 LLM。
            数据缺失时（atr_14 / last_close 任一为 None）保守起见**不**触发，
            让 LLM 仍有机会基于其它信号判断——而不是因为单条因子缺失就不交易。
            把 ``decision_min_atr_pct_15m`` 设为 0 或负数可完全关闭该底线。
        """
        settings = self.factor_aggregator.settings
        threshold = self._safe_float(
            getattr(settings, "decision_min_atr_pct_15m", None)
        )
        if threshold is None or threshold <= 0:
            return None
        if not isinstance(factors, dict):
            return None
        by_tf = factors.get("by_timeframe") or {}
        ms_15m = ((by_tf.get("15m") or {}).get("market_structure")) or {}
        atr_14 = self._safe_float(ms_15m.get("atr_14"))
        last_close = self._safe_float(ms_15m.get("last_close"))
        if atr_14 is None or last_close is None or last_close <= 0:
            return None
        atr_pct = atr_14 / last_close
        if atr_pct < threshold:
            logger.debug(
                "ATR floor 触发：atr_pct_15m=%.5f < threshold=%.5f",
                atr_pct, threshold,
            )
            return "atr_too_low"
        return None

    @staticmethod
    def _make_atr_floor_neutral_signal(reason: str) -> TradingSignal:
        """
        构造一条"ATR floor 拦截"的 neutral 信号
        --------------------------------------------------------------
        参数：
            reason: 触发原因（当前只有 "atr_too_low" 一种）
        返回：
            干净的 neutral TradingSignal（plan 字段全部为 None）。
        说明：
            confidence 一律置 0.0：这条 signal 是被极端风控拦下的，不入库，
            仅用于本轮 API 响应 / 日志展示。
        """
        zh_map = {
            "atr_too_low": "ATR(15m) 占比低于阈值（< 0.25%），市场无可交易波动",
        }
        zh_reason = zh_map.get(reason, reason)
        return TradingSignal(
            bias="neutral",
            confidence=0.0,
            reason=f"[ATR floor:{reason}] {zh_reason}",
            risk=f"极端风控底线触发：{zh_reason}（数学上不可交易，跳过 LLM 调用）",
            suggestion=(
                "本周期建议观望，等待波动率恢复正常水平后再做判断。"
                "仅供参考，不构成交易指令"
            ),
            timeframe_alignment={},
            invalidation_conditions=[],
        )

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
        min_bars = max(6, int(getattr(settings, "mtf_lookback_bars", 80)) // 4)
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
                logger.exception("信号循环 %s warmup 探测失败，稍后重试", symbol)
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
            max(interval, int(self.factor_aggregator.settings.factor_window_seconds))
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

        # 上一轮 source，用于"边沿触发"日志：当 source 以 "atr_floor:" 开头
        # 且与上一轮完全一致时，说明 ATR floor 状态未变化，本轮日志降级到
        # DEBUG，与 generate() 内的 ATR floor 边沿日志保持一致，避免刷屏。
        last_source: Optional[str] = None
        while not self._stopping.is_set():
            try:
                result = await self.generate(symbol)
                signal_payload = result.get("signal") or {}
                source = result["source"]
                # 日志级别决策：
                # - source=="llm(cache)"：命中 LLM 节流缓存，每轮都会出现，
                #   按 INFO 打会刷屏，走 DEBUG。
                # - source 以 "atr_floor:" 开头且与上一轮相同：ATR floor 持续
                #   触发，状态未变，走 DEBUG（与 generate() 边沿日志对齐）。
                # - 其它情况（真实 LLM 调用 / ATR floor 进入或切换 reason /
                #   LLM 不可用 等）走 INFO。
                if source == "llm(cache)":
                    log_method = logger.debug
                elif source.startswith("atr_floor:") and source == last_source:
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
                last_source = source
            except Exception:
                logger.exception("信号生成失败 %s", symbol)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
