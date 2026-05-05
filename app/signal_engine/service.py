"""Signal service: fuses rule engine + LLM agent and persists outputs.

P2 升级：
* 在 INSERT signals 后**同事务节奏**追加一条 ``signal_lifecycle`` 行
  （pending / 或 neutral 的直接 expired），让生命周期跟踪任务能拿到信号；
* 在调用 LLMAgent.analyze 之前从 ``signal_lifecycle`` 读最近 N 条已结算 +
  当前未结算行，作为 ``recent_settled / open_lifecycle`` 透传给 LLM；
  这样 llm_agent 不直接依赖 repos，职责清晰。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.data_storage.repositories import Repositories
from app.factor_engine.aggregator import FactorAggregator
from app.logging_config import get_logger
from app.notification.email_sender import EmailSender
from app.signal_engine.lifecycle import compute_expires_at
from app.signal_engine.llm_agent import LLMAgent, LLMAnalysisResult
from app.signal_engine.rules import RuleEngine
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


class SignalService:
    def __init__(
        self,
        repos: Repositories,
        factor_aggregator: FactorAggregator,
        rule_engine: RuleEngine,
        llm_agent: LLMAgent,
        email_sender: Optional[EmailSender] = None,
    ):
        self.repos = repos
        self.factor_aggregator = factor_aggregator
        self.rule_engine = rule_engine
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
        factors = await self.factor_aggregator.compute(symbol)
        # P2：rule_engine.evaluate 现在是 async（要查 factor_weights 表）
        rule_signal, rule_score, contributions = await self.rule_engine.evaluate(factors)

        # P2：在调 LLM 之前从 signal_lifecycle 读"近 N 条已结算"作为
        # recent_settled 透传给 LLMAgent；llm_agent 不直接依赖 repos，
        # 避免双向耦合。
        # 注：原先一并读取的 open_lifecycle（未结算的最近一条）已废弃——
        # 信号引擎只产建议、不掌握用户实际持仓，把 lifecycle 的 open 行
        # 当成"持仓"是概念错位，会导致 LLM 被自己的旧判断绑架。方向冲突
        # 应由后续持仓管理层处理（见 generate() 末尾 TODO 注释块）。
        recent_settled = await self._fetch_lifecycle_feedback(symbol)

        # llm_agent.analyze 返回 LLMAnalysisResult（包装了 TradingSignal +
        # 思考模式下的 reasoning_content）；调用失败 / 未启用时返回 None。
        llm_result: Optional[LLMAnalysisResult] = await self.llm_agent.analyze(
            symbol=symbol,
            factors=factors,
            rule_signal=rule_signal,
            rule_score=rule_score,
            rule_contributions=contributions,
            recent_settled=recent_settled,
        )

        if llm_result is not None:
            final: TradingSignal = llm_result.signal
            reasoning_content: Optional[str] = llm_result.reasoning_content
            # 区分两种 source：
            # - "rules+llm"        ：本次真正触发了一次 LLM API 调用
            # - "rules+llm(cache)" ：本次命中了 LLM 节流缓存（默认 15 分钟），
            #                       并未实际调用 LLM。它只用于响应/日志展示，
            #                       不会落库（参见下方 should_persist 判断）。
            source = "rules+llm(cache)" if llm_result.from_cache else "rules+llm"
        else:
            final = rule_signal
            reasoning_content = None
            source = "rules"

        # 入库策略：
        # 1) llm_result 为 None（LLM 未启用 / 调用失败 / schema 校验失败）：不入库。
        #    避免把纯规则引擎结果当成 LLM 结论写库，造成大量低价值脏数据。
        # 2) llm_result.from_cache 为 True：本次只是复用 15 分钟内的 LLM 缓存，
        #    并未真正发起新的 LLM 推理；如果继续入库，会出现每 30s 一条
        #    reason/risk/suggestion 完全相同、只有 factors 不同的伪 LLM 记录，
        #    既污染 signals 表，也会让下游分析误以为 LLM 在反复确认同一判断。
        # 3) 只有"真正发起了一次 LLM 调用并成功解析"时（即 from_cache=False），
        #    才把这条结果落库。这样入库节奏就严格对齐 LLM_MIN_INTERVAL_SECONDS
        #    （默认 15 分钟一条），与成本预算一致。
        # 不入库时仍正常返回规则引擎 / 缓存中的判断用于接口响应与日志观察。
        should_persist = llm_result is not None and not llm_result.from_cache
        signal_id: Optional[int] = None
        if should_persist:
            # signals.factors 只保存"原始因子快照 + 规则引擎打分细节"。
            # rule_signal 本身（bias/confidence/reason）可以由 rule_score 重算得到，
            # 不再冗余写入，避免 JSON 体积膨胀。
            # reasoning_content 单独落到 signals.reasoning_content 列，仅作审计，
            # 不参与下游决策、也不会被回灌进下一轮 prompt。
            # P0 升级：把结构化交易计划字段一并落库；neutral 时全部为 None / 空。
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
                factors={
                    "factors": factors,
                    "rule_score": rule_score,
                    "rule_contributions": contributions,
                },
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
            # P2：信号入库成功后立刻挂一条 lifecycle 行（pending）
            # 失败不阻塞主路径——下一轮 generate 会重新写一条新的 signal，
            # lifecycle 任务也会跳过没有对应 lifecycle 行的旧 signal。
            if signal_id is not None and bool(
                getattr(self.factor_aggregator.settings, "enable_lifecycle_tracking", False)
            ):
                try:
                    # 优先用分钟单位的 TTL（默认 90 分钟，与 LLM 生成节奏匹配）；
                    # 老配置走 hours 兼容通道。compute_expires_at 内部按
                    # "minutes 优先于 hours，都缺则默认 90 分钟"的规则解析。
                    ttl_minutes = getattr(
                        self.factor_aggregator.settings,
                        "lifecycle_default_ttl_minutes",
                        None,
                    )
                    ttl_hours = getattr(
                        self.factor_aggregator.settings,
                        "lifecycle_default_ttl_hours",
                        None,
                    )
                    expires_at = compute_expires_at(
                        ttl_minutes=int(ttl_minutes) if ttl_minutes is not None else None,
                        ttl_hours=int(ttl_hours) if ttl_hours is not None else None,
                    )
                    # supersede：在写入新 pending 行之前，把同 symbol 下所有
                    # 旧的 status='pending' 行批量改成 invalidated。
                    # ----------------------------------------------------
                    # 这一步是"新建议作废旧建议"的语义落地——
                    # 旧 pending 行的 entry_zone 在生成几分钟后就已经偏离当前价，
                    # 即便价格回去也已经没有"当时那个 ATR / 结构"的语义；
                    # 留着它们只会等到 TTL 后批量 expired，污染近 N 次成绩单
                    # 的胜率统计。triggered 行不动——那是真正入过场、仍在跟踪
                    # SL/TP 的样本，必须等它自然到 SL/TP/expired 终态。
                    try:
                        invalidated_n = (
                            await self.repos.invalidate_pending_lifecycles_for_symbol(
                                symbol
                            )
                        )
                        if invalidated_n:
                            logger.info(
                                "supersede 旧 pending lifecycle 行 symbol=%s 影响=%d 条",
                                symbol,
                                invalidated_n,
                            )
                    except Exception:
                        logger.warning(
                            "supersede 旧 pending lifecycle 失败 symbol=%s（不阻塞主路径）",
                            symbol,
                            exc_info=True,
                        )
                    await self.repos.insert_signal_lifecycle(
                        signal_id=signal_id,
                        symbol=symbol,
                        bias=final.bias,
                        entry_zone=(
                            list(final.entry_zone) if final.entry_zone else None
                        ),
                        stop_loss=final.stop_loss,
                        take_profit=(
                            list(final.take_profit) if final.take_profit else None
                        ),
                        expires_at=expires_at,
                    )
                except Exception:
                    logger.warning(
                        "写入 signal_lifecycle 失败 signal_id=%s（不影响主路径）",
                        signal_id,
                        exc_info=True,
                    )
        else:
            logger.debug(
                "跳过信号入库 %s：source=%s（llm_present=%s，from_cache=%s）",
                symbol,
                source,
                llm_result is not None,
                llm_result.from_cache if llm_result is not None else None,
            )

        # ----------------------------------------------------------------
        # 邮件提醒：仅在以下三个条件**同时**满足时触发
        #   1) 本轮真正发起了 LLM 调用并落库（should_persist=True）
        #   2) LLM 输出方向明确（bias ∈ {long, short}）—— 观望不发
        #   3) email_sender 已注入且启用（SMTP 凭据齐备）
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
                    rule_score=rule_score,
                    factors=factors,
                    signal_id=signal_id,
                ),
                name=f"signal-email-{symbol}-{signal_id}",
            )
            # 防止任务在 loop 还没调度前被 GC 回收：保持强引用直到任务完成。
            self._email_tasks.add(task)
            task.add_done_callback(self._email_tasks.discard)

        # ----------------------------------------------------------------
        # TODO（持仓管理层 / portfolio sizing 占位）
        # ----------------------------------------------------------------
        # 当前 SignalService 仅产"独立的交易建议"，不维护任何持仓状态：
        #   - LLM 每次基于当前因子矩阵独立判断方向，不会被自己 30 分钟前
        #     的旧 lifecycle 绑架（这是本轮重构的核心目标）；
        #   - 同 symbol 下连续两条信号方向相反完全合法。
        # 但生产实盘 / paper-trading 场景下，需要在**信号下游**接入一层
        # portfolio 模块来处理：
        #   1) 净敞口控制：已有 long 仓时收到新的 long 建议应降低加仓比例；
        #      已有 long 仓时收到 short 建议应优先平仓而不是反向开仓；
        #   2) 风险预算：单 symbol / 全账户的最大同时开仓数与杠杆约束；
        #   3) 成交跟踪：把"实际下单价 / 实际持仓"反馈给 lifecycle 任务，
        #      使 triggered_price 接近真实成交而非 mark price。
        # 留作后续模块（计划路径建议 app/portfolio/）。当前发布版本以"建议"
        # 为唯一交付物，方向冲突由用户/运维判断；此 TODO 不影响信号生成主路径。
        # ----------------------------------------------------------------

        return {
            "id": signal_id,
            "symbol": symbol,
            "source": source,
            "persisted": signal_id is not None,
            "signal": final.model_dump(),
            "rule_signal": rule_signal.model_dump(),
            "rule_score": rule_score,
            "rule_contributions": contributions,
            "factors": factors,
            # 思维链单独以独立字段返回，便于上层接口选择性透出 / 隐藏，
            # 避免污染 signal 主体；未启用思考模式或纯规则路径时为 None。
            "reasoning_content": reasoning_content,
        }

    async def _dispatch_signal_email(
        self,
        *,
        symbol: str,
        signal: TradingSignal,
        rule_score: float,
        factors: Dict[str, Any],
        signal_id: Optional[int],
    ) -> None:
        """
        异步给所有已启用收件人发送一封"明确方向"的交易信号 HTML 邮件
        ---------------------------------------------------------------
        参数：
            symbol     ：合约代码
            signal     ：完整 TradingSignal（已通过 schema 强约束）
            rule_score ：规则引擎打分（[-1, 1] 区间）
            factors    ：本轮因子聚合 dict（用于 HTML 摘要 regime / current_price）
            signal_id  ：signals.id，用于在邮件页脚展示与未来排错
        说明：
            - 本方法在 generate() 的 fire-and-forget 任务里跑，**不允许**抛异常；
              任何 DB / SMTP 错误都吞掉只打日志，避免拖累信号循环。
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
                rule_score=rule_score,
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

    async def _fetch_lifecycle_feedback(
        self, symbol: str
    ) -> List[Dict[str, Any]]:
        """
        拉取 P2 自我反馈所需的"近 N 条已结算"数据
        ---------------------------------------------------------------
        参数：
            symbol: 合约
        返回：
            近 N 条已结算 lifecycle 行（含 join 出来的 signals 字段）。
            空列表 = 自我反馈关闭 / 表为空 / 查询失败。
        说明：
            - enable_llm_self_feedback=False / enable_lifecycle_tracking=False
              时直接返回空，避免无意义查表；
            - 查询失败吞掉异常返回空：自我反馈是"锦上添花"，不能因为它把
              整轮 LLM 调用打挂。
            - 不再读取"未结算的最近一条 open_lifecycle"——历史版本里它被
              注入 prompt 当成"持仓"，但本系统只产建议、不掌握用户是否实际
              下单，是概念错位。方向冲突 / 净敞口控制由后续持仓管理层处理
              （见 generate() 末尾 TODO 注释块）。
        """
        settings = self.factor_aggregator.settings
        if not bool(getattr(settings, "enable_llm_self_feedback", False)):
            return []
        if not bool(getattr(settings, "enable_lifecycle_tracking", False)):
            return []
        recent_n = int(getattr(settings, "llm_feedback_recent_n", 5))
        try:
            return await self.repos.fetch_recent_settled_lifecycles(
                symbol=symbol, limit=max(1, recent_n)
            )
        except Exception:
            logger.warning(
                "读取 %s 最近成绩单失败，本次 LLM 自我反馈降级为空",
                symbol,
                exc_info=True,
            )
            return []

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
        # "Task was destroyed but it is pending"。SMTP 默认 20s 超时，最多等
        # 个位数秒级别即可回收。
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
        日志策略：
            - "立即就绪"路径：单条 INFO 说明耗时与各项 ready 状态。
            - "超时兜底"路径：单条 WARNING 把当时缺失的资源列出来，
              方便排查 OKX 频道掉线 / DB 为空 / 数据采集挂掉等问题。
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
                    "首轮信号可能因数据不足被规则引擎/LLM 跳过",
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
        # ------------------------------------------------------------------
        # 冷启动 warmup：分两条路径
        # ------------------------------------------------------------------
        # 1) enable_mtf_factors=True（默认主路径）
        #    因子聚合走 _compute_mtf，数据全部来自已经持久化在 PostgreSQL 的
        #    klines_<tf> / funding_rates / open_interest / orderbook_snapshots
        #    等表，进程重启后只要 DB 里已经有历史数据，就能立刻算出有意义的
        #    多周期因子。继续硬等 1860s 是过度的（来自老 legacy 路径的遗留）。
        #    所以这里改成"探测式 ready"——周期性检查 5m K 线根数、funding、
        #    orderbook 三件套是否就位，全部就绪立即跑首轮；同时保留一个
        #    硬上限作为首次部署 / 数据库为空时的兜底。
        #
        # 2) enable_mtf_factors=False（灰度回滚通道）
        #    老聚合器从 trades 表重采样近 factor_window_seconds 内的成交，
        #    没有 K 线表可探测，沿用原来的"定时 warmup"逻辑保持行为不变。
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
            if bool(
                getattr(self.factor_aggregator.settings, "enable_mtf_factors", True)
            ):
                await self._wait_until_ready(symbol, hard_timeout=hard_timeout)
            else:
                logger.info(
                    "信号循环 %s 冷启动 warmup %ds（legacy 路径，等待因子窗口攒满）",
                    symbol,
                    hard_timeout,
                )
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(), timeout=hard_timeout
                    )
                except asyncio.TimeoutError:
                    pass
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
                logger.info(
                    "信号[%s] %s confidence=%.2f source=%s",
                    symbol,
                    result["signal"]["bias"],
                    result["signal"]["confidence"],
                    result["source"],
                )
            except Exception:
                logger.exception("信号生成失败 %s", symbol)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
