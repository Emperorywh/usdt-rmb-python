"""Signal service: fuses rule engine + LLM agent and persists outputs.

P2 升级：
* 在 INSERT signals 后**同事务节奏**追加一条 ``signal_lifecycle`` 行
  （pending / 或 neutral 的直接 expired），让生命周期跟踪任务能拿到信号；
* 在调用 LLMAgent.analyze 之前从 ``signal_lifecycle`` 读最近 N 条已结算 +
  当前未结算行，作为 ``recent_settled / open_lifecycle`` 透传给 LLM；
  这样 llm_agent 不直接依赖 repos，职责清晰。

P3 升级（决策层守门员 + 服务端 size 覆盖）：
* 在 LLM 调用**前**插入 2 道闸门：
    闸 1（ATR 过低）—— ``_decision_atr_gate``
    闸 2（连续止损冷静期）—— ``_decision_cooldown_gate``
  任一触发都跳过 LLM 调用，直接构造一条 ``bias=neutral`` 的"被门禁的"
  TradingSignal 落库，``source="rules+llm(gated:<reason>)"``。
* 在 LLM 调用**后**插入 2 道闸门：
    闸 3（方向反转 + 价格微动）—— ``_decision_direction_stability_gate``
    闸 4（规则 vs LLM 长期反向且历史胜率 < 40%）—— ``_decision_rule_conflict_gate``
  任一触发都把 LLM 输出的 bias 改 neutral，但**保留** reason / risk /
  suggestion 文本（写入审计），``confidence`` 收紧到 ≤0.5。
* 在所有闸门通过后，对最终 ``position_size_pct`` 做服务端覆盖
  （``_apply_size_override``）：基于
      base = (conf - 0.5) × 2 × kelly × win_rate_factor
  并 clamp 到 ``[0, decision_max_position_size_pct]``，让最终下单仓位
  与历史胜率、置信度严格挂钩，从根上消除"size 与 conf 脱钩"的旧问题。
所有 P3 行为受 ``settings.enable_decision_gates`` 单一总开关控制，
关闭时退回 P2，便于灰度回滚。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
        # 邮件实际发送链路走 Resend HTTPS REST API（不再依赖 SMTP 端口 / STARTTLS）。
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

        settings = self.factor_aggregator.settings
        gates_enabled = bool(getattr(settings, "enable_decision_gates", False))

        # ====================================================================
        # P3 升级：LLM 前置闸门（闸 1 ATR + 闸 2 连续止损冷静期）
        # --------------------------------------------------------------------
        # 任一闸门触发都直接构造一条 gated-neutral signal 入库，跳过 LLM 调用。
        # 这是"省钱 + 防 whipsaw"的第一道保险——窄幅震荡 / 连续止损时 LLM
        # 价值最低，绕过它直接 neutral 是更经济的选择。
        # ====================================================================
        gated_pre_llm_reason: Optional[str] = None
        if gates_enabled:
            atr_gate_reason = self._decision_atr_gate(factors)
            if atr_gate_reason is not None:
                gated_pre_llm_reason = atr_gate_reason
            else:
                cooldown_reason = await self._decision_cooldown_gate(symbol)
                if cooldown_reason is not None:
                    gated_pre_llm_reason = cooldown_reason

        if gated_pre_llm_reason is not None:
            final = self._make_gated_neutral_signal(
                rule_signal=rule_signal,
                gate_reason=gated_pre_llm_reason,
            )
            reasoning_content = None
            source = f"rules+llm(gated:{gated_pre_llm_reason})"
            llm_result = None
            should_persist = True
            logger.info(
                "P3 决策闸门：symbol=%s 触发前置闸门 reason=%s，跳过 LLM 调用直接 neutral",
                symbol, gated_pre_llm_reason,
            )
        else:
            # llm_agent.analyze 返回 LLMAnalysisResult（包装了 TradingSignal +
            # 思考模式下的 reasoning_content）；调用失败 / 未启用时返回 None。
            llm_result = await self.llm_agent.analyze(
                symbol=symbol,
                factors=factors,
                rule_signal=rule_signal,
                rule_score=rule_score,
                rule_contributions=contributions,
                recent_settled=recent_settled,
            )

            if llm_result is not None:
                final = llm_result.signal
                reasoning_content = llm_result.reasoning_content
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

            # ====================================================================
            # P3：LLM 后置闸门（闸 3 方向稳定性 + 闸 4 规则 vs LLM 冲突保护）
            # --------------------------------------------------------------------
            # 仅在"真正得到一个有方向的 LLM 输出"时检查；neutral / 缓存命中
            # 都跳过——前者已经满足"不交易"目的，后者本来就不会落库。
            # ====================================================================
            if (
                gates_enabled
                and llm_result is not None
                and not llm_result.from_cache
                and final.bias in ("long", "short")
            ):
                gate3_reason = await self._decision_direction_stability_gate(
                    symbol=symbol, llm_signal=final, factors=factors
                )
                if gate3_reason is not None:
                    final = self._force_neutral_preserving_text(
                        signal=final, gate_reason=gate3_reason
                    )
                    source = f"rules+llm(gated:{gate3_reason})"
                    logger.info(
                        "P3 决策闸门：symbol=%s 触发闸 3 方向稳定性，降为 neutral",
                        symbol,
                    )
                else:
                    gate4_reason = await self._decision_rule_conflict_gate(
                        symbol=symbol, llm_signal=final, rule_score=rule_score
                    )
                    if gate4_reason is not None:
                        final = self._force_neutral_preserving_text(
                            signal=final, gate_reason=gate4_reason
                        )
                        source = f"rules+llm(gated:{gate4_reason})"
                        logger.info(
                            "P3 决策闸门：symbol=%s 触发闸 4 规则冲突保护，降为 neutral",
                            symbol,
                        )

            # ====================================================================
            # P3：服务端 size 覆盖
            # --------------------------------------------------------------------
            # 即便所有闸门都通过，最终 position_size_pct 也由服务端按
            # (conf - 0.5) × 2 × win_rate_factor × kelly 重算，clamp 到
            # [0, decision_max_position_size_pct]。这才是"size 与 conf 严格
            # 挂钩"的唯一可信入口——LLM 输出的 size 仅作"上限建议"。
            # ====================================================================
            if (
                gates_enabled
                and llm_result is not None
                and not llm_result.from_cache
                and final.bias in ("long", "short")
            ):
                final = self._apply_size_override(
                    signal=final, recent_settled=recent_settled
                )

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
            # P3 升级：neutral 信号不写 lifecycle——
            #   - 它没有 plan（entry_zone/sl/tp 都为 None），写进去也只会被
            #     lifecycle tracker 立刻判 expired，纯属噪声；
            #   - 闸门触发的 gated-neutral 同样落到这条分支被跳过，与计划一致。
            if (
                signal_id is not None
                and final.bias != "neutral"
                and bool(
                    getattr(
                        self.factor_aggregator.settings,
                        "enable_lifecycle_tracking",
                        False,
                    )
                )
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
        #   3) email_sender 已注入且启用（Resend API Key / 发件人地址齐备）
        # 走 asyncio.create_task 后台发送，绝不阻塞信号生成主路径；任意失败
        # 都不会向上抛错，仅打 warning 日志。Resend SDK 的同步阻塞 HTTP 调用
        # 内部由 EmailSender 通过 asyncio.to_thread 丢到 executor 上跑。
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

    # ==================================================================
    # P3：决策层守门员（4 道闸门）+ 服务端 size 覆盖
    # ==================================================================
    # 设计原则：
    #   - 每个闸门都是纯方法 + 早返回风格；触发返回 reason 字符串，
    #     未触发返回 None。这样上层 ``generate()`` 只需链式 if 判断,
    #     不需要复杂的 result 对象。
    #   - 所有阈值都从 settings 读，可在 .env 即时调参不必改代码。
    #   - 闸门触发只产生"结构性降级"（neutral / 仓位归零），不抛异常,
    #     主路径不会被一道闸门门禁打挂。

    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        """与 evaluator 同名工具函数：把任意输入转 float，失败 / NaN / Inf 返回 None。"""
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        # 与 math.isfinite 等价但不引 import：业务路径里 NaN/Inf 一律视为非法
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f

    def _decision_atr_gate(self, factors: Dict[str, Any]) -> Optional[str]:
        """
        闸 1：ATR 过低门禁
        --------------------------------------------------------------
        参数：
            factors: FactorAggregator.compute 输出
        返回：
            触发时返回 reason 字符串（如 "atr_too_low"），未触发返回 None。
        说明：
            atr_pct_15m = atr_14 / last_close。当此比例低于
            ``decision_min_atr_pct_15m``（默认 0.0025 = 0.25%）时，
            视为窄幅震荡，跳过 LLM 调用直接 neutral。
            数据缺失时（atr_14 / last_close 任一为 None）保守起见**不**触发,
            让 LLM 仍有机会基于其它信号判断——而不是因为单条因子缺失就不交易。
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
                "P3 闸 1 触发：atr_pct_15m=%.5f < threshold=%.5f",
                atr_pct, threshold,
            )
            return "atr_too_low"
        return None

    async def _decision_cooldown_gate(self, symbol: str) -> Optional[str]:
        """
        闸 2：连续止损冷静期
        --------------------------------------------------------------
        参数：
            symbol: 合约
        返回：
            触发时返回 "cooldown_consecutive_sl"，未触发返回 None。
        说明：
            从 fetch_recent_settled_lifecycles 拉最近 N 条已结算行（按
            updated_at DESC 排序）。如果"最近连续 N 条都是 sl_hit"且
            最新一条的 ``exit_at`` 距现在 < ``decision_cooldown_minutes``,
            视为冷静期触发。
            阈值 N = ``decision_cooldown_consecutive_sl_threshold``（默认 2）。
            查询失败 / 数据不足时返回 None：保守起见不无故触发冷静期。
        """
        settings = self.factor_aggregator.settings
        n_threshold = int(
            getattr(settings, "decision_cooldown_consecutive_sl_threshold", 2)
        )
        cooldown_minutes = int(
            getattr(settings, "decision_cooldown_minutes", 60)
        )
        if n_threshold <= 0 or cooldown_minutes <= 0:
            return None
        try:
            rows = await self.repos.fetch_recent_settled_lifecycles(
                symbol=symbol, limit=max(n_threshold, 1)
            )
        except Exception:
            logger.warning(
                "P3 闸 2 查询 fetch_recent_settled_lifecycles 失败 symbol=%s（不阻塞）",
                symbol, exc_info=True,
            )
            return None
        if len(rows) < n_threshold:
            return None
        # 最近 n 条全是 sl_hit
        recent = rows[:n_threshold]
        if not all(str(r.get("status")) == "sl_hit" for r in recent):
            return None
        # 最新一条的 exit_at 距现在 < cooldown_minutes
        latest_exit = recent[0].get("exit_at")
        if latest_exit is None:
            return None
        if latest_exit.tzinfo is None:
            latest_exit = latest_exit.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        elapsed_minutes = (now - latest_exit).total_seconds() / 60.0
        if elapsed_minutes >= cooldown_minutes:
            return None
        logger.debug(
            "P3 闸 2 触发：连续 %d 次 sl_hit，距最新 exit_at 仅 %.1f 分钟（< %d）",
            n_threshold, elapsed_minutes, cooldown_minutes,
        )
        return "cooldown_consecutive_sl"

    async def _decision_direction_stability_gate(
        self,
        symbol: str,
        llm_signal: TradingSignal,
        factors: Dict[str, Any],
    ) -> Optional[str]:
        """
        闸 3：方向稳定性（反向 + 价格微动）
        --------------------------------------------------------------
        参数：
            symbol     : 合约
            llm_signal : 当前 LLM 输出的 TradingSignal（bias ∈ {long, short}）
            factors    : 因子聚合结果（取 ATR(1h) 与 last_close）
        返回：
            触发时返回 "direction_flip_micro_move"，未触发返回 None。
        说明：
            从 ``fetch_latest_signal_judgment`` 拿最近 1 条信号；如果与本次
            方向相反，且 ``|current_price - last_price| < 阈值 × ATR(1h)``,
            视为"价格没有真实位移就被 whipsaw 反向"，触发降级。
            阈值 = ``decision_direction_flip_min_price_move_atr_1h``（默认 0.6）。
            数据缺失（无历史信号 / ATR 缺失）一律返回 None：尚不构成判断条件。
        """
        settings = self.factor_aggregator.settings
        threshold_mult = self._safe_float(
            getattr(settings, "decision_direction_flip_min_price_move_atr_1h", None)
        )
        if threshold_mult is None or threshold_mult <= 0:
            return None
        try:
            last = await self.repos.fetch_latest_signal_judgment(symbol)
        except Exception:
            logger.warning(
                "P3 闸 3 查询 fetch_latest_signal_judgment 失败 symbol=%s（不阻塞）",
                symbol, exc_info=True,
            )
            return None
        if last is None:
            return None
        last_bias = str(last.get("bias") or "").strip()
        if last_bias not in ("long", "short"):
            return None
        # 仅在"方向相反"时检查；同向无需触发本闸
        if (last_bias == "long" and llm_signal.bias == "long") or (
            last_bias == "short" and llm_signal.bias == "short"
        ):
            return None
        # 取 1h ATR + 当前价
        by_tf = (factors or {}).get("by_timeframe") or {}
        ms_1h = ((by_tf.get("1h") or {}).get("market_structure")) or {}
        atr_1h = self._safe_float(ms_1h.get("atr_14"))
        current_price = self._safe_float(ms_1h.get("last_close"))
        if atr_1h is None or atr_1h <= 0 or current_price is None:
            return None
        # 取上次"入场中点"作为价格基准；缺失则退回上次 entry_zone 中点 / mid 估算
        last_entry_low = self._safe_float(last.get("entry_zone_low"))
        last_entry_high = self._safe_float(last.get("entry_zone_high"))
        if last_entry_low is not None and last_entry_high is not None:
            last_price = (last_entry_low + last_entry_high) / 2.0
        else:
            ez = last.get("entry_zone")
            if isinstance(ez, (list, tuple)) and len(ez) == 2:
                a = self._safe_float(ez[0])
                b = self._safe_float(ez[1])
                last_price = (a + b) / 2.0 if a is not None and b is not None else None
            else:
                last_price = None
        if last_price is None:
            return None
        price_move = abs(current_price - last_price)
        threshold = threshold_mult * atr_1h
        if price_move < threshold:
            logger.debug(
                "P3 闸 3 触发：last_bias=%s now=%s |Δprice|=%.4f < %.2f × ATR(1h)=%.4f",
                last_bias, llm_signal.bias, price_move, threshold_mult, threshold,
            )
            return "direction_flip_micro_move"
        return None

    async def _decision_rule_conflict_gate(
        self,
        symbol: str,
        llm_signal: TradingSignal,
        rule_score: float,
    ) -> Optional[str]:
        """
        闸 4：规则 vs LLM 冲突保护
        --------------------------------------------------------------
        参数：
            symbol     : 合约
            llm_signal : 当前 LLM 输出（bias ∈ {long, short}）
            rule_score : 规则引擎本轮打分（[-1, 1]）
        返回：
            触发时返回 "rule_llm_conflict_low_winrate"，未触发返回 None。
        说明：
            仅当本轮 ``sign(rule_score)`` 与 ``llm_signal.bias`` 反向时检查：
                long  vs rule_score < 0
                short vs rule_score > 0
            从 fetch_recent_signals_for_conflict_check 拉近 N 条已结算样本,
            统计"历史上同样反向冲突时 LLM 的胜率"——
                conflict_sample = bias 与 rule_score 反向且已结算
                wins = pnl_pct > 0
                胜率 = wins / conflict_sample
            如果 conflict_sample ≥ window/2 且胜率 < 阈值（默认 0.4），
            触发降级。
            数据不足时不触发：让 LLM 在样本积累期照常出手。
        """
        settings = self.factor_aggregator.settings
        window = int(getattr(settings, "decision_rule_llm_conflict_window", 5))
        winrate_threshold = self._safe_float(
            getattr(settings, "decision_rule_llm_conflict_winrate_threshold", None)
        )
        if window <= 0 or winrate_threshold is None or winrate_threshold <= 0:
            return None
        # 当前是否冲突？
        rs = self._safe_float(rule_score) or 0.0
        bias = llm_signal.bias
        is_current_conflict = (
            (bias == "long" and rs < 0) or (bias == "short" and rs > 0)
        )
        if not is_current_conflict:
            return None
        try:
            rows = await self.repos.fetch_recent_signals_for_conflict_check(
                symbol=symbol, limit=window
            )
        except Exception:
            logger.warning(
                "P3 闸 4 查询 fetch_recent_signals_for_conflict_check 失败 symbol=%s（不阻塞）",
                symbol, exc_info=True,
            )
            return None
        # 历史样本里"同样反向冲突"的部分
        conflict_rows: List[Dict[str, Any]] = []
        for r in rows:
            r_bias = str(r.get("bias") or "").strip()
            r_score = self._safe_float(r.get("rule_score")) or 0.0
            if (r_bias == "long" and r_score < 0) or (
                r_bias == "short" and r_score > 0
            ):
                conflict_rows.append(r)
        # 样本量门槛：至少 window/2 条（不取整不准，向下取整即可）
        min_samples = max(2, window // 2)
        if len(conflict_rows) < min_samples:
            return None
        wins = 0
        decided = 0
        for r in conflict_rows:
            status = str(r.get("lifecycle_status") or "")
            pnl = self._safe_float(r.get("pnl_pct"))
            # 只有 sl_hit / tp_hit 有"决定性结论"，expired/invalidated 跳过
            if status in ("tp1_hit", "tp2_hit"):
                wins += 1
                decided += 1
            elif status == "sl_hit":
                decided += 1
            elif pnl is not None and status in ("expired",):
                # 防御性：如果 expired 但 pnl 明显 > 0，也归为"赢一半"
                # —— 实际生产里 expired 多为持平，这里保留旧逻辑只看 sl/tp。
                pass
        if decided < min_samples:
            return None
        winrate = wins / decided
        if winrate < winrate_threshold:
            logger.debug(
                "P3 闸 4 触发：conflict_samples=%d decided=%d winrate=%.2f < %.2f",
                len(conflict_rows), decided, winrate, winrate_threshold,
            )
            return "rule_llm_conflict_low_winrate"
        return None

    @staticmethod
    def _make_gated_neutral_signal(
        rule_signal: TradingSignal,
        gate_reason: str,
    ) -> TradingSignal:
        """
        构造一条"被前置闸门拦截"的 neutral 信号
        --------------------------------------------------------------
        参数：
            rule_signal : 当轮规则引擎初判（用于继承基础文本字段）
            gate_reason : 闸门触发原因，会拼到 reason / suggestion 中
        返回：
            一条干净的 neutral TradingSignal（plan 字段全部为 None）。
        说明：
            confidence 一律置 0.0：这条 signal 是被守门员拦截的，不应被
            下游误用为"低 conf 但仍有方向倾向"的判断；neutral 时下游也不会
            读取 confidence 来做仓位计算。
        """
        zh_map = {
            "atr_too_low": "ATR(15m) 占比低于阈值，市场无可交易波动",
            "cooldown_consecutive_sl": "近期连续止损触发冷静期，强制观望",
            "direction_flip_micro_move": "方向反转但价格未实质位移，疑似 whipsaw，降级观望",
            "rule_llm_conflict_low_winrate": "规则引擎与 LLM 长期反向且历史胜率不足，降级观望",
        }
        zh_reason = zh_map.get(gate_reason, gate_reason)
        return TradingSignal(
            bias="neutral",
            confidence=0.0,
            reason=f"[决策守门员:{gate_reason}] {zh_reason}",
            risk=f"决策守门员触发：{zh_reason}（参见 settings.enable_decision_gates 与相关阈值）",
            suggestion=(
                "本周期建议观望，等待守门员条件解除后再做判断。"
                "仅供参考，不构成交易指令"
            ),
            timeframe_alignment=dict(rule_signal.timeframe_alignment or {}),
            invalidation_conditions=[],
        )

    @staticmethod
    def _force_neutral_preserving_text(
        signal: TradingSignal,
        gate_reason: str,
    ) -> TradingSignal:
        """
        把一条 long/short 的 LLM 输出降级为 neutral，但**保留**原始文本字段
        --------------------------------------------------------------
        参数：
            signal      : 原 LLM 输出（bias ∈ {long, short}）
            gate_reason : 闸门触发原因，会拼到 reason 字段开头
        返回：
            一条 neutral TradingSignal：
                - bias=neutral，所有 plan 字段被 schema validator 自动清空
                - confidence 收紧到 min(原值, 0.5)
                - reason 前缀加 "[决策守门员:reason] "，保留原文本作为审计
        说明：
            timeframe_alignment 与 invalidation_conditions 保留原值，
            前者作为历史方向投票审计、后者方便在前端继续展示原失效条件。
        """
        return TradingSignal(
            bias="neutral",
            confidence=min(float(signal.confidence), 0.5),
            reason=f"[决策守门员:{gate_reason}] {signal.reason}",
            risk=signal.risk,
            suggestion=signal.suggestion,
            timeframe_alignment=dict(signal.timeframe_alignment or {}),
            invalidation_conditions=list(signal.invalidation_conditions or []),
        )

    def _apply_size_override(
        self,
        signal: TradingSignal,
        recent_settled: List[Dict[str, Any]],
    ) -> TradingSignal:
        """
        服务端 position_size_pct 覆盖（半凯利近似 + 历史胜率因子）
        --------------------------------------------------------------
        参数：
            signal         : 通过所有闸门后的 LLM 输出（bias ∈ {long, short}）
            recent_settled : 近 N 条已结算 lifecycle 行；空时退回保守路径
        返回：
            position_size_pct 被覆盖后的新 TradingSignal（其它字段不变）。
        计算公式：
            base = (conf - 0.5) × 2                      # conf ≤ 0.5 → 0
            win_rate_factor = clamp(2 × win_rate - 0.5, 0.2, 1.0)
                # win_rate 来自近期成绩单；缺数据时按 0.5 处理 → factor=0.5
            kelly = decision_kelly_aggressiveness（默认 0.5，半凯利）
            new_size = base × win_rate_factor × kelly
            clamp 到 [0, decision_max_position_size_pct]
        说明：
            这里没有用完整 Kelly 公式（需要 b/p 估计），用"胜率敏感的乘性
            因子"近似——对中小样本量更稳健，回测中表现也更接近实盘。
            最终 clamp 上限决定"无论 conf 多高，单笔最多打多少"。
        """
        settings = self.factor_aggregator.settings
        max_size = self._safe_float(
            getattr(settings, "decision_max_position_size_pct", None)
        )
        if max_size is None or max_size <= 0:
            return signal
        kelly = self._safe_float(
            getattr(settings, "decision_kelly_aggressiveness", None)
        ) or 0.5

        conf = float(signal.confidence)
        base = max(0.0, (conf - 0.5) * 2.0)
        # win_rate_factor：近期已结算样本的 wins / decided
        wins = 0
        decided = 0
        for r in recent_settled or []:
            status = str(r.get("status") or "")
            if status in ("tp1_hit", "tp2_hit"):
                wins += 1
                decided += 1
            elif status == "sl_hit":
                decided += 1
        if decided > 0:
            wr = wins / decided
            win_rate_factor = max(0.2, min(1.0, 2.0 * wr - 0.5))
        else:
            # 样本不足时给 0.5：既不极端激进、也不极端保守
            win_rate_factor = 0.5
        new_size = base * win_rate_factor * kelly
        new_size = max(0.0, min(float(max_size), new_size))
        # 用 model_copy(update=) 仅替换 position_size_pct；其他字段（含 plan）
        # 完全保持原状。Pydantic v2 的 model_copy 不会重新跑 validators，
        # 但本字段本身受 ge=0 / le=0.25 约束，越界值已在上方 clamp 过。
        return signal.model_copy(update={"position_size_pct": float(round(new_size, 4))})

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
