"""信号评估后台任务（P3 升级核心组件）。

职责
====
- 周期性地（默认 5 分钟一轮）按 (symbol, window_minutes) 组合，
  从 ``signals`` + ``signal_lifecycle`` 联合视图聚合 13 个评估指标，
  写入 ``signal_evaluation`` 表。
- 让 LLM Prompt 注入路径能拿到"系统级评估摘要"，形成
  "决策 → lifecycle → 评估 → 反馈进 prompt"的可量化闭环。

为什么要单独做评估系统
======================
现有的 P2 自我反馈段（``LLMAgent._render_self_feedback``）只会输出"近 5 条
成绩单"逐条表格，对单条信号的细节足够，但缺以下系统级视角：

- **方向翻转率**：连续两条信号方向相反的频次，反映 LLM 在窄幅震荡市的
  whipsaw 倾向。
- **Brier score**：``mean((conf - actual_outcome)^2)``，衡量置信度校准
  程度（< 0.25 表示比随机猜好；> 0.30 说明 conf 与实际结果脱节）。
- **Sharpe 估计**：``avg_pnl / std(pnl)``，无年化，仅做窗口间相对比较。

这些都是"宏观判别"指标，每条样本算不出来，必须在窗口内聚合。

性能预算
========
- 每 ``signal_evaluation_interval_seconds``（默认 300s）一轮；
- 单 symbol × 窗口数（默认 3）× 单次 SQL 拉取（≤ 几百行） + 内存聚合；
- 整轮 < 200ms 可控；写入失败不阻塞主路径，与 lifecycle 任务对齐。
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# 常量：lifecycle 状态分类
# ----------------------------------------------------------------------
# 与 schema CHECK 约束保持一致，集中放这里方便排错时一眼看清归类。
_WIN_STATUSES: frozenset[str] = frozenset({"tp1_hit", "tp2_hit"})
_LOSS_STATUSES: frozenset[str] = frozenset({"sl_hit"})
# 曾入场但超时 = 进了 entry_zone 但没碰 SL/TP，相当于"判断方向准不准未知"。
# 不计入胜率分母（与 _render_self_feedback 同口径）。
_EXPIRED_AFTER_TRIGGER: str = "expired"


def _safe_float(v: Any) -> Optional[float]:
    """
    把任意输入转成 float；失败返回 None
    --------------------------------------------------------------
    参数：
        v: 任意输入（asyncpg 的 Decimal / int / str / None 都可能）
    返回：
        float 或 None。NaN / Inf 一并视为非法返回 None，避免污染统计聚合。
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _stddev(values: Sequence[float]) -> Optional[float]:
    """
    样本标准差（n-1 分母），不足 2 个样本时返回 None
    --------------------------------------------------------------
    参数：
        values: 数值序列
    返回：
        样本标准差 / None
    说明：
        Sharpe 估计的分母用样本标准差更保守。本函数只在评估器内部使用，
        不引入 numpy 依赖（保持评估系统对外只依赖标准库 + asyncpg）。
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    if var < 0:
        return 0.0
    return math.sqrt(var)


class SignalEvaluator:
    """
    信号评估后台任务驱动器
    --------------------------------------------------------------
    用法：
        evaluator = SignalEvaluator(settings, repos, symbols=["ETH-USDT-SWAP"])
        await evaluator.start()  # FastAPI lifespan 启动
        await evaluator.stop()
    """

    def __init__(
        self,
        settings: Settings,
        repos: Repositories,
        symbols: Sequence[str],
    ):
        """
        构造评估器
        --------------------------------------------------------------
        参数：
            settings : 全局配置（读 enable_signal_evaluation /
                       signal_evaluation_interval_seconds /
                       signal_evaluation_windows_minutes）
            repos    : 数据仓储
            symbols  : 待评估的合约列表
        """
        self.settings = settings
        self.repos = repos
        self.symbols: List[str] = list(symbols)
        self._task: Optional[asyncio.Task[Any]] = None
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动后台循环（幂等）。"""
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="signal-evaluator")

    async def stop(self) -> None:
        """优雅关停：通知循环退出 + 等任务结束。"""
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _loop(self) -> None:
        """
        后台循环主体
        --------------------------------------------------------------
        节奏来自 settings.signal_evaluation_interval_seconds（默认 300s）。
        异常吞掉只记日志：评估器是"锦上添花"，不能因为它把整个进程打挂。
        """
        interval = max(
            30, int(getattr(self.settings, "signal_evaluation_interval_seconds", 300))
        )
        windows = list(
            getattr(self.settings, "signal_evaluation_windows_minutes", [60, 360, 1440])
        ) or [1440]
        logger.info(
            "信号评估任务已启动（symbols=%s，windows=%s 分钟，interval=%ds）",
            self.symbols,
            windows,
            interval,
        )
        while not self._stopping.is_set():
            try:
                await self.tick_once(windows=windows)
            except Exception:
                logger.exception("信号评估任务一轮执行失败，下个周期继续")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def tick_once(
        self,
        windows: Optional[Sequence[int]] = None,
    ) -> Dict[str, Dict[int, Optional[int]]]:
        """
        手动 / 周期触发一轮评估
        --------------------------------------------------------------
        参数：
            windows: 评估窗口（分钟）列表；None 时回退到 settings 配置。
        返回：
            {symbol: {window_minutes: signal_evaluation.id 或 None}}
            None 表示该 (symbol, window) 本轮写入失败 / 无样本。
        说明：
            返回值供单元测试与 admin 接口断言；任务循环里只关心是否抛异常。
        """
        result: Dict[str, Dict[int, Optional[int]]] = {}
        if windows is None:
            windows = list(
                getattr(
                    self.settings, "signal_evaluation_windows_minutes", [60, 360, 1440]
                )
            ) or [1440]

        for symbol in self.symbols:
            result[symbol] = {}
            for window_minutes in windows:
                try:
                    inserted_id = await self._evaluate_window(
                        symbol=symbol, window_minutes=int(window_minutes)
                    )
                except Exception:
                    logger.warning(
                        "信号评估失败 symbol=%s window=%d 分钟（不影响主路径）",
                        symbol,
                        int(window_minutes),
                        exc_info=True,
                    )
                    inserted_id = None
                result[symbol][int(window_minutes)] = inserted_id
        return result

    # ------------------------------------------------------------------
    # 评估计算
    # ------------------------------------------------------------------
    async def _evaluate_window(
        self,
        symbol: str,
        window_minutes: int,
    ) -> Optional[int]:
        """
        计算并写入单个 (symbol, window_minutes) 的评估指标
        --------------------------------------------------------------
        参数：
            symbol         : 合约代码
            window_minutes : 评估窗口（分钟）
        返回：
            写入后的 signal_evaluation.id；样本为空时返回 None（不写表）。
        """
        since = datetime.now(timezone.utc) - timedelta(minutes=int(window_minutes))
        rows = await self.repos.fetch_signals_for_evaluation(
            symbol=symbol, since=since
        )
        if not rows:
            logger.debug(
                "信号评估 symbol=%s window=%d 分钟：窗口内无样本，跳过",
                symbol,
                window_minutes,
            )
            return None

        metrics = self.compute_metrics(rows)
        return await self.repos.insert_signal_evaluation(
            symbol=symbol,
            window_minutes=window_minutes,
            metrics=metrics,
        )

    @classmethod
    def compute_metrics(cls, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于一批 signals + lifecycle 联合视图行计算 13 个评估指标
        --------------------------------------------------------------
        参数：
            rows: ``Repositories.fetch_signals_for_evaluation`` 返回的列表，
                  每行至少含：bias / confidence / lifecycle_status / pnl_pct /
                  max_favorable_pct / max_adverse_pct / triggered_at
                  （顺序需按 ts 升序，由 SQL 保证）
        返回：
            指标 dict，可直接 ``**metrics`` 解包给 ``insert_signal_evaluation``。
        说明：
            纯函数实现：不读数据库、不依赖 self，便于单元测试拼装固定输入。
            当样本不足某项指标的最低条件时（如 win_rate 需 wins+losses>=1），
            该项写 None；上层 SQL 列允许 NULL。
        """
        rows_list: List[Dict[str, Any]] = list(rows)
        total = len(rows_list)

        # ----- 1) 触发率段 -----
        triggered_rows = [r for r in rows_list if r.get("triggered_at") is not None]
        triggered_count = len(triggered_rows)
        fill_rate = (triggered_count / total) if total > 0 else None

        # ----- 2) 判断质量段（仅曾入场） -----
        wins = 0
        losses = 0
        expired_after_triggered = 0
        pnl_list: List[float] = []
        mfp_list: List[float] = []
        map_list: List[float] = []
        for r in triggered_rows:
            status = r.get("lifecycle_status")
            if status in _WIN_STATUSES:
                wins += 1
            elif status in _LOSS_STATUSES:
                losses += 1
            elif status == _EXPIRED_AFTER_TRIGGER:
                expired_after_triggered += 1
            pnl = _safe_float(r.get("pnl_pct"))
            if pnl is not None:
                pnl_list.append(pnl)
            mfp = _safe_float(r.get("max_favorable_pct"))
            if mfp is not None:
                mfp_list.append(mfp)
            mav = _safe_float(r.get("max_adverse_pct"))
            if mav is not None:
                map_list.append(mav)

        decided = wins + losses
        win_rate: Optional[float] = (wins / decided) if decided > 0 else None
        avg_pnl_pct: Optional[float] = (
            sum(pnl_list) / len(pnl_list) if pnl_list else None
        )
        total_pnl_pct: Optional[float] = sum(pnl_list) if pnl_list else None
        max_favorable_avg: Optional[float] = (
            sum(mfp_list) / len(mfp_list) if mfp_list else None
        )
        max_adverse_avg: Optional[float] = (
            sum(map_list) / len(map_list) if map_list else None
        )

        # ----- 3) Sharpe 估计（无年化，仅做相对比较） -----
        # 公式：avg_pnl / std(pnl)，n<2 或 std≈0 时为 None。
        sharpe_estimated: Optional[float] = None
        if avg_pnl_pct is not None and len(pnl_list) >= 2:
            sd = _stddev(pnl_list)
            if sd is not None and sd > 1e-12:
                sharpe_estimated = avg_pnl_pct / sd

        # ----- 4) 方向翻转率 -----
        # 相邻两条信号方向相反计 1 次；observe / neutral 也算"方向"参与对比，
        # 因为"信号引擎从看多变成观望再变成看空"已经体现摇摆。
        # 但若中间夹了一条 neutral，long↔short 的"摇摆"应被算 2 次（更敏感）。
        direction_flip_count = 0
        prev_bias: Optional[str] = None
        for r in rows_list:
            cur = (r.get("bias") or "").strip() or None
            if prev_bias is not None and cur is not None and cur != prev_bias:
                direction_flip_count += 1
            prev_bias = cur
        direction_flip_rate: Optional[float] = (
            direction_flip_count / max(total - 1, 1) if total > 1 else None
        )

        # ----- 5) Brier score（置信度校准） -----
        # actual_outcome ∈ {0, 1}：tp_hit -> 1，sl_hit -> 0；
        # neutral / 未结算 / expired_after_triggered 不计入（无明确"中没中"）。
        # Brier = mean((conf - actual)^2)。
        brier_terms: List[float] = []
        for r in rows_list:
            status = r.get("lifecycle_status")
            if status not in _WIN_STATUSES and status not in _LOSS_STATUSES:
                continue
            conf = _safe_float(r.get("confidence"))
            if conf is None:
                continue
            actual = 1.0 if status in _WIN_STATUSES else 0.0
            brier_terms.append((conf - actual) ** 2)
        brier_score: Optional[float] = (
            sum(brier_terms) / len(brier_terms) if brier_terms else None
        )

        return {
            "total_signals": total,
            "triggered_count": triggered_count,
            "fill_rate": fill_rate,
            "wins": wins,
            "losses": losses,
            "expired_after_triggered": expired_after_triggered,
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl_pct,
            "total_pnl_pct": total_pnl_pct,
            "max_favorable_avg": max_favorable_avg,
            "max_adverse_avg": max_adverse_avg,
            "sharpe_estimated": sharpe_estimated,
            "direction_flip_count": direction_flip_count,
            "direction_flip_rate": direction_flip_rate,
            "brier_score": brier_score,
        }
