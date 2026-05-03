"""IC 校准任务（P2 升级核心组件）。

职责
====
- 后台周期任务（默认每天 02:00 UTC 跑一次，由 lifespan 启动一个简单的
  ``asyncio sleep`` 循环驱动，**不引入** APScheduler / cron）。
- 流程：
    1. 取过去 30 天 / 90 天 ``signals`` 表里 ``source ILIKE '%llm%'`` 的全部记录；
    2. 对每条 signal，递归遍历 ``factors`` JSONB，提取所有"原子数值因子"
       （路径形如 ``by_timeframe.{tf}.{group}.{factor_name}``）；
    3. 取 signal.ts 后 1h / 4h / 24h 时刻 ETH-USDT-SWAP 的 K 线 close，
       计算 forward_return = (px_future - px_now) / px_now；
    4. 对每个 (regime, timeframe, factor_group, factor_name) 组合，
       计算因子值序列与 forward_return 序列的 **Spearman 相关系数**
       （numpy 自实现 rank → pearson，**不引入 scipy**）；
    5. 权重更新规则：
         |IC| < 0.02 → weight = 0（视为噪声）
         |IC| ∈ [0.02, 0.05] → weight ∝ |IC|
         |IC| > 0.05 → weight ∝ |IC| × 1.5（强信号加权）
       同一 (regime, timeframe) 下所有 factor 权重归一到 sum = 1。
    6. UPSERT 回 ``factor_weights`` 表；同时把校准报告（每个 regime 下
       top 10 IC 因子）写到 ``logs/ic_calibration_YYYYMMDD.json``。

降级策略
========
- 30 天总样本 < settings.ic_calibrator_min_total_samples（默认 30）：
  跳过整轮校准并保留上一份权重 + ``logger.warning``。
- 单个 (regime, timeframe) 组合下样本 < ic_calibrator_min_group_samples
  （默认 10）：该组合不更新，保留基线值。

设计取舍
========
- 选 Spearman 而非 Pearson：因子值分布大多偏态（CVD slope、OI 变动百分比等
  都有重尾），秩相关对异常值天然鲁棒，且不假设线性关系。
- 选 forward_return 1h / 4h / 24h 三档：与多周期信号生成节奏对齐——
  5m / 15m 因子主要靠 1h 验证；1h / 4h 因子用 4h 验证；1d 因子用 24h 验证。
  IC 计算时按 timeframe 自动选对应窗口（5m/15m → 1h；1h → 4h；4h/1d → 24h）。
- 不引入 scipy：rank 统计 + Pearson 简单到 30 行 numpy 就能写完，引入
  scipy 仅为这一项不划算（带宽 / 安装复杂度都翻倍）。
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger

logger = get_logger(__name__)


# 因子组白名单：只对这 5 类下的"原子数值因子"做 IC，避免 cvd_close
# 之类的累积量 / 时间戳混进相关性计算。
# 命中规则：路径形如 by_timeframe.{tf}.{group}.{factor_name} 才参与。
_ALLOWED_GROUPS: Tuple[str, ...] = (
    "capital_flow",
    "orderbook",
    "derivatives",
    "market_structure",
    "liquidity",
)

# 黑名单：累积量 / 显式价位列表 / 不参与 IC 的字段（即便落在白名单 group 下也跳过）
_FACTOR_BLACKLIST: Tuple[str, ...] = (
    "cvd_close",          # 累积值，量纲不稳定
    "last_close",         # 价格本身，IC 没意义
    "last_price",
    "supports",
    "resistances",
    "best_bid",
    "best_ask",
    "spread",             # 绝对值，按 spread_bp 走
    "available",          # bool，跳过
    "next_settlement_at",
)

# 默认 forward window（按因子的 timeframe 选）：
#   5m / 15m → 1h；1h → 4h；4h / 1d → 24h
_FORWARD_WINDOW_BY_TF: Dict[str, str] = {
    "5m": "1h",
    "15m": "1h",
    "1h": "4h",
    "4h": "24h",
    "1d": "24h",
}

# forward_window 标签到秒数的映射
_WINDOW_SECONDS: Dict[str, int] = {
    "1h": 3600,
    "4h": 4 * 3600,
    "24h": 24 * 3600,
}

# forward_window 标签到 K 线表周期的映射（取该周期"最早一根 ts ≥ target"的 close）
_WINDOW_KLINE_TF: Dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "24h": "1h",  # 24h 用 1h K 线最临近的一根足够；4h 表稀疏取不准
}


@dataclass
class _FactorRecord:
    """
    单条 (signal, factor) 记录：因子值 + signal 元数据
    -----------------------------------------------------------
    用于在内存里按 (regime, timeframe, factor_group, factor_name)
    聚合后再计算 Spearman IC。
    """

    regime: str
    timeframe: str
    factor_group: str
    factor_name: str
    factor_value: float
    forward_return: float


@dataclass
class _GroupReport:
    """
    单个 (regime, timeframe, factor_group, factor_name) 的 IC 计算结果
    """

    regime: str
    timeframe: str
    factor_group: str
    factor_name: str
    ic_30d: Optional[float]
    ic_90d: Optional[float]
    sample_count_30d: int
    sample_count_90d: int
    weight: float = 0.0  # 归一前先填 |IC| 加权值，归一后再覆盖
    ic_abs_weighted: float = 0.0


@dataclass
class CalibrationReport:
    """
    一次校准任务的整体输出报告
    """

    ran_at: datetime
    started_at: datetime
    finished_at: datetime
    skipped: bool
    skipped_reason: Optional[str] = None
    total_signals_30d: int = 0
    total_signals_90d: int = 0
    total_records_30d: int = 0
    total_records_90d: int = 0
    groups_updated: int = 0
    groups_skipped_low_sample: int = 0
    top_ic_by_regime: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    weights_written: int = 0


# ----------------------------------------------------------------------
# 数学小工具：Spearman = rank(x) ↔ rank(y) 的 Pearson
# ----------------------------------------------------------------------
def _rankdata(arr: np.ndarray) -> np.ndarray:
    """
    用 numpy 计算"平均秩"（与 scipy.stats.rankdata(method='average') 一致）
    -----------------------------------------------------------
    参数：
        arr: 一维 ndarray
    返回：
        与 arr 等长的秩向量（float）。并列值取平均秩。
    实现：
        argsort 两次再处理 ties；ties 用"同值 group 起止位置"求平均秩。
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    # ties 处理：扫一遍 sorted 序列，对相等值段统一赋平均秩
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-indexed
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_ic(x: List[float], y: List[float]) -> Optional[float]:
    """
    计算两个等长序列的 Spearman 相关系数（中文：斯皮尔曼秩相关 IC）
    -----------------------------------------------------------
    参数：
        x, y: 等长 list[float]，已剔除 NaN / None
    返回：
        相关系数；当样本 < 5 / 任一序列方差为 0 时返回 None。
    实现：
        rank(x), rank(y) 后做 Pearson；不引入 scipy。
    """
    if len(x) != len(y) or len(x) < 5:
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if a.size < 5:
            return None
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    # Pearson on ranks
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    denom = float(np.sqrt((ra_c**2).sum() * (rb_c**2).sum()))
    if denom < 1e-12:
        return None
    return float((ra_c * rb_c).sum() / denom)


def _ic_to_weight(ic: Optional[float]) -> float:
    """
    把单个 IC 值映射为"未归一权重"
    -----------------------------------------------------------
    规则（与 P2 提示词同源）：
        |IC| < 0.02         → 0                （噪声）
        |IC| ∈ [0.02, 0.05] → |IC|             （线性）
        |IC| > 0.05         → |IC| × 1.5       （强信号加权）
    返回：
        float ≥ 0；归一前；None 视为 0。
    """
    if ic is None or not np.isfinite(ic):
        return 0.0
    abs_ic = abs(float(ic))
    if abs_ic < 0.02:
        return 0.0
    if abs_ic <= 0.05:
        return abs_ic
    return abs_ic * 1.5


# ----------------------------------------------------------------------
# JSONB 因子矩阵 → 原子因子键值对展开
# ----------------------------------------------------------------------
def _extract_atomic_factors(
    factors_blob: Any,
) -> List[Tuple[str, str, str, float]]:
    """
    从 signals.factors JSONB 中递归提取原子数值因子
    -----------------------------------------------------------
    入参：
        factors_blob: signals.factors 反序列化后的对象。两种结构：
            - 新结构：{"factors": <real_factors>, "rule_score": ..., "rule_contributions": ...}
            - 旧结构：直接是 <real_factors>
    返回：
        [(timeframe, factor_group, factor_name, value), ...]
    说明：
        - 只下钻 by_timeframe.{tf}.{group}.{factor}；
        - 同时把根节点上的 liquidity / position_ratios 视为 timeframe='overall'；
        - 黑名单字段 / 非数值字段 / NaN 一律跳过。
    """
    if not isinstance(factors_blob, dict):
        return []
    real = factors_blob.get("factors") if "factors" in factors_blob else factors_blob
    if not isinstance(real, dict):
        return []

    out: List[Tuple[str, str, str, float]] = []

    by_tf = real.get("by_timeframe") or {}
    if isinstance(by_tf, dict):
        for tf, block in by_tf.items():
            if not isinstance(block, dict):
                continue
            for group, payload in block.items():
                if group not in _ALLOWED_GROUPS or not isinstance(payload, dict):
                    continue
                for k, v in payload.items():
                    if k in _FACTOR_BLACKLIST:
                        continue
                    val = _coerce_numeric(v)
                    if val is not None:
                        out.append((str(tf), group, k, val))

    # 根节点上挂的 liquidity / position_ratios 也算入 timeframe='overall'
    for root_group in ("liquidity", "position_ratios"):
        payload = real.get(root_group)
        if isinstance(payload, dict):
            for k, v in payload.items():
                if k in _FACTOR_BLACKLIST:
                    continue
                val = _coerce_numeric(v)
                if val is not None:
                    out.append(("overall", root_group, k, val))

    return out


def _coerce_numeric(v: Any) -> Optional[float]:
    """
    把 JSONB 反序列化结果尝试转 float；非数值 / NaN / 布尔 → None
    -----------------------------------------------------------
    说明：
        - bool 在 Python 里 isinstance(_, int) 为 True，要先排除。
        - 字符串数字（少见但历史脏数据可能有）也尝试一次 float()。
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, str):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except ValueError:
            return None
    return None


def _resolve_regime_from_factors(factors_blob: Any) -> str:
    """
    从 signals.factors 中提取 regime；缺失时回退 'overall'
    """
    if not isinstance(factors_blob, dict):
        return "overall"
    inner = factors_blob.get("factors") if "factors" in factors_blob else factors_blob
    if isinstance(inner, dict):
        regime = inner.get("regime")
        if isinstance(regime, str) and regime:
            return regime
    return "overall"


# ----------------------------------------------------------------------
# 主任务
# ----------------------------------------------------------------------
class ICCalibrator:
    """
    IC 校准任务驱动器
    -----------------------------------------------------------
    用法：
        calib = ICCalibrator(settings, repos)
        await calib.start()       # FastAPI lifespan 启动
        await calib.run_once(...) # 手动触发（admin API 用）
        await calib.stop()        # 关停
    线程模型：
        单后台 task，串行执行 calibrate()；admin 触发也走同一把 lock，
        不会并发跑两轮。
    """

    def __init__(
        self,
        settings: Settings,
        repos: Repositories,
        symbol: str = "ETH-USDT-SWAP",
    ):
        self.settings = settings
        self.repos = repos
        self.symbol = symbol
        self._task: Optional[asyncio.Task[Any]] = None
        self._stopping = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """启动后台循环（lifespan 调用）。"""
        if self._task is not None:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="ic-calibrator")

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
        -----------------------------------------------------------
        节奏：
            首轮启动延迟 ic_calibrator_first_delay_seconds；
            随后每 ic_calibrator_interval_seconds 跑一次。
        """
        first_delay = max(0, int(self.settings.ic_calibrator_first_delay_seconds))
        interval = max(60, int(self.settings.ic_calibrator_interval_seconds))
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=first_delay)
            return
        except asyncio.TimeoutError:
            pass

        while not self._stopping.is_set():
            try:
                await self.run_once(triggered_by="cron")
            except Exception:
                logger.exception("IC 校准任务执行异常，将在下个周期重试")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def run_once(self, triggered_by: str = "manual") -> CalibrationReport:
        """
        手动 / 周期触发一次校准；带串行锁
        -----------------------------------------------------------
        参数：
            triggered_by: 仅用于日志区分 cron / admin / manual
        返回：
            CalibrationReport（无论 skipped 与否）
        """
        async with self._lock:
            return await self._calibrate(triggered_by=triggered_by)

    async def _calibrate(self, triggered_by: str) -> CalibrationReport:
        """
        校准主流程实现
        -----------------------------------------------------------
        步骤参考模块 docstring。
        """
        started_at = datetime.now(timezone.utc)
        logger.info("IC 校准开始，triggered_by=%s", triggered_by)

        # 1) 拉取 90 天 LLM 信号
        since_90d = started_at - timedelta(days=90)
        since_30d = started_at - timedelta(days=30)
        try:
            signals_90d = await self.repos.fetch_signals_with_factors_since(
                symbol=self.symbol, since=since_90d
            )
        except Exception:
            logger.exception("IC 校准拉取 signals 失败，本轮跳过")
            return CalibrationReport(
                ran_at=started_at,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                skipped=True,
                skipped_reason="fetch_signals_failed",
            )
        signals_30d = [s for s in signals_90d if s["ts"] >= since_30d]

        min_total = int(self.settings.ic_calibrator_min_total_samples)
        if len(signals_30d) < min_total:
            logger.warning(
                "IC 校准 30 天样本不足 %d 条（实际 %d），跳过本轮，保留上一份权重",
                min_total,
                len(signals_30d),
            )
            return CalibrationReport(
                ran_at=started_at,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                skipped=True,
                skipped_reason="too_few_total_samples",
                total_signals_30d=len(signals_30d),
                total_signals_90d=len(signals_90d),
            )

        # 2) 展开成 (signal, factor) 记录 + 算 forward_return
        records_30d: List[_FactorRecord] = []
        records_90d: List[_FactorRecord] = []
        # 缓存：同一 signal 的 forward_return 结果，避免对每个原子因子都查一次 DB
        for sig in signals_90d:
            ts: datetime = sig["ts"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # 取 signal 时刻的"当前价"：优先用 1h K 线在该时刻最近一根的 close；
            # 用同一个查询函数，但传入 ts 自身（而非 ts + Δh），近似当前价。
            px_now = await self._safe_kline_close(
                tf="1m", target_ts=ts
            ) or await self._safe_kline_close(tf="5m", target_ts=ts)
            if px_now is None or px_now <= 0:
                continue

            # 各 forward_window 的未来价
            future_px: Dict[str, Optional[float]] = {}
            for win_label, win_seconds in _WINDOW_SECONDS.items():
                tf_for_lookup = _WINDOW_KLINE_TF[win_label]
                future_px[win_label] = await self._safe_kline_close(
                    tf=tf_for_lookup,
                    target_ts=ts + timedelta(seconds=win_seconds),
                )

            atomic = _extract_atomic_factors(sig.get("factors"))
            regime = _resolve_regime_from_factors(sig.get("factors"))

            for tf, group, name, value in atomic:
                win_label = _FORWARD_WINDOW_BY_TF.get(tf, "1h" if tf == "overall" else None)
                if win_label is None:
                    # tf='overall' 没在表里 → 默认用 4h 窗口（介于流动性 / 持仓比的反应节奏）
                    win_label = "4h"
                fut = future_px.get(win_label)
                if fut is None or fut <= 0:
                    continue
                fwd_ret = (fut - px_now) / px_now
                rec = _FactorRecord(
                    regime=regime,
                    timeframe=tf,
                    factor_group=group,
                    factor_name=name,
                    factor_value=value,
                    forward_return=float(fwd_ret),
                )
                records_90d.append(rec)
                if ts >= since_30d:
                    records_30d.append(rec)

        logger.info(
            "IC 校准已展开样本：30d=%d 条原子记录（%d 信号），"
            "90d=%d 条原子记录（%d 信号）",
            len(records_30d),
            len(signals_30d),
            len(records_90d),
            len(signals_90d),
        )

        # 3) 按 (regime, tf, group, name) 聚合 → 算 IC（30d / 90d）
        min_group = int(self.settings.ic_calibrator_min_group_samples)
        groups: Dict[Tuple[str, str, str, str], _GroupReport] = {}
        # 索引：(regime, tf, group, name) → list[(value, fwd_ret)]
        by_key_30d: Dict[Tuple[str, str, str, str], List[Tuple[float, float]]] = {}
        by_key_90d: Dict[Tuple[str, str, str, str], List[Tuple[float, float]]] = {}
        for rec in records_30d:
            by_key_30d.setdefault(
                (rec.regime, rec.timeframe, rec.factor_group, rec.factor_name), []
            ).append((rec.factor_value, rec.forward_return))
        for rec in records_90d:
            by_key_90d.setdefault(
                (rec.regime, rec.timeframe, rec.factor_group, rec.factor_name), []
            ).append((rec.factor_value, rec.forward_return))

        all_keys = set(by_key_30d.keys()) | set(by_key_90d.keys())
        groups_skipped_low_sample = 0
        for key in all_keys:
            samples_30d = by_key_30d.get(key, [])
            samples_90d = by_key_90d.get(key, [])
            ic_30d = (
                _spearman_ic([s[0] for s in samples_30d], [s[1] for s in samples_30d])
                if len(samples_30d) >= min_group
                else None
            )
            ic_90d = (
                _spearman_ic([s[0] for s in samples_90d], [s[1] for s in samples_90d])
                if len(samples_90d) >= min_group
                else None
            )
            if ic_30d is None and ic_90d is None:
                groups_skipped_low_sample += 1
                continue
            # 优先用 30d IC；缺则退回 90d
            ic_for_weight = ic_30d if ic_30d is not None else ic_90d
            groups[key] = _GroupReport(
                regime=key[0],
                timeframe=key[1],
                factor_group=key[2],
                factor_name=key[3],
                ic_30d=ic_30d,
                ic_90d=ic_90d,
                sample_count_30d=len(samples_30d),
                sample_count_90d=len(samples_90d),
                ic_abs_weighted=_ic_to_weight(ic_for_weight),
            )

        # 4) 同一 (regime, timeframe) 下归一化：sum(weight) = 1
        groups_by_rt: Dict[Tuple[str, str], List[_GroupReport]] = {}
        for g in groups.values():
            groups_by_rt.setdefault((g.regime, g.timeframe), []).append(g)
        for (regime, tf), members in groups_by_rt.items():
            total = sum(g.ic_abs_weighted for g in members)
            if total <= 1e-12:
                # 整组都是噪声 → 平均权重，避免该组合下规则引擎打分恒为 0
                avg = 1.0 / len(members) if members else 0.0
                for g in members:
                    g.weight = avg
            else:
                for g in members:
                    g.weight = g.ic_abs_weighted / total

        # 5) UPSERT factor_weights 表
        rows_to_upsert: List[Dict[str, Any]] = [
            {
                "regime": g.regime,
                "timeframe": g.timeframe,
                "factor_group": g.factor_group,
                "factor_name": g.factor_name,
                "weight": round(g.weight, 6),
                "ic_30d": round(g.ic_30d, 6) if g.ic_30d is not None else None,
                "ic_90d": round(g.ic_90d, 6) if g.ic_90d is not None else None,
                "sample_count": g.sample_count_30d or g.sample_count_90d,
                "updated_at": started_at,
            }
            for g in groups.values()
        ]
        weights_written = 0
        try:
            weights_written = await self.repos.upsert_factor_weights(rows_to_upsert)
        except Exception:
            logger.exception("UPSERT factor_weights 失败，本轮报告仍会落盘")

        # 6) 写校准报告
        top_ic_by_regime = self._build_top_ic_by_regime(groups.values(), top_n=10)
        report = CalibrationReport(
            ran_at=started_at,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            skipped=False,
            total_signals_30d=len(signals_30d),
            total_signals_90d=len(signals_90d),
            total_records_30d=len(records_30d),
            total_records_90d=len(records_90d),
            groups_updated=len(groups),
            groups_skipped_low_sample=groups_skipped_low_sample,
            top_ic_by_regime=top_ic_by_regime,
            weights_written=weights_written,
        )
        self._write_report_log(report)
        logger.info(
            "IC 校准完成：耗时 %.1fs，写入权重 %d 条，跳过低样本 %d 组",
            (report.finished_at - started_at).total_seconds(),
            weights_written,
            groups_skipped_low_sample,
        )
        return report

    async def _safe_kline_close(
        self, tf: str, target_ts: datetime
    ) -> Optional[float]:
        """
        包装 fetch_kline_close_at：异常时记 debug 并返回 None
        -----------------------------------------------------------
        说明：
            校准任务量大（每条 signal 查 4 次），不能因为单个时间点 K 线
            缺失就让整轮失败。
        """
        try:
            return await self.repos.fetch_kline_close_at(
                timeframe=tf, symbol=self.symbol, target_ts=target_ts
            )
        except Exception:
            logger.debug("IC 校准 K 线查询失败 tf=%s ts=%s", tf, target_ts, exc_info=True)
            return None

    @staticmethod
    def _build_top_ic_by_regime(
        groups: Any, top_n: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        给每个 regime 选出 |IC_30d| 最大的 top_n 个因子（用于落盘报告）
        """
        by_regime: Dict[str, List[_GroupReport]] = {}
        for g in groups:
            by_regime.setdefault(g.regime, []).append(g)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for regime, members in by_regime.items():
            members.sort(
                key=lambda x: abs(x.ic_30d if x.ic_30d is not None else 0.0),
                reverse=True,
            )
            out[regime] = [
                {
                    "timeframe": m.timeframe,
                    "factor_group": m.factor_group,
                    "factor_name": m.factor_name,
                    "ic_30d": m.ic_30d,
                    "ic_90d": m.ic_90d,
                    "weight": round(m.weight, 6),
                    "samples_30d": m.sample_count_30d,
                }
                for m in members[:top_n]
            ]
        return out

    def _write_report_log(self, report: CalibrationReport) -> None:
        """
        把校准报告写到 logs/ic_calibration_YYYYMMDD.json
        -----------------------------------------------------------
        说明：
            - 不引入 logging.handlers.TimedRotatingFileHandler；直接按日期
              写一份文件；多次同日触发会覆盖（最新一份是当天最准的）。
            - 目录由 settings.ic_calibrator_log_dir 控制；若不存在则递归创建。
            - 写盘失败不抛错，单独 log.warning。
        """
        try:
            log_dir = self.settings.ic_calibrator_log_dir or "logs"
            os.makedirs(log_dir, exist_ok=True)
            date_str = report.ran_at.strftime("%Y%m%d")
            path = os.path.join(log_dir, f"ic_calibration_{date_str}.json")
            payload = {
                "ran_at": report.ran_at.isoformat(),
                "started_at": report.started_at.isoformat(),
                "finished_at": report.finished_at.isoformat(),
                "skipped": report.skipped,
                "skipped_reason": report.skipped_reason,
                "total_signals_30d": report.total_signals_30d,
                "total_signals_90d": report.total_signals_90d,
                "total_records_30d": report.total_records_30d,
                "total_records_90d": report.total_records_90d,
                "groups_updated": report.groups_updated,
                "groups_skipped_low_sample": report.groups_skipped_low_sample,
                "weights_written": report.weights_written,
                "top_ic_by_regime": report.top_ic_by_regime,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("IC 校准报告落盘失败", exc_info=True)
