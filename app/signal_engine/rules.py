"""规则引擎：基于因子的快速预筛（P2 升级版：从 factor_weights 表加载权重 + 缓存 + fallback）。

设计目标
========
- P0：把因子聚合 dict 映射成有方向偏置 + [-1, 1] 的总分，并构造结构化交易计划。
- P1：按 regime 切换三套硬编码权重（trending / ranging / breakout）。
- P2（本版）：
    * 权重不再硬编码：周期性从 ``factor_weights`` 表加载（同一 regime / timeframe
      下所有 factor_name 的 weight 总和应 = 1，由 IC 校准任务保证）；
    * 内存缓存 5 分钟，避免热路径反复打 DB；
    * 任意失败（开关关闭 / 表为空 / 查询异常）自动降级到 P1 三套硬编码权重；
    * ``contributions`` 字段从"P0 的 4 类粗粒度"升级为"原子因子"粒度，
      给归因 API（/signals/{id}/attribution）提供更细的分解。

关于"按 timeframe 累加"的设计取舍
================================
一条信号要落地交易计划，必须用一个标量 score 决定 bias。但 factor_weights
是按 (regime, timeframe, factor_name) 三维存储，IC 把"在 5m 上 cvd_slope 的
有效程度"和"在 1h 上 cvd_slope 的有效程度"分得很开。
这里采取的策略：
    score = Σ_tf  Σ_f  weight(regime, tf, f) × normalize(factor_value(tf, f))
归一化器 normalize 把不同量纲打到 [-1, 1]，让 weight 直接作为加权和的系数。
由于 IC 校准任务在每个 (regime, tf) 内部已经把权重归一到 sum=1，
跨 tf 累加后 score 的最大绝对值上限 = 周期数（5）；最后 clamp 回 [-1, 1]。

Fallback：与 P1 行为对齐
========================
表查询失败 / 影子模式 / 总开关关闭时，使用与 P1 等价的硬编码权重：
    capital_flow=0.35  orderbook=0.15  derivatives=0.20  market_structure=0.30
对应 contributions 仍以"原子因子"为键写入（同一 group 内分摊基线权重），
保证下游归因接口 schema 统一。
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.signal_engine.schemas import TradingSignal

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 基线权重（P1 三套 regime 权重的展开版本，也是 P2 fallback 种子）
# ----------------------------------------------------------------------
# 数据结构：
#   _BASELINE_WEIGHTS[regime][timeframe][factor_name] = weight (∈ [0, 1])
# 含义：
#   - 同一 (regime, tf) 内所有 factor_name 的 weight 总和应 ≈ 1；
#   - 没显式列出的 factor_name 视为 weight=0（IC 任务可后续补全）。
# 该表由 P1 的"四类粗粒度权重"展开而来：
#   trending_*  → capital_flow=0.30 / orderbook=0.10 / derivatives=0.25 / market_structure=0.35
#   ranging     → capital_flow=0.20 / orderbook=0.30 / derivatives=0.20 / market_structure=0.30
#   breakout/down → capital_flow=0.40 / orderbook=0.15 / derivatives=0.25 / market_structure=0.20
#   transitional → 与 trending_up 同（保守兜底）
_TRENDING_TF_WEIGHTS: Dict[str, Dict[str, float]] = {
    "5m": {"net_flow_usd": 0.10, "imbalance": 0.05, "oi_change_pct": 0.10, "trend": 0.20},
    "15m": {"net_flow_usd": 0.10, "imbalance": 0.05, "oi_change_pct": 0.05, "trend": 0.10},
    "1h": {"net_flow_usd": 0.05, "imbalance": 0.00, "oi_change_pct": 0.05, "trend": 0.05},
    "4h": {"net_flow_usd": 0.05, "imbalance": 0.00, "oi_change_pct": 0.05, "trend": 0.00},
    "1d": {"net_flow_usd": 0.00, "imbalance": 0.00, "oi_change_pct": 0.00, "trend": 0.00},
}

_RANGING_TF_WEIGHTS: Dict[str, Dict[str, float]] = {
    "5m": {"net_flow_usd": 0.05, "imbalance": 0.15, "oi_change_pct": 0.05, "trend": 0.10},
    "15m": {"net_flow_usd": 0.05, "imbalance": 0.10, "oi_change_pct": 0.05, "trend": 0.10},
    "1h": {"net_flow_usd": 0.05, "imbalance": 0.05, "oi_change_pct": 0.05, "trend": 0.05},
    "4h": {"net_flow_usd": 0.05, "imbalance": 0.00, "oi_change_pct": 0.05, "trend": 0.05},
    "1d": {"net_flow_usd": 0.00, "imbalance": 0.00, "oi_change_pct": 0.00, "trend": 0.00},
}

_BREAKOUT_TF_WEIGHTS: Dict[str, Dict[str, float]] = {
    "5m": {"net_flow_usd": 0.15, "imbalance": 0.10, "oi_change_pct": 0.10, "trend": 0.15},
    "15m": {"net_flow_usd": 0.15, "imbalance": 0.05, "oi_change_pct": 0.10, "trend": 0.05},
    "1h": {"net_flow_usd": 0.05, "imbalance": 0.00, "oi_change_pct": 0.05, "trend": 0.00},
    "4h": {"net_flow_usd": 0.05, "imbalance": 0.00, "oi_change_pct": 0.00, "trend": 0.00},
    "1d": {"net_flow_usd": 0.00, "imbalance": 0.00, "oi_change_pct": 0.00, "trend": 0.00},
}

_BASELINE_WEIGHTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "trending_up": _TRENDING_TF_WEIGHTS,
    "trending_down": _TRENDING_TF_WEIGHTS,
    "breakout": _BREAKOUT_TF_WEIGHTS,
    "breakdown": _BREAKOUT_TF_WEIGHTS,
    "ranging": _RANGING_TF_WEIGHTS,
    "transitional": _TRENDING_TF_WEIGHTS,
    "overall": _TRENDING_TF_WEIGHTS,
}

# 因子归一化"参考量纲"：用 sigmoid(value / scale) × 2 - 1 把任意值压到 (-1, 1)。
# scale 选取依据：
#   - net_flow_usd：阈值 ±5e4，sigmoid(±1) ≈ ±0.46，量纲合理；
#   - imbalance：原值已在 [-1, 1]，不需要 sigmoid，直接 clamp；
#   - oi_change_pct：阈值 ±0.005，scale 取 0.01；
#   - cvd_slope：经验值，scale 取 1e-4；
#   - 其余未列出的因子默认走"value 直接 tanh"。
_NORMALIZE_SCALES: Dict[str, float] = {
    "net_flow_usd": 5e4,
    "net_flow": 5e4,
    "oi_change_pct": 0.01,
    "cvd_slope": 1e-4,
    "funding_rate_now": 1e-4,
    "funding_rate": 1e-4,
    "imbalance_slope_5m": 1e-3,
    "imbalance_zscore_15m": 2.0,
    "spread_bp": 5.0,
    "atr_14": 50.0,
    "adx_14": 25.0,
    "bb_width": 0.02,
    "wall_distance_pct": 0.005,
}

# trend 取值映射（uptrend=+1 / downtrend=-1 / 其他=0）
_TREND_VALUE_MAP: Dict[str, float] = {
    "uptrend": 1.0,
    "downtrend": -1.0,
    "range": 0.0,
    "neutral": 0.0,
}


def _normalize_factor(name: str, value: Any) -> Optional[float]:
    """
    把单个原子因子的值压到 [-1, 1]
    --------------------------------------------------------------
    参数：
        name : 因子名（决定使用哪种归一化策略）
        value: 原始值
    返回：
        float ∈ [-1, 1]；无法解析时返回 None。
    实现：
        - bool / 'true|false' 映射到 ±1；
        - trend 字段映射到 +1/-1/0；
        - 其他数值走 sigmoid(value / scale) * 2 - 1；
        - imbalance / *_pct 一类已经天然在 [-1, 1]，直接 clamp 不再 sigmoid。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    if isinstance(value, str):
        return _TREND_VALUE_MAP.get(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    # 已经在 [-1, 1] 量纲的，直接 clamp
    if name in ("imbalance", "imbalance_now", "alignment_score") or name.endswith("_ratio"):
        return max(-1.0, min(1.0, v))
    scale = _NORMALIZE_SCALES.get(name)
    if scale is None or scale <= 0:
        return math.tanh(v)
    return math.tanh(v / scale)


def _direction_sign(name: str, value: Any) -> float:
    """
    估算"该因子在多空两个方向上的有效贡献符号"
    --------------------------------------------------------------
    多数因子是"值越大越偏多、越小越偏空"，符号 = sign(normalize(value))；
    少数特殊因子会反过来（例如 retail_long_short_ratio 极高 = 散户狂多 = 反指）。
    本函数集中处理这些方向反转，避免在 evaluate 里散落 if-else。
    返回：
        +1.0 / -1.0 / 0.0
    """
    norm = _normalize_factor(name, value)
    if norm is None:
        return 0.0
    sign = 1.0 if norm > 0 else (-1.0 if norm < 0 else 0.0)
    # 反指因子：散户多空比 / 散户合约比（顶 ↔ 底反相关）
    if name in ("account_long_short_ratio", "account_contract_ratio"):
        return -sign
    return sign


class RuleEngine:
    """
    确定性快速信号生成器（P2 升级：权重表 + 缓存 + 原子粒度 contributions）
    -----------------------------------------------------------------
    职责：
        - 把因子聚合结果映射成有方向的偏置（long/short/neutral）。
        - 输出 [-1, 1] 的总分以及"原子因子"贡献度，方便下游归因。
        - 周期性从 factor_weights 表刷新权重；表为空 / 失败时退回基线。
    """

    # 缓存有效期（秒）。5 分钟是经验取舍：低于此值会让规则引擎对 IC
    # 任务结果反应过快（不利于平稳性），高于此值灰度切换感知会变慢。
    _CACHE_TTL_SECONDS: float = 300.0

    def __init__(self, settings: Settings, repos: Optional[Repositories] = None):
        """
        构造规则引擎
        ---------------------------------------------------------------
        参数：
            settings : 全局配置（含 enable_factor_weights_table /
                       ic_calibrator_shadow_mode 等开关）
            repos    : 数据仓储；P0/P1 模式下可选（None → 直接走基线权重）
        """
        self.settings = settings
        self.repos = repos
        # 缓存：{(regime, timeframe): {factor_name: weight}}
        self._weight_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._weight_cache_ts: float = 0.0
        # 缓存来源标记：'db' / 'baseline' / 'shadow_baseline'
        # 仅用于日志诊断，不影响业务。
        self._weight_cache_source: str = "baseline"

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    async def evaluate(
        self, factors: Dict[str, Any]
    ) -> Tuple[TradingSignal, float, Dict[str, Any]]:
        """
        基于因子 dict 评估信号（P2 升级：异步，需要拉权重表）
        ---------------------------------------------------------------
        参数：
            factors: FactorAggregator.compute 的输出（兼容多周期与老格式）。
        返回：
            (signal, score, contributions)
                signal:        TradingSignal（中文 reason/risk/suggestion）
                score:         [-1, 1] 区间的加权打分
                contributions: dict，键为 "tf:group.factor_name"，值为该原子因子
                               对总打分的贡献（含正负号）。
        """
        regime = factors.get("regime") or "overall"
        await self._ensure_weight_cache()

        # 多周期 / 老格式抽取：构造"按 (timeframe, factor_name) 索引的因子表"
        # 同时保留四类粗粒度块用于 reason 文案 / 交易计划构造。
        atomic_values: Dict[Tuple[str, str, str], Any] = self._collect_atomic_values(factors)
        # 兼容老格式的 4 类粗粒度（仅用于 reason / suggestion 渲染）
        cap_layer, ob_layer, deriv_layer, struct_layer = self._extract_legacy_layers(factors)

        # ---- 计算 contributions（原子因子粒度）+ 总分 ----
        contributions: Dict[str, float] = {}
        score = 0.0
        weight_lookup_count = 0
        for (tf, group, name), value in atomic_values.items():
            weight = self._lookup_weight(regime=regime, tf=tf, factor_name=name)
            if weight <= 0:
                continue
            sign = _direction_sign(name, value)
            if sign == 0.0:
                continue
            contrib = sign * weight
            score += contrib
            contributions[f"{tf}:{group}.{name}"] = round(contrib, 6)
            weight_lookup_count += 1

        # 若没有任何因子命中权重（极端情况：表里只有黑名单因子），
        # 退回老的 4 类粗粒度计算逻辑，保证 score 不长期为 0。
        if weight_lookup_count == 0:
            score, legacy_contribs = self._evaluate_legacy_score(
                regime=regime,
                cap=cap_layer,
                ob=ob_layer,
                deriv=deriv_layer,
                struct=struct_layer,
            )
            contributions.update(legacy_contribs)
            logger.debug(
                "规则引擎未命中任何权重表因子，已退回 P1 等价的四类粗粒度打分"
            )

        score = max(-1.0, min(1.0, score))

        if score >= 0.25:
            bias = "long"
        elif score <= -0.25:
            bias = "short"
        else:
            bias = "neutral"

        confidence = round(min(abs(score), 1.0), 3)

        # ---- 构造 reason / risk / suggestion + 结构化交易计划 ----
        reason = self._render_reason(
            regime=regime,
            cap=cap_layer,
            ob=ob_layer,
            deriv=deriv_layer,
            struct=struct_layer,
            cache_source=self._weight_cache_source,
        )
        risk = self._render_risk(cap=cap_layer, struct=struct_layer)

        sup = [float(s) for s in (struct_layer.get("supports") or []) if s is not None]
        res = [float(r) for r in (struct_layer.get("resistances") or []) if r is not None]
        entry_raw = struct_layer.get("last_price") or struct_layer.get("last_close")
        atr_14 = struct_layer.get("atr_14")
        try:
            entry = float(entry_raw) if entry_raw is not None else None
        except (TypeError, ValueError):
            entry = None
        try:
            atr_val = float(atr_14) if atr_14 is not None else None
        except (TypeError, ValueError):
            atr_val = None

        plan: Optional[Dict[str, Any]] = None
        if bias != "neutral" and entry is not None and entry > 0:
            plan = _build_trade_plan(
                bias=bias,
                entry=entry,
                supports=sup,
                resistances=res,
                atr=atr_val,
            )

        if bias == "long":
            stop_disp = plan["stop_loss"] if plan else (sup[0] if sup else None)
            target_disp = plan["take_profit"][0] if plan else (res[0] if res else None)
            suggestion = (
                f"建议在 {_fmt_price(entry)} 附近分批做多，"
                f"止损放在支撑 {_fmt_price(stop_disp)} 下方，"
                f"目标看 {_fmt_price(target_disp)}（仅供参考，不构成交易指令）。"
            )
        elif bias == "short":
            stop_disp = plan["stop_loss"] if plan else (res[0] if res else None)
            target_disp = plan["take_profit"][0] if plan else (sup[0] if sup else None)
            suggestion = (
                f"建议在 {_fmt_price(entry)} 附近分批做空，"
                f"止损放在阻力 {_fmt_price(stop_disp)} 上方，"
                f"目标看 {_fmt_price(target_disp)}（仅供参考，不构成交易指令）。"
            )
        else:
            suggestion = (
                "建议保持观望，等待多因子方向一致后再入场"
                "（仅供参考，不构成交易指令）。"
            )

        if bias != "neutral" and plan is None:
            logger.info(
                "规则引擎缺少有效价位（entry/sup/res/atr 数据不足），"
                "bias=%s 软降级为 neutral 以满足 TradingSignal 结构化字段约束",
                bias,
            )
            bias = "neutral"
            confidence = round(min(abs(score) * 0.5, 1.0), 3)

        signal_kwargs: Dict[str, Any] = dict(
            bias=bias,
            confidence=confidence,
            reason=reason,
            risk=risk,
            suggestion=suggestion,
        )
        if plan is not None and bias != "neutral":
            signal_kwargs.update(
                entry_zone=plan["entry_zone"],
                stop_loss=plan["stop_loss"],
                take_profit=plan["take_profit"],
                risk_reward_ratio=plan["risk_reward_ratio"],
            )

        signal = TradingSignal(**signal_kwargs)
        return signal, score, contributions

    # ------------------------------------------------------------------
    # 同步 evaluate 包装：service 层目前是 await self.rule_engine.evaluate(...)，
    # 该 wrapper 仅为单元测试 / 旧调用方保持兼容（同步调用会直接报错）。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 权重缓存
    # ------------------------------------------------------------------
    async def _ensure_weight_cache(self) -> None:
        """
        必要时刷新内存权重缓存（5 分钟 TTL）
        ---------------------------------------------------------------
        刷新策略：
            - enable_factor_weights_table=False 或 ic_calibrator_shadow_mode=True
              或 repos 缺失：直接装入基线权重（不查 DB）；
            - 否则查表，按 (regime, timeframe) 聚合 → 写缓存；
            - 任何异常 / 表为空 → 退回基线 + warning。
        """
        if (time.time() - self._weight_cache_ts) < self._CACHE_TTL_SECONDS and self._weight_cache:
            return
        use_db = (
            bool(getattr(self.settings, "enable_factor_weights_table", False))
            and not bool(getattr(self.settings, "ic_calibrator_shadow_mode", True))
            and self.repos is not None
        )
        if not use_db:
            self._weight_cache = self._load_baseline_cache()
            self._weight_cache_ts = time.time()
            self._weight_cache_source = (
                "shadow_baseline"
                if bool(getattr(self.settings, "ic_calibrator_shadow_mode", True))
                else "baseline"
            )
            return

        try:
            rows = await self.repos.fetch_all_factor_weights()
        except Exception:
            logger.warning(
                "factor_weights 查询失败，规则引擎降级到基线权重", exc_info=True
            )
            self._weight_cache = self._load_baseline_cache()
            self._weight_cache_ts = time.time()
            self._weight_cache_source = "baseline"
            return

        if not rows:
            logger.warning(
                "factor_weights 表为空，规则引擎降级到基线权重（IC 任务尚未跑过？）"
            )
            self._weight_cache = self._load_baseline_cache()
            self._weight_cache_ts = time.time()
            self._weight_cache_source = "baseline"
            return

        cache: Dict[Tuple[str, str], Dict[str, float]] = {}
        for r in rows:
            key = (str(r["regime"]), str(r["timeframe"]))
            try:
                weight = float(r["weight"])
            except (TypeError, ValueError):
                continue
            cache.setdefault(key, {})[str(r["factor_name"])] = weight

        self._weight_cache = cache
        self._weight_cache_ts = time.time()
        self._weight_cache_source = "db"

    @classmethod
    def _load_baseline_cache(cls) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        把 _BASELINE_WEIGHTS 装成与 DB 同结构的缓存
        """
        out: Dict[Tuple[str, str], Dict[str, float]] = {}
        for regime, tf_map in _BASELINE_WEIGHTS.items():
            for tf, factor_map in tf_map.items():
                out[(regime, tf)] = dict(factor_map)
        return out

    def _lookup_weight(self, regime: str, tf: str, factor_name: str) -> float:
        """
        三级查找：(regime, tf) → ('overall', tf) → 0
        ---------------------------------------------------------------
        说明：
            regime 在 detect_regime 输出 transitional 时也会走这里，
            如果该 regime 下没有显式权重，退到 overall 兜底维度。
        """
        v = self._weight_cache.get((regime, tf), {}).get(factor_name)
        if v is not None:
            return float(v)
        v = self._weight_cache.get(("overall", tf), {}).get(factor_name)
        if v is not None:
            return float(v)
        return 0.0

    # ------------------------------------------------------------------
    # 因子展开 / 抽取
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_atomic_values(
        factors: Dict[str, Any],
    ) -> Dict[Tuple[str, str, str], Any]:
        """
        把多周期因子矩阵 + 老格式都展平到 (timeframe, group, name) → value 的 dict
        """
        out: Dict[Tuple[str, str, str], Any] = {}
        # 多周期路径
        by_tf = factors.get("by_timeframe") or {}
        if isinstance(by_tf, dict):
            for tf, block in by_tf.items():
                if not isinstance(block, dict):
                    continue
                for group in (
                    "capital_flow", "orderbook", "derivatives", "market_structure"
                ):
                    payload = block.get(group)
                    if not isinstance(payload, dict):
                        continue
                    for k, v in payload.items():
                        out[(str(tf), group, k)] = v
        # 老格式（无 by_timeframe）：当 15m / overall 看待
        if not by_tf:
            for group in ("capital_flow", "orderbook", "derivatives", "market_structure"):
                payload = factors.get(group)
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        out[("15m", group, k)] = v
        # 根节点 liquidity / position_ratios → timeframe='overall'
        for root_group in ("liquidity", "position_ratios"):
            payload = factors.get(root_group)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    out[("overall", root_group, k)] = v
        return out

    @staticmethod
    def _extract_legacy_layers(
        factors: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        抽取代表周期（默认 15m，回退 5m → 1h → 4h → 1d）的四类块
        ---------------------------------------------------------------
        说明：
            仅用于 reason / risk / suggestion 文案生成 + 交易计划构造，
            真正的打分已经走 _collect_atomic_values 多周期展开了。
        """
        if "by_timeframe" not in factors:
            return (
                factors.get("capital_flow", {}) or {},
                factors.get("orderbook", {}) or {},
                factors.get("derivatives", {}) or {},
                factors.get("market_structure", {}) or {},
            )
        by_tf = factors.get("by_timeframe") or {}
        chosen: Optional[Dict[str, Any]] = None
        for tf in ("15m", "5m", "1h", "4h", "1d"):
            block = by_tf.get(tf)
            if block:
                chosen = block
                break
        chosen = chosen or {}
        return (
            chosen.get("capital_flow", {}) or {},
            chosen.get("orderbook", {}) or {},
            chosen.get("derivatives", {}) or {},
            chosen.get("market_structure", {}) or {},
        )

    def _evaluate_legacy_score(
        self,
        regime: str,
        cap: Dict[str, Any],
        ob: Dict[str, Any],
        deriv: Dict[str, Any],
        struct: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        权重表完全无效时的兜底打分 —— 与 P1 行为等价，但 contributions 仍按
        "原子因子键"输出，保证下游归因接口 schema 不变。
        """
        net_flow = float(cap.get("net_flow_usd") or cap.get("net_flow") or 0.0)
        imbalance = float(ob.get("imbalance") or 0.0)
        funding = float(deriv.get("funding_rate_now") or deriv.get("funding_rate") or 0.0)
        oi_change = deriv.get("oi_change_pct")
        trend = struct.get("trend") or "range"

        # 与 P0/P1 等价的"四类粗粒度权重"
        weights = {"capital_flow": 0.35, "orderbook": 0.15, "derivatives": 0.20, "market_structure": 0.30}

        cap_score = (
            1.0 if net_flow > self.settings.rule_net_flow_usd_threshold else
            -1.0 if net_flow < -self.settings.rule_net_flow_usd_threshold else 0.0
        )
        ob_score = (
            1.0 if imbalance > self.settings.rule_orderbook_imbalance_threshold else
            -1.0 if imbalance < -self.settings.rule_orderbook_imbalance_threshold else 0.0
        )
        oi_score = 0.0
        if oi_change is not None:
            if oi_change > self.settings.rule_oi_change_threshold and funding > self.settings.rule_funding_rate_threshold:
                oi_score = 1.0
            elif oi_change > self.settings.rule_oi_change_threshold and funding < -self.settings.rule_funding_rate_threshold:
                oi_score = -1.0
            elif oi_change < -self.settings.rule_oi_change_threshold:
                oi_score = -0.5
        struct_score = +1.0 if trend == "uptrend" else -1.0 if trend == "downtrend" else 0.0

        score = (
            cap_score * weights["capital_flow"]
            + ob_score * weights["orderbook"]
            + oi_score * weights["derivatives"]
            + struct_score * weights["market_structure"]
        )
        contribs = {
            "15m:capital_flow.net_flow_usd": round(cap_score * weights["capital_flow"], 6),
            "15m:orderbook.imbalance": round(ob_score * weights["orderbook"], 6),
            "15m:derivatives.oi_change_pct": round(oi_score * weights["derivatives"], 6),
            "15m:market_structure.trend": round(struct_score * weights["market_structure"], 6),
        }
        return score, contribs

    def _render_reason(
        self,
        regime: str,
        cap: Dict[str, Any],
        ob: Dict[str, Any],
        deriv: Dict[str, Any],
        struct: Dict[str, Any],
        cache_source: str,
    ) -> str:
        """
        生成中文 reason 摘要
        ---------------------------------------------------------------
        说明：
            cache_source 写在末尾，便于排查 LLM 是否拿到了影子模式 / DB 权重。
        """
        net_flow = float(cap.get("net_flow_usd") or cap.get("net_flow") or 0.0)
        imbalance = float(ob.get("imbalance") or 0.0)
        funding = float(deriv.get("funding_rate_now") or deriv.get("funding_rate") or 0.0)
        oi_change = deriv.get("oi_change_pct")
        trend = struct.get("trend") or "range"

        oi_change_str = "暂无" if oi_change is None else f"{oi_change:+.4%}"
        return (
            f"规则引擎(regime={regime}, weights={cache_source}): "
            f"净流入={net_flow:+.0f} USDT, "
            f"盘口失衡={imbalance:+.3f}, "
            f"资金费率={funding:+.6f}, "
            f"持仓量变动={oi_change_str}, "
            f"趋势={_trend_zh(trend)}"
        )

    def _render_risk(
        self, cap: Dict[str, Any], struct: Dict[str, Any]
    ) -> str:
        """
        生成中文 risk 摘要
        """
        net_flow = float(cap.get("net_flow_usd") or cap.get("net_flow") or 0.0)
        trend = struct.get("trend") or "range"
        risk_bits: List[str] = []
        if abs(net_flow) < self.settings.rule_net_flow_usd_threshold:
            risk_bits.append("资金流方向不明")
        if trend in ("range", "neutral"):
            risk_bits.append("无明确趋势")
        if not risk_bits:
            risk_bits.append("关注资金费率反转与对侧大单墙")
        return "；".join(risk_bits)


# ----------------------------------------------------------------------
# 与 P0/P1 完全相同的交易计划构造（保持兼容；逻辑未改动）
# ----------------------------------------------------------------------
def _build_trade_plan(
    bias: str,
    entry: float,
    supports: List[float],
    resistances: List[float],
    atr: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    基于支撑 / 阻力 / ATR 构造一份满足 TradingSignal schema 的粗粒度交易计划
    -------------------------------------------------------------------
    说明：
        与 P0/P1 完全一致，未做任何调整；P2 升级只改打分通路，不改交易
        计划构造逻辑，避免一次性引入两个变化导致回归难度上升。
    """
    if entry <= 0:
        return None
    band_unit = atr if atr and atr > 0 else max(entry * 0.005, 1e-6)
    half_band = max(band_unit * 0.1, entry * 0.001)
    ez_low = entry - half_band
    ez_high = entry + half_band

    if bias == "long":
        valid_sup = [s for s in supports if s < ez_low]
        if valid_sup:
            sl = max(valid_sup)
        else:
            sl = ez_low - max(band_unit * 1.0, entry * 0.005)
        valid_res = sorted({round(r, 6) for r in resistances if r > ez_high})
        if len(valid_res) >= 2:
            tp1, tp2 = float(valid_res[0]), float(valid_res[1])
        elif len(valid_res) == 1:
            tp1 = float(valid_res[0])
            tp2 = tp1 + max(band_unit * 1.0, entry * 0.01)
        else:
            tp1 = ez_high + max(band_unit * 2.0, entry * 0.01)
            tp2 = tp1 + max(band_unit * 1.0, entry * 0.01)
        if not (sl < ez_low <= ez_high < tp1 < tp2):
            return None
        entry_mid = (ez_low + ez_high) / 2
        risk_per_unit = abs(entry_mid - sl)
        reward_per_unit = abs(tp1 - entry_mid)
    elif bias == "short":
        valid_res = [r for r in resistances if r > ez_high]
        if valid_res:
            sl = min(valid_res)
        else:
            sl = ez_high + max(band_unit * 1.0, entry * 0.005)
        valid_sup = sorted({round(s, 6) for s in supports if s < ez_low}, reverse=True)
        if len(valid_sup) >= 2:
            tp1, tp2 = float(valid_sup[0]), float(valid_sup[1])
        elif len(valid_sup) == 1:
            tp1 = float(valid_sup[0])
            tp2 = tp1 - max(band_unit * 1.0, entry * 0.01)
        else:
            tp1 = ez_low - max(band_unit * 2.0, entry * 0.01)
            tp2 = tp1 - max(band_unit * 1.0, entry * 0.01)
        if not (sl > ez_high >= ez_low > tp1 > tp2):
            return None
        entry_mid = (ez_low + ez_high) / 2
        risk_per_unit = abs(entry_mid - sl)
        reward_per_unit = abs(tp1 - entry_mid)
    else:
        return None

    if risk_per_unit <= 1e-9:
        return None
    rr = round(reward_per_unit / risk_per_unit, 4)
    if rr < 1.5:
        return None
    return {
        "entry_zone": (round(ez_low, 4), round(ez_high, 4)),
        "stop_loss": round(float(sl), 4),
        "take_profit": [round(float(tp1), 4), round(float(tp2), 4)],
        "risk_reward_ratio": rr,
    }


def _trend_zh(trend: str) -> str:
    """
    把英文趋势枚举映射到中文标签
    """
    mapping = {
        "uptrend": "上升",
        "downtrend": "下降",
        "range": "震荡",
        "neutral": "暂无",
    }
    return mapping.get(trend, trend)


def _fmt_price(value: Any) -> str:
    """
    格式化价格：None 显示为'未知'，其他保留原值
    """
    if value is None:
        return "未知"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
