"""规则引擎：基于因子的快速预筛。

将聚合后的因子 dict 通过一组确定性启发式规则映射成
:class:`TradingSignal` 与 [-1, 1] 区间的打分。打分还会作为
额外上下文喂给 LLM Agent。

所有阈值都从 :class:`Settings` 中读取，避免在代码里出现魔法数字。

输出语言：reason / risk / suggestion 三个字段统一使用简体中文，
bias 仍然保持 long/short/neutral 英文枚举（与表 CHECK 约束对齐）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.config import Settings
from app.signal_engine.schemas import TradingSignal


class RuleEngine:
    """
    确定性快速信号生成器
    -----------------------------------------------------------------
    职责：
        - 把因子聚合结果映射成有方向的偏置（long/short/neutral）。
        - 输出归一到 [-1, 1] 的总分以及各因子组的贡献度，方便给 LLM
          做二次决策时参考。
    """

    def __init__(self, settings: Settings):
        """
        构造规则引擎
        ---------------------------------------------------------------
        参数：
            settings: 全局配置，提供四个核心阈值与权重的来源。
        """
        self.settings = settings

    def evaluate(
        self, factors: Dict[str, Any]
    ) -> Tuple[TradingSignal, float, Dict[str, Any]]:
        """
        基于因子 dict 评估信号
        ---------------------------------------------------------------
        参数：
            factors: :class:`FactorAggregator.compute` 的输出（兼容多周期与老格式）。
        返回：
            ``(signal, score, contributions)``：
                signal:        TradingSignal 实例（中文 reason/risk/suggestion）
                score:         [-1, 1] 区间的加权打分
                contributions: 每个因子组的原始贡献度（-1 / 0 / +1 / -0.5 等）
        说明：
            P0 升级后兼容两种输入：
                1) 老格式（enable_mtf_factors=False）：单层 dict，含 capital_flow 等四类；
                2) 新格式（enable_mtf_factors=True）：含 by_timeframe；规则引擎默认抽取
                   15m 周期的因子作为代表（与原 30 分钟窗口的语义最接近），并把
                   net_flow_usd 用作老 net_flow 的等价值。
        """
        # 阈值统一从 Settings 拉取
        net_flow_thr = self.settings.rule_net_flow_usd_threshold
        ob_thr = self.settings.rule_orderbook_imbalance_threshold
        oi_thr = self.settings.rule_oi_change_threshold
        funding_thr = self.settings.rule_funding_rate_threshold

        cap, ob, deriv, struct = self._extract_layers(factors)

        contributions: Dict[str, float] = {}

        # ---- 资金流贡献度 ----
        # 兼容：老格式用 net_flow（USD）；新格式用 net_flow_usd
        net_flow = float(cap.get("net_flow_usd") or cap.get("net_flow") or 0.0)
        if net_flow > net_flow_thr:
            contributions["capital_flow"] = +1.0
        elif net_flow < -net_flow_thr:
            contributions["capital_flow"] = -1.0
        else:
            contributions["capital_flow"] = 0.0

        # ---- 订单簿贡献度 ----
        imbalance = float(ob.get("imbalance") or 0.0)
        if imbalance > ob_thr:
            contributions["orderbook"] = +1.0
        elif imbalance < -ob_thr:
            contributions["orderbook"] = -1.0
        else:
            contributions["orderbook"] = 0.0

        # ---- 衍生品贡献度（funding × OI 变动）----
        funding = float(
            deriv.get("funding_rate_now") or deriv.get("funding_rate") or 0.0
        )
        oi_change = deriv.get("oi_change_pct")
        oi_signal = 0.0
        if oi_change is not None:
            if oi_change > oi_thr and funding > funding_thr:
                oi_signal = +1.0
            elif oi_change > oi_thr and funding < -funding_thr:
                oi_signal = -1.0
            elif oi_change < -oi_thr:
                oi_signal = -0.5
        contributions["derivatives"] = oi_signal

        # ---- 市场结构贡献度 ----
        trend = struct.get("trend") or "range"
        contributions["market_structure"] = (
            +1.0 if trend == "uptrend" else -1.0 if trend == "downtrend" else 0.0
        )

        # 链上因子已下线，剩余四类重新分配权重，保证总和为 1.0
        weights = {
            "capital_flow": 0.35,
            "orderbook": 0.15,
            "derivatives": 0.20,
            "market_structure": 0.30,
        }
        score = sum(contributions[k] * weights[k] for k in weights)
        score = max(min(score, 1.0), -1.0)

        if score >= 0.25:
            bias = "long"
        elif score <= -0.25:
            bias = "short"
        else:
            bias = "neutral"

        confidence = round(min(abs(score), 1.0), 3)

        # ---- 中文 reason ----
        oi_change_str = (
            "暂无" if oi_change is None else f"{oi_change:+.4%}"
        )
        reason_parts = [
            f"净流入={net_flow:+.0f} USDT",
            f"盘口失衡={imbalance:+.3f}",
            f"资金费率={funding:+.6f}",
            f"持仓量变动={oi_change_str}",
            f"趋势={_trend_zh(trend)}",
        ]
        reason = "规则引擎: " + ", ".join(reason_parts)

        # ---- 中文 risk ----
        risk_bits = []
        if abs(net_flow) < net_flow_thr:
            risk_bits.append("资金流方向不明")
        if trend == "range" or trend == "neutral":
            risk_bits.append("无明确趋势")
        if not risk_bits:
            risk_bits.append("关注资金费率反转与对侧大单墙")
        risk = "；".join(risk_bits)

        # ---- 中文 suggestion ----
        sup = struct.get("supports") or []
        res = struct.get("resistances") or []
        # 兼容新老 schema：market_structure 在新格式里用 last_close
        entry = struct.get("last_price") or struct.get("last_close")
        if bias == "long":
            stop = sup[0] if sup else None
            target = res[0] if res else None
            suggestion = (
                f"建议在 {_fmt_price(entry)} 附近分批做多，"
                f"止损放在支撑 {_fmt_price(stop)} 下方，"
                f"目标看 {_fmt_price(target)}（仅供参考，不构成交易指令）。"
            )
        elif bias == "short":
            stop = res[0] if res else None
            target = sup[0] if sup else None
            suggestion = (
                f"建议在 {_fmt_price(entry)} 附近分批做空，"
                f"止损放在阻力 {_fmt_price(stop)} 上方，"
                f"目标看 {_fmt_price(target)}（仅供参考，不构成交易指令）。"
            )
        else:
            suggestion = (
                "建议保持观望，等待多因子方向一致后再入场"
                "（仅供参考，不构成交易指令）。"
            )

        signal = TradingSignal(
            bias=bias,
            confidence=confidence,
            reason=reason,
            risk=risk,
            suggestion=suggestion,
        )
        return signal, score, contributions


    @staticmethod
    def _extract_layers(
        factors: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        从因子聚合结果中抽取四类因子层（兼容老 / 新格式）
        ---------------------------------------------------------------
        参数：
            factors: 来自 FactorAggregator.compute 的输出
        返回：
            (capital_flow, orderbook, derivatives, market_structure)
        说明：
            - 老格式（无 by_timeframe）：直接返回顶层四个键。
            - 新格式（多周期矩阵）：默认抽取 15m 周期作为规则引擎代表周期，
              这与历史"30 分钟单一窗口"的语义最接近，最大化与旧阈值的兼容性。
              如果 15m 不存在则按 5m → 1h 顺序回退。
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


def _trend_zh(trend: str) -> str:
    """
    把英文趋势枚举映射到中文标签
    -------------------------------------------------------------------
    参数：
        trend: 'uptrend' / 'downtrend' / 'range' / 'neutral' 等
    返回：
        中文标签字符串，未知值原样返回。
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
    -------------------------------------------------------------------
    参数：
        value: 数字或 None
    返回：
        中文友好的价格字符串。
    """
    if value is None:
        return "未知"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
