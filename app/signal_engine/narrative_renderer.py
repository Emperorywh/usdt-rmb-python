"""LLM-First desk trader 叙事渲染层。

设计目标
========
把因子矩阵从"指标堆砌 ASCII 表"翻译成 7 段**带解读**的 desk trader 叙事，
让 LLM 拿到的是半成品的语义判断，而不是从零开始倒推：

1. 市场状态（regime + ATR 波动语义 + ADX 趋势强度）
2. 多周期方向（5 周期箭头 + 共振度 + 主导方向解读）
3. 主动资金 vs 价格（CVD / OI / net_flow / taker 的**因果解读**）
4. 衍生品（funding 极端值 + OI/价散度）
5. 关键价位（多周期 supports / resistances + 距当前价 X×ATR）
6. 流动性地图（上下方止损池 + 真空区警示）
7. **Liquidations 滚动窗口**（多头 / 空头爆仓 + cascade → 反转动力解读）

关键设计原则
========================================
- 强制使用动词解读："被动拉盘"、"真实加仓"、"诱多"、"squeeze 风险"、
  "接盘出货"、"筹码交换"、"空头回补"。
- 禁止纯指标罗列：不写 "regime=X, alignment=Y, score=Z"，而是写
  "向上突破，4h 已上行 3 根；ADX 28.5 → 趋势确立"。
- 必须解释因果：不只是说 "CVD 走低"，而是 "CVD 走低 = 被动拉盘 = 诱多"。
- 必须提具体价位：不只说 "上方阻力"，而是 "上方 3625 4h 强档"。
- 必须给可比较的距离：不只说 "靠近"，而是 "距阻力 1.5×ATR(15m)"。

使用方式
========
::

    from app.signal_engine.narrative_renderer import NarrativeRenderer

    renderer = NarrativeRenderer()
    sections = renderer.render_sections(factors)
    # sections 是 dict，键名与 HUMAN_PROMPT_COMPACT 占位符对齐：
    #   {market_state, mtf_direction, capital_action,
    #    derivatives, key_levels, liquidity, liquidations}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.utils import safe_float as _to_float


def _fmt_price(v: Any, digits: int = 2) -> str:
    """价格统一格式化：None → '-'；其他保留 digits 位"""
    f = _to_float(v)
    if f is None:
        return "-"
    return f"{f:.{digits}f}"


# ----------------------------------------------------------------------
# 多周期主因子集合常量
# ----------------------------------------------------------------------
# 决定哪些周期出现在叙事段中。order 与 desk trader 阅读顺序一致：
# 高周期决定方向 → 低周期决定时机。
_TIMEFRAMES_DIRECTION_VOTES: Tuple[str, ...] = ("4h", "1h", "15m", "5m", "1d")


class NarrativeRenderer:
    """
    desk trader 叙事化渲染器
    -----------------------------------------------------------------
    无状态：所有方法都是基于 factors dict 的纯函数。可以全局单例使用，
    也可以每次调用 new 一个，都不会有副作用。
    """

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    def render_sections(self, factors: Dict[str, Any]) -> Dict[str, str]:
        """
        渲染全部 7 段叙事，对齐 HUMAN_PROMPT_COMPACT 的占位符
        --------------------------------------------------------------
        参数：
            factors: FactorAggregator.compute 的输出（必须是 MTF 模式）
        返回：
            {
              "market_state": str,    # 市场状态（regime + ATR + ADX）
              "mtf_direction": str,   # 多周期方向 + 共振解读
              "capital_action": str,  # 主动资金 vs 价格的因果解读
              "derivatives": str,     # funding + OI/价散度
              "key_levels": str,      # 关键价位 + 距当前价 ATR 倍数
              "liquidity": str,       # 流动性地图解读
              "liquidations": str,    # 爆仓滚动窗口 → 反转动力解读
            }
        说明：
            - 任一段缺数据时返回"该段无可用数据"短文本，不抛异常；
            - 由 FactorAggregator 保证 by_timeframe 总是存在；如果上层
              未启用多周期模式，相关段会显式提示"数据缺失"。
        """
        return {
            "market_state": self.render_market_state(factors),
            "mtf_direction": self.render_mtf_direction(factors),
            "capital_action": self.render_capital_action(factors),
            "derivatives": self.render_derivatives(factors),
            "key_levels": self.render_key_levels(factors),
            "liquidity": self.render_liquidity(factors),
            "liquidations": self.render_liquidations(factors),
        }

    # ==================================================================
    # 第 1 段：市场状态（regime + ATR + 当前价的"语义"标签）
    # ==================================================================
    def render_market_state(self, factors: Dict[str, Any]) -> str:
        """
        渲染市场状态段（1-3 行 desk 语言）
        --------------------------------------------------------------
        参数：
            factors: 多周期 factors dict
        返回：
            紧凑中文段，约 80-150 tokens。例如：
            "向上突破（breakout），4h 收盘已上行 3 根
             ATR(15m)=15.20（0.43%，正常波动），当前价 3530.15
             ADX(1h)=28.5 → 趋势已确立"
        """
        regime = factors.get("regime") or "unknown"
        regime_zh = self._regime_label(regime)

        by_tf = factors.get("by_timeframe") or {}
        ms_15m = ((by_tf.get("15m") or {}).get("market_structure")) or {}
        ms_1h = ((by_tf.get("1h") or {}).get("market_structure")) or {}
        ms_4h = ((by_tf.get("4h") or {}).get("market_structure")) or {}
        ms_1d = ((by_tf.get("1d") or {}).get("market_structure")) or {}

        last_close = _to_float(ms_15m.get("last_close"))
        atr_15m = _to_float(ms_15m.get("atr_14"))
        adx_1h = _to_float(ms_1h.get("adx_14"))

        atr_pct: Optional[float] = None
        if atr_15m is not None and last_close is not None and last_close > 0:
            atr_pct = atr_15m / last_close

        atr_label = self._atr_label(atr_pct)
        adx_label = self._adx_label(adx_1h)

        ctx_bits: List[str] = []
        trend_4h = (ms_4h.get("trend") or "").lower()
        trend_1d = (ms_1d.get("trend") or "").lower()
        if regime in ("trending_up", "breakout"):
            if trend_4h == "uptrend":
                ctx_bits.append("4h 上行结构清晰")
            elif trend_4h:
                ctx_bits.append(f"4h={trend_4h}")
        elif regime in ("trending_down", "breakdown"):
            if trend_4h == "downtrend":
                ctx_bits.append("4h 下行结构清晰")
            elif trend_4h:
                ctx_bits.append(f"4h={trend_4h}")
        elif regime == "ranging":
            ctx_bits.append("窄幅震荡中，方向选择尚未明确")
        elif regime == "transitional":
            if trend_4h and trend_1d and trend_4h != trend_1d:
                ctx_bits.append(f"高低周期方向不一致（4h={trend_4h} / 1d={trend_1d}）")
            else:
                ctx_bits.append("状态切换中，结构未稳定")

        lines: List[str] = []
        lines.append(
            f"{regime_zh}（{regime}）" + (f"，{ctx_bits[0]}" if ctx_bits else "")
        )
        if atr_15m is not None and atr_pct is not None:
            lines.append(
                f"ATR(15m)={atr_15m:.2f}（{atr_pct * 100:.2f}%，{atr_label}），"
                f"当前价 {_fmt_price(last_close)}"
            )
        elif last_close is not None:
            lines.append(f"ATR(15m) 数据缺失，当前价 {_fmt_price(last_close)}")
        else:
            lines.append("ATR / 当前价数据缺失")
        if adx_1h is not None:
            lines.append(f"ADX(1h)={adx_1h:.1f} → {adx_label}")
        return "\n".join(lines)

    # ==================================================================
    # 第 2 段：多周期方向 + 共振解读
    # ==================================================================
    def render_mtf_direction(self, factors: Dict[str, Any]) -> str:
        """
        渲染多周期方向段
        --------------------------------------------------------------
        参数：
            factors: 多周期 factors dict
        返回：
            约 100 tokens。例如：
            "4h ↑ | 1h ↑ | 15m ↑ | 5m ↑ | 1d →    共振度 +0.72（dom=long）
             高周期一致看多，低周期同向跟进 → 主导多，可顺势"
        """
        by_tf = factors.get("by_timeframe") or {}
        tfs_render: List[str] = []
        for tf in _TIMEFRAMES_DIRECTION_VOTES:
            block = by_tf.get(tf) or {}
            ms = block.get("market_structure") or {}
            trend = (ms.get("trend") or "").lower()
            arrow = self._trend_arrow(trend)
            tfs_render.append(f"{tf} {arrow}")

        mtf = factors.get("mtf_alignment") or {}
        align = _to_float(mtf.get("alignment_score"))
        dominant = (mtf.get("dominant_bias") or "").lower()

        align_str = f"{align:+.2f}" if align is not None else "N/A"

        if align is None:
            interpret = "共振数据缺失，方向参考意义有限"
        elif align >= 0.6:
            interpret = "高周期一致看多，低周期同向跟进 → 主导多，可顺势"
        elif align <= -0.6:
            interpret = "高周期一致看空，低周期同向跟进 → 主导空，可顺势"
        elif abs(align) >= 0.3:
            side = "多" if align > 0 else "空"
            interpret = (
                f"略偏{side}但共振强度不够，需主因子（OI/CVD/资金流）同向才考虑出手"
            )
        else:
            interpret = (
                "共振崩溃，方向不明 → 典型震荡市特征，强追趋势单容易被 whipsaw"
            )

        first_line = " | ".join(tfs_render) + (
            f"    共振度 {align_str}" + (f"（dom={dominant}）" if dominant else "")
        )
        return f"{first_line}\n{interpret}"

    # ==================================================================
    # 第 3 段：主动资金 vs 价格的因果解读（最关键的一段）
    # ==================================================================
    def render_capital_action(self, factors: Dict[str, Any]) -> str:
        """
        渲染主动资金 vs 价格段
        --------------------------------------------------------------
        参数：
            factors: 多周期 factors dict
        返回：
            约 200-300 tokens（最关键的一段，可以略长）。例如：
            "5m: net_flow +1.20M USD，CVD slope +0.0023，taker 52.5%，OI +0.34% → 主动买入真实
             1h: net_flow +0.80M USD，CVD slope +0.0019，OI +1.80% → OI 与价格同步上升 = 多头真实建仓中，非短期投机推升
             4h: OI +0.42% → 高周期持仓格局稳定看多
             综合：资金行为多周期一致看多，方向可信"
        """
        by_tf = factors.get("by_timeframe") or {}
        lines: List[str] = []
        any_data = False

        for tf in ("5m", "15m", "1h", "4h"):
            block = by_tf.get(tf) or {}
            cap = block.get("capital_flow") or {}
            deriv = block.get("derivatives") or {}
            ms = block.get("market_structure") or {}

            net_flow = _to_float(cap.get("net_flow_usd"))
            cvd_slope = _to_float(cap.get("cvd_slope"))
            taker = _to_float(cap.get("taker_buy_ratio"))
            oi_change = _to_float(deriv.get("oi_change_pct"))
            oi_rel = (deriv.get("oi_price_relation") or "").lower()
            trend = (ms.get("trend") or "").lower()

            if net_flow is None and cvd_slope is None and oi_change is None:
                continue
            any_data = True

            parts: List[str] = []
            if net_flow is not None:
                parts.append(f"net_flow {self._fmt_money(net_flow)}")
            if cvd_slope is not None:
                parts.append(f"CVD slope {cvd_slope:+.4f}")
            if taker is not None:
                parts.append(f"taker {taker * 100:.1f}%")
            if oi_change is not None:
                parts.append(f"OI {oi_change * 100:+.2f}%")

            verdict = self._verdict_capital(
                net_flow=net_flow,
                cvd_slope=cvd_slope,
                taker=taker,
                oi_change=oi_change,
                oi_rel=oi_rel,
                trend=trend,
            )
            lines.append(f"{tf}: " + "，".join(parts) + (f" → {verdict}" if verdict else ""))

        if not any_data:
            return "（资金流 / CVD / OI 数据缺失，无法判断资金行为）"

        comprehensive = self._comprehensive_capital_judgment(by_tf)
        if comprehensive:
            lines.append(f"综合：{comprehensive}")
        return "\n".join(lines)

    # ==================================================================
    # 第 4 段：衍生品（funding 极端值 + OI/价散度）
    # ==================================================================
    def render_derivatives(self, factors: Dict[str, Any]) -> str:
        """
        渲染衍生品段
        --------------------------------------------------------------
        参数：
            factors: 多周期 factors dict
        返回：
            约 50-100 tokens。例如：
            "funding +12.00bp（偏高，多头持仓拥挤；任何反向触发都容易诱发挤压）
             OI/价散度：OI 创新高但价格未创新高 → potential_top"
        """
        by_tf = factors.get("by_timeframe") or {}
        deriv_1h = ((by_tf.get("1h") or {}).get("derivatives")) or {}
        funding = _to_float(deriv_1h.get("funding_rate_now"))
        oi_divergence = (deriv_1h.get("oi_price_divergence") or "").lower()

        lines: List[str] = []

        funding_desc = self._funding_label(funding)
        lines.append(funding_desc if funding is not None else "funding 数据缺失")

        if oi_divergence and oi_divergence != "none":
            oi_div_zh = self._oi_divergence_label(oi_divergence)
            lines.append(f"OI/价散度：{oi_div_zh}")

        return "\n".join(lines) if lines else "（衍生品数据缺失）"

    # ==================================================================
    # 第 5 段：关键价位 + 距当前价 ATR 倍数
    # ==================================================================
    def render_key_levels(self, factors: Dict[str, Any]) -> str:
        """
        渲染关键价位段（从大周期到小周期）
        --------------------------------------------------------------
        参数：
            factors: 多周期 factors dict
        返回：
            约 150 tokens。例如：
            "4h: 阻力 3625, 3680     支撑 3505, 3470
             1h: 阻力 3585, 3625     支撑 3518, 3505
             15m: 阻力 3585          支撑 3520
             当前 3530.15 ─ 距上方近阻 3585 仅 1.5×ATR(15m)；距下方近撑 3518 0.8×ATR(15m)"
        """
        by_tf = factors.get("by_timeframe") or {}
        ms_15m = ((by_tf.get("15m") or {}).get("market_structure")) or {}
        last_close = _to_float(ms_15m.get("last_close"))
        atr_15m = _to_float(ms_15m.get("atr_14"))

        lines: List[str] = []
        for tf in ("4h", "1h", "15m"):
            block = by_tf.get(tf) or {}
            struct = block.get("market_structure") or {}
            sup = self._normalize_levels(struct.get("supports"))
            res = self._normalize_levels(struct.get("resistances"))
            sup_str = ", ".join(_fmt_price(s) for s in sup[:3]) if sup else "-"
            res_str = ", ".join(_fmt_price(r) for r in res[:3]) if res else "-"
            lines.append(f"{tf}: 阻力 {res_str}    支撑 {sup_str}")

        if last_close is not None and atr_15m is not None and atr_15m > 0:
            all_res: List[float] = []
            all_sup: List[float] = []
            for tf in ("4h", "1h", "15m"):
                struct = ((by_tf.get(tf) or {}).get("market_structure")) or {}
                for r in self._normalize_levels(struct.get("resistances")):
                    if r > last_close:
                        all_res.append(r)
                for s in self._normalize_levels(struct.get("supports")):
                    if s < last_close:
                        all_sup.append(s)
            nearest_res = min(all_res) if all_res else None
            nearest_sup = max(all_sup) if all_sup else None
            distance_bits: List[str] = []
            if nearest_res is not None:
                d = (nearest_res - last_close) / atr_15m
                distance_bits.append(
                    f"距上方近阻 {_fmt_price(nearest_res)} {d:.1f}×ATR(15m)"
                )
            if nearest_sup is not None:
                d = (last_close - nearest_sup) / atr_15m
                distance_bits.append(
                    f"距下方近撑 {_fmt_price(nearest_sup)} {d:.1f}×ATR(15m)"
                )
            if distance_bits:
                lines.append(
                    f"当前 {_fmt_price(last_close)} ─ " + "；".join(distance_bits)
                )

        return "\n".join(lines) if lines else "（多周期关键价位数据缺失）"

    # ==================================================================
    # 第 6 段：流动性地图
    # ==================================================================
    def render_liquidity(self, factors: Dict[str, Any]) -> str:
        """
        渲染流动性地图段
        --------------------------------------------------------------
        参数：
            factors: 多周期 factors dict
        返回：
            约 80-120 tokens。例如：
            "上方止损池: 3625.00 (strong) | 3680.00 (medium)
             下方止损池: 3505.00 (medium) | 3470.00 (weak)
             订单簿: 上方真空 → 突破后下跌空间大，价位容易被快速穿透
             距离: 上方最近池 2.69%"
        """
        liq = factors.get("liquidity") or {}
        above = (liq.get("liquidity_pool_above") or [])[:3]
        below = (liq.get("liquidity_pool_below") or [])[:3]
        nearest_above_pct = _to_float(liq.get("nearest_above_pct"))
        nearest_below_pct = _to_float(liq.get("nearest_below_pct"))

        lines: List[str] = []
        if above:
            lines.append("上方止损池: " + " | ".join(self._fmt_pool(p) for p in above))
        else:
            lines.append("上方止损池: 无（突破后上行空间无明显阻力堆积）")
        if below:
            lines.append("下方止损池: " + " | ".join(self._fmt_pool(p) for p in below))
        else:
            lines.append("下方止损池: 无（真空区，跌破后下跌空间大）")

        by_tf = factors.get("by_timeframe") or {}
        ob_5m = ((by_tf.get("5m") or {}).get("orderbook")) or {}
        vacuum_above = ob_5m.get("liquidity_vacuum_above")
        vacuum_below = ob_5m.get("liquidity_vacuum_below")
        vacuum_bits: List[str] = []
        if vacuum_above is True:
            vacuum_bits.append("上方真空 → 突破后价位容易被快速穿透，追多需谨慎")
        if vacuum_below is True:
            vacuum_bits.append("下方真空 → 止损易被插针扫到，追空止损需放宽")
        if vacuum_bits:
            lines.append("订单簿: " + "；".join(vacuum_bits))

        dist_bits: List[str] = []
        if nearest_above_pct is not None:
            dist_bits.append(f"上方最近池 {nearest_above_pct * 100:.2f}%")
        if nearest_below_pct is not None:
            dist_bits.append(f"下方最近池 {nearest_below_pct * 100:.2f}%")
        if dist_bits:
            lines.append("距离: " + "，".join(dist_bits))

        return "\n".join(lines) if lines else "（流动性地图数据缺失）"

    # ==================================================================
    # 第 7 段：Liquidations 滚动窗口（plan 第 2.4 节新增）
    # ==================================================================
    def render_liquidations(self, factors: Dict[str, Any]) -> str:
        """
        渲染爆仓滚动窗口段 + cascade 判读
        --------------------------------------------------------------
        参数：
            factors: 因子聚合输出，要求顶层挂 ``liquidations`` 字典；
                     缺失时返回数据缺失提示。
        返回：
            约 100-150 tokens。例如：
            "近 5m:  long_liq 8.20 ETH | short_liq 0.30 ETH → 多头被清量级偏大
             近 15m: long_liq 12.40 ETH | short_liq 0.50 ETH | cascade=true → 多头清仓在进行
             近 1h:  long_liq 45.60 ETH | short_liq 2.10 ETH（多头爆仓主导）
             → 持续大额多头爆仓，短期反弹动力多来自空头回补而非新增买盘"
        说明：
            爆仓数据是 desk trader 的核心反转信号：
              * 大额多头爆仓 + cascade → 多头止损被引爆，短期内可能见到空头回补反弹；
              * 大额空头爆仓 + cascade → 空头止损被引爆，短期内可能见到多头平仓回落；
              * 双向爆仓量级相近 → 拉锯，方向中性。
        """
        liq = factors.get("liquidations") or {}
        if not liq:
            return "（爆仓数据缺失）"

        lines: List[str] = []
        long_totals: List[float] = []
        short_totals: List[float] = []
        windows = [(5, "近 5m"), (15, "近 15m"), (60, "近 1h")]
        cascade = bool(liq.get("cascade_signal"))

        for w_min, w_label in windows:
            long_v = _to_float(liq.get(f"long_{w_min}m"))
            short_v = _to_float(liq.get(f"short_{w_min}m"))
            if long_v is None and short_v is None:
                continue
            bits: List[str] = []
            if long_v is not None:
                bits.append(f"long_liq {long_v:.2f} ETH")
                long_totals.append(long_v)
            if short_v is not None:
                bits.append(f"short_liq {short_v:.2f} ETH")
                short_totals.append(short_v)
            row = f"{w_label}: " + " | ".join(bits)
            if w_min == 15 and cascade:
                row += " | cascade=true"
            # 单窗口语义标签
            tag = self._liquidation_window_tag(long_v, short_v)
            if tag:
                row += f" → {tag}"
            lines.append(row)

        if not lines:
            return "（爆仓窗口数据缺失）"

        overall = self._liquidation_overall_verdict(
            long_totals=long_totals,
            short_totals=short_totals,
            cascade=cascade,
        )
        if overall:
            lines.append(f"→ {overall}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 内部辅助：标签映射 + 因果判断
    # ------------------------------------------------------------------
    @staticmethod
    def _regime_label(regime: str) -> str:
        """regime 英文 → desk 中文标签"""
        mapping = {
            "trending_up": "趋势上行",
            "trending_down": "趋势下行",
            "breakout": "向上突破",
            "breakdown": "向下破位",
            "ranging": "窄幅震荡",
            "transitional": "状态切换",
            "overall": "未明",
            "unknown": "未明",
        }
        return mapping.get(regime, regime)

    @staticmethod
    def _atr_label(atr_pct: Optional[float]) -> str:
        """ATR 占比 → 波动语义标签"""
        if atr_pct is None:
            return "未知"
        if atr_pct < 0.0025:
            return "极低，缺乏可交易波动"
        if atr_pct < 0.005:
            return "偏低，谨慎设 SL"
        if atr_pct < 0.012:
            return "正常波动"
        if atr_pct < 0.020:
            return "偏高，止损要给空间"
        return "极端高波动，关注 squeeze 风险"

    @staticmethod
    def _adx_label(adx: Optional[float]) -> str:
        """ADX → 趋势强度语义"""
        if adx is None:
            return "ADX 数据缺失"
        if adx >= 25:
            return "趋势已确立"
        if adx >= 18:
            return "弱趋势 / 过渡区间"
        return "震荡市，方向单容易被 whipsaw"

    @staticmethod
    def _trend_arrow(trend: str) -> str:
        """trend 枚举 → 单字符方向符号"""
        return {
            "uptrend": "↑",
            "downtrend": "↓",
            "range": "→",
            "neutral": "→",
        }.get(trend, "─")

    @staticmethod
    def _funding_label(funding: Optional[float]) -> str:
        """funding 阈值 → desk 语义（带 long_squeeze / short_squeeze 判读）"""
        if funding is None:
            return "funding 数据缺失"
        bp = funding * 10000
        if bp >= 30:
            return (
                f"funding {bp:+.2f}bp 已偏高，多头持仓拥挤；"
                "任何反向触发（如 4h MA 跌破）都容易引发 long_squeeze"
            )
        if bp >= 10:
            return f"funding {bp:+.2f}bp 偏多，关注是否继续累积形成挤压风险"
        if bp <= -30:
            return (
                f"funding {bp:+.2f}bp 已偏空，空头持仓拥挤；"
                "任何反向触发都容易引发 short_squeeze"
            )
        if bp <= -10:
            return f"funding {bp:+.2f}bp 偏空，关注是否继续累积形成挤压风险"
        return f"funding {bp:+.2f}bp 中性，远未挤压"

    @staticmethod
    def _oi_divergence_label(divergence: str) -> str:
        """oi_price_divergence → desk 中文"""
        return {
            "potential_top": "OI 创新高但价格未创新高 → 资金在入场但价格乏力，潜在顶部",
            "potential_bottom": "OI 创新低但价格未创新低 → 空头撤退但价格企稳，潜在底部",
        }.get(divergence, divergence)

    @staticmethod
    def _fmt_money(usd: float) -> str:
        """USD 金额 desk 化展示：±1.20M / ±230K / ±450"""
        abs_v = abs(usd)
        sign = "+" if usd >= 0 else "-"
        if abs_v >= 1e6:
            return f"{sign}{abs_v / 1e6:.2f}M USD"
        if abs_v >= 1e3:
            return f"{sign}{abs_v / 1e3:.1f}K USD"
        return f"{sign}{abs_v:.0f} USD"

    @classmethod
    def _verdict_capital(
        cls,
        net_flow: Optional[float],
        cvd_slope: Optional[float],
        taker: Optional[float],
        oi_change: Optional[float],
        oi_rel: str,
        trend: str,
    ) -> str:
        """
        单周期资金行为综合判断（强制因果解读）
        --------------------------------------------------------------
        逻辑：
            1) OI 与价格四象限（最优先，最有判断意义）
            2) net_flow + CVD 是否一致（不一致 → 被动拉盘 / 砸盘）
            3) taker 偏向（弱信号，仅在数据足够时叠加）
        """
        bits: List[str] = []

        if oi_change is not None:
            if oi_rel == "uptrend" and oi_change > 0:
                bits.append("OI 与价格同步上升 = 多头真实建仓中，非短期投机推升")
            elif oi_rel == "uptrend" and oi_change < 0:
                bits.append("价格上行但 OI 减仓 = 空头平仓推升，趋势力度偏弱")
            elif oi_rel == "downtrend" and oi_change > 0:
                bits.append("OI 与价格同步下降 = 空头真实建仓中，趋势力度强")
            elif oi_rel == "downtrend" and oi_change < 0:
                bits.append("价格下行但 OI 减仓 = 多头平仓砸盘，趋势力度偏弱")

        if net_flow is not None and cvd_slope is not None:
            net_flow_dir = 1 if net_flow > 0 else -1 if net_flow < 0 else 0
            cvd_dir = 1 if cvd_slope > 0 else -1 if cvd_slope < 0 else 0
            if net_flow_dir > 0 and cvd_dir > 0:
                bits.append("做多证据：net_flow↑ + CVD↑ = 主动买盘真实；做空证据：无")
            elif net_flow_dir < 0 and cvd_dir < 0:
                bits.append("做多证据：无；做空证据：net_flow↓ + CVD↓ = 主动卖盘真实")
            elif net_flow_dir > 0 and cvd_dir < 0:
                bits.append(
                    "做多证据：无（价格上行但主动买盘未跟随，非真实推升）；"
                    "做空证据：被动拉盘 = 上涨持续性存疑，可能为空头回补"
                )
            elif net_flow_dir < 0 and cvd_dir > 0:
                bits.append(
                    "做多证据：被动砸盘 = 下跌持续性存疑，可能为多头止损回补；"
                    "做空证据：无（价格下行但主动卖盘未跟随，非真实推跌）"
                )

        return "；".join(bits)

    @classmethod
    def _comprehensive_capital_judgment(
        cls, by_tf: Dict[str, Any]
    ) -> Optional[str]:
        """
        多周期资金行为综合判断（用于第 3 段末尾的"综合"行）
        """
        cvd_dirs: List[int] = []
        nf_dirs: List[int] = []
        deception_tfs_bull: List[str] = []
        deception_tfs_bear: List[str] = []
        for tf in ("5m", "15m", "1h"):
            block = by_tf.get(tf) or {}
            cap = block.get("capital_flow") or {}
            net_flow = _to_float(cap.get("net_flow_usd"))
            cvd_slope = _to_float(cap.get("cvd_slope"))
            if net_flow is None or cvd_slope is None:
                continue
            nf_dir = 1 if net_flow > 0 else -1 if net_flow < 0 else 0
            cvd_dir = 1 if cvd_slope > 0 else -1 if cvd_slope < 0 else 0
            if nf_dir != 0:
                nf_dirs.append(nf_dir)
            if cvd_dir != 0:
                cvd_dirs.append(cvd_dir)
            if nf_dir > 0 and cvd_dir < 0:
                deception_tfs_bull.append(tf)
            elif nf_dir < 0 and cvd_dir > 0:
                deception_tfs_bear.append(tf)
        if not nf_dirs and not cvd_dirs:
            return None
        parts: List[str] = []
        if deception_tfs_bull:
            parts.append(
                f"{'/'.join(deception_tfs_bull)} net_flow↑ 与 CVD↓ 背离 → "
                "做多证据：无（价格上行但主动买盘未跟随）；"
                "做空证据：被动拉盘 = 上涨非资金推动，方向可信度下降"
            )
        if deception_tfs_bear:
            parts.append(
                f"{'/'.join(deception_tfs_bear)} net_flow↓ 与 CVD↑ 背离 → "
                "做空证据：无（价格下行但主动卖盘未跟随）；"
                "做多证据：被动砸盘 = 下跌非资金推动，方向可信度下降"
            )
        if parts:
            return "；".join(parts)
        if nf_dirs and all(d == nf_dirs[0] for d in nf_dirs) and cvd_dirs and all(
            d == cvd_dirs[0] for d in cvd_dirs
        ):
            side = "多" if nf_dirs[0] > 0 else "空"
            return f"资金行为多周期一致看{side}，方向可信"
        return "资金行为多周期分歧，需等待方向确认"

    @staticmethod
    def _normalize_levels(v: Any) -> List[float]:
        """支持 supports/resistances 是 list[float] 或 list[dict]"""
        if not isinstance(v, list):
            return []
        out: List[float] = []
        for x in v:
            if isinstance(x, (int, float)):
                f = _to_float(x)
                if f is not None:
                    out.append(f)
            elif isinstance(x, dict):
                f = _to_float(x.get("price") or x.get("level"))
                if f is not None:
                    out.append(f)
        return out

    @staticmethod
    def _fmt_pool(pool: Any) -> str:
        """流动性池条目格式化"""
        if not isinstance(pool, dict):
            return str(pool)
        price = _to_float(pool.get("price") or pool.get("level"))
        strength = pool.get("strength") or pool.get("type") or "?"
        if price is None:
            return f"({strength})"
        return f"{price:.2f} ({strength})"

    # ------------------------------------------------------------------
    # 爆仓段辅助：单窗口标签 + 整体反转动力判读
    # ------------------------------------------------------------------
    @staticmethod
    def _liquidation_window_tag(
        long_v: Optional[float], short_v: Optional[float]
    ) -> str:
        """
        单窗口爆仓量级 → desk 语义标签
        --------------------------------------------------------------
        判断"哪边量级显著占优"，便于在该窗口行末尾给一句解读。
        """
        if long_v is None and short_v is None:
            return ""
        long_v = long_v or 0.0
        short_v = short_v or 0.0
        # 任一方 < 1 ETH 量级太小，不打标签（避免噪声）
        if max(long_v, short_v) < 1.0:
            return ""
        # 一侧 ≥ 另一侧 5 倍视为显著主导
        if long_v >= short_v * 5 and long_v >= 1.0:
            return "多头被清量级显著占优"
        if short_v >= long_v * 5 and short_v >= 1.0:
            return "空头被清量级显著占优"
        return ""

    @staticmethod
    def _liquidation_overall_verdict(
        long_totals: List[float],
        short_totals: List[float],
        cascade: bool,
    ) -> str:
        """
        多窗口整体反转动力判读
        --------------------------------------------------------------
        逻辑：
            * 多窗口都偏多头爆仓 → 多头清仓潮，反弹动力多来自空头回补；
            * 多窗口都偏空头爆仓 → 空头清仓潮，回落动力多来自多头平仓；
            * 双方拉锯 → 双向流动性吃透中，方向中性；
            * cascade=true 时显式强调"清仓潮在加速"。
        """
        if not long_totals and not short_totals:
            return ""
        long_sum = sum(long_totals) if long_totals else 0.0
        short_sum = sum(short_totals) if short_totals else 0.0
        if max(long_sum, short_sum) < 1.0:
            return "爆仓量级偏小，对结构影响有限"
        if long_sum >= short_sum * 3 and long_sum >= 1.0:
            tail = "，cascade 在进行 → 清仓潮正在加速" if cascade else ""
            return (
                "持续大额多头爆仓，短期反弹动力多来自空头回补而非新增买盘"
                + tail
            )
        if short_sum >= long_sum * 3 and short_sum >= 1.0:
            tail = "，cascade 在进行 → 清仓潮正在加速" if cascade else ""
            return (
                "持续大额空头爆仓，短期回落动力多来自多头平仓而非新增卖盘"
                + tail
            )
        return "双向爆仓量级接近，流动性双向被吃透，方向中性"
