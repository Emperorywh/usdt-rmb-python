"""分析结果序列化助手（前端友好视图）。

职责
=====
- 把 ``signals`` 表（+ ``signal_lifecycle`` JOIN）出来的原始 DB row
  转换成 **对前端友好** 的 JSON 结构：
    * 所有 Decimal / datetime 都转成 float / ISO8601 字符串；
    * 给 bias / confidence / lifecycle_status 等枚举字段额外补一个中文 ``label``
      与 ``color``（前端可直接用作 Tag/Badge 的样式 key），无需自己再写映射；
    * 把 ``signals.factors`` JSONB 拆成 ``factors_snapshot``（原值）+
      ``rule_engine``（rule_score / 排序后的 top contributions）+ ``regime`` /
      ``mtf_alignment`` / ``current_price`` 等顶层摘要字段；
    * 把 ``timeframe_alignment`` 与 ``invalidation_conditions`` 这种 JSONB 列
      统一成 ``dict`` / ``list``（asyncpg 在某些场景下会返回字符串）；
    * 把 ``entry_zone`` / ``take_profit`` 同时给 ``raw`` 与 ``formatted``
      字段，formatted 已经按方向语义排序，前端直接渲染即可；
    * 计算 ``time_ago_seconds`` / ``time_ago_human``（"刚刚 / 3 分钟前 / 2 小时前"）
      给列表页用。

设计取舍
========
- 这层代码 **不查表**、**不做计算**：纯字典转换 + 字面量映射；
  让 routes.py 保持瘦层，只负责"读 row → 序列化 → 返回"。
- 顶层 ``summary`` 字段把"前端常用的 4~5 个字段"提到根节点，
  即便不 parse 完整结构也能渲染卡片摘要。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional


# ----------------------------------------------------------------------
# 字面量映射表
# ----------------------------------------------------------------------
# 方向偏置 → 中文 / 颜色（颜色 key 与 Ant Design Tag 的 status 对齐）。
_BIAS_LABEL: Dict[str, str] = {
    "long": "做多",
    "short": "做空",
    "neutral": "观望",
}
_BIAS_COLOR: Dict[str, str] = {
    "long": "success",
    "short": "error",
    "neutral": "default",
}

# 信号来源 → 中文标签
_SOURCE_LABEL: Dict[str, str] = {
    "rules": "规则引擎",
    "rules+llm": "规则+LLM",
    "rules+llm(cache)": "规则+LLM(缓存)",
    "rules+llm(shadow)": "规则+LLM(影子)",
}

# 生命周期状态 → 中文 / 颜色 / 是否结算
# 与 schema.sql 中 signal_lifecycle.status CHECK 约束严格对齐。
_LIFECYCLE_LABEL: Dict[str, str] = {
    "pending": "待入场",
    "triggered": "进行中",
    "tp1_hit": "止盈 1 触发",
    "tp2_hit": "止盈 2 触发",
    "sl_hit": "止损触发",
    "expired": "已过期",
    "invalidated": "失效",
}
_LIFECYCLE_COLOR: Dict[str, str] = {
    "pending": "processing",
    "triggered": "warning",
    "tp1_hit": "success",
    "tp2_hit": "success",
    "sl_hit": "error",
    "expired": "default",
    "invalidated": "default",
}
_LIFECYCLE_SETTLED = {"tp1_hit", "tp2_hit", "sl_hit", "expired", "invalidated"}

# 多周期共振 trend → 与 bias 对齐的方向标签
_TREND_TO_BIAS: Dict[str, str] = {
    "uptrend": "long",
    "downtrend": "short",
    "range": "neutral",
    "neutral": "neutral",
}


# ----------------------------------------------------------------------
# 通用类型转换
# ----------------------------------------------------------------------
def _to_float(v: Any) -> Optional[float]:
    """
    把 Decimal / int / str 等数值类型安全转成 float
    -----------------------------------------------------------------
    参数：
        v: 任意值
    返回：
        float；None 与无法转换时返回 None。
    说明：
        asyncpg 把 NUMERIC 列返回 Decimal，前端 JSON 不友好；统一转 float。
    """
    if v is None:
        return None
    if isinstance(v, Decimal):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_iso(v: Any) -> Optional[str]:
    """
    把 datetime 转成带时区 ISO8601 字符串
    -----------------------------------------------------------------
    参数：
        v: datetime 或 None
    返回：
        ISO8601 字符串；非 datetime 时返回 None。
    说明：
        DB 列基本都是 TIMESTAMPTZ；UTC 直接输出，前端按本地时区 format。
    """
    if not isinstance(v, datetime):
        return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.isoformat()


def _coerce_json(v: Any) -> Any:
    """
    确保 JSONB 列在 Python 端是 dict / list（asyncpg 偶尔以 str 返回）
    -----------------------------------------------------------------
    参数：
        v: dict / list / str(JSON) / None
    返回：
        反序列化后的 dict / list；解析失败原样返回。
    说明：
        ``signals.factors`` 等 JSONB 列在大多数 asyncpg 配置下会自动反序列化
        为 dict；但若用户没装 codec / 自定义类型，asyncpg 会以字符串返回，
        这里加一层防御，保证下游所有字段访问都按 dict / list 处理。
    """
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, (bytes, bytearray)):
        try:
            return json.loads(v.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _human_time_ago(seconds: Optional[float]) -> str:
    """
    把"距今多少秒"转成人类可读的中文相对时间
    -----------------------------------------------------------------
    参数：
        seconds: 距今秒数；None 返回 '未知'
    返回：
        '刚刚' / 'N 秒前' / 'N 分钟前' / 'N 小时前' / 'N 天前'
    说明：
        负数（信号在未来，理论不可能但要防御）返回 '刚刚'。
    """
    if seconds is None:
        return "未知"
    if seconds < 30:
        return "刚刚"
    if seconds < 60:
        return f"{int(seconds)} 秒前"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时前"
    return f"{int(seconds // 86400)} 天前"


def _confidence_label(conf: Optional[float]) -> str:
    """
    把 [0, 1] 置信度映射为高 / 中 / 低标签
    -----------------------------------------------------------------
    参数：
        conf: 置信度
    返回：
        '高' (>=0.7) / '中' (>=0.4) / '低' / '未知'
    """
    if conf is None:
        return "未知"
    if conf >= 0.7:
        return "高"
    if conf >= 0.4:
        return "中"
    return "低"


# ----------------------------------------------------------------------
# 子结构序列化
# ----------------------------------------------------------------------
def _serialize_entry_zone(raw: Any) -> Optional[Dict[str, Any]]:
    """
    序列化 entry_zone（JSONB 列存的是 [low, high]）
    -----------------------------------------------------------------
    参数：
        raw: list / tuple / None
    返回：
        {low, high, mid, width_pct} 或 None
    说明：
        - mid 取 (low + high) / 2，方便前端画"中点参考线"；
        - width_pct = (high - low) / mid，让前端可直接展示"区间宽度 0.35%"。
    """
    raw = _coerce_json(raw)
    if not raw or not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        low = float(min(raw[0], raw[1]))
        high = float(max(raw[0], raw[1]))
    except (TypeError, ValueError):
        return None
    mid = (low + high) / 2 if (low + high) else None
    width_pct = ((high - low) / mid) if mid else None
    return {
        "low": low,
        "high": high,
        "mid": mid,
        "width_pct": width_pct,
    }


def _serialize_take_profit(
    raw: Any,
    bias: str,
    entry_mid: Optional[float],
) -> List[Dict[str, Any]]:
    """
    序列化 take_profit 列表，并附带"距入场中点收益百分比"
    -----------------------------------------------------------------
    参数：
        raw       : take_profit JSONB（list of float）
        bias      : 'long' / 'short' / 'neutral'
        entry_mid : 入场区间中点；用于算 reward_pct
    返回：
        [{level, price, reward_pct}, ...]，按多空方向排序好。
    """
    raw = _coerce_json(raw)
    if not raw or not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, p in enumerate(raw, start=1):
        try:
            price = float(p)
        except (TypeError, ValueError):
            continue
        reward_pct: Optional[float] = None
        if entry_mid and entry_mid > 0:
            if bias == "long":
                reward_pct = (price - entry_mid) / entry_mid
            elif bias == "short":
                reward_pct = (entry_mid - price) / entry_mid
        out.append({
            "level": f"tp{idx}",
            "price": price,
            "reward_pct": reward_pct,
        })
    return out


def _serialize_timeframe_alignment(raw: Any) -> List[Dict[str, str]]:
    """
    把 ``signals.timeframe_alignment`` JSONB 转成有序列表
    -----------------------------------------------------------------
    参数：
        raw: dict like {"5m": "long", "15m": "neutral", ...}
    返回：
        [{timeframe, bias, label, color}, ...]，固定按 5m/15m/1h/4h/1d 顺序。
    """
    raw = _coerce_json(raw) or {}
    if not isinstance(raw, dict):
        return []
    order = ["5m", "15m", "1h", "4h", "1d"]
    out: List[Dict[str, str]] = []
    for tf in order:
        bias = raw.get(tf)
        if bias is None:
            continue
        bias = str(bias)
        out.append({
            "timeframe": tf,
            "bias": bias,
            "label": _BIAS_LABEL.get(bias, bias),
            "color": _BIAS_COLOR.get(bias, "default"),
        })
    # 兜底：不在白名单顺序里的也追加（容错）
    extras = [tf for tf in raw.keys() if tf not in order]
    for tf in extras:
        bias = str(raw[tf])
        out.append({
            "timeframe": str(tf),
            "bias": bias,
            "label": _BIAS_LABEL.get(bias, bias),
            "color": _BIAS_COLOR.get(bias, "default"),
        })
    return out


def _serialize_rule_contributions(
    contribs: Any, top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    把 rule_contributions 字典展开成"按贡献度绝对值降序"的扁平列表
    -----------------------------------------------------------------
    参数：
        contribs: P2 升级后的 contributions 结构，key 形如 "5m:capital_flow.cvd_zscore"
        top_n   : 截断的 top N
    返回：
        [{key, timeframe, group, factor, contribution, abs_contribution}, ...]
    说明：
        前端可以直接用作"归因热力条"，不需要再二次解析 key 字符串。
    """
    contribs = _coerce_json(contribs) or {}
    if not isinstance(contribs, dict):
        return []
    items: List[Dict[str, Any]] = []
    for k, v in contribs.items():
        val = _to_float(v)
        if val is None:
            continue
        timeframe = ""
        group = ""
        factor = str(k)
        # 兼容两种 key 形态：
        # - 新："tf:group.factor_name"
        # - 旧："group.factor_name" 或 "factor_name"
        if isinstance(k, str):
            tf_part = k
            if ":" in k:
                tf_part, rest = k.split(":", 1)
                timeframe = tf_part
                if "." in rest:
                    group, factor = rest.split(".", 1)
                else:
                    group, factor = "", rest
            elif "." in k:
                group, factor = k.split(".", 1)
        items.append({
            "key": str(k),
            "timeframe": timeframe,
            "group": group,
            "factor": factor,
            "contribution": val,
            "abs_contribution": abs(val),
        })
    items.sort(key=lambda x: x["abs_contribution"], reverse=True)
    return items[:max(1, int(top_n))]


def _extract_inner_factors(factors_blob: Any) -> Dict[str, Any]:
    """
    从 signals.factors JSONB 里取出"真正的因子快照"
    -----------------------------------------------------------------
    参数：
        factors_blob: dict（service.py 写入时是 {"factors": {...}, "rule_score": ..., "rule_contributions": ...}）
    返回：
        内层 factors dict；老格式 / 缺字段时返回原 dict / {}
    """
    blob = _coerce_json(factors_blob) or {}
    if not isinstance(blob, dict):
        return {}
    inner = blob.get("factors")
    if isinstance(inner, dict):
        return inner
    return blob


# ----------------------------------------------------------------------
# 主入口：单条信号 → 前端友好 dict
# ----------------------------------------------------------------------
def serialize_signal_full(
    row: Mapping[str, Any],
    *,
    include_factors_snapshot: bool = True,
    include_reasoning: bool = False,
    rule_contributions_top_n: int = 10,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    把 fetch_latest_signal_full / fetch_recent_signals_full 出来的一行
    转成前端友好的完整分析视图
    -----------------------------------------------------------------
    参数：
        row                         : DB row（asyncpg Record / dict 均可）
        include_factors_snapshot    : 是否在响应中携带"全量因子快照"，
                                       前端列表页可关闭以节省带宽
        include_reasoning           : 是否携带 reasoning_content 全文，
                                       默认只返回 ``reasoning_available`` + 字符长度
        rule_contributions_top_n    : 归因贡献度截取 top N（默认 10）
        now                         : 用于计算 time_ago 的"现在"，缺省取 utcnow
    返回：
        包含以下顶层字段的 dict（详见模块 docstring）：
            signal_id / symbol / source / source_label
            ts / time_ago_seconds / time_ago_human
            summary（最关键的卡片字段）
            decision（bias / confidence / 文本判断）
            trading_plan（结构化交易计划）
            timeframe_alignment（多周期方向）
            invalidation_conditions（量化失效条件）
            market_context（regime / current_price / mtf_alignment / liquidations 摘要）
            rule_engine（rule_score + 排序好的 top contributions）
            lifecycle（实战表现）
            reasoning_available / reasoning_total_chars / [reasoning_content]
            factors_snapshot（可选）
    """
    now = now or datetime.now(timezone.utc)

    # ---- 基础元信息 ----
    signal_ts: Optional[datetime] = row.get("ts")
    if isinstance(signal_ts, datetime) and signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)
    age_seconds: Optional[float] = (
        (now - signal_ts).total_seconds() if isinstance(signal_ts, datetime) else None
    )
    bias = str(row.get("bias") or "neutral")
    source = str(row.get("source") or "rules")
    confidence = _to_float(row.get("confidence"))
    reasoning_text: Optional[str] = row.get("reasoning_content") or None
    reasoning_chars = len(reasoning_text) if isinstance(reasoning_text, str) else 0

    # ---- 分支 1：交易计划 ----
    entry_zone = _serialize_entry_zone(row.get("entry_zone"))
    entry_mid = entry_zone.get("mid") if entry_zone else None
    take_profit = _serialize_take_profit(
        row.get("take_profit"), bias=bias, entry_mid=entry_mid
    )
    stop_loss = _to_float(row.get("stop_loss"))
    stop_loss_pct: Optional[float] = None
    if stop_loss is not None and entry_mid:
        if bias == "long":
            stop_loss_pct = (entry_mid - stop_loss) / entry_mid
        elif bias == "short":
            stop_loss_pct = (stop_loss - entry_mid) / entry_mid
    rr = _to_float(row.get("risk_reward_ratio"))
    pos_pct = _to_float(row.get("position_size_pct"))

    # ---- 分支 2：因子快照 / 规则引擎细节 ----
    factors_blob = _coerce_json(row.get("factors")) or {}
    inner_factors = _extract_inner_factors(factors_blob)
    rule_score = (
        _to_float(factors_blob.get("rule_score"))
        if isinstance(factors_blob, dict)
        else None
    )
    rule_contribs = _serialize_rule_contributions(
        factors_blob.get("rule_contributions") if isinstance(factors_blob, dict) else None,
        top_n=rule_contributions_top_n,
    )
    regime: Optional[str] = None
    current_price: Optional[float] = None
    mtf_alignment_summary: Dict[str, Any] = {}
    liquidations_summary: Dict[str, Any] = {}
    liquidity_summary: Dict[str, Any] = {}
    if isinstance(inner_factors, dict):
        regime = inner_factors.get("regime")
        liquidity = inner_factors.get("liquidity") or {}
        if isinstance(liquidity, dict):
            current_price = _to_float(liquidity.get("current_price"))
            liquidity_summary = {
                "current_price": current_price,
                "nearest_above_pct": _to_float(liquidity.get("nearest_above_pct")),
                "nearest_below_pct": _to_float(liquidity.get("nearest_below_pct")),
                "pool_above_count": len(liquidity.get("liquidity_pool_above") or []),
                "pool_below_count": len(liquidity.get("liquidity_pool_below") or []),
            }
        mtf = inner_factors.get("mtf_alignment") or {}
        if isinstance(mtf, dict):
            mtf_alignment_summary = {
                "alignment_score": _to_float(mtf.get("alignment_score")),
                "dominant_bias": mtf.get("dominant_bias"),
                "trend_votes": mtf.get("trend_votes") or {},
            }
        liq = inner_factors.get("liquidations") or {}
        if isinstance(liq, dict):
            liquidations_summary = {
                k: _to_float(v) if isinstance(v, (Decimal, int, float)) else v
                for k, v in liq.items()
            }

    # ---- 分支 3：lifecycle 实战表现 ----
    lifecycle_status_raw = row.get("lifecycle_status")
    lifecycle_status = str(lifecycle_status_raw) if lifecycle_status_raw else None
    lifecycle = {
        "status": lifecycle_status,
        "status_label": _LIFECYCLE_LABEL.get(lifecycle_status or "", "未跟踪"),
        "status_color": _LIFECYCLE_COLOR.get(lifecycle_status or "", "default"),
        "is_settled": lifecycle_status in _LIFECYCLE_SETTLED,
        "is_open": lifecycle_status in ("pending", "triggered"),
        "triggered_at": _to_iso(row.get("lifecycle_triggered_at")),
        "triggered_price": _to_float(row.get("triggered_price")),
        "exit_at": _to_iso(row.get("lifecycle_exit_at")),
        "exit_price": _to_float(row.get("exit_price")),
        "pnl_pct": _to_float(row.get("pnl_pct")),
        "max_favorable_pct": _to_float(row.get("max_favorable_pct")),
        "max_adverse_pct": _to_float(row.get("max_adverse_pct")),
        "expires_at": _to_iso(row.get("lifecycle_expires_at")),
        "updated_at": _to_iso(row.get("lifecycle_updated_at")),
    }

    # ---- 分支 4：决策、判断 ----
    decision = {
        "bias": bias,
        "bias_label": _BIAS_LABEL.get(bias, bias),
        "bias_color": _BIAS_COLOR.get(bias, "default"),
        "confidence": confidence,
        "confidence_pct": (round(confidence * 100, 2) if confidence is not None else None),
        "confidence_label": _confidence_label(confidence),
        "reason": row.get("reason") or "",
        "risk": row.get("risk") or "",
        "suggestion": row.get("suggestion") or "",
    }

    # ---- 分支 5：交易计划聚合 ----
    trading_plan = {
        "has_plan": bias != "neutral" and entry_zone is not None,
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "stop_loss_pct": stop_loss_pct,
        "take_profit": take_profit,
        "risk_reward_ratio": rr,
        "position_size_pct": pos_pct,
        "position_size_label": (
            f"{round(pos_pct * 100, 2)}%" if pos_pct is not None else None
        ),
    }

    # ---- 分支 6：失效条件 / 多周期 ----
    invalidation = _coerce_json(row.get("invalidation_conditions")) or []
    if not isinstance(invalidation, list):
        invalidation = []
    timeframe_alignment = _serialize_timeframe_alignment(
        row.get("timeframe_alignment")
    )

    # ---- 顶层 summary：给"卡片预览"用 ----
    summary = {
        "bias": bias,
        "bias_label": _BIAS_LABEL.get(bias, bias),
        "bias_color": _BIAS_COLOR.get(bias, "default"),
        "confidence": confidence,
        "confidence_label": _confidence_label(confidence),
        "regime": regime,
        "current_price": current_price,
        "risk_reward_ratio": rr,
        "position_size_pct": pos_pct,
        "lifecycle_status": lifecycle_status,
        "lifecycle_status_label": _LIFECYCLE_LABEL.get(
            lifecycle_status or "", "未跟踪"
        ),
        "pnl_pct": lifecycle["pnl_pct"],
        "headline": _build_headline(
            bias=bias,
            confidence_label=decision["confidence_label"],
            regime=regime,
            current_price=current_price,
            symbol=row.get("symbol"),
        ),
    }

    payload: Dict[str, Any] = {
        "signal_id": row.get("id"),
        "symbol": row.get("symbol"),
        "source": source,
        "source_label": _SOURCE_LABEL.get(source, source),
        "ts": _to_iso(signal_ts),
        "time_ago_seconds": age_seconds,
        "time_ago_human": _human_time_ago(age_seconds),
        "summary": summary,
        "decision": decision,
        "trading_plan": trading_plan,
        "timeframe_alignment": timeframe_alignment,
        "invalidation_conditions": invalidation,
        "market_context": {
            "regime": regime,
            "current_price": current_price,
            "mtf_alignment": mtf_alignment_summary,
            "liquidations": liquidations_summary,
            "liquidity": liquidity_summary,
        },
        "rule_engine": {
            "rule_score": rule_score,
            "top_contributions": rule_contribs,
        },
        "lifecycle": lifecycle,
        "reasoning_available": bool(reasoning_text),
        "reasoning_total_chars": reasoning_chars,
    }

    if include_reasoning and reasoning_text:
        payload["reasoning_content"] = reasoning_text
    if include_factors_snapshot:
        payload["factors_snapshot"] = inner_factors

    return payload


def _build_headline(
    *,
    bias: str,
    confidence_label: str,
    regime: Optional[str],
    current_price: Optional[float],
    symbol: Any,
) -> str:
    """
    生成一行中文摘要，用于卡片标题 / 通知推送
    -----------------------------------------------------------------
    参数：
        bias             : long / short / neutral
        confidence_label : 高 / 中 / 低 / 未知
        regime           : market regime（如 trending_up），可空
        current_price    : 最新价，可空
        symbol           : 合约代码
    返回：
        例： "ETH-USDT-SWAP 做多（高置信） · 趋势上行 · 当前价 3120.50"
    """
    parts: List[str] = []
    sym_text = str(symbol) if symbol else ""
    if sym_text:
        parts.append(sym_text)
    bias_text = _BIAS_LABEL.get(bias, bias)
    parts.append(f"{bias_text}（{confidence_label}置信）")
    if regime:
        parts.append(str(regime))
    if current_price is not None:
        parts.append(f"当前价 {current_price:.4f}".rstrip("0").rstrip("."))
    return " · ".join(parts)
