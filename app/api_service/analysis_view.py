"""分析结果序列化助手（前端友好视图，LLM-First 架构）。

职责
=====
把 ``signals`` 表查出来的原始 DB row 转换成 **对前端友好** 的 JSON 结构：

* 所有 Decimal / datetime 都转成 float / ISO8601 字符串；
* 给 bias / confidence 等枚举字段额外补一个中文 ``label`` 与 ``color``
  （前端可直接用作 Tag/Badge 的样式 key），无需自己再写映射；
* 把 ``signals.factors`` JSONB 拆成 ``factors_snapshot``（原值）+ ``regime`` /
  ``mtf_alignment`` / ``current_price`` 等顶层摘要字段；
* 把 ``timeframe_alignment`` 与 ``invalidation_conditions`` 这种 JSONB 列
  统一成 ``dict`` / ``list``（asyncpg 在某些场景下会返回字符串）；
* 把 ``entry_zone`` / ``take_profit`` 同时给 ``raw`` 与 ``formatted`` 字段，
  formatted 已经按方向语义排序，前端直接渲染即可；
* 计算 ``time_ago_seconds`` / ``time_ago_human``（"刚刚 / 3 分钟前 / 2 小时前"）
  给列表页用。

LLM-First 架构调整
==================
原先这层会把 ``signal_lifecycle`` JOIN 出的字段拼成 ``lifecycle`` 区块，
并从 ``signals.factors`` 里抽 ``rule_score`` / ``rule_contributions`` 拼成
``rule_engine`` 区块。重构后这两个区块整体删除——LLM 拥有 100% 决策权，
没有规则引擎打分；信号生命周期跟踪也已退场（plan 第 1.1 / 1.5 / 1.6 节）。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional

from app.utils import safe_float as _to_float


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

# 信号来源 → 中文标签（LLM-First 架构下只剩 llm / llm(cache) / atr_floor:*）
_SOURCE_LABEL: Dict[str, str] = {
    "llm": "LLM 决策",
    "llm(cache)": "LLM 决策（缓存）",
    "llm_unavailable": "LLM 不可用",
    "atr_floor:atr_too_low": "ATR 极端风控（波动过低）",
}


# _to_float 已由 from app.utils import safe_float as _to_float 提供


def _to_iso(v: Any) -> Optional[str]:
    """把 datetime 转成带时区 ISO8601 字符串"""
    if not isinstance(v, datetime):
        return None
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.isoformat()


def _coerce_json(v: Any) -> Any:
    """确保 JSONB 列在 Python 端是 dict / list（asyncpg 偶尔以 str 返回）"""
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
    """把"距今多少秒"转成人类可读的中文相对时间"""
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
    """把 [0, 1] 置信度映射为高 / 中 / 低标签"""
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
    """序列化 entry_zone（JSONB 列存的是 [low, high]）"""
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
    """序列化 take_profit 列表，并附带"距入场中点收益百分比" """
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
    """把 signals.timeframe_alignment JSONB 转成有序列表"""
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


def _extract_inner_factors(factors_blob: Any) -> Dict[str, Any]:
    """从 signals.factors JSONB 里取出"真正的因子快照"

    LLM-First 架构下 service.py 写库时 factors 仅包含 {"factors": {...}}，
    不再有 rule_score / rule_contributions 等顶层伴生字段。
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
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    把 fetch_latest_signal_full / fetch_recent_signals_full 出来的一行
    转成前端友好的完整分析视图
    -----------------------------------------------------------------
    参数：
        row                       : DB row（asyncpg Record / dict 均可）
        include_factors_snapshot  : 是否在响应中携带"全量因子快照"，
                                     前端列表页可关闭以节省带宽
        include_reasoning         : 是否携带 reasoning_content 全文，
                                     默认只返回 reasoning_available + 字符长度
        now                       : 用于计算 time_ago 的"现在"，缺省取 utcnow
    返回：
        包含以下顶层字段的 dict：
            signal_id / symbol / source / source_label
            ts / time_ago_seconds / time_ago_human
            summary（最关键的卡片字段）
            decision（bias / confidence / 文本判断）
            trading_plan（结构化交易计划）
            timeframe_alignment（多周期方向）
            invalidation_conditions（量化失效条件）
            market_context（regime / current_price / mtf_alignment / liquidations 摘要）
            reasoning_available / reasoning_total_chars / [reasoning_content]
            factors_snapshot（可选）
    """
    now = now or datetime.now(timezone.utc)

    signal_ts: Optional[datetime] = row.get("ts")
    if isinstance(signal_ts, datetime) and signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)
    age_seconds: Optional[float] = (
        (now - signal_ts).total_seconds() if isinstance(signal_ts, datetime) else None
    )
    bias = str(row.get("bias") or "neutral")
    source = str(row.get("source") or "llm")
    confidence = _to_float(row.get("confidence"))
    # reasoning_content 单条可达数十 KB，列表视图不会展示全文，
    # ``fetch_recent_signals_full`` 已经改为只 SELECT 派生列
    # ``reasoning_available`` / ``reasoning_total_chars``，避免一次性
    # 拉 100 条思维链把网络/超时打爆。这里同时兼容两种 row：
    #   - 详情页/最新一条：含 reasoning_content 全文
    #   - 列表页：只含派生列，row.get("reasoning_content") 为 None
    reasoning_text: Optional[str] = row.get("reasoning_content") or None
    if isinstance(reasoning_text, str):
        reasoning_chars = len(reasoning_text)
        reasoning_available = True
    else:
        raw_chars = row.get("reasoning_total_chars")
        try:
            reasoning_chars = int(raw_chars) if raw_chars is not None else 0
        except (TypeError, ValueError):
            reasoning_chars = 0
        raw_available = row.get("reasoning_available")
        if raw_available is None:
            reasoning_available = reasoning_chars > 0
        else:
            reasoning_available = bool(raw_available)

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

    factors_blob = _coerce_json(row.get("factors")) or {}
    inner_factors = _extract_inner_factors(factors_blob)
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

    invalidation = _coerce_json(row.get("invalidation_conditions")) or []
    if not isinstance(invalidation, list):
        invalidation = []
    timeframe_alignment = _serialize_timeframe_alignment(
        row.get("timeframe_alignment")
    )

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
        "reasoning_available": reasoning_available,
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
    """生成一行中文摘要，用于卡片标题 / 通知推送

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
