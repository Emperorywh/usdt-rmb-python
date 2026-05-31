"""HTTP routes（LLM-First 架构）。

LLM-First 重构后，本路由表只保留三大类接口：

1. ``/health`` / ``/healthz``                 ：基础健康度
2. ``/factors`` / ``/signal*`` / ``/analysis*``：行情因子与 LLM 信号读写
3. ``/emails*``                                ：邮件通知收件人 CRUD（运营）

已删除（与 plan 第 1.5 / 1.6 节对齐）：
- ``GET  /signals/{signal_id}/attribution``：规则引擎归因
- ``GET  /factors/weights/current``         ：因子权重表
- ``GET  /signals/lifecycle/stats``         ：信号生命周期统计
- ``POST /admin/calibrate-ic``              ：手动触发 IC 校准

这些接口依赖的 ``factor_weights`` / ``signal_lifecycle`` / IC 校准器在
LLM-First 架构下整体退场，相关 DB 表也已 DROP。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api_service.analysis_view import serialize_signal_full
from app.api_service.deps import (
    get_container,
    get_factor_aggregator,
    get_signal_service,
)
from app.container import AppContainer
from app.factor_engine.aggregator import FactorAggregator
from app.signal_engine.service import SignalService

router = APIRouter()


# 简单的邮箱格式校验：避免引入 email-validator 这一可选依赖。
# pydantic 的 EmailStr 需要 email-validator 才能工作，作为兜底。
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _validate_email_str(email: str) -> str:
    """
    轻量校验邮箱格式，去掉首尾空格并统一小写
    --------------------------------------------------------------
    参数：
        email: 原始邮箱字符串
    返回：
        清洗后的邮箱
    异常：
        HTTPException(400) - 格式不合法
    """
    cleaned = (email or "").strip()
    if not cleaned or not _EMAIL_RE.match(cleaned):
        raise HTTPException(status_code=400, detail=f"邮箱格式不合法: {email!r}")
    return cleaned.lower()


class NotificationEmailCreate(BaseModel):
    """新增邮件收件人入参"""

    email: str = Field(..., description="收件邮箱")
    name: Optional[str] = Field(default=None, max_length=128, description="备注名")
    enabled: bool = Field(default=True, description="是否启用")


class NotificationEmailUpdate(BaseModel):
    """更新邮件收件人入参（PATCH 语义）"""

    email: Optional[str] = Field(default=None, description="收件邮箱")
    name: Optional[str] = Field(default=None, max_length=128, description="备注名")
    enabled: Optional[bool] = Field(default=None, description="是否启用")


class TestEmailRequest(BaseModel):
    """/emails/test 入参：发送一封测试邮件到指定地址"""

    email: str = Field(..., description="测试邮箱地址")


@router.get("/health", tags=["meta"])
async def health(container: AppContainer = Depends(get_container)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "symbols": container.settings.ingestion.symbols,
        "exchanges": container.settings.ingestion.exchanges,
        "llm_enabled": container.llm_agent.enabled,
    }


@router.get("/healthz", tags=["meta"])
async def healthz(container: AppContainer = Depends(get_container)) -> Dict[str, Any]:
    """
    采集通道详细健康度
    -------------------------------------------------------------------
    返回字段：
        - status:  'ok' / 'degraded'，是否所有 WS 频道都在新鲜窗口内
        - ws:      {symbol: {kind: {age_seconds, last_event_at}}}
        - rest:    {op_name: {state, consecutive_failures, ...}}
    """
    ws_snapshot = (
        container.ingestion_runner.ws_health_snapshot()
        if container.ingestion_runner is not None
        else {}
    )
    rest_snapshot = container.okx_rest.health_snapshot()

    degraded = False
    for symbol_view in ws_snapshot.values():
        for kind, info in symbol_view.items():
            age = float(info.get("age_seconds") or 0.0)
            if kind == "funding_rate" and age > 5 * 60:
                degraded = True
            elif kind == "open_interest" and age > 60:
                degraded = True
            elif kind in ("trade", "orderbook") and age > 60:
                degraded = True

    return {
        "status": "degraded" if degraded else "ok",
        "ws": ws_snapshot,
        "rest": rest_snapshot,
    }


def _resolve_symbol(symbol: Optional[str], container: AppContainer) -> str:
    if symbol:
        return symbol
    if container.settings.ingestion.symbols:
        return container.settings.ingestion.symbols[0]
    raise HTTPException(status_code=400, detail="No symbol configured")


@router.get("/factors", tags=["factors"])
async def get_factors(
    symbol: Optional[str] = Query(default=None),
    container: AppContainer = Depends(get_container),
    aggregator: FactorAggregator = Depends(get_factor_aggregator),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    return await aggregator.compute(sym)


@router.get("/signal", tags=["signal"])
async def get_signal(
    symbol: Optional[str] = Query(default=None),
    include_reasoning: bool = Query(
        default=False,
        description="是否返回思考模式下的 reasoning_content 原文（可能很长）",
    ),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    row = await container.repos.fetch_latest_signal(sym)
    if not row:
        raise HTTPException(status_code=404, detail="No signal yet, try /signal/refresh")
    payload: Dict[str, Any] = {
        "timestamp": row["ts"].isoformat(),
        "symbol": row["symbol"],
        "source": row["source"],
        "signal": {
            "bias": row["bias"],
            "confidence": float(row["confidence"]),
            "reason": row["reason"],
            "risk": row["risk"],
            "suggestion": row["suggestion"],
        },
        "factors": row["factors"],
        # 仅暴露"是否存在思维链"作为元信息；具体内容只在显式请求时返回，
        # 避免默认响应被一两万字的思维链撑爆带宽与日志。
        "reasoning_available": bool(row.get("reasoning_content")),
    }
    if include_reasoning:
        payload["reasoning_content"] = row.get("reasoning_content")
    return payload


@router.post("/signal/refresh", tags=["signal"])
async def refresh_signal(
    symbol: Optional[str] = Query(default=None),
    include_reasoning: bool = Query(
        default=False,
        description="是否返回本次 LLM 的 reasoning_content 原文（可能很长）",
    ),
    container: AppContainer = Depends(get_container),
    signal_service: SignalService = Depends(get_signal_service),
) -> Dict[str, Any]:
    sym = _resolve_symbol(symbol, container)
    result = await signal_service.generate(sym)
    payload: Dict[str, Any] = {
        "timestamp": result["factors"]["computed_at"],
        "symbol": sym,
        "source": result["source"],
        "signal": result["signal"],
        # LLM-First 架构下不再返回 rule_signal / rule_score：
        # 决策路径已经只剩 LLM 一条线。
        "persisted": result["persisted"],
        "reasoning_available": result["reasoning_content"] is not None,
    }
    if include_reasoning:
        payload["reasoning_content"] = result["reasoning_content"]
    return payload


# ======================================================================
# 前端综合分析视图（最近一次 / 历史列表 / 单条详情）
# ======================================================================
@router.get("/analysis/latest", tags=["analysis"])
async def get_latest_analysis(
    symbol: Optional[str] = Query(default=None, description="合约代码；缺省取 settings.symbols[0]"),
    include_factors: bool = Query(
        default=True,
        description="是否携带完整 factors 快照；前端列表页可置 False 节省带宽",
    ),
    include_reasoning: bool = Query(
        default=False,
        description="是否携带 LLM 思维链全文（reasoning_content，可能很长）",
    ),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    返回最新一次综合分析（前端友好视图）
    -------------------------------------------------------------------
    包含：LLM 判断 / 结构化交易计划 / 多周期共振 / regime / 流动性地图 /
    失效条件 / 信号元数据。
    """
    sym = _resolve_symbol(symbol, container)
    row = await container.repos.fetch_latest_signal_full(sym)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=(
                "暂无分析结果，请稍后再试或调用 POST /signal/refresh 主动触发"
            ),
        )
    return serialize_signal_full(
        row,
        include_factors_snapshot=bool(include_factors),
        include_reasoning=bool(include_reasoning),
    )


@router.get("/analysis/history", tags=["analysis"])
async def get_analysis_history(
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100, description="返回条数（1~100）"),
    bias: Optional[str] = Query(
        default=None,
        description="按 bias 过滤：long / short / neutral",
    ),
    source: Optional[str] = Query(
        default=None,
        description="source ILIKE 模式（如 '%llm%' 只看 LLM 路径）",
    ),
    include_factors: bool = Query(
        default=False,
        description="列表场景默认不返回 factors 快照，节省带宽；详情页再单独取",
    ),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    返回最近 N 条综合分析（按时间倒序，前端时间线 / 历史列表用）
    """
    if bias is not None and bias not in ("long", "short", "neutral"):
        raise HTTPException(
            status_code=400, detail="bias 必须是 long / short / neutral"
        )
    sym = _resolve_symbol(symbol, container)
    rows = await container.repos.fetch_recent_signals_full(
        symbol=sym,
        limit=int(limit),
        bias=bias,
        source_like=source,
    )
    items = [
        serialize_signal_full(
            r,
            include_factors_snapshot=bool(include_factors),
            include_reasoning=False,
        )
        for r in rows
    ]
    return {
        "symbol": sym,
        "count": len(items),
        "filters": {"bias": bias, "source_like": source, "limit": int(limit)},
        "items": items,
    }


@router.get("/analysis/{signal_id}", tags=["analysis"])
async def get_analysis_by_id(
    signal_id: int,
    include_factors: bool = Query(default=True),
    include_reasoning: bool = Query(default=False),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """
    按 signals.id 读取单条综合分析详情
    -------------------------------------------------------------------
    与 /analysis/latest 字段一致；详情页通常打开 include_reasoning=true
    看思维链全文。
    """
    row = await container.repos.fetch_signal_full_by_id(signal_id)
    if not row:
        raise HTTPException(status_code=404, detail="signal_id not found")
    return serialize_signal_full(
        row,
        include_factors_snapshot=bool(include_factors),
        include_reasoning=bool(include_reasoning),
    )


# ======================================================================
# 邮件通知收件人 CRUD（notification_emails）
# ----------------------------------------------------------------------
# 设计原则：
#   - 路径 /emails 而非 /admin/emails，方便前端配置页直接调用；
#   - 所有写操作都做"邮箱格式预校验 + 唯一约束错误友好转换"；
#   - 输出统一走 _serialize_notification_email_row。
# ======================================================================
def _serialize_notification_email_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """把 notification_emails 表行序列化为前端友好的 dict"""
    created_at = row.get("created_at")
    updated_at = row.get("updated_at")
    return {
        "id": row.get("id"),
        "email": row.get("email"),
        "name": row.get("name"),
        "enabled": bool(row.get("enabled")),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else None,
    }


@router.get("/emails", tags=["notification"])
async def list_notification_emails(
    only_enabled: bool = Query(default=False, description="仅返回启用的收件人"),
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """列出所有邮件通知收件人"""
    rows = await container.repos.list_notification_emails(only_enabled=only_enabled)
    items = [_serialize_notification_email_row(r) for r in rows]
    return {"count": len(items), "items": items}


@router.post("/emails", tags=["notification"], status_code=201)
async def create_notification_email(
    payload: NotificationEmailCreate,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """新增一条邮件通知收件人。错误：400 格式 / 409 唯一冲突。"""
    email = _validate_email_str(payload.email)
    try:
        row = await container.repos.insert_notification_email(
            email=email,
            name=(payload.name or None),
            enabled=bool(payload.enabled),
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "notification_emails_unique" in msg or "duplicate key" in msg.lower():
            raise HTTPException(
                status_code=409, detail=f"邮箱已存在：{email}"
            ) from exc
        raise HTTPException(status_code=500, detail=f"新增邮箱失败：{msg}") from exc
    return _serialize_notification_email_row(row)


@router.get("/emails/{email_id}", tags=["notification"])
async def get_notification_email(
    email_id: int,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """按 id 查询单条邮件通知收件人详情"""
    row = await container.repos.fetch_notification_email_by_id(email_id)
    if not row:
        raise HTTPException(status_code=404, detail="未找到对应邮箱")
    return _serialize_notification_email_row(row)


@router.put("/emails/{email_id}", tags=["notification"])
async def update_notification_email(
    email_id: int,
    payload: NotificationEmailUpdate,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """更新一条邮件通知收件人（PATCH 语义：未提供的字段保持不变）"""
    new_email: Optional[str] = None
    if payload.email is not None:
        new_email = _validate_email_str(payload.email)
    try:
        row = await container.repos.update_notification_email(
            email_id,
            email=new_email,
            name=payload.name,
            enabled=payload.enabled,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "notification_emails_unique" in msg or "duplicate key" in msg.lower():
            raise HTTPException(
                status_code=409, detail=f"邮箱与他人冲突：{new_email}"
            ) from exc
        raise HTTPException(status_code=500, detail=f"更新邮箱失败：{msg}") from exc
    if not row:
        raise HTTPException(status_code=404, detail="未找到对应邮箱")
    return _serialize_notification_email_row(row)


@router.delete("/emails/{email_id}", tags=["notification"])
async def delete_notification_email(
    email_id: int,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """删除一条邮件通知收件人"""
    ok = await container.repos.delete_notification_email(email_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到对应邮箱")
    return {"ok": True, "id": email_id}


@router.post("/emails/test", tags=["notification"])
async def send_test_notification_email(
    payload: TestEmailRequest,
    container: AppContainer = Depends(get_container),
) -> Dict[str, Any]:
    """发送一封测试邮件（用于校验 Resend 配置是否正确）"""
    email = _validate_email_str(payload.email)
    sender = container.email_sender
    if sender is None or not sender.enabled:
        raise HTTPException(
            status_code=503,
            detail="邮件通知未启用或 Resend 凭据未配置（检查 ENABLE_EMAIL_NOTIFICATION / RESEND_API_KEY / RESEND_FROM）",
        )
    try:
        resp = await sender.send_test_email(email)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=(
                f"测试邮件发送失败：{exc}。"
                f"常见原因：1) RESEND_FROM 的域名未在 Resend 控制台 DNS 验证；"
                f"2) RESEND_API_KEY 无效或权限不足；"
                f"3) 仍在使用 onboarding@resend.dev 测试地址，但收件人不是 Resend 账号本人。"
            ),
        ) from exc
    msg_id = (resp or {}).get("id") if isinstance(resp, dict) else None
    return {"ok": True, "to": email, "resend_id": msg_id}
