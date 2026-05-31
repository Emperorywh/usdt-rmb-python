"""Shared helper functions for all repository modules."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_dec(v: Any) -> Optional[Decimal]:
    """Coerce numerics into ``Decimal`` for asyncpg NUMERIC columns."""
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))


def parse_delete_count(status: str) -> int:
    """
    从 asyncpg ``execute`` 返回的命令状态字符串中解析受影响行数
    --------------------------------------------------------------
    参数：
        status: 形如 ``"DELETE 123"`` 的命令完成状态串
    返回：
        删除的行数；解析失败时返回 0（仅用于日志，不影响业务正确性）
    说明：
        asyncpg 的 ``Connection.execute`` 不像 ``fetch`` 那样直接给出
        rowcount，需要从 PG 协议返回的 CommandComplete 字符串里抠数字。
    """
    if not status:
        return 0
    parts = status.strip().split()
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0
