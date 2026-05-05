"""compute_expires_at 的边界与优先级单元测试
=================================================================
覆盖：
    - ttl_minutes 优先于 ttl_hours
    - 仅给 ttl_hours（向后兼容路径）
    - 都不给（默认 90 分钟）
    - 边界保护：负数 / 0 被 clamp 到 1
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.signal_engine.lifecycle import compute_expires_at


# ---------------------------------------------------------------
# 固定一个 base 时间，避免依赖 datetime.now()
# ---------------------------------------------------------------
_BASE = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def test_ttl_minutes_takes_priority() -> None:
    """同时给 minutes 和 hours，必须按 minutes 计算"""
    out = compute_expires_at(now=_BASE, ttl_hours=24, ttl_minutes=90)
    assert out == _BASE + timedelta(minutes=90)


def test_ttl_minutes_only_default_path() -> None:
    """只给 ttl_minutes：标准新通道"""
    out = compute_expires_at(now=_BASE, ttl_minutes=45)
    assert out == _BASE + timedelta(minutes=45)


def test_ttl_hours_backward_compat() -> None:
    """只给 ttl_hours：兼容历史调用方"""
    out = compute_expires_at(now=_BASE, ttl_hours=24)
    assert out == _BASE + timedelta(hours=24)


def test_no_ttl_defaults_to_90_minutes() -> None:
    """都不给：默认 90 分钟（与 settings.lifecycle_default_ttl_minutes 对齐）"""
    out = compute_expires_at(now=_BASE)
    assert out == _BASE + timedelta(minutes=90)


def test_ttl_minutes_zero_clamped_to_one() -> None:
    """ttl_minutes=0 不能让 expires_at 与 now 相等（会立刻 expired）"""
    out = compute_expires_at(now=_BASE, ttl_minutes=0)
    assert out == _BASE + timedelta(minutes=1)


def test_ttl_minutes_negative_clamped_to_one() -> None:
    """负数 ttl 被 clamp 到 1 分钟，不会回到过去"""
    out = compute_expires_at(now=_BASE, ttl_minutes=-30)
    assert out == _BASE + timedelta(minutes=1)


def test_ttl_hours_zero_clamped_to_one() -> None:
    """ttl_hours=0 同理被 clamp 到 1 小时（保持兼容路径的语义）"""
    out = compute_expires_at(now=_BASE, ttl_hours=0)
    assert out == _BASE + timedelta(hours=1)


def test_default_now_uses_utc() -> None:
    """不显式传 now 时取 UTC now，结果必须带 tzinfo"""
    out = compute_expires_at(ttl_minutes=10)
    assert out.tzinfo is not None
    assert out.tzinfo.utcoffset(out) == timedelta(0)
