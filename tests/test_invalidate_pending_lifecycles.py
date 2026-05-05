"""Repositories.invalidate_pending_lifecycles_for_symbol 单元测试
=================================================================
不实际起 PostgreSQL，使用 mock 链路验证：
    1) 发出的 SQL 仅作用在 status='pending'（不会动 triggered / 已结算行）；
    2) 把 status 写为 'invalidated' 且会同时写 exit_at / updated_at；
    3) 解析 'UPDATE n' 的返回值正确转成行数；
    4) symbol 作为唯一参数透传，不会越过 symbol 边界。

测试只关心 repository 层的 SQL 形状与返回值解析，
不验证业务上下游（service.py supersede 调用），那部分由 service 实测覆盖。
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import pytest

from app.data_storage.repositories import Repositories


# ----------------------------------------------------------------------
# 极简 db stub：模拟 self.db.acquire().__aenter__() 返回 conn
# 以及 self.db.run_with_retry(op, op_name=...) 直接 await op()
# ----------------------------------------------------------------------
class _ConnStub:
    """模拟 asyncpg connection，记录 execute 调用参数与返回 'UPDATE n' tag"""

    def __init__(self, update_tag: str = "UPDATE 0"):
        self.update_tag = update_tag
        # 把每一次 conn.execute 的 (sql, *args) 都记下来
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return self.update_tag


class _AcquireCtx:
    """模拟 ``async with self.db.acquire() as conn`` 上下文"""

    def __init__(self, conn: _ConnStub):
        self._conn = conn

    async def __aenter__(self) -> _ConnStub:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _DbStub:
    """
    模拟 Repositories.db 字段
    -----------------------------------------------------------------
    - acquire(): 返回一个产出固定 conn 的 async context manager
    - run_with_retry(op, op_name=...): 直接 await op()，不做退避（测试场景下足够）
    """

    def __init__(self, conn: _ConnStub):
        self.conn = conn

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self.conn)

    async def run_with_retry(
        self,
        op: Callable[[], Awaitable[Any]],
        *,
        op_name: str,
    ) -> Any:
        # 直接 await 一次；不做实际重试逻辑
        return await op()


def _build_repos(update_tag: str = "UPDATE 0") -> tuple[Repositories, _ConnStub]:
    """构造一个绑定 stub db 的 Repositories 实例"""
    conn = _ConnStub(update_tag=update_tag)
    db_stub = _DbStub(conn=conn)
    repos = Repositories.__new__(Repositories)
    repos.db = db_stub  # type: ignore[attr-defined]
    return repos, conn


# ----------------------------------------------------------------------
# 测试用例
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_only_targets_pending_rows() -> None:
    """SQL 必须含 ``status = 'pending'`` 的 WHERE 条件，确保不动 triggered 行"""
    repos, conn = _build_repos(update_tag="UPDATE 3")

    n = await repos.invalidate_pending_lifecycles_for_symbol("ETH-USDT-SWAP")

    assert n == 3
    assert len(conn.calls) == 1
    sql, args = conn.calls[0]
    assert "status = 'pending'" in sql
    assert "status     = 'invalidated'" in sql or "status = 'invalidated'" in sql
    assert "exit_at" in sql
    assert "updated_at" in sql
    assert args == ("ETH-USDT-SWAP",)


@pytest.mark.asyncio
async def test_does_not_touch_triggered_or_settled() -> None:
    """SQL 不允许出现 status IN (...) 这种会顺带捎上 triggered / 已结算行的写法"""
    repos, conn = _build_repos(update_tag="UPDATE 0")

    await repos.invalidate_pending_lifecycles_for_symbol("ETH-USDT-SWAP")

    sql, _ = conn.calls[0]
    assert "triggered" not in sql.lower()
    assert "tp1_hit" not in sql
    assert "tp2_hit" not in sql
    assert "sl_hit" not in sql


@pytest.mark.asyncio
async def test_parses_update_tag_to_row_count() -> None:
    """必须把 asyncpg 风格的 'UPDATE 7' 字符串转成数字 7"""
    repos, _ = _build_repos(update_tag="UPDATE 7")
    n = await repos.invalidate_pending_lifecycles_for_symbol("ETH-USDT-SWAP")
    assert n == 7


@pytest.mark.asyncio
async def test_parses_zero_update_tag() -> None:
    """无任何行被改时返回 0，不抛异常"""
    repos, _ = _build_repos(update_tag="UPDATE 0")
    n = await repos.invalidate_pending_lifecycles_for_symbol("ETH-USDT-SWAP")
    assert n == 0


@pytest.mark.asyncio
async def test_handles_unexpected_tag_gracefully() -> None:
    """tag 解析失败时降级返回 0，不能向上抛错把信号主路径打挂"""
    repos, _ = _build_repos(update_tag="")
    n = await repos.invalidate_pending_lifecycles_for_symbol("ETH-USDT-SWAP")
    assert n == 0


@pytest.mark.asyncio
async def test_symbol_param_isolation() -> None:
    """传入的 symbol 必须作为唯一 SQL 参数，不越过 symbol 边界"""
    repos, conn = _build_repos(update_tag="UPDATE 1")

    await repos.invalidate_pending_lifecycles_for_symbol("BTC-USDT-SWAP")

    _, args = conn.calls[0]
    assert args == ("BTC-USDT-SWAP",)
