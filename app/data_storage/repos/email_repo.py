"""Email repository: notification_emails table CRUD."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.logging_config import get_logger

from .base import BaseRepo
from .helpers import parse_delete_count

logger = get_logger(__name__)


class EmailRepo(BaseRepo):
    """notification_emails 表仓储：邮件通知收件人 CRUD。"""

    # ------------------------------------------------------------------
    # notification_emails（邮件通知收件人）
    # ------------------------------------------------------------------
    # 设计要点：
    #   - 仅作管理员维护表使用，量级 < 100 行，所有读路径都走全表扫描
    #     即可，无需缓存。
    #   - email 列 UNIQUE：插入冲突时由调用方捕获 UniqueViolationError
    #     转成 409 给 API 调用方，避免静默丢错。
    #   - 写操作不走 db.run_with_retry：手工管理动作量极少，幂等性也由
    #     UNIQUE 保证；与高频行情写入路径解耦更清晰。

    async def insert_notification_email(
        self,
        email: str,
        name: Optional[str] = None,
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        新增一条邮件通知收件人
        --------------------------------------------------------------
        参数：
            email   ：收件邮箱（UNIQUE）
            name    ：备注名（可空），如 "风控小组"
            enabled ：是否启用，默认 True
        返回：
            新建行的完整 dict（含 id / created_at / updated_at）
        异常：
            asyncpg.exceptions.UniqueViolationError - 邮箱已存在；
            由路由层捕获后返回 409 Conflict。
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO notification_emails (email, name, enabled)
                VALUES ($1, $2, $3)
                RETURNING id, email, name, enabled, created_at, updated_at
                """,
                email,
                name,
                bool(enabled),
            )
        return dict(row)

    async def list_notification_emails(
        self, only_enabled: bool = False
    ) -> List[Dict[str, Any]]:
        """
        列出所有邮件通知收件人
        --------------------------------------------------------------
        参数：
            only_enabled : True 时只返回 enabled=TRUE 的行（用于发件路径）
        返回：
            按 created_at 升序的 dict 列表
        """
        sql = """
            SELECT id, email, name, enabled, created_at, updated_at
            FROM notification_emails
            {where}
            ORDER BY created_at ASC
        """.format(where="WHERE enabled = TRUE" if only_enabled else "")
        async with self._db.acquire() as conn:
            rows = await conn.fetch(sql)
        return [dict(r) for r in rows]

    async def fetch_notification_email_by_id(
        self, email_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        按主键 id 读取一条邮件通知收件人
        --------------------------------------------------------------
        参数：
            email_id : notification_emails.id
        返回：
            dict 或 None
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, name, enabled, created_at, updated_at
                FROM notification_emails
                WHERE id = $1
                """,
                email_id,
            )
        return dict(row) if row else None

    async def update_notification_email(
        self,
        email_id: int,
        *,
        email: Optional[str] = None,
        name: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        增量更新一条邮件通知收件人
        --------------------------------------------------------------
        参数：
            email_id ：主键
            email    ：新的收件邮箱（可空表示不改）
            name     ：新的备注名（可空表示不改；显式置空请单独发空字符串）
            enabled  ：是否启用（可空表示不改）
        返回：
            更新后的完整 dict；找不到时返回 None
        说明：
            使用 COALESCE($n, 当前列) 让单条 UPDATE 一次跑完，不传的字段不变。
            updated_at 总是被刷新到 NOW()。
        """
        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE notification_emails SET
                    email      = COALESCE($2, email),
                    name       = COALESCE($3, name),
                    enabled    = COALESCE($4, enabled),
                    updated_at = NOW()
                WHERE id = $1
                RETURNING id, email, name, enabled, created_at, updated_at
                """,
                email_id,
                email,
                name,
                enabled,
            )
        return dict(row) if row else None

    async def delete_notification_email(self, email_id: int) -> bool:
        """
        按 id 删除一条邮件通知收件人
        --------------------------------------------------------------
        参数：
            email_id ：主键
        返回：
            True - 已删除一行；False - 该 id 不存在
        """
        async with self._db.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM notification_emails WHERE id = $1",
                email_id,
            )
        return parse_delete_count(result) > 0
