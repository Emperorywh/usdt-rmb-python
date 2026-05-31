"""Base repository class shared by all repo modules."""
from __future__ import annotations

from typing import AsyncIterator

from app.data_storage.database import Database


class BaseRepo:
    """所有 Repo 的基类，持有 Database 实例并提供 acquire() 快捷方式。"""

    def __init__(self, db: Database):
        self._db = db

    @property
    def pool(self):
        """直接访问 asyncpg 连接池（向后兼容）。"""
        return self._db.pool

    def acquire(self):
        """代理 Database.acquire()，返回 async context manager。"""
        return self._db.acquire()

    @property
    def db(self) -> Database:
        """暴露底层 Database 实例（供需要 run_with_retry 的场景使用）。"""
        return self._db
