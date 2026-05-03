"""
数据库初始化脚本
---------------------------------------------------------------
作用：
    1. 连接到 .env 中 DATABASE_URL 指定的 PostgreSQL 服务器
    2. 若目标数据库（例如 eth_analysis）不存在，则自动创建
    3. 在目标数据库内执行项目根目录下的 schema.sql，建好所有表和索引

使用方法（项目根目录，已激活 .venv）：
    python scripts/init_database.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import asyncpg
from dotenv import load_dotenv


# 项目根目录（scripts/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# schema.sql 的绝对路径
SCHEMA_FILE = PROJECT_ROOT / "schema.sql"


def _split_database_url(database_url: str) -> tuple[str, str]:
    """
    拆分 DATABASE_URL
    ---------------------------------------------------------------
    参数:
        database_url: 形如 postgresql://user:pass@host:5432/dbname 的连接串
    返回:
        (server_url, db_name):
            server_url: 指向内置 postgres 维护库的连接串（用于 CREATE DATABASE）
            db_name:    需要创建/使用的目标库名
    """
    parsed = urlparse(database_url)
    # path 形如 "/eth_analysis"，去掉最前面的 "/"
    db_name = parsed.path.lstrip("/") or "postgres"
    # 构造一个指向 postgres 系统库的连接串，用来执行 CREATE DATABASE
    server_parsed = parsed._replace(path="/postgres")
    server_url = urlunparse(server_parsed)
    return server_url, db_name


async def _ensure_database(server_url: str, db_name: str) -> bool:
    """
    确保目标数据库存在，不存在则创建
    ---------------------------------------------------------------
    参数:
        server_url: 指向 postgres 维护库的连接串
        db_name:    目标数据库名
    返回:
        True  表示本次调用实际创建了数据库
        False 表示数据库已存在，未做任何改动
    """
    conn = await asyncpg.connect(server_url)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            db_name,
        )
        if exists:
            print(f"[skip] 数据库 {db_name!r} 已存在")
            return False

        # CREATE DATABASE 不能放在事务里，也不支持参数化占位符，需要手动转义
        safe_name = db_name.replace('"', '""')
        await conn.execute(f'CREATE DATABASE "{safe_name}"')
        print(f"[ok]   已创建数据库 {db_name!r}")
        return True
    finally:
        await conn.close()


async def _apply_schema(database_url: str, schema_sql: str) -> None:
    """
    在目标数据库上执行 schema.sql
    ---------------------------------------------------------------
    参数:
        database_url: 指向目标业务库的完整连接串
        schema_sql:   schema.sql 的全部文本内容
    """
    conn = await asyncpg.connect(database_url)
    try:
        await conn.execute(schema_sql)
        print("[ok]   schema.sql 执行完毕，表与索引已就绪")
    finally:
        await conn.close()


async def main() -> int:
    """
    入口函数
    ---------------------------------------------------------------
    流程：加载 .env -> 拆解 URL -> 建库 -> 建表
    返回:
        进程退出码，0 表示成功，非 0 表示失败
    """
    # 加载项目根目录下的 .env
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[error] .env 中未找到 DATABASE_URL", file=sys.stderr)
        return 1

    if not SCHEMA_FILE.exists():
        print(f"[error] 找不到 schema 文件: {SCHEMA_FILE}", file=sys.stderr)
        return 1

    server_url, db_name = _split_database_url(database_url)
    print(f"[info] 目标服务器已解析, 目标库名: {db_name}")

    # 1. 建库
    await _ensure_database(server_url, db_name)

    # 2. 建表
    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
    await _apply_schema(database_url, schema_sql)

    print("[done] 数据库初始化完成，可以启动服务了")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except asyncpg.PostgresError as exc:
        print(f"[error] PostgreSQL 错误: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 初始化失败: {exc}", file=sys.stderr)
        sys.exit(3)
