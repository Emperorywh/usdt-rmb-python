"""asyncpg connection pool wrapper."""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Optional, Tuple, Type, TypeVar

import asyncpg

from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# 视为"瞬时连接错误"的异常集合
# ----------------------------------------------------------------
# - ConnectionDoesNotExistError: 池子里的连接已被对端/OS 静默关闭
# - InterfaceError:              asyncpg 协议层错误（多数为连接层）
# - ConnectionResetError:        TCP RST
# - OSError:                     Windows WinError 121 / 64 等 socket 异常
# - asyncio.TimeoutError:        命令级别超时
# - InternalClientError:         asyncpg 状态机错乱
#                                （"cannot switch to state X; another
#                                operation in progress"），上一次查询超时
#                                后 pool release 时残留的协议状态，连接已
#                                不可用，需要丢弃 + 重建。
# 这些错误在写操作上都是可以重试的——前提是写入本身幂等
# （本项目的 INSERT 全部带 ON CONFLICT DO NOTHING/UPDATE）。
TRANSIENT_DB_ERRORS: Tuple[Type[BaseException], ...] = (
    asyncpg.exceptions.ConnectionDoesNotExistError,
    asyncpg.exceptions.InterfaceError,
    asyncpg.exceptions.ConnectionFailureError,
    asyncpg.exceptions.InternalClientError,
    ConnectionResetError,
    OSError,
    asyncio.TimeoutError,
)


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register JSON codecs so ``dict``/``list`` are auto-encoded for JSONB."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


class Database:
    """Thin wrapper around an asyncpg pool with connection-level JSON codecs."""

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
        max_inactive_connection_lifetime: float = 60.0,
        write_max_retries: int = 2,
        write_retry_backoff: float = 0.2,
    ):
        """
        构造数据库门面对象
        ---------------------------------------------------------------
        参数：
            dsn:                                PG 连接串
            min_size / max_size:                连接池大小
            max_inactive_connection_lifetime:   空闲连接最大存活秒数；
                                                Windows 下空闲 TCP 容易被
                                                防火墙/OS 静默断开，缩短此
                                                值能极大降低"僵尸连接"概率
            write_max_retries:                  幂等写入瞬时失败时的重试次数
            write_retry_backoff:                首次重试前的等待秒数
                                                （之后按 2^n 指数退避）
        """
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._max_inactive = float(max_inactive_connection_lifetime)
        self._write_max_retries = max(0, int(write_max_retries))
        self._write_retry_backoff = max(0.0, float(write_retry_backoff))
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        logger.info(
            "正在连接 PostgreSQL 连接池 大小=%d-%d 空闲回收=%.0fs",
            self._min_size,
            self._max_size,
            self._max_inactive,
        )
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            init=_init_connection,
            command_timeout=30,
            max_inactive_connection_lifetime=self._max_inactive,
        )
        logger.info("PostgreSQL 连接池就绪")

    async def disconnect(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL 连接池已关闭")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool is not initialised. Call connect() first.")
        return self._pool

    def acquire(self):
        return self.pool.acquire()

    async def run_with_retry(
        self,
        op: Callable[[], Awaitable[T]],
        *,
        op_name: str,
    ) -> T:
        """
        执行一个幂等写操作，遇到瞬时连接错误时按指数退避自动重试
        ---------------------------------------------------------------
        参数：
            op:      无参 callable，每次调用返回一个 **新的** coroutine。
                     注意不能传入已经 await 过的 awaitable，否则无法重试。
            op_name: 操作名称（仅用于日志）
        返回：
            最后一次成功执行 op() 的返回值
        异常：
            - 非瞬时错误：原样抛出，调用方按业务处理
            - 重试用尽仍失败：抛出最后一次的异常
        说明：
            - 触发重试的异常清单见模块级 ``TRANSIENT_DB_ERRORS``。
            - 所有传入的 op 必须是幂等写（ON CONFLICT DO NOTHING/UPDATE
              或 SELECT），否则可能产生重复数据。
        """
        attempt = 0
        while True:
            try:
                return await op()
            except TRANSIENT_DB_ERRORS as exc:
                if attempt >= self._write_max_retries:
                    raise
                delay = self._write_retry_backoff * (2 ** attempt)
                logger.warning(
                    "数据库瞬时错误 %s：%s：%s；%.2fs 后第 %d/%d 次重试",
                    op_name,
                    exc.__class__.__name__,
                    str(exc).strip() or "(无消息)",
                    delay,
                    attempt + 1,
                    self._write_max_retries,
                )
                await asyncio.sleep(delay)
                attempt += 1
