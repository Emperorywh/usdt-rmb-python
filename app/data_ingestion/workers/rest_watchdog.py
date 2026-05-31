"""REST 兜底看门狗 Worker。

当 WS 的 funding-rate / open-interest 频道长时间没有推送时，
发起 REST 兜底拉取并落库。正常情况下完全不发请求。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.config import Settings
from app.data_ingestion.okx_rest import CircuitOpenError
from app.data_storage.repos.derivatives_repo import DerivativesRepo
from app.data_ingestion.base import ExchangeRestClient
from app.logging_config import get_logger

logger = get_logger(__name__)


class RestWatchdog:
    """REST 兜底看门狗。

    职责：
    - 监控 WS funding_rate / open_interest 频道活跃度
    - 频道陈旧时发起 REST 兜底拉取
    - 写入 DB（幂等，与 WS 路径共享唯一约束）
    """

    def __init__(
        self,
        deriv_repo: DerivativesRepo,
        rest_client: ExchangeRestClient,
        settings: Settings,
        stopping: Optional[asyncio.Event] = None,
        # 共享的 WS 健康状态引用（由 Runner 持有并维护）
        ws_event_at: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self._deriv_repo = deriv_repo
        self._rest_client = rest_client
        self._settings = settings
        self._stopping = stopping or asyncio.Event()
        self._ws_event_at = ws_event_at
        self._task: Optional[asyncio.Task[Any]] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run(), name="worker-RestWatchdog")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """REST 兜底看门狗主循环。"""
        # 启动宽限：避免冷启动瞬间 WS 还没握手就触发兜底
        try:
            await asyncio.wait_for(
                self._stopping.wait(),
                timeout=float(self._settings.ingestion.watchdog_grace_seconds),
            )
        except asyncio.TimeoutError:
            pass

        try:
            while not self._stopping.is_set():
                tasks: List[Awaitable[Any]] = []
                for symbol in self._settings.ingestion.symbols:
                    if self._is_ws_stale(
                        symbol, "funding_rate",
                        float(self._settings.ingestion.ws_stale_funding_seconds),
                    ):
                        tasks.append(
                            self._fallback_one(
                                symbol,
                                "funding_rate",
                                self._rest_client.fetch_funding_rate,
                                self._persist_funding_rate,
                            )
                        )
                    if self._is_ws_stale(
                        symbol, "open_interest",
                        float(self._settings.ingestion.ws_stale_oi_seconds),
                    ):
                        tasks.append(
                            self._fallback_one(
                                symbol,
                                "open_interest",
                                self._rest_client.fetch_open_interest,
                                self._persist_open_interest,
                            )
                        )
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(),
                        timeout=float(self._settings.ingestion.watchdog_tick_seconds),
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _is_ws_stale(self, symbol: str, kind: str, threshold_seconds: float) -> bool:
        """判断 (symbol, kind) 通道是否陈旧到需要 REST 兜底。"""
        if self._ws_event_at is None:
            return True
        last = self._ws_event_at.get((symbol, kind))
        if last is None:
            return True
        return (time.monotonic() - last) >= threshold_seconds

    async def _fallback_one(
        self,
        symbol: str,
        kind: str,
        fetcher: Callable[[str], Awaitable[Dict[str, Any]]],
        persister: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """执行一次 REST 兜底拉取并落库。"""
        try:
            payload = await fetcher(symbol)
            await persister(payload)
            logger.info("REST 兜底成功：%s/%s（WS 已陈旧）", symbol, kind)
        except CircuitOpenError:
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "REST 兜底失败 %s/%s：%s", symbol, kind, exc.__class__.__name__,
            )

    async def _persist_funding_rate(self, payload: Dict[str, Any]) -> None:
        """把 REST 返回的 funding-rate dict 写入 DB。"""
        await self._deriv_repo.insert_funding_rate(
            exchange=payload["exchange"],
            symbol=payload["symbol"],
            ts=payload["ts"],
            funding_rate=payload["funding_rate"],
            next_funding_ts=payload.get("next_funding_ts"),
        )

    async def _persist_open_interest(self, payload: Dict[str, Any]) -> None:
        """把 REST 返回的 open-interest dict 写入 DB。"""
        await self._deriv_repo.insert_open_interest(
            exchange=payload["exchange"],
            symbol=payload["symbol"],
            ts=payload["ts"],
            oi=payload["oi"],
            oi_ccy=payload.get("oi_ccy"),
        )
