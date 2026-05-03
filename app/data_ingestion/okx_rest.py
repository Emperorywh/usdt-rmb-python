"""OKX V5 public REST client.

Used as a fallback / supplementary source for funding rate and open interest
data, plus a stub for liquidation orders (only available with auth in OKX V5,
so we just return [] when unauthorised).
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import httpx

from app.data_ingestion.base import ExchangeRestClient
from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# 对这些异常做重试；业务层错误（如 OKX 返回 code != "0"）不重试
_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def _ms_to_dt(ms: str | int | None) -> Optional[datetime]:
    """
    毫秒时间戳 → UTC datetime
    ---------------------------------------------------------------
    参数：
        ms: 毫秒级时间戳字符串或整型，为空时返回 None
    返回：
        带 tzinfo=UTC 的 datetime，或 None
    """
    if not ms:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


class OKXRestClient(ExchangeRestClient):
    """
    OKX V5 公共 REST 客户端
    ---------------------------------------------------------------
    职责：
        - funding-rate / open-interest / instruments 元数据的拉取入口
        - 对瞬时网络抖动（代理握手失败、连接被重置等）做指数退避重试，
          避免上层轮询每次都打印完整 traceback
    代理：
        - 默认 trust_env=False，不读取系统 HTTP_PROXY/HTTPS_PROXY，
          因为系统代理经常 TLS 握手失败（见生产日志）。
        - 如需走代理，显式传入 proxy="http://user:pass@host:port"，
          或把 trust_env 设成 True 并保留系统代理变量。
    """

    name = "okx"

    def __init__(
        self,
        base_url: str = "https://www.okx.com",
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 0.8,
        trust_env: bool = False,
        proxy: Optional[str] = None,
    ):
        """
        构造函数
        ---------------------------------------------------------------
        参数：
            base_url:       OKX 公共 REST 基础 URL
            timeout:        单次 HTTP 请求超时（秒）
            max_retries:    最大重试次数（总请求数 = 1 + max_retries）
            retry_backoff:  指数退避基数，第 n 次重试等待 backoff * 2**n 秒
            trust_env:      是否让 httpx 读取系统代理/证书环境变量
            proxy:          显式代理 URL，优先级高于 trust_env
        """
        self.base_url = base_url.rstrip("/")
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = float(retry_backoff)

        client_kwargs: Dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout,
            "trust_env": trust_env,
        }
        if proxy:
            client_kwargs["proxy"] = proxy
        self._client = httpx.AsyncClient(**client_kwargs)

    async def close(self) -> None:
        """
        关闭底层 httpx 连接池
        ---------------------------------------------------------------
        """
        await self._client.aclose()

    async def _request_with_retry(
        self,
        op_name: str,
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        """
        带指数退避重试的请求执行器
        ---------------------------------------------------------------
        参数：
            op_name: 日志中标识本次操作的名字（如 "funding-rate"）
            fn:      实际发起请求并返回结果的无参协程
        返回：
            fn 的返回值
        异常：
            多次重试仍失败时，抛出最后一次的异常
        说明：
            只对 _RETRYABLE_EXC 中列出的网络类异常重试；业务错误
            （例如 OKX 返回非零 code）直接抛出，不做重试。
        """
        attempt = 0
        while True:
            try:
                return await fn()
            except _RETRYABLE_EXC as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff * (2 ** attempt) + random.random() * 0.2
                logger.warning(
                    "OKX REST %s transient error (attempt %d/%d): %s; retry in %.2fs",
                    op_name,
                    attempt + 1,
                    self.max_retries,
                    exc.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        拉取指定合约的最新资金费率
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码，如 'ETH-USDT-SWAP'
        返回：
            dict，包含 exchange / symbol / ts / funding_rate / next_funding_ts
        异常：
            网络错误（重试耗尽后）或 OKX 返回 code != "0" 时抛出
        """
        async def _do() -> Dict[str, Any]:
            resp = await self._client.get(
                "/api/v5/public/funding-rate", params={"instId": symbol}
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != "0" or not body.get("data"):
                raise RuntimeError(f"OKX funding-rate error: {body}")
            item = body["data"][0]
            return {
                "exchange": self.name,
                "symbol": symbol,
                "ts": _ms_to_dt(item.get("ts")) or datetime.now(timezone.utc),
                "funding_rate": float(item["fundingRate"]),
                "next_funding_ts": _ms_to_dt(item.get("nextFundingTime")),
            }

        return await self._request_with_retry(f"funding-rate[{symbol}]", _do)

    async def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        """
        拉取指定合约的当前持仓量
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码，如 'ETH-USDT-SWAP'
        返回：
            dict，包含 exchange / symbol / ts / oi / oi_ccy
        """
        inst_type = "SWAP" if symbol.endswith("-SWAP") else "FUTURES"

        async def _do() -> Dict[str, Any]:
            resp = await self._client.get(
                "/api/v5/public/open-interest",
                params={"instType": inst_type, "instId": symbol},
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != "0" or not body.get("data"):
                raise RuntimeError(f"OKX open-interest error: {body}")
            item = body["data"][0]
            return {
                "exchange": self.name,
                "symbol": symbol,
                "ts": _ms_to_dt(item.get("ts")) or datetime.now(timezone.utc),
                "oi": float(item.get("oi") or 0),
                "oi_ccy": float(item.get("oiCcy") or 0) or None,
            }

        return await self._request_with_retry(f"open-interest[{symbol}]", _do)

    async def fetch_instrument_meta(self, symbol: str) -> Dict[str, Any]:
        """
        拉取单个合约的元数据
        ---------------------------------------------------------------
        作用：
            从 OKX `/api/v5/public/instruments` 拉取合约规格，
            用来获取 ctVal（合约面值，单位为基础币种）等关键参数。
            因为 trades / books 推送中的 sz 是"张数"，需要乘以
            ctVal 才能换算成基础币种数量（如 ETH）。
        参数：
            symbol: 合约代码，如 'ETH-USDT-SWAP'
        返回：
            dict，至少包含：
              - ct_val:      每张合约面值（float），缺省 1.0
              - ct_val_ccy:  面值的计价币种（如 'ETH'）
              - inst_type:   合约类型（'SWAP' / 'FUTURES'）
              - raw:         OKX 原始返回（便于诊断）
        异常：
            网络错误（重试耗尽）或返回 code != "0" 时抛出 RuntimeError。
        """
        inst_type = "SWAP" if symbol.endswith("-SWAP") else "FUTURES"

        async def _do() -> Dict[str, Any]:
            resp = await self._client.get(
                "/api/v5/public/instruments",
                params={"instType": inst_type, "instId": symbol},
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != "0" or not body.get("data"):
                raise RuntimeError(f"OKX instruments error: {body}")
            item = body["data"][0]
            return {
                "ct_val": float(item.get("ctVal") or 1.0),
                "ct_val_ccy": item.get("ctValCcy") or "",
                "inst_type": inst_type,
                "raw": item,
            }

        return await self._request_with_retry(f"instruments[{symbol}]", _do)

    async def fetch_liquidation_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        拉取爆仓订单历史（best-effort）
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码
        返回：
            OKX 原始 data 列表；任意失败时返回空列表，保证服务不挂
        说明：
            OKX 的 /api/v5/public/liquidation-orders 有速率限制，
            这里不做重试，失败直接吞掉，避免轮询打印噪声。
        """
        inst_type = "SWAP" if symbol.endswith("-SWAP") else "FUTURES"
        try:
            resp = await self._client.get(
                "/api/v5/public/liquidation-orders",
                params={"instType": inst_type, "instId": symbol, "state": "filled"},
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get("code") != "0":
                return []
            return body.get("data") or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("liquidation-orders unavailable: %s", exc)
            return []
