"""OKX V5 public REST client.

Used as a fallback / supplementary source for funding rate and open interest
data (the primary path is the WebSocket; see ``okx_ws.py``), plus a stub for
liquidation orders (only available with auth in OKX V5, so we just return []
when unauthorised).

The client is wrapped with a per-endpoint **circuit breaker** + summarised
logging so that a flaky egress to ``www.okx.com`` (a common situation for
Chinese mainland deployments) cannot spam the console with one WARNING per
retry per minute. State machine:

* ``closed``     -> normal, every request goes through.
* ``open``       -> last N requests failed; requests fail fast and we stay
                   off the wire until ``cooldown_until`` elapses, doubling
                   the cooldown each time we re-open.
* On the first success the breaker fully resets to ``closed``.

Failure logs are summarised: the first failure inside a window prints a
WARNING; subsequent ones increment a counter and we print one INFO summary
every ``_SUMMARY_INTERVAL`` seconds (``in last Xs: N failures``).
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
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

# 熔断打开后第一次冷却时长（秒）；后续每次 trip 都会翻倍直到 _BREAKER_MAX_COOLDOWN
_BREAKER_BASE_COOLDOWN = 60.0
_BREAKER_MAX_COOLDOWN = 15 * 60.0
# 连续失败多少次后跳闸
_BREAKER_FAIL_THRESHOLD = 3
# 失败摘要日志的最小输出间隔（秒）
_SUMMARY_INTERVAL = 5 * 60.0


class CircuitOpenError(RuntimeError):
    """
    熔断已打开
    ---------------------------------------------------------------
    说明：
        - 熔断器处于 ``open`` 状态时，调用方再发起请求会直接抛出该异常，
          调用方应当把它当作"通道暂时不可用"来处理（例如降级、跳过、
          沿用上次缓存值），而不是无脑 retry。
        - 不属于 _RETRYABLE_EXC，重试逻辑不会再追加退避。
    """


@dataclass
class _BreakerState:
    """
    单 endpoint 熔断状态机
    ---------------------------------------------------------------
    字段：
        consecutive_failures: 连续失败次数（成功一次清零）
        opened_at:            最近一次跳闸时刻（monotonic 秒），None 表示未跳闸
        cooldown_until:       熔断状态结束的时刻（monotonic 秒）
        next_cooldown:        下一次跳闸将采用的冷却时长，跳闸后翻倍直到上限
        total_failures:       本窗口内累计失败次数（用于摘要日志）
        last_summary_at:      上一次输出摘要日志的时刻
        last_error_class:     最后一次失败的异常类名（写入摘要日志）
    """

    consecutive_failures: int = 0
    opened_at: Optional[float] = None
    cooldown_until: float = 0.0
    next_cooldown: float = _BREAKER_BASE_COOLDOWN
    total_failures: int = 0
    last_summary_at: float = 0.0
    last_error_class: str = ""


@dataclass
class EndpointHealth:
    """
    供外部观测的 endpoint 健康度快照
    ---------------------------------------------------------------
    字段：
        state:                'closed' / 'open'
        consecutive_failures: 连续失败次数
        cooldown_remaining:   熔断剩余秒数（state=='closed' 时为 0）
        last_error:           最近一次错误的异常类名
        last_success_at:      最近一次成功的 UTC 时间，None 表示从未成功
    """

    state: str = "closed"
    consecutive_failures: int = 0
    cooldown_remaining: float = 0.0
    last_error: str = ""
    last_success_at: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


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
        # 每个 op_name（如 "funding-rate[ETH-USDT-SWAP]"）独立一份熔断状态
        # —— 不同 endpoint / symbol 的网络命运是独立的，全局熔断会误伤。
        self._breakers: Dict[str, _BreakerState] = {}
        # 同上，为外部观测保留的健康度快照
        self._health: Dict[str, EndpointHealth] = {}

    async def close(self) -> None:
        """
        关闭底层 httpx 连接池
        ---------------------------------------------------------------
        """
        await self._client.aclose()

    # ------------------------------------------------------------------
    # circuit breaker helpers
    # ------------------------------------------------------------------
    def _get_breaker(self, op_name: str) -> _BreakerState:
        """
        懒初始化指定 op 的熔断状态机
        ---------------------------------------------------------------
        参数：
            op_name: endpoint 标识符，例如 "funding-rate[ETH-USDT-SWAP]"
        返回：
            该 op 对应的 _BreakerState（首次调用会现场创建）
        """
        st = self._breakers.get(op_name)
        if st is None:
            st = _BreakerState()
            self._breakers[op_name] = st
        return st

    def _get_health(self, op_name: str) -> EndpointHealth:
        """
        懒初始化指定 op 的健康度快照
        ---------------------------------------------------------------
        参数：
            op_name: endpoint 标识符
        返回：
            EndpointHealth 实例，供 /healthz 等外部观测读取
        """
        h = self._health.get(op_name)
        if h is None:
            h = EndpointHealth()
            self._health[op_name] = h
        return h

    def health_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        导出所有 endpoint 的健康度快照
        ---------------------------------------------------------------
        返回：
            {op_name: {state, consecutive_failures, cooldown_remaining,
                       last_error, last_success_at, success_count,
                       failure_count}}
        说明：
            被 /healthz 路由用来回报"REST 通道目前是否可用"。
            cooldown_remaining 是相对于调用时刻动态计算的剩余秒数。
        """
        now = time.monotonic()
        out: Dict[str, Dict[str, Any]] = {}
        for op_name, health in self._health.items():
            breaker = self._breakers.get(op_name)
            cooldown = 0.0
            if breaker and breaker.opened_at is not None:
                cooldown = max(0.0, breaker.cooldown_until - now)
            out[op_name] = {
                "state": health.state,
                "consecutive_failures": health.consecutive_failures,
                "cooldown_remaining": round(cooldown, 1),
                "last_error": health.last_error,
                "last_success_at": (
                    health.last_success_at.isoformat()
                    if health.last_success_at
                    else None
                ),
                "success_count": health.success_count,
                "failure_count": health.failure_count,
            }
        return out

    def _on_success(self, op_name: str) -> None:
        """
        请求成功时复位熔断器与健康度
        ---------------------------------------------------------------
        参数：
            op_name: endpoint 标识符
        说明：
            一次成功就完全 reset，不做"半开"过渡 —— 公开行情这种幂等
            读接口不需要 half-open 灰度，简单可读优先。
        """
        breaker = self._get_breaker(op_name)
        was_open = breaker.opened_at is not None
        if was_open or breaker.consecutive_failures > 0:
            logger.info(
                "OKX REST %s 已恢复（之前累计 %d 次失败）",
                op_name,
                breaker.total_failures,
            )
        breaker.consecutive_failures = 0
        breaker.opened_at = None
        breaker.cooldown_until = 0.0
        breaker.next_cooldown = _BREAKER_BASE_COOLDOWN
        breaker.total_failures = 0
        breaker.last_summary_at = 0.0
        breaker.last_error_class = ""
        health = self._get_health(op_name)
        health.state = "closed"
        health.consecutive_failures = 0
        health.cooldown_remaining = 0.0
        health.last_error = ""
        health.last_success_at = datetime.now(timezone.utc)
        health.success_count += 1

    def _on_failure(self, op_name: str, exc: BaseException) -> None:
        """
        请求失败时推进熔断状态 + 失败摘要日志
        ---------------------------------------------------------------
        参数：
            op_name: endpoint 标识符
            exc:     触发失败的异常对象
        说明：
            - 累计失败 ≥ _BREAKER_FAIL_THRESHOLD 时跳闸，cooldown 翻倍递增。
            - 第一条失败打 WARNING；其余每 _SUMMARY_INTERVAL 秒打一条 INFO
              摘要，避免刷屏。
        """
        now = time.monotonic()
        breaker = self._get_breaker(op_name)
        breaker.consecutive_failures += 1
        breaker.total_failures += 1
        breaker.last_error_class = exc.__class__.__name__

        if breaker.consecutive_failures == 1:
            logger.warning(
                "OKX REST %s 失败：%s：%s（进入失败摘要模式，后续按 %ds 汇总）",
                op_name,
                exc.__class__.__name__,
                exc,
                int(_SUMMARY_INTERVAL),
            )
            breaker.last_summary_at = now
        elif now - breaker.last_summary_at >= _SUMMARY_INTERVAL:
            logger.info(
                "OKX REST %s 持续失败：最近 %ds 内累计 %d 次（最后一次：%s）",
                op_name,
                int(now - breaker.last_summary_at),
                breaker.total_failures,
                breaker.last_error_class,
            )
            breaker.last_summary_at = now
            breaker.total_failures = 0  # 摘要窗口重置

        if breaker.consecutive_failures >= _BREAKER_FAIL_THRESHOLD:
            breaker.opened_at = now
            breaker.cooldown_until = now + breaker.next_cooldown
            cooldown_used = breaker.next_cooldown
            breaker.next_cooldown = min(
                breaker.next_cooldown * 2.0, _BREAKER_MAX_COOLDOWN
            )
            logger.warning(
                "OKX REST %s 熔断打开：连续失败 %d 次，冷却 %.0fs",
                op_name,
                breaker.consecutive_failures,
                cooldown_used,
            )

        health = self._get_health(op_name)
        health.consecutive_failures = breaker.consecutive_failures
        health.last_error = breaker.last_error_class
        health.failure_count += 1
        if breaker.opened_at is not None:
            health.state = "open"
            health.cooldown_remaining = max(0.0, breaker.cooldown_until - now)

    def _check_breaker(self, op_name: str) -> None:
        """
        请求开始前检查熔断状态，必要时直接抛 CircuitOpenError
        ---------------------------------------------------------------
        参数：
            op_name: endpoint 标识符
        异常：
            CircuitOpenError - 当前 endpoint 处于 open 状态且未到冷却时刻
        """
        breaker = self._breakers.get(op_name)
        if breaker is None or breaker.opened_at is None:
            return
        now = time.monotonic()
        if now < breaker.cooldown_until:
            remaining = breaker.cooldown_until - now
            raise CircuitOpenError(
                f"{op_name} 熔断中，剩余 {remaining:.0f}s（last_error="
                f"{breaker.last_error_class}）"
            )
        # 冷却结束：保留累计失败计数，但允许"试探"一次
        # （opened_at 暂存为 None，让 _check_breaker 放行；如果再失败，
        # _on_failure 会用翻倍后的 next_cooldown 重新跳闸）
        breaker.opened_at = None

    async def _request_with_retry(
        self,
        op_name: str,
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        """
        带熔断 + 指数退避重试的请求执行器
        ---------------------------------------------------------------
        参数：
            op_name: 日志中标识本次操作的名字（含 symbol，便于熔断隔离）
            fn:      实际发起请求并返回结果的无参协程
        返回：
            fn 的返回值
        异常：
            - CircuitOpenError: endpoint 当前在熔断窗口内（不重试）
            - 多次重试仍失败时抛出最后一次的网络异常
            - 业务错误（OKX code != "0"）按原异常直接抛出
        说明：
            只对 _RETRYABLE_EXC 列出的网络异常做指数退避；
            每次失败都会推进熔断状态，连续失败到阈值后直接 fail-fast。
        """
        self._check_breaker(op_name)
        attempt = 0
        while True:
            try:
                result = await fn()
            except _RETRYABLE_EXC as exc:
                if attempt >= self.max_retries:
                    self._on_failure(op_name, exc)
                    raise
                # 单次循环内的轻量重试只 DEBUG 一下，避免每次重试都打 WARNING
                delay = self.retry_backoff * (2 ** attempt) + random.random() * 0.2
                logger.debug(
                    "OKX REST %s 瞬时错误（第 %d/%d 次重试）：%s；%.2fs 后重试",
                    op_name,
                    attempt + 1,
                    self.max_retries,
                    exc.__class__.__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            except Exception as exc:
                # 业务错误（如 OKX code != 0）也算失败，推进熔断
                self._on_failure(op_name, exc)
                raise
            else:
                self._on_success(op_name)
                return result

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
            logger.debug("爆仓订单接口不可用：%s", exc)
            return []
