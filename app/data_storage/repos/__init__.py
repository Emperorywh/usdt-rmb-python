"""按领域拆分的独立 Repository 模块。

每个 Repo 类接受一个 Database 实例，拥有各自领域的方法。
``Repositories`` 聚合所有子 Repo 并委托全部调用，保持原有
``container.repos.xxx(...)`` API 完全不变。
"""
from __future__ import annotations

from app.data_storage.database import Database

from app.data_storage.repos.base import BaseRepo
from app.data_storage.repos.trade_repo import TradeRepo
from app.data_storage.repos.kline_repo import KlineRepo
from app.data_storage.repos.orderbook_repo import OrderbookRepo
from app.data_storage.repos.derivatives_repo import DerivativesRepo
from app.data_storage.repos.signal_repo import SignalRepo
from app.data_storage.repos.email_repo import EmailRepo

__all__ = [
    "BaseRepo",
    "TradeRepo",
    "KlineRepo",
    "OrderbookRepo",
    "DerivativesRepo",
    "SignalRepo",
    "EmailRepo",
    "Repositories",
]


class Repositories:
    """
    聚合所有表级仓储方法的 Facade 对象。

    所有方法调用都委托给对应的子 Repo 实例，对外 API 完全不变：
        ``container.repos.insert_trades(...)``  等价于
        ``container.repos._trade.insert_trades(...)``。
    """

    def __init__(self, db: Database):
        self.db = db
        self._trade = TradeRepo(db)
        self._kline = KlineRepo(db)
        self._orderbook = OrderbookRepo(db)
        self._derivatives = DerivativesRepo(db)
        self._signal = SignalRepo(db)
        self._email = EmailRepo(db)

    # ------------------------------------------------------------------
    # trades
    # ------------------------------------------------------------------
    async def insert_trades(self, *args, **kwargs):
        return await self._trade.insert_trades(*args, **kwargs)

    async def fetch_recent_trades(self, *args, **kwargs):
        return await self._trade.fetch_recent_trades(*args, **kwargs)

    async def aggregate_trades_in_window(self, *args, **kwargs):
        return await self._trade.aggregate_trades_in_window(*args, **kwargs)

    async def delete_trades_older_than(self, *args, **kwargs):
        return await self._trade.delete_trades_older_than(*args, **kwargs)

    # ------------------------------------------------------------------
    # klines
    # ------------------------------------------------------------------
    async def upsert_kline(self, *args, **kwargs):
        return await self._kline.upsert_kline(*args, **kwargs)

    async def fetch_recent_klines(self, *args, **kwargs):
        return await self._kline.fetch_recent_klines(*args, **kwargs)

    async def fetch_latest_kline(self, *args, **kwargs):
        return await self._kline.fetch_latest_kline(*args, **kwargs)

    # ------------------------------------------------------------------
    # orderbook
    # ------------------------------------------------------------------
    async def insert_orderbook(self, *args, **kwargs):
        return await self._orderbook.insert_orderbook(*args, **kwargs)

    async def fetch_latest_orderbook(self, *args, **kwargs):
        return await self._orderbook.fetch_latest_orderbook(*args, **kwargs)

    async def delete_orderbook_older_than(self, *args, **kwargs):
        return await self._orderbook.delete_orderbook_older_than(*args, **kwargs)

    async def insert_orderbook_metric(self, *args, **kwargs):
        return await self._orderbook.insert_orderbook_metric(*args, **kwargs)

    async def fetch_orderbook_metrics_since(self, *args, **kwargs):
        return await self._orderbook.fetch_orderbook_metrics_since(*args, **kwargs)

    async def delete_orderbook_metrics_older_than(self, *args, **kwargs):
        return await self._orderbook.delete_orderbook_metrics_older_than(*args, **kwargs)

    # ------------------------------------------------------------------
    # derivatives (funding_rates / open_interest / liquidations)
    # ------------------------------------------------------------------
    async def insert_funding_rate(self, *args, **kwargs):
        return await self._derivatives.insert_funding_rate(*args, **kwargs)

    async def fetch_latest_funding(self, *args, **kwargs):
        return await self._derivatives.fetch_latest_funding(*args, **kwargs)

    async def fetch_funding_rates_since(self, *args, **kwargs):
        return await self._derivatives.fetch_funding_rates_since(*args, **kwargs)

    async def delete_funding_older_than(self, *args, **kwargs):
        return await self._derivatives.delete_funding_older_than(*args, **kwargs)

    async def insert_open_interest(self, *args, **kwargs):
        return await self._derivatives.insert_open_interest(*args, **kwargs)

    async def fetch_recent_oi(self, *args, **kwargs):
        return await self._derivatives.fetch_recent_oi(*args, **kwargs)

    async def delete_oi_older_than(self, *args, **kwargs):
        return await self._derivatives.delete_oi_older_than(*args, **kwargs)

    async def insert_liquidations(self, *args, **kwargs):
        return await self._derivatives.insert_liquidations(*args, **kwargs)

    async def fetch_liquidations_since(self, *args, **kwargs):
        return await self._derivatives.fetch_liquidations_since(*args, **kwargs)

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------
    async def insert_signal(self, *args, **kwargs):
        return await self._signal.insert_signal(*args, **kwargs)

    async def fetch_latest_signal(self, *args, **kwargs):
        return await self._signal.fetch_latest_signal(*args, **kwargs)

    async def fetch_latest_signal_full(self, *args, **kwargs):
        return await self._signal.fetch_latest_signal_full(*args, **kwargs)

    async def fetch_recent_signals_full(self, *args, **kwargs):
        return await self._signal.fetch_recent_signals_full(*args, **kwargs)

    async def fetch_signal_full_by_id(self, *args, **kwargs):
        return await self._signal.fetch_signal_full_by_id(*args, **kwargs)

    async def fetch_latest_signal_judgment(self, *args, **kwargs):
        return await self._signal.fetch_latest_signal_judgment(*args, **kwargs)

    async def delete_signals_older_than(self, *args, **kwargs):
        return await self._signal.delete_signals_older_than(*args, **kwargs)

    # ------------------------------------------------------------------
    # notification_emails
    # ------------------------------------------------------------------
    async def insert_notification_email(self, *args, **kwargs):
        return await self._email.insert_notification_email(*args, **kwargs)

    async def list_notification_emails(self, *args, **kwargs):
        return await self._email.list_notification_emails(*args, **kwargs)

    async def fetch_notification_email_by_id(self, *args, **kwargs):
        return await self._email.fetch_notification_email_by_id(*args, **kwargs)

    async def update_notification_email(self, *args, **kwargs):
        return await self._email.update_notification_email(*args, **kwargs)

    async def delete_notification_email(self, *args, **kwargs):
        return await self._email.delete_notification_email(*args, **kwargs)
