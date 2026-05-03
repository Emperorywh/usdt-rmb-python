"""因子聚合器：从 PG 拉取最新数据并计算各类因子。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.factor_engine.capital_flow import compute_capital_flow
from app.factor_engine.derivatives import compute_derivatives_factors
from app.factor_engine.market_structure import compute_market_structure
from app.factor_engine.orderbook import compute_orderbook_factors
from app.logging_config import get_logger

logger = get_logger(__name__)


class FactorAggregator:
    """
    因子聚合器
    -------------------------------------------------------------------
    将四类因子（资金流 / 订单簿 / 衍生品 / 市场结构）聚合为单一 dict。

    备注：
        - 链上因子（onchain）和参与者画像（participants）依赖真实链上数据
          源；mock 数据被下线后这两块从聚合结果中暂时移除，等接入
          Glassnode/Nansen/Etherscan 等真实数据源后再恢复。
    """

    def __init__(self, repos: Repositories, settings: Settings):
        """
        构造因子聚合器
        ---------------------------------------------------------------
        参数：
            repos:    数据仓储集合，提供成交、盘口、资金费率、OI 读接口
            settings: 全局配置，提供窗口大小、流动性墙阈值等
        """
        self.repos = repos
        self.settings = settings

    async def compute(self, symbol: str) -> Dict[str, Any]:
        """
        计算并返回当前因子快照
        ---------------------------------------------------------------
        参数：
            symbol: 合约代码，例如 'ETH-USDT-SWAP'
        返回：
            一个统一结构的因子 dict，键包含：
                symbol / window_seconds / computed_at / capital_flow /
                orderbook / derivatives / market_structure
        """
        window = self.settings.factor_window_seconds
        since = datetime.now(timezone.utc) - timedelta(seconds=window)

        trades = await self.repos.fetch_recent_trades(symbol, since=since, limit=20000)
        orderbook = await self.repos.fetch_latest_orderbook(symbol)
        funding = await self.repos.fetch_latest_funding(symbol)
        oi_history = await self.repos.fetch_recent_oi(symbol, since=since, limit=2000)

        capital = compute_capital_flow(trades)
        ob = compute_orderbook_factors(
            orderbook, wall_multiplier=self.settings.liquidity_wall_multiplier
        )
        deriv = compute_derivatives_factors(funding, oi_history)
        struct = compute_market_structure(trades)

        return {
            "symbol": symbol,
            "window_seconds": window,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "capital_flow": capital,
            "orderbook": ob,
            "derivatives": deriv,
            "market_structure": struct,
        }
