"""scripts/replay_signals.py —— 历史信号重放骨架（P2 上线灰度对比工具）。

用途
====
- 在切换 RuleEngine 权重源（factor_weights 表 vs 基线硬编码权重）之前，
  对历史 ``signals`` 行做"用新权重重新打分 → 对照 lifecycle 的真实 PnL"，
  输出 Sharpe / 胜率 / 平均 RR / 命中率分布等指标。
- 本脚本是骨架：核心结构 + 关键函数 + CLI 入口完整给出，
  不要求一次跑通所有指标，预留 TODO 让后续按需扩展。

非目标
======
- 不做"基于历史 K 线模拟交易撮合"——这块属于回测系统范畴，远超 P2 范围。
  我们只重算"打分 → bias / confidence"，PnL 直接复用 lifecycle 表里
  跟踪任务记录的真实结果。这样新旧权重的对比不会受撮合细节干扰。

使用
====
    python scripts/replay_signals.py \
        --symbol ETH-USDT-SWAP \
        --since 2026-04-01T00:00:00Z \
        --weights-source baseline | db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# 允许从 scripts/ 直接 python 起脚本
sys.path.insert(0, ".")

from app.config import get_settings  # noqa: E402
from app.data_storage.database import Database  # noqa: E402
from app.data_storage.repositories import Repositories  # noqa: E402
from app.signal_engine.rules import RuleEngine  # noqa: E402


@dataclass
class ReplayMetrics:
    """
    一次回放的统计指标
    """

    total: int = 0
    wins: int = 0
    losses: int = 0
    expired: int = 0
    pnl_list: List[float] = field(default_factory=list)
    rr_list: List[float] = field(default_factory=list)
    bias_distribution: Dict[str, int] = field(default_factory=lambda: {
        "long": 0, "short": 0, "neutral": 0
    })
    bias_disagreement: int = 0  # 新旧打分给出方向不同的次数

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def avg_pnl(self) -> float:
        return sum(self.pnl_list) / len(self.pnl_list) if self.pnl_list else 0.0

    @property
    def avg_rr(self) -> float:
        return sum(self.rr_list) / len(self.rr_list) if self.rr_list else 0.0

    @property
    def sharpe_proxy(self) -> float:
        """
        给信号级 PnL 列表的 Sharpe-like 代理指标
        --------------------------------------------------------------
        说明：
            标准 Sharpe 需要"每日收益率序列 + 年化"，这里用"每条信号的
            PnL%"近似：mean / std；Sharpe ∝ 不变性，足够横向对比新旧权重。
        """
        if len(self.pnl_list) < 5:
            return 0.0
        mean = self.avg_pnl
        std = math.sqrt(
            sum((x - mean) ** 2 for x in self.pnl_list) / max(1, len(self.pnl_list) - 1)
        )
        if std < 1e-9:
            return 0.0
        return mean / std

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "wins": self.wins,
            "losses": self.losses,
            "expired": self.expired,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "avg_rr": self.avg_rr,
            "sharpe_proxy": self.sharpe_proxy,
            "bias_distribution": self.bias_distribution,
            "bias_disagreement": self.bias_disagreement,
        }


# ----------------------------------------------------------------------
# 核心：用规则引擎对 historical signals 重新打分，对照真实 PnL
# ----------------------------------------------------------------------
async def replay(
    symbol: str,
    since: datetime,
    weights_source: str,
    limit: int = 5000,
) -> ReplayMetrics:
    """
    重放一个 symbol 在 since 之后的全部已结算信号
    --------------------------------------------------------------
    参数：
        symbol         : 合约
        since          : 起始 UTC 时间
        weights_source : 'baseline' 用 P1 硬编码权重；'db' 用 factor_weights 表
        limit          : 防御性上限
    返回：
        ReplayMetrics
    流程：
        1) 拉取 [since, now] 内 signals 全表 + lifecycle 真实退出结果；
        2) 对每条 signal：
            - 取出快照中的 factors（signals.factors.factors，老格式直接 .factors）；
            - 调用 RuleEngine.evaluate(snapshot_factors) 算出新 bias / score；
            - 与 lifecycle.pnl_pct / status 对照，更新 ReplayMetrics。
        3) 汇总输出。
    设计取舍：
        - 不重算 LLM 路径：LLM 调用受 prompt / 思考模式 / token 成本约束，
          离线"重放 LLM"成本极高且不收敛（每次都不一样）。这里只评估
          规则引擎权重切换对最终 bias 的影响幅度。
    """
    settings = get_settings()
    db = Database(
        dsn=settings.database_url,
        min_size=1,
        max_size=2,
        max_inactive_connection_lifetime=settings.db_max_inactive_connection_lifetime,
    )
    await db.connect()
    repos = Repositories(db)

    # 控制 RuleEngine 加载哪一份权重：
    # - 'baseline'：暂时把 enable_factor_weights_table 关闭，让 RuleEngine 走基线；
    # - 'db'      ：保持开启 + ic_calibrator_shadow_mode=False，让它去 DB 拉。
    if weights_source == "baseline":
        object.__setattr__(settings, "enable_factor_weights_table", False)
        object.__setattr__(settings, "ic_calibrator_shadow_mode", True)
    elif weights_source == "db":
        object.__setattr__(settings, "enable_factor_weights_table", True)
        object.__setattr__(settings, "ic_calibrator_shadow_mode", False)
    else:
        raise ValueError(f"unknown weights_source: {weights_source}")

    rule_engine = RuleEngine(settings=settings, repos=repos)

    # 拉历史信号 + 对应 lifecycle（仅 join 已结算的）
    rows = await _fetch_settled_signals(repos, symbol, since, limit)

    metrics = ReplayMetrics()
    for row in rows:
        snapshot_factors = _extract_snapshot_factors(row.get("factors"))
        if not isinstance(snapshot_factors, dict):
            continue
        try:
            new_signal, _new_score, _contrib = await rule_engine.evaluate(
                snapshot_factors
            )
        except Exception:
            continue

        # 旧 bias 直接来自历史 signals.bias；新 bias 来自重打分
        old_bias = row.get("bias") or "neutral"
        new_bias = new_signal.bias

        metrics.total += 1
        metrics.bias_distribution[new_bias] = (
            metrics.bias_distribution.get(new_bias, 0) + 1
        )
        if old_bias != new_bias:
            metrics.bias_disagreement += 1

        # 真实 PnL 来自 lifecycle —— 注意：当新 bias 与 lifecycle 跟踪用的旧 bias
        # 不一致时，PnL 需要翻号（粗略近似：实际进出价位会变，这里只看方向影响）
        pnl_pct = row.get("pnl_pct")
        status = row.get("status")
        if pnl_pct is not None:
            pnl_value = float(pnl_pct)
            if old_bias != new_bias and new_bias != "neutral":
                pnl_value = -pnl_value
            metrics.pnl_list.append(pnl_value)
        if status in ("tp1_hit", "tp2_hit"):
            metrics.wins += 1
        elif status == "sl_hit":
            metrics.losses += 1
        elif status == "expired":
            metrics.expired += 1

        rr = row.get("risk_reward_ratio")
        if rr is not None:
            metrics.rr_list.append(float(rr))

    await db.disconnect()
    return metrics


async def _fetch_settled_signals(
    repos: Repositories, symbol: str, since: datetime, limit: int
) -> List[Dict[str, Any]]:
    """
    JOIN signals × signal_lifecycle 拉取所有已结算的历史信号（含真实 PnL）
    --------------------------------------------------------------
    说明：
        本函数是 replay 专用，没必要写成 Repositories 方法（仅脚本调用）。
    """
    async with repos.db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT s.id, s.ts, s.symbol, s.bias, s.confidence,
                   s.factors, s.risk_reward_ratio,
                   sl.status, sl.pnl_pct, sl.triggered_price,
                   sl.exit_price, sl.max_favorable_pct, sl.max_adverse_pct
            FROM signals s
            JOIN signal_lifecycle sl ON sl.signal_id = s.id
            WHERE s.symbol = $1
              AND s.ts >= $2
              AND sl.status IN ('sl_hit', 'tp1_hit', 'tp2_hit',
                                'expired', 'invalidated')
            ORDER BY s.ts ASC
            LIMIT $3
            """,
            symbol,
            since,
            limit,
        )
    return [dict(r) for r in rows]


def _extract_snapshot_factors(blob: Any) -> Optional[Dict[str, Any]]:
    """
    取出 signals.factors 列里的"原始因子矩阵"（兼容新旧 schema）
    """
    if not isinstance(blob, dict):
        return None
    return blob.get("factors") if "factors" in blob else blob


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """
    解析 CLI 参数
    """
    p = argparse.ArgumentParser(description="重放历史信号 + 对比新旧权重打分质量")
    p.add_argument("--symbol", default="ETH-USDT-SWAP", help="合约代码")
    p.add_argument(
        "--since",
        default=None,
        help="起始 UTC 时间，ISO 格式，例如 2026-04-01T00:00:00Z；缺省 = 14 天前",
    )
    p.add_argument(
        "--weights-source",
        choices=("baseline", "db"),
        default="baseline",
        help="规则引擎使用的权重来源",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="同时跑 baseline + db 两组并打印对照",
    )
    p.add_argument("--limit", type=int, default=5000)
    return p.parse_args()


def _resolve_since(since_arg: Optional[str]) -> datetime:
    if since_arg is None:
        return datetime.now(timezone.utc).replace(microsecond=0) - timedelta_days(14)
    raw = since_arg.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SystemExit(f"无法解析 --since={since_arg}: {exc}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def timedelta_days(days: int):
    """
    简单等价 timedelta(days=days)，单独函数化方便单元测试 mock。
    """
    from datetime import timedelta

    return timedelta(days=days)


async def _main() -> int:
    args = _parse_args()
    since = _resolve_since(args.since)

    if args.compare:
        baseline_metrics = await replay(
            symbol=args.symbol,
            since=since,
            weights_source="baseline",
            limit=args.limit,
        )
        db_metrics = await replay(
            symbol=args.symbol,
            since=since,
            weights_source="db",
            limit=args.limit,
        )
        print(json.dumps({
            "symbol": args.symbol,
            "since": since.isoformat(),
            "baseline": baseline_metrics.as_dict(),
            "db": db_metrics.as_dict(),
        }, ensure_ascii=False, indent=2))
    else:
        metrics = await replay(
            symbol=args.symbol,
            since=since,
            weights_source=args.weights_source,
            limit=args.limit,
        )
        print(json.dumps({
            "symbol": args.symbol,
            "since": since.isoformat(),
            "weights_source": args.weights_source,
            "metrics": metrics.as_dict(),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
