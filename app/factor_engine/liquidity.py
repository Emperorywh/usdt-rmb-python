"""流动性地图：聚合 swing 高低点 + 整数关口 + 价值区上下沿 → 双向止损池。

输出结构挂在因子矩阵根节点 ``liquidity``：

```
{
  "liquidity_pool_above": [
      {"price": 3450.0, "strength": "strong", "source": "4h_swing_high"},
      ...
  ],
  "liquidity_pool_below": [...],
  "nearest_above_pct": 0.0123,   # 距当前价的相对距离
  "nearest_below_pct": 0.0089,
  "current_price": 3401.5
}
```

设计原则
========
- 不引入新表 / 新 IO：所有数据来自 by_timeframe.{1h,4h}.market_structure，
  以及当前价（从 5m / 15m 块兜底推导）。
- 强度等级 strength ∈ {'strong','medium','weak'}：
    strong - 4h swing 极值
    medium - 1h swing 极值 / value_area 边界
    weak   - 整数关口（每 50 USD 一档）
- 强度同时决定排序：strong 在前；同强度按距离当前价从近到远排序。
- 单次计算 < 1ms。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.utils import safe_float as _to_float


# strength → 排序权重；越大越优先
_STRENGTH_WEIGHT = {"strong": 3, "medium": 2, "weak": 1}


def build_liquidity_map(
    by_timeframe: Dict[str, Dict[str, Any]],
    round_step_usd: float = 50.0,
    max_levels_per_side: int = 5,
    rounds_above: int = 4,
    rounds_below: int = 4,
) -> Dict[str, Any]:
    """
    构造双向流动性池
    -----------------------------------------------------------------
    参数：
        by_timeframe:        多周期因子矩阵
        round_step_usd:      整数关口的格子大小（USD）
        max_levels_per_side: 每个方向最多保留的档位
        rounds_above/below:  整数关口枚举的档数（向上/下各 N 个）
    返回：
        含 liquidity_pool_above / liquidity_pool_below /
        nearest_above_pct / nearest_below_pct / current_price 的 dict
    说明：
        当前价取 5m last_close → 15m last_close → 1h last_close 的优先顺序，
        全部缺失时返回 current_price=None，调用方应跳过该字段。
    """
    current_price = _resolve_current_price(by_timeframe)
    raw_above: List[Dict[str, Any]] = []
    raw_below: List[Dict[str, Any]] = []

    if current_price is None or current_price <= 0:
        return {
            "liquidity_pool_above": [],
            "liquidity_pool_below": [],
            "nearest_above_pct": None,
            "nearest_below_pct": None,
            "current_price": current_price,
        }

    # ---- 1) 从 4h / 1h 抽 swing 高低点 ----
    for tf, strength in (("4h", "strong"), ("1h", "medium")):
        block = (by_timeframe or {}).get(tf) or {}
        ms = block.get("market_structure") or {}
        for r in (ms.get("resistances") or []):
            price = _to_float(r)
            if price is not None and price > current_price:
                raw_above.append(
                    {
                        "price": round(price, 4),
                        "strength": strength,
                        "source": f"{tf}_swing_high",
                    }
                )
        for s in (ms.get("supports") or []):
            price = _to_float(s)
            if price is not None and price < current_price:
                raw_below.append(
                    {
                        "price": round(price, 4),
                        "strength": strength,
                        "source": f"{tf}_swing_low",
                    }
                )
        # value_area 上下沿：medium 强度（区间策略关键位）
        vah = _to_float(ms.get("value_area_high"))
        val = _to_float(ms.get("value_area_low"))
        if vah is not None and vah > current_price:
            raw_above.append(
                {
                    "price": round(vah, 4),
                    "strength": "medium",
                    "source": f"{tf}_value_area_high",
                }
            )
        if val is not None and val < current_price:
            raw_below.append(
                {
                    "price": round(val, 4),
                    "strength": "medium",
                    "source": f"{tf}_value_area_low",
                }
            )

    # ---- 2) 整数关口：每 round_step_usd 一档 ----
    if round_step_usd > 0:
        # 向上枚举
        # base = ceil(current_price / step) * step
        base_up = (int(current_price // round_step_usd) + 1) * round_step_usd
        for i in range(rounds_above):
            price = base_up + i * round_step_usd
            raw_above.append(
                {
                    "price": round(float(price), 4),
                    "strength": "weak",
                    "source": "round_level",
                }
            )
        # 向下枚举
        base_dn = (int(current_price // round_step_usd)) * round_step_usd
        for i in range(rounds_below):
            price = base_dn - i * round_step_usd
            if price > 0 and price < current_price:
                raw_below.append(
                    {
                        "price": round(float(price), 4),
                        "strength": "weak",
                        "source": "round_level",
                    }
                )

    # ---- 3) 去重：同价位保留强度最高的一档 ----
    above = _dedup_and_rank(raw_above, current_price, side="above")
    below = _dedup_and_rank(raw_below, current_price, side="below")

    # ---- 4) 截断到 max_levels_per_side ----
    above = above[:max_levels_per_side]
    below = below[:max_levels_per_side]

    nearest_above_pct = (
        round((above[0]["price"] - current_price) / current_price, 6)
        if above
        else None
    )
    nearest_below_pct = (
        round((current_price - below[0]["price"]) / current_price, 6)
        if below
        else None
    )

    return {
        "liquidity_pool_above": above,
        "liquidity_pool_below": below,
        "nearest_above_pct": nearest_above_pct,
        "nearest_below_pct": nearest_below_pct,
        "current_price": round(float(current_price), 4),
    }


def _dedup_and_rank(
    items: List[Dict[str, Any]],
    current_price: float,
    side: str,
) -> List[Dict[str, Any]]:
    """
    同价位保留 strength 最高的一档；按 strength + 距离当前价排序
    -----------------------------------------------------------------
    参数：
        items:         原始档位列表
        current_price: 当前价
        side:          'above' / 'below'，决定距离计算方向
    返回：
        排序去重后的列表（强度优先，同强度近优先）。
    """
    by_price: Dict[float, Dict[str, Any]] = {}
    for it in items:
        p = float(it["price"])
        prev = by_price.get(p)
        if prev is None or _STRENGTH_WEIGHT.get(it["strength"], 0) > _STRENGTH_WEIGHT.get(
            prev["strength"], 0
        ):
            by_price[p] = it

    def _key(rec: Dict[str, Any]) -> tuple:
        weight = _STRENGTH_WEIGHT.get(rec["strength"], 0)
        if side == "above":
            distance = float(rec["price"]) - current_price
        else:
            distance = current_price - float(rec["price"])
        return (-weight, distance)

    return sorted(by_price.values(), key=_key)


def _resolve_current_price(
    by_timeframe: Dict[str, Dict[str, Any]],
) -> Optional[float]:
    """
    依次尝试 5m → 15m → 1h 的 last_close
    """
    for tf in ("5m", "15m", "1h"):
        block = (by_timeframe or {}).get(tf) or {}
        ms = block.get("market_structure") or {}
        v = _to_float(ms.get("last_close"))
        if v is not None and v > 0:
            return v
    return None
