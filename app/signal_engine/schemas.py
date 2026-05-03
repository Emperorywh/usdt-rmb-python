"""TradingSignal Pydantic schema（P0 升级：结构化交易计划）。

设计目标
=========
- 旧字段（``bias / confidence / reason / risk / suggestion``）保持原样，
  让规则引擎、缓存重建、老 LLM 路径无需任何修改即可兼容。
- 新增 7 个结构化字段，让 LLM 输出的"建议"从一段自然语言升级为
  机器可执行的交易计划：
    - ``entry_zone``：入场区间（区间，不是单点）
    - ``stop_loss``：止损价
    - ``take_profit``：止盈位列表（≥ 2 档）
    - ``risk_reward_ratio``：盈亏比（tp1 vs sl）
    - ``position_size_pct``：建议仓位占总资金比例 [0, 0.25]
    - ``timeframe_alignment``：5 个周期方向投票
    - ``invalidation_conditions``：≥ 2 条量化失效条件

强约束（model_validator）
=========================
1. ``bias != "neutral"`` 时，entry_zone / stop_loss / take_profit 必须非空，
   且 ``len(take_profit) >= 2``。
2. ``bias == "long"`` 时：
   ``stop_loss < entry_zone[0] <= entry_zone[1] < take_profit[0] < take_profit[1]``。
3. ``bias == "short"`` 时方向反过来：
   ``stop_loss > entry_zone[0] >= entry_zone[1] > take_profit[0] > take_profit[1]``。
4. ``risk_reward_ratio < 1.5`` 时把 bias 强制降级为 neutral 并清空 entry/SL/TP，
   并在日志里 warning。
5. ``position_size_pct ∈ [0, 0.25]``。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class TradingSignal(BaseModel):
    """
    结构化交易信号（规则引擎 + LLM 共用的输出 schema）
    -----------------------------------------------------------------
    字段语义见模块顶部 docstring。所有约束在 ``_post_validate`` 中实施。
    """

    # ---- 旧字段（保留兼容） ----
    bias: Literal["long", "short", "neutral"] = Field(
        description="方向偏置（long / short / neutral）。"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="置信度，[0, 1] 区间。",
    )
    reason: str = Field(
        description="判断依据（中文摘要）。",
    )
    risk: str = Field(
        description="主要风险 / 失效条件（中文摘要）。",
    )
    suggestion: str = Field(
        description="操作建议（中文段落，仅作参考；不构成交易指令）。",
    )

    # ---- P0 新增结构化字段 ----
    entry_zone: Optional[Tuple[float, float]] = Field(
        default=None,
        description="可执行入场区间 [low, high]，浮点元组；neutral 时为 None。",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="止损价；neutral 时为 None。",
    )
    take_profit: List[float] = Field(
        default_factory=list,
        description="止盈位列表（≥ 2 档）；neutral 时为空。",
    )
    risk_reward_ratio: Optional[float] = Field(
        default=None,
        description="盈亏比 = |tp1 - entry_mid| / |entry_mid - stop_loss|；neutral 时为 None。",
    )
    position_size_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=0.25,
        description="建议仓位占比 [0, 0.25]；neutral 时为 None。",
    )
    timeframe_alignment: Dict[str, str] = Field(
        default_factory=dict,
        description="5 个周期方向 {'5m': 'long'|'short'|'neutral', '15m': ..., ...}",
    )
    invalidation_conditions: List[str] = Field(
        default_factory=list,
        description="量化失效条件（≥ 2 条），中文短句。",
    )

    model_config = {"extra": "forbid"}

    # ------------------------------------------------------------------
    # 字段级标准化
    # ------------------------------------------------------------------
    @field_validator("entry_zone", mode="before")
    @classmethod
    def _coerce_entry_zone(cls, v: Any) -> Any:
        """
        允许 LLM 把 entry_zone 输出成 list（[a, b]）；统一转为有序 (low, high) 元组
        ---------------------------------------------------------------
        参数：
            v: 原始值
        返回：
            排序后的 (low, high) 元组；None 透传。
        """
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = float(v[0]), float(v[1])
            return (min(a, b), max(a, b))
        return v

    # ------------------------------------------------------------------
    # 整体约束
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _post_validate(self) -> "TradingSignal":
        """
        模型级强约束：方向、价位顺序、RR、仓位
        ---------------------------------------------------------------
        说明：
            1. neutral 信号清空所有结构化字段，避免下游误用残留值；
            2. long/short 信号必须给齐 entry_zone / stop_loss / ≥2 档 take_profit；
            3. 价位顺序按方向校验；
            4. RR < 1.5 强制降级为 neutral；
            5. position_size_pct 已在字段层 ge/le 限制了 [0, 0.25]，
               这里再防御性 clamp 一次。
        """
        # 1) neutral：清空所有结构化字段
        if self.bias == "neutral":
            object.__setattr__(self, "entry_zone", None)
            object.__setattr__(self, "stop_loss", None)
            object.__setattr__(self, "take_profit", [])
            object.__setattr__(self, "risk_reward_ratio", None)
            object.__setattr__(self, "position_size_pct", None)
            return self

        # 2) 非 neutral：必须给齐核心字段
        if self.entry_zone is None or self.stop_loss is None or len(self.take_profit) < 2:
            raise ValueError(
                "bias 非 neutral 时必须给齐 entry_zone / stop_loss / "
                "至少 2 档 take_profit"
            )

        ez_low, ez_high = self.entry_zone
        tps = list(self.take_profit)
        sl = float(self.stop_loss)

        # 3) 价位顺序校验
        if self.bias == "long":
            if not (sl < ez_low <= ez_high < tps[0] < tps[1]):
                raise ValueError(
                    "long 信号需满足：stop_loss < entry_low ≤ entry_high < tp1 < tp2，"
                    f"实际：sl={sl}, ez=({ez_low},{ez_high}), tps={tps[:2]}"
                )
        else:  # short
            # entry_zone 在 _coerce_entry_zone 里被统一规整成 (low, high)，
            # short 信号的校验顺序：stop_loss > entry_high ≥ entry_low > tp1 > tp2
            if not (sl > ez_high >= ez_low > tps[0] > tps[1]):
                raise ValueError(
                    "short 信号需满足：stop_loss > entry_high ≥ entry_low > tp1 > tp2，"
                    f"实际：sl={sl}, ez=({ez_low},{ez_high}), tps={tps[:2]}"
                )

        # 4) RR 校验：< 1.5 直接降级 neutral，避免拿低 EV 计划下场
        rr = self.risk_reward_ratio
        if rr is None:
            entry_mid = (ez_low + ez_high) / 2
            risk_per_unit = abs(entry_mid - sl)
            reward_per_unit = abs(tps[0] - entry_mid)
            rr = (
                round(reward_per_unit / risk_per_unit, 4)
                if risk_per_unit > 1e-9
                else None
            )
            object.__setattr__(self, "risk_reward_ratio", rr)
        if rr is None or rr < 1.5:
            logger.warning(
                "TradingSignal RR=%s 不足 1.5，强制降级为 neutral 并清空交易计划", rr
            )
            object.__setattr__(self, "bias", "neutral")
            object.__setattr__(self, "entry_zone", None)
            object.__setattr__(self, "stop_loss", None)
            object.__setattr__(self, "take_profit", [])
            object.__setattr__(self, "risk_reward_ratio", None)
            object.__setattr__(self, "position_size_pct", None)
            return self

        # 5) position_size_pct 防御性 clamp
        if self.position_size_pct is not None:
            object.__setattr__(
                self,
                "position_size_pct",
                max(0.0, min(float(self.position_size_pct), 0.25)),
            )
        return self
