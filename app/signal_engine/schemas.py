"""TradingSignal Pydantic schema（LLM-Native 单一路径）。

设计目标
=========
LLM-First 重构后：

* 信号 ``bias / confidence / reason / risk / suggestion`` 五个核心字段
  仍是 LLM 输出的主体（reason/risk/suggestion 简体中文 desk 语气）；
* 结构化交易计划 7 个字段（entry_zone / stop_loss / take_profit /
  risk_reward_ratio / position_size_pct / timeframe_alignment /
  invalidation_conditions）保证"建议"是机器可执行的交易计划而非散文；
* schema 只做 **数学自洽** 校验：价位顺序 + risk>0 + RR>0 + 仓位范围；
  **不再** 因为 "RR 不到 2.0" / "胜率不到 50%" 这类业务下限强制 neutral——
  方向判断 100% 听 LLM，业务下限由 LLM 在 prompt 中自行权衡。

强约束（model_validator）
=========================
1. ``bias != "neutral"`` 时，entry_zone / stop_loss / take_profit 必须非空，
   且 ``len(take_profit) >= 2``；neutral 时这些字段强制清空。
2. ``bias == "long"`` 时：
   ``stop_loss < entry_zone[0] <= entry_zone[1] < take_profit[0] < take_profit[1]``。
3. ``bias == "short"`` 时方向反过来：
   ``stop_loss > entry_zone[0] >= entry_zone[1] > take_profit[0] > take_profit[1]``。
4. ``risk_reward_ratio`` 仅做"数学自洽"校验：``risk > 0`` 且 ``rr > 0``；
   不再有 RR < 2.0 → 强制 neutral 的业务路径。
5. ``position_size_pct ∈ [0, 0.25]``（字段层 ge/le 已强约束，再防御性 clamp）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class TradingSignal(BaseModel):
    """
    结构化交易信号（LLM 输出 schema）
    -----------------------------------------------------------------
    字段语义见模块顶部 docstring。所有约束在 ``_post_validate`` 中实施。
    """

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

    @model_validator(mode="after")
    def _post_validate(self) -> "TradingSignal":
        """
        模型级强约束：方向、价位顺序、数学自洽、仓位
        ---------------------------------------------------------------
        说明：
            1. neutral 信号清空所有结构化字段，避免下游误用残留值；
            2. long/short 信号必须给齐 entry_zone / stop_loss / ≥2 档 take_profit；
            3. 价位顺序按方向校验；
            4. 数学自洽：risk_per_unit > 0 且 risk_reward_ratio > 0
               （RR 业务下限不再 schema 强制，由 LLM 自己决定）；
            5. position_size_pct 已在字段层 ge/le 限制了 [0, 0.25]，
               这里再防御性 clamp 一次。
        """
        if self.bias == "neutral":
            object.__setattr__(self, "entry_zone", None)
            object.__setattr__(self, "stop_loss", None)
            object.__setattr__(self, "take_profit", [])
            object.__setattr__(self, "risk_reward_ratio", None)
            object.__setattr__(self, "position_size_pct", None)
            return self

        if self.entry_zone is None or self.stop_loss is None or len(self.take_profit) < 2:
            raise ValueError(
                "bias 非 neutral 时必须给齐 entry_zone / stop_loss / "
                "至少 2 档 take_profit"
            )

        ez_low, ez_high = self.entry_zone
        tps = list(self.take_profit)
        sl = float(self.stop_loss)

        if self.bias == "long":
            if not (sl < ez_low <= ez_high < tps[0] < tps[1]):
                raise ValueError(
                    "long 信号需满足：stop_loss < entry_low ≤ entry_high < tp1 < tp2，"
                    f"实际：sl={sl}, ez=({ez_low},{ez_high}), tps={tps[:2]}"
                )
        else:  # short
            if not (sl > ez_high >= ez_low > tps[0] > tps[1]):
                raise ValueError(
                    "short 信号需满足：stop_loss > entry_high ≥ entry_low > tp1 > tp2，"
                    f"实际：sl={sl}, ez=({ez_low},{ez_high}), tps={tps[:2]}"
                )

        # 数学自洽：risk_per_unit > 0；RR 必须能算出且 > 0
        entry_mid = (ez_low + ez_high) / 2
        risk_per_unit = abs(entry_mid - sl)
        if risk_per_unit <= 1e-9:
            raise ValueError(
                f"风险距离 |entry_mid - sl|={risk_per_unit} 接近 0，无法构成有效计划"
            )

        rr = self.risk_reward_ratio
        if rr is None:
            reward_per_unit = abs(tps[0] - entry_mid)
            rr = round(reward_per_unit / risk_per_unit, 4)
            object.__setattr__(self, "risk_reward_ratio", rr)

        if rr is None or rr <= 0:
            logger.warning(
                "TradingSignal RR=%s 数学不自洽（risk=0 或负值），降级为 neutral", rr,
            )
            object.__setattr__(self, "bias", "neutral")
            object.__setattr__(self, "entry_zone", None)
            object.__setattr__(self, "stop_loss", None)
            object.__setattr__(self, "take_profit", [])
            object.__setattr__(self, "risk_reward_ratio", None)
            object.__setattr__(self, "position_size_pct", None)
            return self

        if self.position_size_pct is not None:
            object.__setattr__(
                self,
                "position_size_pct",
                max(0.0, min(float(self.position_size_pct), 0.25)),
            )
        return self
