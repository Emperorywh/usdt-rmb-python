"""LLM-First 决策核心：基于 DeepSeek（OpenAI 兼容协议）的 LangChain Agent。

设计目标
========
本系统是 **LLM-Native** 而非"LLM 外壳 + 量化内核"。``LLMAgent`` 负责：

1. 把因子矩阵交给 :class:`~app.signal_engine.narrative_renderer.NarrativeRenderer`
   渲染成 7 段 desk trader 叙事 + 当前价；
2. 拼成紧凑 prompt（system + 1 条 few-shot + human）调用 DeepSeek；
3. 校验 LLM 输出的"数学自洽性"（schema 强约束 + 轻量 post-check）；
4. 通过基于 PostgreSQL ``signals`` 表 ts 的**固定窗口**节流缓存控制成本。

** 不做 ** 的事情：
- 不渲染任何 ASCII 表格 / 数值堆砌段；
- 不注入"历史成绩单 / 系统级评估摘要 / 规则引擎打分"作为参考；
- 不做 RR / SL 业务下限校验（schema 已保证数学自洽，业务下限由 LLM 自决）；
- 不做"价格漂移强制刷新"补丁，也不做"按波动率自适应节流"——
  节流就是固定的 ``LLM_MIN_INTERVAL_SECONDS``（默认 1800s），一处控制。

输出语言：reason / risk / suggestion 简体中文 desk 语气，bias 保持
long / short / neutral 英文枚举（与 ``signals.bias`` CHECK 约束对齐）。

思考模式下额外返回 ``reasoning_content``（思维链原文），由外层 service
负责落库做审计；不会进入下一轮 prompt。
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger
from app.signal_engine.narrative_renderer import NarrativeRenderer
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


# --------------------------------------------------------------------
# 防 prompt injection 的 symbol 白名单
# --------------------------------------------------------------------
# 业务上当前只支持 OKX 永续合约形如 ``ETH-USDT-SWAP``：
#   * 首字符必须是字母或数字；
#   * 仅允许 [A-Z0-9_-]，长度 3–32；
# 任何越界字符都被拒绝，避免用户控制的 symbol 被拼到 prompt 里
# 进行 prompt injection（参考：OWASP LLM Top 10 2025 #1）。
_SYMBOL_PATTERN: re.Pattern[str] = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,31}$")


def _to_float_safe(v: Any) -> Optional[float]:
    """
    把任意输入转 float，失败 / NaN / Inf 一律返回 None
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


# ------------------------------------------------------------------
# DeepSeek 思考模式 reasoning_content 透传补丁
# ------------------------------------------------------------------
# 背景：
#   langchain-openai (<= 0.3.x) 的 ``_convert_dict_to_message`` 只识别
#   OpenAI 标准字段（content / function_call / tool_calls / audio），
#   对 DeepSeek 在 assistant 消息上额外返回的 ``reasoning_content``
#   字段（即"思维链原文"）会**直接丢弃**。
# 修复方案：
#   子类化 ChatOpenAI，重写 ``_create_chat_result`` 把 reasoning_content
#   塞回 AIMessage.additional_kwargs，让 _extract_reasoning 能命中。
def _build_deepseek_chat_openai_class():
    """
    构造一个支持 DeepSeek ``reasoning_content`` 透传的 ChatOpenAI 子类
    --------------------------------------------------------------
    放在函数里延迟导入，避免在没装 langchain-openai 的环境里 import
    本模块就直接挂掉（与 ``_build_chain`` 的延迟导入策略保持一致）。
    """
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI

    class _DeepSeekChatOpenAI(ChatOpenAI):
        """
        DeepSeek 思考模式专用的 ChatOpenAI 子类
        ----------------------------------------------------------
        唯一改动：在 ``_create_chat_result`` 中把响应里每个 choice 的
        ``message.reasoning_content`` 透传到对应 AIMessage 的
        ``additional_kwargs["reasoning_content"]``。
        """

        def _create_chat_result(self, response, generation_info=None):
            chat_result = super()._create_chat_result(response, generation_info)
            response_dict = (
                response if isinstance(response, dict) else response.model_dump()
            )
            choices = response_dict.get("choices") or []
            for idx, choice in enumerate(choices):
                if idx >= len(chat_result.generations):
                    break
                message_dict = choice.get("message") or {}
                reasoning = message_dict.get("reasoning_content") or message_dict.get(
                    "reasoning"
                )
                if not isinstance(reasoning, str) or not reasoning.strip():
                    continue
                generated_message = chat_result.generations[idx].message
                if isinstance(generated_message, AIMessage):
                    generated_message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )
            return chat_result

    return _DeepSeekChatOpenAI


@dataclass(frozen=True)
class LLMAnalysisResult:
    """
    LLM 分析结果的轻量包装
    --------------------------------------------------------------
    字段：
        signal             : 解析后的结构化 TradingSignal
        reasoning_content  : DeepSeek 思考模式下的"思维链"原文文本；
                             未启用思考模式 / 模型未返回时为 None。
                             仅作审计用途，绝不参与下一轮提示词拼接。
        from_cache         : 本次结果是否来自节流缓存（即并未真正发起 LLM
                             API 调用）。外层 service 用它来决定是否落库——
                             只保留"真正由 LLM 产出"的那一条记录。
    """

    signal: TradingSignal
    reasoning_content: Optional[str] = None
    from_cache: bool = False


# ====================================================================
# Prompt 模板（精简到 4 块；plan 第 3.1 / 3.2 / 3.3 节）
# --------------------------------------------------------------------
# SYSTEM_PROMPT ~600 tokens：
#   1) 角色定位（资深 desk trader）
#   2) 工作方式（5 条决策风格，**不是**硬约束）
#   3) 硬约束（schema 强约束的口语版，4 条）
#   4) 输出语言
# 没有 P1 / P2 / P3 强约束、没有 6 级决策优先级、没有 5 档校准锚点；
# 让 LLM 看叙事数据自己判断方向。
# ====================================================================
SYSTEM_PROMPT = """\
你是一位资深加密永续合约 desk trader，专注 ETH-USDT-SWAP 中频交易（持仓数小时到 1 日）。
该信号仅作交易建议，不会被自动下单。

## 你的工作方式（像 desk trader，不像量化研究员）

1. 先识别市场状态（趋势 / 震荡 / 突破 / 假突破 / 诱多诱空）。
2. 找当前的"核心矛盾"——价格行为与资金流是否一致？是否有 squeeze / 大资金撤退 / 订单簿真空？
3. 多周期共振优先信任高周期（4h/1d 决定方向，1h 决定波段，15m/5m 决定时机）。
4. 输出像在 trading desk 跟同事讨论，**禁止纯指标罗列**，必须解读因果。
5. **不知道就说不知道**，证据不充分就 bias=neutral；不要为了出方向编造逻辑。

## 硬约束（违反任一条都会被 schema 拒绝）

1. **价位顺序**：long → sl < entry_low ≤ entry_high < tp1 < tp2；short 反向。
2. **必填**：bias=long/short 时必须给齐 entry_zone（[low, high] 区间）、stop_loss、take_profit（≥ 2 档）；
   neutral 时这些字段必须为 null / 空数组。
3. **invalidation_conditions**：≥ 2 条量化失效条件，每条带具体价位 / 阈值
   （如 "1h 收盘跌破 3505"、"funding 转负"）。
4. **suggestion 末尾必须有**："仅供参考，不构成交易指令"。

## 输出语言

- bias / timeframe_alignment 的 value 固定 long / short / neutral 英文枚举（不要翻译）。
- reason / risk / suggestion 用简体中文 desk 语气。
- 对照写法（不要写左边，要写右边）：

不要写："CVD 走低 + alignment=0.20，bearish_warning"
要写："1h CVD 持续走低与价格背离 = 上方有人在接盘出货；散户狂多 + 精英反向 = 典型诱多顶部结构；
       共振崩溃，强行做趋势会被 whipsaw。"

下面通过 1 个 few-shot 示例演示"什么时候不出手"——这是最难学也最关键的能力。
"""


# --------------------------------------------------------------------
# 紧凑 few-shot（仅 1 条拒绝交易示例 ≈ 1000 tokens）
# --------------------------------------------------------------------
# 选"拒绝交易"而非"做多 / 做空"的理由（plan 第 3.3 节）：
#   - 趋势单（long / short）LLM 天然会做，不需要示例；
#   - "什么时候不出手"是 LLM 最难学也最关键的能力；
#   - 1 条对比原 3 条省 ~2000 tokens。
FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
    (
        # human：transitional + 共振崩溃 + funding 偏高 + 散户精英背离 + 多头爆仓
        """合约: ETH-USDT-SWAP    时间: 2026-04-16 14:30    最新价: 2342.00

【市场状态】
状态切换（transitional），高低周期方向不一致（4h=uptrend / 1d=downtrend）
ATR(15m)=14.50（0.62%，正常波动），当前价 2342.00
ADX(1h)=20.5 → 弱趋势 / 过渡区间

【多周期方向】
4h ↑ | 1h ↓ | 15m ↑ | 5m ↓ | 1d →    共振度 +0.10（dom=neutral）
共振崩溃，方向不明 → 典型震荡市特征，强追趋势单容易被 whipsaw

【主动资金 vs 价格】
5m: net_flow -0.30K USD，CVD slope +0.0011 → 价格被动下压，主动卖盘未跟随 → 更像多头止损 / 诱空嫌疑
1h: OI -0.40% → OI 减仓但价格未跟随，双方都在撤退
综合：资金行为多周期分歧，需等待方向确认

【衍生品】
funding +45.00bp 已偏高，多头持仓拥挤；任何反向触发（如 4h MA 跌破）都容易引发 long_squeeze
散户多空比 2.45（极端多头 → 强反指利空）
精英持仓比 0.78（偏空）
散户/精英：散户狂多 + 精英反向 → 典型诱多顶部结构，警惕反转

【关键价位】
4h: 阻力 2360.00    支撑 2310.00, 2290.00
1h: 阻力 2360.00    支撑 2318.00, 2310.00
15m: 阻力 2360.00   支撑 2320.00
当前 2342.00 ─ 距上方近阻 2360.00 1.2×ATR(15m)；距下方近撑 2320.00 1.5×ATR(15m)

【流动性地图】
上方止损池: 2360.00 (weak)
下方止损池: 2310.00 (medium)
订单簿: 上方真空 → 突破后价位容易被快速穿透，追多需谨慎
距离: 上方最近池 0.77%

【Liquidations 滚动窗口】
近 5m:  long_liq 2.40 ETH | short_liq 0.10 ETH → 多头被清量级显著占优
近 15m: long_liq 5.20 ETH | short_liq 0.30 ETH | cascade=true → 多头被清量级显著占优
近 1h:  long_liq 14.50 ETH | short_liq 1.20 ETH（多头爆仓主导）
→ 持续大额多头爆仓，短期反弹动力多来自空头回补而非新增买盘，cascade 在进行 → 清仓潮正在加速

请基于以上市场状态判断方向，并按 schema 输出完整 JSON。
""",
        # ai：诚实拒绝交易
        """{
  "bias": "neutral",
  "confidence": 0.0,
  "reason": "共振崩溃明显——4h 仍上行但 1h 已转头向下，5m 跌破短期支撑；高低周期方向不一致。散户多空比 2.45 极端多头 + 精英持仓 0.78 偏空 = 典型诱多顶部结构，funding +45bp 偏高多头持仓拥挤，long_squeeze 一触即发；多头爆仓 cascade 在进行，反弹动力多来自空头回补而非新增买盘。当前 2342 距上沿 2360 仅 0.77% 做多无空间；做空又卡在区间正中（追空容易被反弹打止损）。震荡末端 + 顶部背离同时出现，方向判断 = 抛硬币，不出手。",
  "risk": "transitional + 共振崩溃 + funding 偏高 + 多头清仓潮叠加，任何方向都属于负期望博弈。",
  "suggestion": "本周期观望，等以下任一信号确认再考虑入场：(a) 4h 收盘跌破 2310 配合 funding 转负 → 高确定性做空；(b) 4h 收盘站上 2360 + OI 同步加仓 → 高确定性做多突破；(c) 共振度恢复到 ±0.5 以上方向明确。仅供参考，不构成交易指令。",
  "entry_zone": null,
  "stop_loss": null,
  "take_profit": [],
  "risk_reward_ratio": null,
  "position_size_pct": null,
  "timeframe_alignment": {"5m": "short", "15m": "long", "1h": "short", "4h": "long", "1d": "neutral"},
  "invalidation_conditions": []
}""",
    ),
]


# 思考模式（deepseek-reasoner）专用的格式硬约束。
# 注意 schema 占位符里的 `{` / `}` 会被 ChatPromptTemplate 当成变量解析，
# 所以放进去之前要把所有大括号双写转义；下面用 .partial 注入而不是 f-string，
# 既避免双写转义问题，又能在运行期一次性绑定。
THINKING_OUTPUT_INSTRUCTIONS = """\

【输出格式硬约束】
请只输出**一个**符合下述 JSON Schema 的 JSON 对象，**不要**输出任何其他文字
（不要 markdown 代码围栏、不要前后说明、不要思考过程）。如果你需要思考，
全部放在 reasoning 通道里，最终回复正文必须只有那一个 JSON 对象。

JSON Schema:
{format_instructions}
"""


# --------------------------------------------------------------------
# 紧凑 HUMAN_PROMPT（7 段 desk 叙事 + 1 段当前价）
# --------------------------------------------------------------------
# 与老 12 段 ASCII 表对照：
#   * 删除指标堆砌段（mtf_factor_table / orderbook_text / regime_text 等），
#     改由 NarrativeRenderer 输出 desk 叙事；
#   * 删除自反馈 / 离线评估 / 浅模型打分 等"反 LLM"参考视角段；
#   * 新增 liquidations 第 7 段（plan 第 2.4 节）。
HUMAN_PROMPT = """\
合约: {symbol}    时间: {ts}    最新价: {last_close}

【市场状态】
{market_state}

【多周期方向】
{mtf_direction}

【主动资金 vs 价格】
{capital_action}

【衍生品】
{derivatives}

【关键价位】（从大到小）
{key_levels}

【流动性地图】
{liquidity}

【Liquidations 滚动窗口】
{liquidations}

请基于以上市场状态判断方向，并按 schema 输出完整 JSON。
"""


class LLMAgent:
    """
    LangChain DeepSeek 调用封装（基于 signals 表的 DB 节流）。
    --------------------------------------------------------------
    节流策略（**固定窗口**，不再做"按波动率自适应"）：
    - "上一次判断时间"取自 signals 表里该 symbol 最后一条记录的 ts。
    - 进程重启 / 多副本部署时所有实例共享同一份"上次调用时间"
      （PostgreSQL 是唯一真源），节流不会被绕过。
    - 在 ``self.min_interval``（即 ``LLM_MIN_INTERVAL_SECONDS``，默认 1800s）
      窗口内，analyze 不会真正发起 LLM 请求，而是从 signals 表读出最近一条
      LLM 判断，重建 ``LLMAnalysisResult`` 返回（带 ``from_cache=True`` 标记）。
    - 上层 service 仅在 ``from_cache=False`` 时落库，保证入库节奏 =
      LLM 真实调用节奏 = 节流窗口节奏。
    - 通过按 symbol 的 ``asyncio.Lock`` 防止同一 symbol 在窗口刚到
      时被并发调用打多次接口。
    - 设置 ``LLM_MIN_INTERVAL_SECONDS=0`` 可完全关闭节流（调试用）。

    设计取舍：之前版本曾按 ATR 比 / regime / alignment 做"自适应节流"
    （高波动 300s / 中波动 600s / 默认 1800s），但实际运行中"高波动"
    分支命中过于频繁，把 LLM 调用节奏压到 5–10 分钟一次，单日 DeepSeek
    调用成本明显超预算。当前策略：**完全忽略波动**，固定 1800s 一次，
    每次都拿数据库最新因子矩阵喂给 LLM 重新判断。
    """

    def __init__(self, settings: Settings, repos: Repositories):
        self.settings = settings
        self.repos = repos
        self._chain = None
        # 思考模式 vs 非思考模式只影响"如何解析返回值"，build 时确定一次。
        self._chain_thinking_mode: bool = False
        # 按 symbol 维度的并发锁，保证窗口刚到时不会被并发调用打多次接口。
        # 跨进程并发由"以 DB 时间戳为准"的语义保证，本锁只防同进程内竞态。
        self._locks: Dict[str, asyncio.Lock] = {}
        self._renderer = NarrativeRenderer()

    def _build_chain(self):
        """
        构建 LangChain 调用链。
        --------------------------------------------------------------
        - 通过 ChatOpenAI 的 OpenAI 兼容协议调用 DeepSeek。
        - **思考模式（deepseek-reasoner）**：
          * 不支持 ``tool_choice`` → 走"提示词约束 + JSON 解析"路线。
          * 使用 ``_build_deepseek_chat_openai_class()`` 透传 reasoning_content。
          * extra_body={"thinking": {"type": "enabled"}} 开启思维链。
          * temperature 在思考模式下不生效，不传入。
        - **非思考模式（deepseek-chat）**：
          * 走 ``with_structured_output(method="function_calling")``。
        """
        from langchain_openai import ChatOpenAI

        if not self.settings.deepseek_api_key:
            logger.warning("未配置 DEEPSEEK_API_KEY，运行时将跳过 LLM 分析")

        thinking_enabled = bool(self.settings.deepseek_thinking_enabled)
        chat_openai_cls = (
            _build_deepseek_chat_openai_class() if thinking_enabled else ChatOpenAI
        )

        llm_kwargs: Dict[str, Any] = {
            "model": self.settings.deepseek_model,
            "api_key": self.settings.deepseek_api_key or "missing-key",
            "base_url": self.settings.deepseek_base_url,
            "timeout": self.settings.llm_timeout,
            "max_retries": 2,
        }

        if thinking_enabled:
            effort = (self.settings.deepseek_reasoning_effort or "high").lower()
            if effort not in {"high", "max"}:
                logger.warning(
                    "未知的 reasoning_effort=%r，回退为 'high'", effort
                )
                effort = "high"
            llm_kwargs["reasoning_effort"] = effort
            llm_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            logger.info(
                "DeepSeek 思考模式已启用（model=%s，effort=%s，json_via=prompt）",
                self.settings.deepseek_model,
                effort,
            )
        else:
            llm_kwargs["temperature"] = self.settings.llm_temperature
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            logger.info(
                "DeepSeek 思考模式已禁用（model=%s，temperature=%.2f，json_via=function_calling）",
                self.settings.deepseek_model,
                self.settings.llm_temperature,
            )

        llm = chat_openai_cls(**llm_kwargs)

        # ----------------------------------------------------------------
        # few-shot 示例以 message history 注入（最佳实践）
        # ----------------------------------------------------------------
        # 把 (human_text, ai_text) 转成 HumanMessage / AIMessage 实例，
        # 插在 system 与运行期 human 之间。直接用 BaseMessage 实例可以
        # 跳过 ChatPromptTemplate 的 f-string 渲染（不需要把 JSON 中的
        # ``{`` ``}`` 双写转义），也不会被 partial 注入污染。
        few_shot_messages: List[Any] = []
        for human_text, ai_text in FEW_SHOT_EXAMPLES:
            few_shot_messages.append(HumanMessage(content=human_text))
            few_shot_messages.append(AIMessage(content=ai_text))

        base_messages: List[Any] = [("system", SYSTEM_PROMPT), *few_shot_messages]

        logger.info(
            "构建 LangChain 调用链（thinking=%s，few-shot=%d）",
            thinking_enabled, len(FEW_SHOT_EXAMPLES),
        )

        if thinking_enabled:
            parser = PydanticOutputParser(pydantic_object=TradingSignal)
            messages = base_messages + [
                ("system", THINKING_OUTPUT_INSTRUCTIONS),
                ("human", HUMAN_PROMPT),
            ]
            prompt = ChatPromptTemplate.from_messages(messages).partial(
                format_instructions=parser.get_format_instructions()
            )
            self._chain_thinking_mode = True
            return prompt | llm

        structured = llm.with_structured_output(
            TradingSignal, method="function_calling", include_raw=True
        )
        prompt = ChatPromptTemplate.from_messages(
            base_messages + [("human", HUMAN_PROMPT)]
        )
        self._chain_thinking_mode = False
        return prompt | structured

    @property
    def enabled(self) -> bool:
        """是否启用 LLM。以是否配置了 DeepSeek API Key 为准。"""
        return bool(self.settings.deepseek_api_key)

    @property
    def min_interval(self) -> int:
        """
        LLM 调用最小间隔（秒）—— 固定窗口节流，唯一真源
        --------------------------------------------------------------
        从 ``settings.llm_min_interval_seconds`` 读取（默认 1800s = 30 分钟），
        单元测试可通过 monkeypatch settings 来调整。
        设为 0 / 负数则完全关闭节流（调试用）。
        """
        return max(0, int(self.settings.llm_min_interval_seconds))

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        """获取或创建指定 symbol 对应的并发锁。"""
        lock = self._locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[symbol] = lock
        return lock

    @staticmethod
    def _collect_cached_plan_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        把 signals 表行里的"结构化交易计划"列规整成 TradingSignal 构造 kwargs
        --------------------------------------------------------------
        类型转换说明：
            - NUMERIC 列经 asyncpg 默认返回 ``decimal.Decimal``；TradingSignal
              的字段是 float，这里统一转 float（精度损失对交易计划无影响）。
            - JSONB 列在 database._init_connection 里已经注册了 codec，
              直接拿到 list / dict / None。
            - 任何字段为 None / 空 / 类型不符时，原样不放进 kwargs，
              让 TradingSignal 走默认值；如果 bias != neutral 又凑不齐
              entry_zone / stop_loss / 2 档 take_profit，model_validator
              会抛 ValueError，调用方 catch 后会让本轮走真 LLM 调用。
        """
        def _to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        kwargs: Dict[str, Any] = {}

        ez = row.get("entry_zone")
        if isinstance(ez, (list, tuple)) and len(ez) == 2:
            ez_low = _to_float(ez[0])
            ez_high = _to_float(ez[1])
            if ez_low is not None and ez_high is not None:
                kwargs["entry_zone"] = (ez_low, ez_high)

        sl = _to_float(row.get("stop_loss"))
        if sl is not None:
            kwargs["stop_loss"] = sl

        tp_raw = row.get("take_profit")
        if isinstance(tp_raw, (list, tuple)) and tp_raw:
            tp_list: List[float] = []
            for t in tp_raw:
                tv = _to_float(t)
                if tv is not None:
                    tp_list.append(tv)
            if tp_list:
                kwargs["take_profit"] = tp_list

        rr = _to_float(row.get("risk_reward_ratio"))
        if rr is not None:
            kwargs["risk_reward_ratio"] = rr

        psp = _to_float(row.get("position_size_pct"))
        if psp is not None:
            kwargs["position_size_pct"] = psp

        tfa = row.get("timeframe_alignment")
        if isinstance(tfa, dict) and tfa:
            kwargs["timeframe_alignment"] = {
                str(k): str(v) for k, v in tfa.items() if isinstance(v, str)
            }

        ic = row.get("invalidation_conditions")
        if isinstance(ic, list) and ic:
            kwargs["invalidation_conditions"] = [
                str(x) for x in ic if isinstance(x, str)
            ]

        return kwargs

    async def _load_recent_judgment(
        self,
        symbol: str,
        min_interval: Optional[int] = None,
    ) -> Optional[LLMAnalysisResult]:
        """
        若指定 symbol 在节流窗口内已有 LLM 判断（落在 signals 表里），
        则把它重建成 LLMAnalysisResult 返回；否则返回 None 让上层调用 LLM。
        --------------------------------------------------------------
        参数：
            symbol      : 合约代码
            min_interval: 节流窗口（秒）。None 时回退到 self.min_interval。
        步骤：
            1) ``effective_interval <= 0``：节流关闭，直接返回 None（每次都真调）。
            2) 查 signals 表里该 symbol 最近一条记录的 ts；表为空 / 查询失败
               都返回 None 让上层去打 LLM。
            3) 计算 ``now - ts``；
               - ``< effective_interval``：节流命中，把那一行的 bias / confidence
                 / reason / risk / suggestion / reasoning_content / plan 字段
                 重建成 LLMAnalysisResult，置 ``from_cache=True`` 返回。
               - ``>= effective_interval``：节流过期，返回 None。
        异常处理：
            DB 查询失败 / 历史脏行无法重建 schema 时不应阻塞 LLM 调用；
            记一行 warning 后返回 None，让上层退化为"直接打 LLM"。
        """
        effective_interval = (
            int(min_interval) if min_interval is not None else self.min_interval
        )
        if effective_interval <= 0:
            return None
        try:
            row = await self.repos.fetch_latest_signal_judgment(symbol)
        except Exception:
            logger.warning(
                "查询 %s 最近一条信号时间戳失败；按未命中缓存处理",
                symbol,
                exc_info=True,
            )
            return None
        if row is None:
            return None

        last_ts: Optional[datetime] = row.get("ts")
        if last_ts is None:
            return None
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
        if elapsed >= effective_interval:
            return None

        cached_plan_kwargs = self._collect_cached_plan_kwargs(row)
        try:
            cached_signal = TradingSignal(
                bias=row.get("bias") or "neutral",
                confidence=float(row["confidence"]),
                reason=row.get("reason") or "",
                risk=row.get("risk") or "",
                suggestion=row.get("suggestion") or "",
                **cached_plan_kwargs,
            )
        except Exception:
            logger.warning(
                "无法从 signals 表行重建 %s 的 TradingSignal（多半是历史脏行"
                " / 结构化字段缺失）；将按未命中缓存重新调用 LLM",
                symbol,
                exc_info=True,
            )
            return None

        remaining = effective_interval - elapsed
        # 节流命中是常态轮询事件（每个 signal 周期都会触发），打 INFO 会刷屏。
        # 真正想观察"什么时候才会发起下一次 LLM 调用"时，把日志级别调到 DEBUG 即可。
        logger.debug(
            "LLM 数据库节流命中 %s（已过 %.0fs，下次调用还需 %.0fs，min_interval=%ds）",
            symbol,
            elapsed,
            remaining,
            effective_interval,
        )
        return LLMAnalysisResult(
            signal=cached_signal,
            reasoning_content=row.get("reasoning_content"),
            from_cache=True,
        )

    # ------------------------------------------------------------------
    # Prompt 构造（NarrativeRenderer 7 段叙事 + 当前价）
    # ------------------------------------------------------------------
    def _build_prompt_inputs(
        self,
        symbol: str,
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        渲染 HUMAN_PROMPT 占位符所需的 dict
        --------------------------------------------------------------
        参数：
            symbol  : 合约代码（必须通过 _SYMBOL_PATTERN 白名单）
            factors : FactorAggregator.compute 输出
        返回：
            ChatPromptTemplate.format 所需的全部 key-value 字典。
        说明：
            - symbol 必须通过白名单校验，防 prompt injection
              （symbol 会被原样拼到 system / human / 下游 reason 中）。
            - NarrativeRenderer 内部容错：任一段数据缺失时返回提示文本，
              不会因为一个字段空导致 KeyError。
        """
        if not _SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"非法 symbol={symbol!r}：仅允许 [A-Z0-9_-]，"
                f"长度 3–32（首字符必须为字母或数字）"
            )

        ts = factors.get("computed_at") or ""

        by_tf = factors.get("by_timeframe") or {}
        ms_15m = ((by_tf.get("15m") or {}).get("market_structure")) or {}
        last_close_v = _to_float_safe(ms_15m.get("last_close"))
        last_close = f"{last_close_v:.2f}" if last_close_v is not None else "-"

        sections = self._renderer.render_sections(factors)

        return {
            "symbol": symbol,
            "ts": ts,
            "last_close": last_close,
            "market_state": sections["market_state"],
            "mtf_direction": sections["mtf_direction"],
            "capital_action": sections["capital_action"],
            "derivatives": sections["derivatives"],
            "key_levels": sections["key_levels"],
            "liquidity": sections["liquidity"],
            "liquidations": sections["liquidations"],
        }

    @staticmethod
    def _extract_token_usage(raw_message: Any) -> Tuple[int, int, int]:
        """
        从 LangChain AIMessage 中提取 (input_tokens, output_tokens, total_tokens)
        --------------------------------------------------------------
        兼容两套常见来源（langchain-openai 0.3+ 同时透传两套）：
        1) ``AIMessage.usage_metadata``（langchain 标准化字段）：
           ``{"input_tokens": int, "output_tokens": int, "total_tokens": int}``。
        2) ``AIMessage.response_metadata["token_usage"]``（OpenAI 协议原貌）：
           ``{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}``。
        任一字段缺失时填 -1，不抛异常——可观测性不能阻塞主决策路径。
        """
        if raw_message is None:
            return (-1, -1, -1)

        in_tok = -1
        out_tok = -1
        total_tok = -1

        usage_meta = getattr(raw_message, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            in_tok_v = usage_meta.get("input_tokens")
            out_tok_v = usage_meta.get("output_tokens")
            total_v = usage_meta.get("total_tokens")
            if isinstance(in_tok_v, (int, float)):
                in_tok = int(in_tok_v)
            if isinstance(out_tok_v, (int, float)):
                out_tok = int(out_tok_v)
            if isinstance(total_v, (int, float)):
                total_tok = int(total_v)

        if in_tok < 0 or out_tok < 0 or total_tok < 0:
            response_meta = getattr(raw_message, "response_metadata", None)
            if isinstance(response_meta, dict):
                tu = response_meta.get("token_usage")
                if isinstance(tu, dict):
                    if in_tok < 0:
                        v = tu.get("prompt_tokens")
                        if isinstance(v, (int, float)):
                            in_tok = int(v)
                    if out_tok < 0:
                        v = tu.get("completion_tokens")
                        if isinstance(v, (int, float)):
                            out_tok = int(v)
                    if total_tok < 0:
                        v = tu.get("total_tokens")
                        if isinstance(v, (int, float)):
                            total_tok = int(v)

        return (in_tok, out_tok, total_tok)

    @staticmethod
    def _extract_reasoning(raw_message: Any) -> Optional[str]:
        """
        从 LangChain AIMessage / dict 中抽取 DeepSeek 思考模式的思维链原文
        --------------------------------------------------------------
        DeepSeek 在思考模式下把思维链放在 ``reasoning_content`` 字段，
        我们子类化 ChatOpenAI 后透传到 ``additional_kwargs["reasoning_content"]``。
        部分版本也可能放到 response_metadata 里，这里同时兼容。
        """
        if raw_message is None:
            return None

        candidates: list[Any] = []

        additional = getattr(raw_message, "additional_kwargs", None)
        if isinstance(additional, dict):
            candidates.append(additional.get("reasoning_content"))
            candidates.append(additional.get("reasoning"))
        response_meta = getattr(raw_message, "response_metadata", None)
        if isinstance(response_meta, dict):
            candidates.append(response_meta.get("reasoning_content"))
            candidates.append(response_meta.get("reasoning"))

        if isinstance(raw_message, dict):
            candidates.append(raw_message.get("reasoning_content"))
            candidates.append(raw_message.get("reasoning"))
            ak = raw_message.get("additional_kwargs")
            if isinstance(ak, dict):
                candidates.append(ak.get("reasoning_content"))
                candidates.append(ak.get("reasoning"))

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _parse_json_content(content: Any, symbol: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 原始 content 文本中抽出 JSON 对象。
        --------------------------------------------------------------
        思考模式下我们让模型直出 JSON，但模型偶尔会：
          - 在 JSON 外面包一层 ```json ... ``` 代码围栏
          - 在 JSON 前后多写几句解释
        这里做容错：先尝试整体 json.loads，失败再用第一个 ``{`` 到最后一个
        ``}`` 之间的子串重试。仍失败则记 ERROR 并返回 None，外层降级。
        """
        if isinstance(content, list):
            text_parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            logger.error("LLM 为 %s 返回了空内容", symbol)
            return None

        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.error(
            "LLM 输出 JSON 解析失败 %s；原始内容（前 500 字符）：%s",
            symbol,
            content[:500],
        )
        return None

    # ------------------------------------------------------------------
    # LLM 输出 deterministic post-check（极简版）
    # ------------------------------------------------------------------
    # 设计原则：
    #   schema 已经强约束：价位顺序 + 必填字段 + 数学自洽（risk>0, RR>0）。
    #   post-check 只做"内部一致性"的最后一道防御，**不**包含任何业务下限
    #   （如 RR < 2.0 / SL < 1.5×ATR 之类），方向判断完全交给 LLM。
    # 仅保留 2 项 hard 校验：
    #   1) RR 自报诚实性：自报 RR 与按 |tp1-entry_mid|/|entry_mid-sl|
    #      复算的 RR 偏差 ≤ 5%；否则视为 LLM 自报失真。
    #   2) take_profit ≥ 2 档（schema 已强约束，防御重复校验）。
    _POST_CHECK_RR_TOLERANCE: float = 0.05

    @classmethod
    def _post_check_signal(
        cls,
        signal: TradingSignal,
    ) -> Tuple[TradingSignal, List[str]]:
        """
        对 LLM 输出做极简版 deterministic post-check
        --------------------------------------------------------------
        参数：
            signal : LLM 已通过 schema 校验的 TradingSignal
        返回：
            (final_signal, issues)
            - issues 为空：原 signal 透传
            - issues 非空：返回降级为 neutral 的新 signal
        校验项（hard）—— 任一不通过 → 降级 neutral：
          1) RR 自报诚实性：|self - recalc| / self > tolerance（默认 5%）
          2) take_profit 数量 < 2（schema 已强约束，防御）

        校验项（soft）—— 仅记录，不触发降级：
          3) invalidation_conditions 数量 < 2（prompt 要求但 schema 未强约束）
          4) timeframe_alignment 缺周期（不齐 5 个）
        """
        if signal.bias == "neutral":
            return signal, []

        hard_issues: List[str] = []
        soft_issues: List[str] = []

        ez = signal.entry_zone
        sl = signal.stop_loss
        tps = list(signal.take_profit or [])

        if ez is None or sl is None or len(tps) < 2:
            hard_issues.append(
                f"plan 字段不完整：entry_zone={ez} stop_loss={sl} take_profit_n={len(tps)}"
            )
            return cls._force_neutral_for_post_check(signal, hard_issues), hard_issues

        entry_mid = (float(ez[0]) + float(ez[1])) / 2.0
        sl_f = float(sl)
        tp1 = float(tps[0])
        risk = abs(entry_mid - sl_f)
        reward = abs(tp1 - entry_mid)

        if risk < 1e-9:
            hard_issues.append(
                f"entry_mid({entry_mid:.4f}) ≈ stop_loss({sl_f:.4f})，无效计划"
            )
            return cls._force_neutral_for_post_check(signal, hard_issues), hard_issues

        rr_recalc = reward / risk
        rr_self = signal.risk_reward_ratio

        if (
            rr_self is not None
            and rr_self > 0
            and abs(rr_recalc - float(rr_self)) / float(rr_self) > cls._POST_CHECK_RR_TOLERANCE
        ):
            hard_issues.append(
                f"自报 RR={float(rr_self):.3f} 与复算 RR={rr_recalc:.3f} 偏差 "
                f"{abs(rr_recalc - float(rr_self)) / float(rr_self) * 100:.1f}% "
                f"> 容忍 {cls._POST_CHECK_RR_TOLERANCE * 100:.0f}%"
            )

        if len(tps) < 2:
            hard_issues.append(f"take_profit 仅 {len(tps)} 档 < 2")

        if len(signal.invalidation_conditions) < 2:
            soft_issues.append(
                f"invalidation_conditions 仅 {len(signal.invalidation_conditions)} 条 < 2 "
                "（prompt 要求 ≥ 2 条，但不触发强制降级）"
            )

        tfa = signal.timeframe_alignment or {}
        missing_tf = [tf for tf in ("5m", "15m", "1h", "4h", "1d") if tf not in tfa]
        if missing_tf:
            soft_issues.append(
                f"timeframe_alignment 缺周期 {missing_tf}（不触发强制降级）"
            )

        if soft_issues:
            logger.warning(
                "LLM post-check soft issues（不强制降级，仅记录）: bias=%s issues=%s",
                signal.bias,
                soft_issues,
            )

        if hard_issues:
            logger.warning(
                "LLM post-check 强制降级 neutral: bias=%s issues=%s",
                signal.bias,
                hard_issues,
            )
            return cls._force_neutral_for_post_check(signal, hard_issues), hard_issues

        return signal, soft_issues

    @staticmethod
    def _force_neutral_for_post_check(
        signal: TradingSignal,
        issues: List[str],
    ) -> TradingSignal:
        """
        把 LLM 的 long/short 输出强制降级为 neutral，并把 issues 写进 reason
        --------------------------------------------------------------
        参数：
            signal : 原 LLM 输出（bias ∈ {long, short}）
            issues : post-check 触发的 hard 校验失败原因
        返回：
            一条 neutral TradingSignal：
                - bias=neutral，schema 自动清空 plan 字段
                - confidence=0.0
                - reason 拼接 [LLM post-check 强制降级] 前缀 + issues 列表
                - timeframe_alignment 保留原始 5 周期投票（事后审计有用）
                - invalidation_conditions 清空（neutral 时无意义）
        """
        issues_text = "；".join(issues) if issues else "未知"
        new_reason = (
            f"[LLM post-check 强制降级] 触发 {len(issues)} 项校验失败："
            f"{issues_text}。原 LLM reason 已截断到 reasoning_content（思考模式）"
            f" / 日志（非思考模式）以便排查。"
        )
        return TradingSignal(
            bias="neutral",
            confidence=0.0,
            reason=new_reason,
            risk=(
                f"原 LLM 输出 bias={signal.bias}，但因 deterministic post-check 校验失败"
                "强制降级观望。原 risk 字段：" + (signal.risk or "（空）")
            ),
            suggestion=(
                "本周期 LLM 输出未通过内部一致性校验（RR 自报失真 / plan 不完整），"
                "强制降级观望，等待下一轮重新判断。仅供参考，不构成交易指令"
            ),
            timeframe_alignment=dict(signal.timeframe_alignment or {}),
            invalidation_conditions=[],
        )

    async def analyze(
        self,
        symbol: str,
        factors: Dict[str, Any],
    ) -> Optional[LLMAnalysisResult]:
        """
        执行一次 LLM 分析，带按 symbol 的节流缓存。
        --------------------------------------------------------------
        参数：
            symbol  : 合约代码，缓存与并发锁的隔离粒度
            factors : 因子聚合结果（JSON 可序列化字典）
        返回：
            LLMAnalysisResult  ：包含 TradingSignal 与 reasoning_content
                                 （来自缓存时同样会带回首次调用的思维链原文）
            None               ：LLM 未启用 / 链构建失败 / 调用失败 / schema 校验失败
        节流策略：
            "上一次判断时间"取自 signals 表里该 symbol 最后一条记录的 ts。
            同一 symbol 在 ``self.min_interval``（即 ``LLM_MIN_INTERVAL_SECONDS``，
            默认 1800s）窗口内的请求会直接把那条记录的判断字段重建成
            LLMAnalysisResult 返回（``from_cache=True``），不会真正发起
            API 调用，也不会再次落库。**完全不考虑波动率 / regime / alignment**——
            每次真实调用都是基于当下数据库里最新的因子矩阵重新判断。
        """
        if not self.enabled:
            logger.info("LLM 已禁用（未配置 DEEPSEEK_API_KEY），返回 None")
            return None

        interval = self.min_interval

        # 快速路径：先在锁外查一次 DB，避免抢锁
        cached = await self._load_recent_judgment(symbol, min_interval=interval)
        if cached is not None:
            return cached

        async with self._get_lock(symbol):
            # 慢速路径：拿锁后再查一次（double-checked locking）
            cached = await self._load_recent_judgment(symbol, min_interval=interval)
            if cached is not None:
                return cached

            if self._chain is None:
                try:
                    self._chain = self._build_chain()
                except Exception:
                    logger.exception("构建 LangChain 调用链失败")
                    return None

            try:
                logger.info(
                    "LLM 发起调用 -> %s（固定节流间隔=%ds）",
                    symbol,
                    interval,
                )
                prompt_inputs = self._build_prompt_inputs(symbol=symbol, factors=factors)
                _llm_call_start = time.monotonic()
                result = await self._chain.ainvoke(prompt_inputs)
                _llm_call_latency_ms = int(
                    (time.monotonic() - _llm_call_start) * 1000
                )
            except Exception:
                logger.exception("LangChain 调用失败，返回 None")
                return None

            # 两条返回路径：
            # 1) 思考模式：result 是 AIMessage，content 是 JSON 文本，需手动解析。
            # 2) 非思考模式：result 是 dict {"raw": AIMessage, "parsed": ...,
            #    "parsing_error": ...}，由 langchain 通过 function calling 解析好。
            # 同时兼容旧版 / 被 monkeypatch 替换成直接返回 TradingSignal 的情况。
            parsed: Any
            raw_message: Any = None
            parsing_error: Any = None

            if isinstance(result, dict) and ("parsed" in result or "raw" in result):
                parsed = result.get("parsed")
                raw_message = result.get("raw")
                parsing_error = result.get("parsing_error")
            elif isinstance(result, TradingSignal):
                parsed = result
            elif hasattr(result, "content"):
                raw_message = result
                content = getattr(result, "content", "")
                parsed = self._parse_json_content(content, symbol)
                if parsed is None:
                    return None
            else:
                parsed = result

            if parsing_error is not None:
                logger.error("LLM 结构化解析错误 %s：%s", symbol, parsing_error)
                return None

            signal: Optional[TradingSignal]
            if isinstance(parsed, TradingSignal):
                signal = parsed
            elif isinstance(parsed, dict):
                try:
                    signal = TradingSignal.model_validate(parsed)
                except Exception:
                    logger.exception("LLM 输出未通过 schema 校验")
                    return None
            else:
                logger.error("LLM 解析结果类型异常 %s：%r", symbol, type(parsed))
                return None

            reasoning_content = self._extract_reasoning(raw_message)

            in_tok, out_tok, total_tok = self._extract_token_usage(raw_message)
            logger.info(
                "LLM 调用完成 symbol=%s latency_ms=%d in_tok=%d out_tok=%d total_tok=%d "
                "thinking=%s reasoning_len=%d",
                symbol,
                _llm_call_latency_ms,
                in_tok,
                out_tok,
                total_tok,
                self._chain_thinking_mode,
                len(reasoning_content) if reasoning_content else 0,
            )
            if reasoning_content:
                logger.debug(
                    "已捕获 LLM 思维链 %s（%d 字符）",
                    symbol, len(reasoning_content),
                )
            elif self.settings.deepseek_thinking_enabled:
                logger.debug(
                    "思考模式已启用，但 %s 的原始消息中未找到 reasoning_content",
                    symbol,
                )

            try:
                signal, post_check_issues = self._post_check_signal(signal=signal)
                if post_check_issues:
                    logger.info(
                        "LLM post-check 命中 %s：原 bias=%s 校验失败 %d 项 → 已处理",
                        symbol,
                        signal.bias,
                        len(post_check_issues),
                    )
            except Exception:
                logger.exception(
                    "LLM post-check 异常 symbol=%s（信号原样透传）",
                    symbol,
                )

            return LLMAnalysisResult(
                signal=signal,
                reasoning_content=reasoning_content,
                from_cache=False,
            )
