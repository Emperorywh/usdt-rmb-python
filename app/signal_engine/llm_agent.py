"""基于 DeepSeek（OpenAI 兼容协议）的 LangChain 分析 Agent。

用 ``langchain-openai`` 的 ``ChatOpenAI``，并把 ``base_url`` 指到 DeepSeek。
通过 ``with_structured_output(...)``（function-calling）把输出约束到
:class:`TradingSignal`，保证返回严格 JSON，否则直接抛错。

输出语言：reason / risk / suggestion 三个字段统一使用简体中文，
bias 字段保持 long/short/neutral 英文枚举（与 ``signals.bias`` 的
CHECK 约束保持一致）。

思考模式下额外返回 ``reasoning_content``（思维链原文），由外层 service
负责落库做审计；它不参与下游决策、也不会进入提示词上下文。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


# ------------------------------------------------------------------
# DeepSeek 思考模式 reasoning_content 透传补丁
# ------------------------------------------------------------------
# 背景：
#   langchain-openai (<= 0.3.x) 的 ``_convert_dict_to_message`` 只识别
#   OpenAI 标准字段（content / function_call / tool_calls / audio），
#   对 DeepSeek 在 assistant 消息上额外返回的 ``reasoning_content``
#   字段（即"思维链原文"）会**直接丢弃**。
#   表现就是：开启思考模式后，调用方从 AIMessage.additional_kwargs 与
#   AIMessage.response_metadata 里都拿不到 reasoning_content，最终
#   signals.reasoning_content 列永远为 NULL。
#
# 修复方案：
#   子类化 ChatOpenAI，重写 ``_create_chat_result``：
#   1) 让父类先按原逻辑构建 ChatResult；
#   2) 再从原始响应字典里把每个 choice 的 ``message.reasoning_content``
#      取出来，塞进对应 AIMessage.additional_kwargs["reasoning_content"]。
#   这样 ``LLMAgent._extract_reasoning`` 现有的查找路径就能命中，
#   service 层也无需改动入库逻辑。
#
# 设计取舍：
#   - 不去 monkey-patch langchain 内部函数：影响面不可控、阻碍未来升级。
#   - 不绕开 langchain 直接裸调 openai SDK：会让 prompt / 解析 /
#     重试 / 日志全部要重写，得不偿失。
#   - 不在 chain 上挂一层 RunnableLambda 后处理：拿不到原始 dict，
#     只能拿到已经被 langchain 转换后丢失了 reasoning_content 的 AIMessage。
#   重写 _create_chat_result 是改动面最小、最稳妥的做法。
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
            # 先让父类完成标准转换；它会把 response 序列化成 dict 再走
            # _convert_dict_to_message。我们这里需要的也是同一份 dict，
            # 因此再独立 dump 一次以拿到原始 reasoning_content。
            chat_result = super()._create_chat_result(response, generation_info)

            response_dict = (
                response if isinstance(response, dict) else response.model_dump()
            )
            choices = response_dict.get("choices") or []

            for idx, choice in enumerate(choices):
                if idx >= len(chat_result.generations):
                    break
                message_dict = choice.get("message") or {}
                # DeepSeek 把思维链放在 message.reasoning_content 中；
                # 部分网关/代理也会放在 message.reasoning，这里一并兼容。
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
                             我们只想保留"真正由 LLM 产出"的那一条记录，
                             cache 命中的请求不应再次写入 signals 表，
                             否则 30s 一次的循环会产生大量 reason/risk/
                             suggestion 完全相同、只有 factors 在变的脏数据。
    设计说明：
        之所以单独包一层，是为了不污染 TradingSignal 的 schema
        （它要直接 model_dump 给 API 调用方与数据库 reason/risk/suggestion
        三个字段，不应混入推理过程）。
    """

    signal: TradingSignal
    reasoning_content: Optional[str] = None
    from_cache: bool = False


SYSTEM_PROMPT = """\
你是一名资深加密衍生品量化交易分析师。系统会给你一份多周期因子矩阵
（5m / 15m / 1h / 4h / 1d）、规则引擎初判、爆仓滚动窗口、多周期关键价位列表，
以及（P1 升级新增）：market regime（市场状态）、流动性地图、订单簿时序指标、
散户/精英多空比。

请基于这些信息输出**一个**严格符合给定 schema 的 JSON 对象，作为
完整可执行的交易计划。该信号仅作交易建议，不会被自动下单。

【输出语言要求】
- reason / risk / suggestion 三个字段必须使用**简体中文**。
- bias 字段保持 long / short / neutral 三个英文枚举值之一，**不要翻译**。
- timeframe_alignment 的 value 也保持 long/short/neutral 英文枚举。

【因子使用要点】
- 综合考虑四类因子：资金流（capital_flow）、订单簿（orderbook）、
  衍生品（derivatives）、市场结构（market_structure）。
- 注意：当前没有链上数据与参与者画像，请勿引用任何 on-chain / whale /
  smart money 相关信息，也不要编造。

【多周期共振原则】
- 因子数据按 5m / 15m / 1h / 4h / 1d 五个周期分层提供。
  周期越大权重越高：4h、1d 决定主方向，1h 决定波段，
  15m、5m 决定入场时机。
- 出现冲突时，优先信任高周期，并在 reason 中显式说明你为什么
  舍弃了哪些低周期信号。
- timeframe_alignment 必须把 5 个周期的方向都填上，缺一不可。

【可执行交易计划】
你必须输出完整可执行交易计划：
- entry_zone：**区间，不是单点**，宽度建议 0.2 × ATR(15m) ~ 0.5 × ATR(15m)。
- stop_loss：必须落在结构关键位（HH/HL swing 点 / 4h 支撑或阻力）的另一侧，
  且与 entry 中点的距离 ≥ 1 × ATR(15m)。
- take_profit：至少 2 档，对应附近的对侧关键位 / 流动性池
  （例如：tp1 = 1h 阻力，tp2 = 4h 阻力 / 上一波等量目标）。
- risk_reward_ratio：必须 ≥ 1.5（按 |tp1 − entry_mid| / |entry_mid − sl| 计算）。
- position_size_pct：基于 1% 账户风险预算计算
  （= 0.01 / (|entry_mid - stop_loss| / entry_mid)），
  最大不超过 0.25。
- invalidation_conditions：≥ 2 条**量化**失效条件（必须含具体价位 / 阈值），
  例如："4h 收盘跌破 3500"、"1h CVD slope 转负且持续 30 分钟以上"。

【降级规则】
- 如果不满足 RR ≥ 1.5、或多周期方向严重冲突（alignment_score 绝对值 < 0.4
  且 dominant_bias=neutral），必须输出 neutral，不得硬给方向。
- neutral 时 entry_zone / stop_loss / take_profit / RR / position_size_pct
  全部置 null（schema 强约束）。
- reason 字段务必引用具体数值（净流入 USD、CVD slope、OI 变动百分比、
  funding_rate、爆仓 imbalance、关键价位等），不要写"看起来"、
  "似乎"等模糊表述。
- suggestion 字段为面向用户的简体中文建议段落，文末必须注明
  "仅供参考，不构成交易指令"。

【P1 强约束（必须严格遵守）】
1) 当 regime=ranging 时禁止给 trending 仓位建议：必须降为 neutral，
   或在最近 supports/resistances 之间做"区间策略"，并把
   position_size_pct ≤ 0.05；reason 必须显式提到 "regime=ranging"。
2) 当 regime=breakout 或 regime=breakdown 时，stop_loss 必须放在
   "被突破的结构位另一侧"（breakout → SL 放在被刺破的 swing high 下方少量
   buffer；breakdown → SL 放在被刺破的 swing low 上方）。
3) 当 retail_vs_smart_divergence='bearish_warning' 时（散户狂多 + 精英反向）：
   confidence ≤ 0.6；risk 字段必须明确写出"散户/精英背离风险"。
   bullish_warning 同理处理（不准盲目追多）。
4) 当 funding_extreme='long_squeeze_risk' 且 bias='long' 时，confidence ≤ 0.5；
   反之 'short_squeeze_risk' + bias='short' 同理。
5) 当 liquidity_vacuum_above=true 且 bias='long' 时，take_profit 至少
   一档要落在"上方流动性池中第一档 strong/medium 节点"附近 ±0.3% 范围内；
   vacuum_below + 'short' 同理。

【P2 强约束（自我反馈机制 - 必须严格遵守）】
1) 你将看到自己最近 N 次判断的实际 PnL（来自 signal_lifecycle 表，含
   sl_hit / tp1_hit / tp2_hit / expired 四类终态）。如果近 5 次中有
   ≥ 3 次为 sl_hit 或 expired（即胜率 ≤ 40%），必须显著降低本次 confidence
   到 < 0.5；同时在 reason 字段中明确反思失败模式（例如：
   "近 5 次中 3 次 sl_hit，多发生在 regime=ranging 时强行做趋势"
   或 "近 5 次 4 次 expired，入场区间过窄导致始终未触发"）。
2) 如果上一条信号仍处于 triggered 状态（未结算），新判断方向**不能**
   与其相反。除非有明显的反转证据（例如同时出现 swept_high +
   cvd_price_divergence + 大周期趋势翻转），否则你应输出 neutral
   等待上一条结算，并在 reason 中显式提到 "上一条信号 #<id> 仍持仓中"。
3) 当近 5 次判断的 sample_count < 3（冷启动期）时不应用上述硬约束，
   但 reason 中要注明 "样本不足，未触发自我反馈降权"。
4) 对历史成绩的解释必须客观：不要因为 1 次大额胜利就过度自信，
   也不要因为 1 次最大不利波动就强行翻转方向；优先看胜率 / 平均 RR
   / 最大回撤三者的组合。
"""

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

HUMAN_PROMPT = """\
{self_feedback_block}合约: {symbol}
时间: {ts}

规则引擎: bias={rule_bias} score={rule_score} confidence={rule_confidence}
规则引擎贡献度: {rule_contributions}

===== 多周期因子矩阵 =====
{mtf_factor_table}

===== 多周期共振 =====
{mtf_alignment_text}

===== 衍生品 =====
{derivatives_text}

===== 爆仓（滚动窗口）=====
{liquidations_table}

===== 关键价位（从大周期到小周期）=====
{key_levels_text}

===== 订单簿快照（最新一次，跨周期共享）=====
{orderbook_text}

===== 市场状态 =====
{regime_text}

===== 订单簿动态 =====
{orderbook_dynamic_text}

===== 主力 / 散户 =====
{position_ratios_text}

===== 流动性地图 =====
{liquidity_text}

请输出完整的 TradingSignal JSON，必须包含：
bias / confidence / reason / risk / suggestion / entry_zone /
stop_loss / take_profit（≥2 档）/ risk_reward_ratio / position_size_pct /
timeframe_alignment（5 个周期都填）/ invalidation_conditions（≥2 条）。
"""


class LLMAgent:
    """
    LangChain DeepSeek 调用封装（基于 signals 表的 DB 节流）。
    --------------------------------------------------------------
    设计目标：
    - DeepSeek API 是按 token 收费的，但规则引擎 / 信号循环每 30s
      就会触发一次 ``analyze``。如果每次都真打 LLM，一天会产生上千次
      调用，浪费且没必要——加密市场短期波动并没有这么快。
    - 节流的"上一次判断时间"以 **signals 表里该 symbol 最后一条记录
      的 ts 为准**，而不是进程内存。这样：
        * 进程重启后状态不会丢失，不会出现"重启即重打 LLM"。
        * 多副本横向部署时，所有实例共享同一份"上次调用时间"
          （PostgreSQL 是唯一真源），节流不会被绕过。
        * 调试时手动清空 signals 表，下一轮即触发新一次 LLM 调用。
    - 在 ``settings.llm_min_interval_seconds`` 窗口内（默认 900 秒，
      即 15 分钟），analyze 不会真正发起 LLM 请求；而是直接从
      signals 表读出最近一条 LLM 判断，重建 ``LLMAnalysisResult``
      返回给上层（带 ``from_cache=True`` 标记）。
    - 上层 service 仅在 ``from_cache=False`` 时落库，从而保证：
      入库节奏 = LLM 真实调用节奏 = ``LLM_MIN_INTERVAL_SECONDS`` 节奏。
    - 通过按 symbol 的 ``asyncio.Lock`` 防止同一 symbol 在窗口刚到
      时被并发调用打多次接口（DB 写入与读出之间存在窄竞态窗口）。
    - 设置 ``LLM_MIN_INTERVAL_SECONDS=0`` 可完全关闭节流（调试用）。
    """

    def __init__(self, settings: Settings, repos: Repositories):
        self.settings = settings
        # repos：用于查询 signals 表里"最后一条 LLM 判断的时间戳"以执行
        # DB 节流，以及在节流命中时把判断字段重建成 LLMAnalysisResult。
        self.repos = repos
        self._chain = None  # build lazily
        # 标记当前 chain 是否走"思考模式"（即未使用 function_calling，
        # 模型直接吐 JSON 文本，需要在 analyze() 里手动 parse）。
        self._chain_thinking_mode: bool = False
        # 按 symbol 维度的并发锁，保证窗口刚到时不会被并发调用打多次接口。
        # 注意：跨进程的并发还是要靠"以 DB 时间戳为准"的语义来保证，本锁
        # 只防同一 worker 进程内的高并发竞争。
        self._locks: Dict[str, asyncio.Lock] = {}

    def _build_chain(self):
        """
        构建 LangChain 调用链。
        --------------------------------------------------------------
        关键点：
        - 通过 ChatOpenAI 的 OpenAI 兼容协议调用 DeepSeek。
        - **思考模式（deepseek-reasoner）**：
          * 服务端会把 deepseek-v4-pro 等带思考能力的别名路由到
            ``deepseek-reasoner`` 引擎。
          * **该引擎不支持 ``tool_choice``**（即不支持 function calling），
            如果在这种模式下调用 ``with_structured_output(method="function_calling")``
            会被服务端 400：``deepseek-reasoner does not support this tool_choice``。
          * 因此思考模式下改走"提示词约束 + JSON 解析"路线：
            用 PydanticOutputParser 把 schema 注入 system prompt，模型
            直接以 JSON 文本作为 content 返回，由 ``analyze`` 手动解析。
          * extra_body={"thinking": {"type": "enabled"}} 透传给底层 OpenAI
            client 开启思维链；reasoning_effort 控制思考强度。这两个参数
            ChatOpenAI 都支持作为顶层 kwargs，比塞进 model_kwargs 更干净，
            也能避免 langchain 抛 "should be specified explicitly" 的 UserWarning。
          * 思考模式下 temperature 不生效，因此不再传入。
        - **非思考模式（deepseek-chat）**：
          * 仍然走 ``with_structured_output(method="function_calling")``，
            走 schema 校验路径，准确率最高。

        返回的是已经构建好的 Runnable；同时把"是否思考模式"记到
        ``self._chain_thinking_mode``，供 ``analyze`` 决定如何解析返回值。

        ChatOpenAI 类的选择策略：
            - 思考模式：使用 ``_build_deepseek_chat_openai_class()`` 返回的
              子类，它会把 DeepSeek 的 ``reasoning_content`` 透传到
              AIMessage.additional_kwargs，否则 langchain-openai 会丢字段，
              导致 signals.reasoning_content 永远写入 NULL。
            - 非思考模式：原生 ``ChatOpenAI`` 即可（响应里本就没有该字段）。
        """
        from langchain_openai import ChatOpenAI

        if not self.settings.deepseek_api_key:
            logger.warning(
                "未配置 DEEPSEEK_API_KEY，运行时将跳过 LLM 分析"
            )

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

        if thinking_enabled:
            # 思考模式：deepseek-reasoner 不支持 tool_choice，走 prompt + 手动解析。
            parser = PydanticOutputParser(pydantic_object=TradingSignal)
            system_messages = [
                ("system", SYSTEM_PROMPT),
                ("system", THINKING_OUTPUT_INSTRUCTIONS),
            ]
            prompt = ChatPromptTemplate.from_messages(
                system_messages + [("human", HUMAN_PROMPT)]
            ).partial(format_instructions=parser.get_format_instructions())
            self._chain_thinking_mode = True
            # 直接返回 AIMessage（不挂解析器），analyze() 里负责 reasoning_content
            # 抽取 + JSON 解析；如果在链上挂解析器，解析失败会在 chain 内抛错，
            # 我们就拿不到原始 content 用于排错日志了。
            return prompt | llm

        # 非思考模式：走 function calling，用 langchain 的 structured_output
        # 直接给我们 dict {"raw": AIMessage, "parsed": TradingSignal, ...}。
        structured = llm.with_structured_output(
            TradingSignal, method="function_calling", include_raw=True
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        self._chain_thinking_mode = False
        return prompt | structured

    @property
    def enabled(self) -> bool:
        """
        是否启用 LLM。
        --------------------------------------------------------------
        以是否配置了 DeepSeek API Key 为准；未配置时整个 LLM 链路降级，
        外层 service 会回退到纯规则引擎结果。
        """
        return bool(self.settings.deepseek_api_key)

    @property
    def min_interval(self) -> int:
        """
        LLM 调用最小间隔（秒）—— P0 静态版，作为自适应节流的兜底
        --------------------------------------------------------------
        从 settings 读取，方便单元测试通过 monkeypatch 修改 settings 调整。
        compute_min_interval(factors) 抛任何异常时统一回退到该值。
        """
        return max(0, int(self.settings.llm_min_interval_seconds))

    def compute_min_interval(self, factors: Optional[Dict[str, Any]]) -> int:
        """
        P1 自适应 LLM 节流：根据当前因子矩阵动态计算下一次允许调用的最小间隔
        --------------------------------------------------------------
        参数：
            factors: FactorAggregator.compute 的输出（多周期模式）；
                     None 或老格式时直接走兜底
        返回：
            建议的 min_interval 秒数；失败 / 关闭时回退到 self.min_interval
        规则（与 P1 提示词同源）：
            volatility_ratio = atr_5m × 12 / atr_1h
                （把 5m ATR 折算到 1h 尺度后与 1h ATR 比较，> 1 表示 5 分钟波动异常放大）
            volatility_ratio >= 1.5 或 regime ∈ {breakout, breakdown} → 180s（3 分钟）
            volatility_ratio >= 1.2 或 |alignment_score| >= 0.75      → 600s（10 分钟）
            其他                                                      → 1800s（30 分钟）
        上下限通过 settings.llm_min_interval_seconds_min/max 钳制。
        """
        # 关闭开关：直接回退到 P0 静态节流
        if not bool(getattr(self.settings, "enable_adaptive_throttle", False)):
            return self.min_interval
        try:
            base_default = int(self.min_interval) if self.min_interval > 0 else 1800
            lo = int(getattr(self.settings, "llm_min_interval_seconds_min", 180))
            hi = int(getattr(self.settings, "llm_min_interval_seconds_max", 1800))
            if not factors or not isinstance(factors, dict):
                return self._clamp_interval(base_default, lo, hi)

            by_tf = factors.get("by_timeframe") or {}
            ms_5m = ((by_tf.get("5m") or {}).get("market_structure")) or {}
            ms_1h = ((by_tf.get("1h") or {}).get("market_structure")) or {}
            atr_5m = ms_5m.get("atr_14")
            atr_1h = ms_1h.get("atr_14")
            volatility_ratio: Optional[float] = None
            try:
                if atr_5m is not None and atr_1h is not None and float(atr_1h) > 0:
                    volatility_ratio = float(atr_5m) * 12.0 / float(atr_1h)
            except (TypeError, ValueError):
                volatility_ratio = None

            regime = factors.get("regime")
            mtf = factors.get("mtf_alignment") or {}
            try:
                alignment_score = abs(float(mtf.get("alignment_score") or 0.0))
            except (TypeError, ValueError):
                alignment_score = 0.0

            if (volatility_ratio is not None and volatility_ratio >= 1.5) or regime in (
                "breakout",
                "breakdown",
            ):
                interval = 180
            elif (
                volatility_ratio is not None and volatility_ratio >= 1.2
            ) or alignment_score >= 0.75:
                interval = 600
            else:
                interval = 1800

            clamped = self._clamp_interval(interval, lo, hi)
            logger.debug(
                "自适应 LLM 节流：volatility_ratio=%s regime=%s alignment=%.3f → %ds (clamped to %ds)",
                ("%.3f" % volatility_ratio) if volatility_ratio is not None else "-",
                regime,
                alignment_score,
                interval,
                clamped,
            )
            return clamped
        except Exception:
            logger.warning(
                "compute_min_interval 计算失败，回退到 P0 默认 %ds",
                self.min_interval,
                exc_info=True,
            )
            return self.min_interval

    @staticmethod
    def _clamp_interval(interval: int, lo: int, hi: int) -> int:
        """
        把建议节流秒数钳制到 [lo, hi]
        """
        if lo > hi:
            lo, hi = hi, lo
        return int(max(lo, min(int(interval), hi)))

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        """
        获取或创建指定 symbol 对应的并发锁。
        --------------------------------------------------------------
        每个 symbol 独立加锁，避免多 symbol 之间互相阻塞；同一个 symbol
        即使在缓存 miss 的瞬间被并发调用，也只会真正打一次 LLM。
        """
        lock = self._locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[symbol] = lock
        return lock

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
            symbol:       合约代码
            min_interval: P1 自适应节流窗口（秒）。None 时回退到 self.min_interval。
        步骤：
            1) ``effective_interval <= 0``：节流关闭，直接返回 None（每次都真调）。
            2) 查 signals 表里该 symbol 最近一条记录的 ts；
               表为空则视为未节流，让上层去打 LLM。
            3) 计算 ``now - ts``；
               - ``< effective_interval``：节流命中，把那一行的 bias / confidence
                 / reason / risk / suggestion / reasoning_content 重建成
                 LLMAnalysisResult，并把 ``from_cache`` 置 True 返回。
                 service 层据此跳过本次入库，避免重复行。
               - ``>= effective_interval``：节流过期，返回 None。
        异常处理：
            DB 查询失败时（如连接被 reset），不应阻塞 LLM 调用；记一行
            warning 后返回 None，让上层退化为"直接打 LLM"——成本上界仍是
            effective_interval，最差也只是少一次节流命中。
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
        # 兼容部分驱动 / 历史数据返回 naive datetime 的情形：把它当作 UTC。
        # signals.ts 的 schema 是 TIMESTAMPTZ，正常情况下 asyncpg 会带 tzinfo。
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
        if elapsed >= effective_interval:
            return None

        try:
            # 缓存重建只为了"跳过本轮 LLM 调用"，本身不会影响下游交易计划
            # （结构化字段 entry_zone / SL / TP 等在节流命中时不会被透出执行）。
            # P0 升级后 model_validator 对 long/short 强约束 entry_zone/SL/TP 必填，
            # 但历史 signals 行可能没有这些列；为了不破坏节流通路，缓存重建时
            # 一律把 bias 当作"展示用 bias"，并强制走 neutral 路径让校验通过：
            #   - 如果上层只读 bias / confidence 用于日志，结果是一致的；
            #   - 真正的执行计划只来自"真正调用 LLM 的那次"，缓存命中本来就不入库。
            cached_signal = TradingSignal(
                bias="neutral",
                confidence=float(row["confidence"]),
                reason=row.get("reason") or "",
                risk=row.get("risk") or "",
                suggestion=row.get("suggestion") or "",
            )
            # 然后把真实 bias 透出来供日志展示（model_validator 已通过，
            # 这里直接 setattr 不会再触发约束）。
            object.__setattr__(cached_signal, "bias", row["bias"])
        except Exception:
            # 历史脏数据 / schema 不匹配时，不强行卡死节流通道；记日志后
            # 当作没有缓存处理，让本轮去真打一次 LLM 重建判断。
            logger.warning(
                "无法从 signals 表行重建 %s 的 TradingSignal；"
                "将按未命中缓存重新调用 LLM",
                symbol,
                exc_info=True,
            )
            return None

        remaining = effective_interval - elapsed
        logger.info(
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
    # Prompt 构造（多周期因子矩阵 → 紧凑中文表格）
    # ------------------------------------------------------------------
    def _build_prompt_inputs(
        self,
        symbol: str,
        factors: Dict[str, Any],
        rule_signal: TradingSignal,
        rule_score: float,
        rule_contributions: Dict[str, float],
        recent_settled: Optional[List[Dict[str, Any]]] = None,
        open_lifecycle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        把多周期因子矩阵渲染成紧凑中文表格，作为 ChatPromptTemplate 输入
        ---------------------------------------------------------------
        参数：
            symbol / factors / rule_signal / rule_score / rule_contributions:
                同 analyze() 形参
        返回：
            ChatPromptTemplate.format 所需的全部 key-value 字典。
        说明：
            - 多周期模式下逐周期渲染表格行；缺失字段统一显示 '-'。
            - 老格式（无 by_timeframe）下，自动降级为单行 / 单段表格，
              保证回滚通道 LLM 仍然能拿到结构化输入。
        """
        is_mtf = "by_timeframe" in factors

        ts = factors.get("computed_at") or ""

        if is_mtf:
            mtf_factor_table = self._render_mtf_factor_table(factors)
            mtf_alignment_text = self._render_alignment(factors)
            derivatives_text = self._render_derivatives_text(factors)
            liquidations_table = self._render_liquidations_table(factors)
            key_levels_text = self._render_key_levels(factors)
            orderbook_text = self._render_orderbook(factors, mtf=True)
            regime_text = self._render_regime(factors)
            orderbook_dynamic_text = self._render_orderbook_dynamic(factors)
            position_ratios_text = self._render_position_ratios(factors)
            liquidity_text = self._render_liquidity(factors)
        else:
            mtf_factor_table = self._render_legacy_factor_block(factors)
            mtf_alignment_text = "（老聚合器模式：未提供多周期共振）"
            derivatives_text = self._render_legacy_derivatives(factors)
            liquidations_table = "（老聚合器模式：未提供爆仓窗口）"
            key_levels_text = self._render_legacy_key_levels(factors)
            orderbook_text = self._render_orderbook(factors, mtf=False)
            regime_text = "regime: -（老聚合器模式不提供 regime）"
            orderbook_dynamic_text = "（老聚合器模式：未提供订单簿时序）"
            position_ratios_text = "（老聚合器模式：未提供散户/精英多空比）"
            liquidity_text = "（老聚合器模式：未提供流动性地图）"

        # P2：自我反馈段（注入到 HUMAN_PROMPT 的最前面）
        # ----------------------------------------------------------
        # 只有 enable_llm_self_feedback=True 且 recent_settled / open_lifecycle
        # 至少一项非空时，才渲染该段；否则给空字符串占位（不破坏模板）。
        if bool(getattr(self.settings, "enable_llm_self_feedback", False)):
            self_feedback_block = self._render_self_feedback(
                symbol=symbol,
                recent_settled=recent_settled or [],
                open_lifecycle=open_lifecycle,
            )
        else:
            self_feedback_block = ""

        return {
            "symbol": symbol,
            "ts": ts,
            "rule_bias": rule_signal.bias,
            "rule_confidence": round(float(rule_signal.confidence), 4),
            "rule_score": round(rule_score, 4),
            "rule_contributions": json.dumps(rule_contributions, ensure_ascii=False),
            "mtf_factor_table": mtf_factor_table,
            "mtf_alignment_text": mtf_alignment_text,
            "derivatives_text": derivatives_text,
            "liquidations_table": liquidations_table,
            "key_levels_text": key_levels_text,
            "orderbook_text": orderbook_text,
            "regime_text": regime_text,
            "orderbook_dynamic_text": orderbook_dynamic_text,
            "position_ratios_text": position_ratios_text,
            "liquidity_text": liquidity_text,
            "self_feedback_block": self_feedback_block,
        }

    @classmethod
    def _render_self_feedback(
        cls,
        symbol: str,
        recent_settled: List[Dict[str, Any]],
        open_lifecycle: Optional[Dict[str, Any]],
    ) -> str:
        """
        渲染"近 5 次成绩单 + 未结算持仓"段（P2 自我反馈）
        ---------------------------------------------------------------
        参数：
            symbol           : 合约
            recent_settled   : 近 N 条已结算 lifecycle（含 join 出来的 signals 字段）
            open_lifecycle   : 当前未结算的最近一条；可选
        返回：
            紧凑中文表格（含统计摘要 + 未结算告警）；总长度严格控制在
            ~1500 token 以内（用 markdown 表格而非 JSON），避免反复推高
            prompt 成本（详见模块顶部 P2 强约束）。
        """
        lines: List[str] = []
        lines.append(f"===== 你最近 {len(recent_settled)} 次判断的实际成绩（{symbol}）=====")

        if not recent_settled:
            lines.append("（样本不足：lifecycle 表里还没有该 symbol 的已结算记录）")
        else:
            lines.append(
                "| ts | bias | conf | regime | 入场区间 | 触发价 | 退出原因 | PnL% | 最大有利% | 最大不利% |"
            )
            lines.append(
                "|----|------|------|--------|----------|--------|----------|------|-----------|-----------|"
            )
            wins = 0
            losses = 0
            expired_cnt = 0
            pnl_acc: List[float] = []
            for row in recent_settled:
                ts = row.get("signal_ts")
                ts_str = ts.strftime("%m-%d %H:%M") if hasattr(ts, "strftime") else "-"
                bias = row.get("bias") or "-"
                conf = row.get("confidence")
                conf_str = f"{float(conf):.2f}" if conf is not None else "-"
                regime = cls._extract_regime_from_factors(row.get("factors"))
                ez_low = row.get("entry_zone_low")
                ez_high = row.get("entry_zone_high")
                ez_str = (
                    f"[{float(ez_low):.2f},{float(ez_high):.2f}]"
                    if ez_low is not None and ez_high is not None
                    else "-"
                )
                trig = row.get("triggered_price")
                trig_str = f"{float(trig):.2f}" if trig is not None else "-"
                status = row.get("status") or "-"
                pnl = row.get("pnl_pct")
                pnl_str = f"{float(pnl) * 100:+.2f}" if pnl is not None else "-"
                mfp = row.get("max_favorable_pct")
                mfp_str = f"{float(mfp) * 100:+.2f}" if mfp is not None else "-"
                map_ = row.get("max_adverse_pct")
                map_str = f"{float(map_) * 100:+.2f}" if map_ is not None else "-"
                lines.append(
                    f"| {ts_str} | {bias} | {conf_str} | {regime} | {ez_str} | "
                    f"{trig_str} | {status} | {pnl_str} | {mfp_str} | {map_str} |"
                )
                if status in ("tp1_hit", "tp2_hit"):
                    wins += 1
                elif status == "sl_hit":
                    losses += 1
                elif status == "expired":
                    expired_cnt += 1
                if pnl is not None:
                    pnl_acc.append(float(pnl))
            total = len(recent_settled)
            win_rate = wins / total if total else 0.0
            avg_pnl = sum(pnl_acc) / len(pnl_acc) if pnl_acc else 0.0
            lines.append(
                f"统计：胜率={win_rate:.0%}（{wins}赢/{losses}输/{expired_cnt}超时） "
                f"平均PnL={avg_pnl * 100:+.2f}%"
            )

        # 未结算持仓提醒（P2 强约束 #2）
        if open_lifecycle:
            sid = open_lifecycle.get("signal_id")
            ob = open_lifecycle.get("bias")
            ostat = open_lifecycle.get("status")
            otrig = open_lifecycle.get("triggered_price")
            otrig_str = f"@{float(otrig):.2f}" if otrig is not None else ""
            lines.append(
                f"⚠️ 上一条信号 #{sid} ({ob}, {ostat}{otrig_str}) **仍未结算**：除非你有明确反转证据，"
                "否则应保持同向或输出 neutral 等待结算。"
            )
        lines.append("")  # 空行作为段落分隔
        return "\n".join(lines) + "\n"

    @staticmethod
    def _extract_regime_from_factors(factors_blob: Any) -> str:
        """
        从 signals.factors JSONB 中尽力取出 regime 字段（成绩单展示用）
        """
        if not isinstance(factors_blob, dict):
            return "-"
        inner = factors_blob.get("factors") if "factors" in factors_blob else factors_blob
        if isinstance(inner, dict):
            r = inner.get("regime")
            if isinstance(r, str) and r:
                return r
        return "-"

    @staticmethod
    def _fmt(value: Any, digits: int = 4) -> str:
        """
        统一的数值格式化：None -> '-'；float 保留指定位数；其它走 str
        ---------------------------------------------------------------
        参数：
            value:  原始值
            digits: float 保留的小数位
        返回：
            渲染后的字符串。
        """
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int,)):
            return str(value)
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _render_mtf_factor_table(cls, factors: Dict[str, Any]) -> str:
        """
        渲染多周期"主因子表"（Markdown 风格 ASCII 表）
        ---------------------------------------------------------------
        列：周期 / trend / net_flow_usd / cvd_slope / taker_buy% /
            oi_change% / oi_price / atr_14 / last_close
        """
        by_tf = factors.get("by_timeframe") or {}
        header = (
            "| 周期 | trend | net_flow_usd | cvd_slope | taker_buy% | "
            "oi_change% | oi_price | atr_14 | last_close |"
        )
        sep = "|------|-------|--------------|-----------|------------|-----------|----------|--------|------------|"
        rows = [header, sep]
        for tf in ("5m", "15m", "1h", "4h", "1d"):
            block = by_tf.get(tf) or {}
            cap = block.get("capital_flow") or {}
            deriv = block.get("derivatives") or {}
            struct = block.get("market_structure") or {}
            taker = cap.get("taker_buy_ratio")
            taker_pct = (
                f"{taker * 100:.2f}%" if isinstance(taker, (int, float)) else "-"
            )
            oi_chg = deriv.get("oi_change_pct")
            oi_pct = (
                f"{oi_chg * 100:+.3f}%"
                if isinstance(oi_chg, (int, float))
                else "-"
            )
            rows.append(
                f"| {tf:<4} | {struct.get('trend') or '-':<7} | "
                f"{cls._fmt(cap.get('net_flow_usd'), 0):>12} | "
                f"{cls._fmt(cap.get('cvd_slope'), 6):>9} | "
                f"{taker_pct:>10} | "
                f"{oi_pct:>9} | "
                f"{deriv.get('oi_price_relation') or '-':<9} | "
                f"{cls._fmt(struct.get('atr_14'), 4):>6} | "
                f"{cls._fmt(struct.get('last_close'), 4):>10} |"
            )
        return "\n".join(rows)

    @classmethod
    def _render_alignment(cls, factors: Dict[str, Any]) -> str:
        """
        渲染多周期共振指标
        """
        mtf = factors.get("mtf_alignment") or {}
        votes = mtf.get("trend_votes") or {}
        return (
            f"trend_votes={json.dumps(votes, ensure_ascii=False)}  "
            f"alignment_score={cls._fmt(mtf.get('alignment_score'), 3)}  "
            f"dominant_bias={mtf.get('dominant_bias') or '-'}"
        )

    @classmethod
    def _render_derivatives_text(cls, factors: Dict[str, Any]) -> str:
        """
        渲染衍生品概要（funding 全周期共享，因此抽 1h block 即可）
        """
        by_tf = factors.get("by_timeframe") or {}
        deriv = ((by_tf.get("1h") or {}).get("derivatives")) or {}
        funding = deriv.get("funding_rate_now")
        next_settlement = deriv.get("next_settlement_at") or "-"
        return (
            f"funding_rate={cls._fmt(funding, 8)}（最新一次结算时间 {next_settlement}）"
        )

    @classmethod
    def _render_liquidations_table(cls, factors: Dict[str, Any]) -> str:
        """
        渲染爆仓滚动窗口表
        """
        liq = factors.get("liquidations") or {}
        header = "| 窗口 | long_liq(ETH) | short_liq(ETH) | imbalance | cascade |"
        sep = "|------|----------------|-----------------|-----------|---------|"
        rows = [header, sep]
        for w in (5, 15, 60):
            rows.append(
                f"| {w}m | "
                f"{cls._fmt(liq.get(f'long_{w}m'), 4):>14} | "
                f"{cls._fmt(liq.get(f'short_{w}m'), 4):>15} | "
                f"{cls._fmt(liq.get(f'imbalance_{w}m'), 3):>9} | "
                f"{cls._fmt(liq.get('cascade_signal'))} |"
            )
        rows.append(
            f"last_minute_size={cls._fmt(liq.get('last_minute_size'), 4)} "
            f"last_hour_avg_per_min={cls._fmt(liq.get('last_hour_avg_per_min'), 4)}"
        )
        return "\n".join(rows)

    @classmethod
    def _render_key_levels(cls, factors: Dict[str, Any]) -> str:
        """
        渲染从大周期到小周期的 supports / resistances（4h / 1h / 15m）
        """
        by_tf = factors.get("by_timeframe") or {}
        lines = []
        for tf in ("4h", "1h", "15m"):
            block = by_tf.get(tf) or {}
            struct = block.get("market_structure") or {}
            sup = struct.get("supports") or []
            res = struct.get("resistances") or []
            lines.append(
                f"{tf}: supports={sup}  resistances={res}  "
                f"last_close={struct.get('last_close')}"
            )
        return "\n".join(lines)

    @classmethod
    def _render_orderbook(cls, factors: Dict[str, Any], mtf: bool) -> str:
        """
        渲染最新订单簿快照（多周期模式下取 5m 那份；老模式下取顶层）
        """
        if mtf:
            block = (factors.get("by_timeframe") or {}).get("5m") or {}
            ob = block.get("orderbook") or {}
        else:
            ob = factors.get("orderbook") or {}
        if not ob.get("available"):
            return "（订单簿快照不可用）"
        return (
            f"best_bid={ob.get('best_bid')}  best_ask={ob.get('best_ask')}  "
            f"spread={ob.get('spread')}  imbalance={ob.get('imbalance')}  "
            f"bid_walls={ob.get('bid_walls')}  ask_walls={ob.get('ask_walls')}"
        )

    @classmethod
    def _render_legacy_factor_block(cls, factors: Dict[str, Any]) -> str:
        """
        老聚合器模式下渲染单段因子摘要（保留回滚通道）
        """
        cap = factors.get("capital_flow") or {}
        struct = factors.get("market_structure") or {}
        return (
            f"capital_flow={json.dumps(cap, ensure_ascii=False, default=str)}\n"
            f"market_structure={json.dumps(struct, ensure_ascii=False, default=str)}"
        )

    @classmethod
    def _render_legacy_derivatives(cls, factors: Dict[str, Any]) -> str:
        """
        老聚合器模式下渲染衍生品摘要
        """
        deriv = factors.get("derivatives") or {}
        return json.dumps(deriv, ensure_ascii=False, default=str)

    @classmethod
    def _render_legacy_key_levels(cls, factors: Dict[str, Any]) -> str:
        """
        老聚合器模式下渲染单一窗口的关键价位
        """
        struct = factors.get("market_structure") or {}
        return (
            f"supports={struct.get('supports')}  "
            f"resistances={struct.get('resistances')}  "
            f"last_price={struct.get('last_price')}"
        )

    @classmethod
    def _render_regime(cls, factors: Dict[str, Any]) -> str:
        """
        渲染 regime 段：当前市场状态 + 决定性指标（1h_adx_14 / 4h_bb_width）
        """
        regime = factors.get("regime") or "-"
        by_tf = factors.get("by_timeframe") or {}
        ms_1h = ((by_tf.get("1h") or {}).get("market_structure")) or {}
        ms_4h = ((by_tf.get("4h") or {}).get("market_structure")) or {}
        return (
            f"regime: {regime}   "
            f"1h_adx_14={cls._fmt(ms_1h.get('adx_14'), 2)}   "
            f"4h_bb_width={cls._fmt(ms_4h.get('bb_width'), 6)}   "
            f"4h_trend={ms_4h.get('trend') or '-'}"
        )

    @classmethod
    def _render_orderbook_dynamic(cls, factors: Dict[str, Any]) -> str:
        """
        渲染订单簿时序段：取 5m 周期块（订单簿因子主要挂在 5m / 15m）
        """
        by_tf = factors.get("by_timeframe") or {}
        ob = ((by_tf.get("5m") or {}).get("orderbook")) or {}
        if not ob.get("available"):
            return "（订单簿时序不可用）"
        return (
            f"imbalance_now={cls._fmt(ob.get('imbalance_now'), 4)}  "
            f"slope_5m={cls._fmt(ob.get('imbalance_slope_5m'), 8)}  "
            f"zscore_15m={cls._fmt(ob.get('imbalance_zscore_15m'), 3)}\n"
            f"vacuum_above={cls._fmt(ob.get('liquidity_vacuum_above'))}  "
            f"vacuum_below={cls._fmt(ob.get('liquidity_vacuum_below'))}\n"
            f"bid_wall_persistence_avg_s={cls._fmt(ob.get('bid_wall_persistence_avg_s'), 1)}  "
            f"ask_wall_persistence_avg_s={cls._fmt(ob.get('ask_wall_persistence_avg_s'), 1)}\n"
            f"wall_distance_pct={ob.get('wall_distance_pct')}  "
            f"spread_bp_now={cls._fmt(ob.get('spread_bp_now'), 2)}"
        )

    @classmethod
    def _render_position_ratios(cls, factors: Dict[str, Any]) -> str:
        """
        渲染散户/精英多空比段
        """
        pr = factors.get("position_ratios") or {}
        return (
            f"account_long_short_ratio={cls._fmt(pr.get('account_long_short_ratio'), 4)}（散户币种，反指）\n"
            f"account_contract_ratio={cls._fmt(pr.get('account_contract_ratio'), 4)}（散户合约，反指）\n"
            f"top_trader_position_ratio={cls._fmt(pr.get('top_trader_position_ratio'), 4)}（精英持仓，顺指）\n"
            f"divergence={pr.get('retail_vs_smart_divergence') or 'unknown'}"
        )

    @classmethod
    def _render_liquidity(cls, factors: Dict[str, Any]) -> str:
        """
        渲染流动性地图段：上下方各取前 3 档（按 strength 排序）
        """
        liq = factors.get("liquidity") or {}
        above = (liq.get("liquidity_pool_above") or [])[:3]
        below = (liq.get("liquidity_pool_below") or [])[:3]
        cur = liq.get("current_price")
        return (
            f"current_price={cur}\n"
            f"上方止损池: {above}  nearest_above_pct={liq.get('nearest_above_pct')}\n"
            f"下方止损池: {below}  nearest_below_pct={liq.get('nearest_below_pct')}"
        )

    @staticmethod
    def _extract_reasoning(raw_message: Any) -> Optional[str]:
        """
        从 LangChain AIMessage / dict 中抽取 DeepSeek 思考模式的思维链原文
        --------------------------------------------------------------
        说明：
            DeepSeek（OpenAI 兼容协议）在思考模式下，会把思维链放在响应
            消息的 ``reasoning_content`` 字段里。``langchain-openai`` 的
            ChatOpenAI 在解析时，会把这个字段透传到 AIMessage 的
            ``additional_kwargs["reasoning_content"]``。
            部分版本也可能把它放到 ``response_metadata`` 里，这里同时兼容。
        参数：
            raw_message: with_structured_output(include_raw=True) 返回的
                         "raw" 字段（可能是 AIMessage 实例，也可能是 dict）
        返回：
            思维链字符串；未拿到时返回 None。
        """
        if raw_message is None:
            return None

        candidates: list[Any] = []

        # AIMessage 实例
        additional = getattr(raw_message, "additional_kwargs", None)
        if isinstance(additional, dict):
            # 优先用 reasoning_content（DeepSeek 原生字段名 / 我们子类透传名）；
            # 兜底再看 reasoning（部分网关/未来 langchain 升级后的别名）。
            candidates.append(additional.get("reasoning_content"))
            candidates.append(additional.get("reasoning"))
        response_meta = getattr(raw_message, "response_metadata", None)
        if isinstance(response_meta, dict):
            candidates.append(response_meta.get("reasoning_content"))
            candidates.append(response_meta.get("reasoning"))

        # dict 形态（旧版本或测试桩）
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
        ``}`` 之间的子串重试。仍失败则记 ERROR 并返回 None，外层降级到规则引擎。
        """
        if isinstance(content, list):
            # OpenAI 多模态格式：content 可能是 [{"type":"text","text":"..."}]
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
        # 去掉 ```json ... ``` / ``` ... ``` 代码围栏
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

    async def analyze(
        self,
        symbol: str,
        factors: Dict[str, Any],
        rule_signal: TradingSignal,
        rule_score: float,
        rule_contributions: Dict[str, float],
        recent_settled: Optional[List[Dict[str, Any]]] = None,
        open_lifecycle: Optional[Dict[str, Any]] = None,
    ) -> Optional[LLMAnalysisResult]:
        """
        执行一次 LLM 分析，带按 symbol 的节流缓存。
        --------------------------------------------------------------
        参数：
            symbol             ：合约代码，缓存与并发锁的隔离粒度
            factors            ：因子聚合结果（JSON 可序列化字典）
            rule_signal        ：规则引擎的初判 TradingSignal
            rule_score         ：规则引擎打分（[-1, 1] 区间）
            rule_contributions ：各因子对规则打分的贡献度
        返回：
            LLMAnalysisResult  ：包含 TradingSignal 与 reasoning_content
                                 （来自缓存时同样会带回首次调用的思维链原文）
            None               ：LLM 未启用 / 链构建失败 / 调用失败 / schema 校验失败
        节流策略：
            "上一次判断时间"取自 signals 表里该 symbol 最后一条记录的 ts。
            同一 symbol 在 ``min_interval`` 秒内的请求会直接把那条记录的
            判断字段重建成 LLMAnalysisResult 返回（``from_cache=True``），
            不会真正发起 API 调用，也不会再次落库。
        """

        if not self.enabled:
            logger.info("LLM 已禁用（未配置 DEEPSEEK_API_KEY），将使用规则引擎结果")
            return None

        # P1 自适应节流：每轮根据当前因子矩阵动态计算 min_interval；
        # 失败时方法内部已自动回退到 self.min_interval（P0 行为）。
        adaptive_interval = self.compute_min_interval(factors)

        # 快速路径：先在锁外查一次 DB，避免抢锁。
        cached = await self._load_recent_judgment(
            symbol, min_interval=adaptive_interval
        )
        if cached is not None:
            return cached

        # 慢速路径：拿锁后再查一次（double-checked locking），
        # 防止同一 symbol 在节流刚过期的瞬间被并发调用打多次接口。
        async with self._get_lock(symbol):
            cached = await self._load_recent_judgment(
                symbol, min_interval=adaptive_interval
            )
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
                    "LLM 发起调用 -> %s（自适应最小间隔=%ss，P0 默认=%ss）",
                    symbol,
                    adaptive_interval,
                    self.min_interval,
                )
                prompt_inputs = self._build_prompt_inputs(
                    symbol=symbol,
                    factors=factors,
                    rule_signal=rule_signal,
                    rule_score=rule_score,
                    rule_contributions=rule_contributions,
                    recent_settled=recent_settled,
                    open_lifecycle=open_lifecycle,
                )
                result = await self._chain.ainvoke(prompt_inputs)
            except Exception:
                logger.exception("LangChain 调用失败，回退到规则引擎")
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
                # 思考模式 AIMessage：手动从 content 抽 JSON。
                raw_message = result
                content = getattr(result, "content", "")
                parsed = self._parse_json_content(content, symbol)
                if parsed is None:
                    return None
            else:
                parsed = result

            if parsing_error is not None:
                logger.error(
                    "LLM 结构化解析错误 %s：%s", symbol, parsing_error
                )
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
                logger.error(
                    "LLM 解析结果类型异常 %s：%r", symbol, type(parsed)
                )
                return None

            reasoning_content = self._extract_reasoning(raw_message)
            if reasoning_content:
                logger.debug(
                    "已捕获 LLM 思维链 %s（%d 字符）",
                    symbol,
                    len(reasoning_content),
                )
            elif self.settings.deepseek_thinking_enabled:
                # 思考模式下仍未拿到 reasoning_content，多半是 LangChain
                # 没把该字段透传过来；不致命，但记一行日志便于排查。
                logger.debug(
                    "思考模式已启用，但 %s 的原始消息中未找到 reasoning_content",
                    symbol,
                )

            # from_cache=False 标记本次是真正发起了一次 LLM 调用，service
            # 层会据此把这条结果写入 signals 表——这条 INSERT 同时也是下一
            # 轮 _load_recent_judgment 的"上次调用时间戳"来源，构成隐式缓存。
            return LLMAnalysisResult(
                signal=signal,
                reasoning_content=reasoning_content,
                from_cache=False,
            )
