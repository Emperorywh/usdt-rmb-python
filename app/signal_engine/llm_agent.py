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


def _to_float_safe(v: Any) -> Optional[float]:
    """
    把任意输入转 float，失败 / NaN / Inf 一律返回 None
    --------------------------------------------------------------
    P3 评估段渲染时大量用到（asyncpg 的 Decimal、None、字符串等都可能
    出现），统一在模块顶部声明避免在多个 classmethod 内部重复实现。
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math as _math
    if not _math.isfinite(f):
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
散户/精英多空比；（P2 升级新增）你最近 N 次判断的成绩单；（P3 升级新增）
近 24h 系统级评估指标（方向翻转率 / 胜率 / Sharpe / Brier）。

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

【可执行交易计划】（P3 升级：SL / RR / size 全部抬高门槛）
你必须输出完整可执行交易计划：
- entry_zone：**区间，不是单点**，宽度建议 0.2 × ATR(15m) ~ 0.5 × ATR(15m)。
- stop_loss：必须落在结构关键位（HH/HL swing 点 / 4h 支撑或阻力）的另一侧，
  并同时满足以下两条最小距离约束（取较大者）：
    a) |entry_mid - stop_loss| ≥ **1.5 × ATR(15m)**（旧值 1×ATR，已上调）；
    b) |entry_mid - stop_loss| / entry_mid ≥ **0.5%**（绝对百分比下限）。
  动机：ETH 永续 1m 随机噪声常见 0.05–0.15%，0.2–0.3% 的 SL 是高频陷阱。
- take_profit：至少 2 档，对应附近的对侧关键位 / 流动性池
  （例如：tp1 = 1h 阻力，tp2 = 4h 阻力 / 上一波等量目标）。
- risk_reward_ratio：必须 ≥ **2.0**（旧值 1.5，已上调）。
  按 |tp1 − entry_mid| / |entry_mid − sl| 计算。
- position_size_pct：建议值上限 **0.10**（旧 cap 0.25，已下调）。
  请基于 1% 账户风险预算计算 = 0.01 / (|entry_mid - stop_loss| / entry_mid)，
  再按 conf 缩放（conf ≤ 0.5 给 0；conf ∈ (0.5, 0.7] 上限 0.05；
  conf > 0.7 上限 0.10）。
  注意：服务端会基于历史胜率 + 半凯利对该值再做最终 clamp，
  你的输出只是"上限建议"，不是最终下单仓位。
- invalidation_conditions：≥ 2 条**量化**失效条件（必须含具体价位 / 阈值），
  例如："4h 收盘跌破 3500"、"1h CVD slope 转负且持续 30 分钟以上"。

【降级规则】
- 当 RR < 2.0 时，禁止给出趋势性 long/short 方向；可以走"区间策略"
  （见下方），或在区间策略也不成立时输出 neutral。
- 多周期方向严重冲突（alignment_score 绝对值 < 0.4 且 dominant_bias=neutral）
  时，**不再强制 neutral**。优先级如下：
    1) 必须先按下方【区间策略检查清单】逐项核对 5m **与** 15m 两个周期是否
       存在可交易区间；任一周期检查通过即允许走"区间策略"，并按区间策略
       约束输出。
    2) 若两个周期检查清单都不通过，才输出 neutral，并在 reason 中明确说明
       "区间不可交易：<5m 失败原因> / <15m 失败原因>"（必须分别给出，
       不允许笼统写"无区间"）。

【区间策略检查清单（强制结构化校验，禁止跳过任何一项）】
当满足"alignment_score 绝对值 < 0.4 且 dominant_bias=neutral"或
"regime ∈ {{ranging, transitional}}"时，你必须在 reason 中**按下表逐行**
列出 5m 与 15m 两个周期的检查结果，缺一不可（缺数据的字段写"N/A"）：

| 周期 | sup[0] | res[0] | last_close | 闸 A: sup[0]<close<res[0] | 区间宽度 W | 0.6×ATR(15m) | 闸 B: W ≥ 0.6×ATR(15m) | 距离上沿 | 距离下沿 | 闸 C: 哪一侧更近 |
|------|--------|--------|------------|---------------------------|------------|---------------|------------------------|----------|----------|------------------|
| 5m   |        |        |            |                           |            |               |                        |          |          |                  |
| 15m  |        |        |            |                           |            |               |                        |          |          |                  |

判定规则：
- **闸 A 与 闸 B 同时通过**的周期 = 该周期"区间可交易"，可作为区间策略的
  边界来源；如果 5m / 15m 都通过，**优先选 15m 作为边界来源**（更稳定），
  仅当 15m 缺数据时才回退到 5m。
- 闸 A 不通过的常见原因：res[0] 缺失（resistances=[]）/ sup[0] 缺失 /
  last_close 已经位于区间外（sweep 已发生）。直接写出哪一项缺。
- 闸 B 不通过的常见原因：区间太窄。即使 ATR(15m) 缺失，也必须用
  ATR(5m) × √3 估算 ATR(15m) 后再比较，**禁止以"ATR(15m) 缺失"为由
  跳过该闸**。
- **边界来源白名单**：sup[0] / res[0] 必须取自因子矩阵中
  by_timeframe.<tf>.market_structure.supports / resistances 列表的首元素。
  **严禁**使用 liquidity_pool_above / liquidity_pool_below 中的
  round_level / weak 节点作为区间边界（流动性池仅可作为 take_profit[1]
  的辅助参考，不能用来替代结构关键位）。

如果"5m 通过"或"15m 通过"任一成立，按下列约束输出区间策略：
  * bias 取"距离更近"的那一侧（贴近 supports → bias=long 做多反弹；
    贴近 resistances → bias=short 做空回踩）；如果价格几乎在区间正中
    且没有明显单边动能（CVD slope / 资金流 / 订单簿失衡都不指向同侧），
    才允许输出 neutral。
  * **禁止贴边**：entry_zone 必须距对侧边界 ≥ 0.4 × 区间宽度
    （避免"贴在支撑一侧做多 → 1 根插针就到 SL"的结构性陷阱）。
  * stop_loss 必须落在该边界另一侧 ≥ 1.5 × ATR(15m) buffer 之外，
    同时仍满足"|entry_mid - SL| / entry_mid ≥ 0.5%"。
  * take_profit[0] 取区间另一侧关键位附近（≥ 0.7 倍区间宽度），
    take_profit[1] 取区间另一侧 + 一档流动性池节点；
  * RR 仍必须 ≥ 2.0；做不到就退回 neutral，并在 reason 中显式写出
    "区间策略 RR=<计算值> < 2.0，退回 neutral"，**不要再写"区间不清晰"**——
    这两种失败原因不同，混淆会让自我反馈机制无法识别真正瓶颈。
  * **position_size_pct 必须 ≤ 0.05**（强制小仓位）；
  * reason 必须显式写出"区间策略"四个字 + 选用了哪个周期（5m/15m）的
    边界 + 区间上下沿的具体价位 + 为什么不做趋势单的核心因子证据。
- neutral 时 entry_zone / stop_loss / take_profit / RR / position_size_pct
  全部置 null（schema 强约束）。
- reason 字段务必引用具体数值（净流入 USD、CVD slope、OI 变动百分比、
  funding_rate、爆仓 imbalance、关键价位等），不要写"看起来"、
  "似乎"等模糊表述。
- suggestion 字段为面向用户的简体中文建议段落，文末必须注明
  "仅供参考，不构成交易指令"。

【区间策略示例 A：检查清单通过 + RR 不达标 → 退回 neutral（参考写法，不要照抄数值）】
当 regime=transitional、当前价 = 2330.72、
5m supports=[2321.8, 2318.88, 2314.3]、5m resistances=[2331.3, 2385.67]、
15m supports=[2314.3]、15m resistances=[]、
5m ATR(14)=5.305（→ ATR(15m) 估算 ≈ 5.305 × √3 ≈ 9.19）时，
reason 中必须先写出检查清单（示意，省略表格框）：
  - 5m: sup[0]=2321.8, res[0]=2331.3, close=2330.72,
        闸A 2321.8<2330.72<2331.3 ✅, W=9.5, 0.6×ATR(15m)≈5.51,
        闸B 9.5≥5.51 ✅, 距上沿 0.58, 距下沿 8.92, 闸C → 贴近上沿
  - 15m: sup[0]=2314.3, res[0]=N/A, 闸A ❌（resistances 为空）→ 该周期不可用
  - 边界来源选 5m，bias=short（贴近上沿做回踩）
然后按区间策略约束算 RR：
  - 区间宽度 9.5 × 0.4 = 3.8 → EZ 必须距下沿 ≥ 3.8 + sup[0]=2321.8,
    即 EZ 下沿 ≥ 2325.6；实际取 EZ ≈ [2329.0, 2331.0]，中点 2330.0；
  - stop_loss ≥ res[0] + 1.5 × ATR(15m) = 2331.3 + 13.79 ≈ 2345.1，
    同时 |2330 - 2345.1| / 2330 = 0.65% > 0.5% ✅，取 SL = 2345.1；
  - take_profit[0] 取下沿附近 ≈ 2321.8（≥ 0.7 × 宽度 = 6.65 ✅）；
  - RR = |2321.8 - 2330| / |2330 - 2345.1| = 8.2 / 15.1 ≈ 0.54。
  - **RR=0.54 < 2.0 → 退回 neutral**，reason 必须显式写
    "区间策略 RR=0.54 < 2.0，退回 neutral"（**禁止改写为"区间不清晰"**）。
此例展示了 P3 升级后的"严格 RR + SL 距离 + EZ 不贴边"三重约束在窄幅震荡里
多数会推回 neutral，这是预期行为：与其频繁亏损不如不交易。

【区间策略示例 B：检查清单两侧均不通过 → 输出 neutral】
当 5m/15m 的 supports 或 resistances 都为空、或最近一根 K 已经扫破区间一侧
（swept_high_recent=true / swept_low_recent=true）时，两个周期闸 A 都失败，
reason 必须写：
  "区间不可交易：5m 失败原因=<具体>（如 res 为空 / 价格已突破 res[0]）；
   15m 失败原因=<具体>"。
**禁止**只写一句"区间不清晰"——必须把两个周期的失败原因分别列出，
便于事后通过 fill_rate / 评估器复盘"是结构缺位还是 RR 不足"。

【P1 强约束（必须严格遵守）】
1) 当 regime ∈ {{ranging, transitional}} 时禁止给 trending 仓位建议：必须降为
   neutral，或在最近 supports/resistances 之间做"区间策略"，并把
   position_size_pct ≤ 0.05；reason 必须显式提到 "regime=<值>"。
   （P3 升级把 transitional 与 ranging 同等对待——切换期同样不该重仓趋势单。）
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

【P2 强约束（自我反馈机制 - P3 升级版，必须严格遵守）】
1) 你将看到自己最近 N 次判断的实际成绩单，且会**按"是否曾入场"分两段**：
   - 【判断质量段】：仅统计 triggered_at 非空（价格曾走进 entry_zone）的样本。
     胜率定义为 wins / (wins + losses)，wins = tp1/tp2_hit，losses = sl_hit。
     "曾入场但超时（expired-after-triggered）"不计入胜率分母。
     **硬约束（P3 收紧）**：
       (a) 当 (wins + losses) ≥ 2 且胜率 ≤ 50% 时，必须显著降低本次
           confidence 到 < 0.5，并在 reason 中具体反思失败模式（例如：
           "近 4 次曾入场样本中 3 次 sl_hit，多发生在 regime=ranging
            时强行做趋势"）；
       (b) 当 (wins + losses) ≥ 1 且胜率 = 0% 时，confidence 上限 0.4。
   - 【触发率段】：fill_rate = 曾入场样本 / 总样本，反映 entry_zone 设计。
     **硬约束（P3 升级）**：fill_rate < 30% 时，本次 entry_zone 中点必须距
     当前价 ≤ 0.3%（不再仅"反思"，必须收紧入场区间），同时在 reason 中
     说明"上 N 笔 fill_rate < 30%，本次贴近当前价挂单"。
2) 当 (wins + losses) < 2（"判断质量段"样本不足，无统计意义）时不应用
   上述硬约束；但 reason 中要注明 "曾入场样本不足，未触发自我反馈降权"。
3) 对历史成绩的解释必须客观：不要因为 1 次大额胜利就过度自信，
   也不要因为 1 次最大不利波动就强行翻转方向；优先看胜率 / 平均 PnL
   / 最大回撤三者的组合。
4) 本系统是"信号建议"层，不持仓、不下单。**不要**因为成绩单里出现某条
   方向就认为"还在仓位里"或被它绑架——每一次判断都应基于当前因子矩阵
   独立做出，方向冲突 / 净敞口控制是后续持仓管理层的职责，不在你的范围。

【P3 强约束（系统级评估指标 - 必须严格遵守）】
你将看到一段近 24h 的系统级评估摘要，包括：方向翻转率、触发率、胜率、
平均 PnL、估算 Sharpe、Brier score。它代表"系统**整体**最近一天的判断
质量"，比单条成绩单更宏观。使用规则：
1) 当 direction_flip_rate > 30% 时，市场近 24h 偏震荡 / 系统在 whipsaw，
   本次判断必须**明显倾向 neutral 或区间策略**：禁止 trending 单且
   position_size_pct ≤ 0.05；如果选择 long/short，confidence 上限 0.5。
2) 当 sharpe_estimated < 0 或 avg_pnl_pct < 0 时，系统近 24h 是**净亏损**
   状态，本次 confidence 上限 0.5；如果同时 brier_score > 0.30
   （置信度严重失校），confidence 上限 0.4 且必须在 reason 中明确写出
   "系统近 24h 处于负期望，本次降权"。
3) 当评估摘要为"无数据"（评估器尚未跑过 / 该窗口无样本）时，跳过本节
   约束，但要在 reason 中注明"尚无系统级评估数据，按因子原始结论输出"。
4) 上述系统级反馈优先级**高于**因子矩阵——即使因子矩阵看起来非常一致，
   只要 24h 系统状态恶劣，必须降权或转 neutral。
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
{self_feedback_block}{evaluation_block}合约: {symbol}
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
        规则（P3 升级版）：
            volatility_ratio = atr_5m × 12 / atr_1h
                （把 5m ATR 折算到 1h 尺度后与 1h ATR 比较，> 1 表示 5 分钟波动异常放大）
            volatility_ratio >= 1.5 或 regime ∈ {breakout, breakdown} → 900s（15 分钟）
            volatility_ratio >= 1.2 或 |alignment_score| >= 0.5       → 600s（10 分钟）
            其他                                                      → 1800s（30 分钟）

            P3 强制下限：当 regime ∈ {ranging, transitional} 时，无论上面命中
            哪一档都强制 interval = max(interval, 1800)。原因：在窄幅震荡 /
            状态切换期里 LLM 价值最低（叙事好但预测准确率接近随机），把节流
            拉满省钱也避免 whipsaw。

            alignment 阈值从 0.75 降到 0.5：原值要求 5/5 周期共振才能加速,
            实测过严，多数有意义的趋势期都打不到；0.5 对应"5 票里有 3 票
            同向"，更接近真实可交易场景。

            上下限通过 settings.llm_min_interval_seconds_min/max 钳制。
        """
        # 关闭开关：直接回退到 P0 静态节流
        if not bool(getattr(self.settings, "enable_adaptive_throttle", False)):
            return self.min_interval
        try:
            base_default = int(self.min_interval) if self.min_interval > 0 else 1800
            lo = int(getattr(self.settings, "llm_min_interval_seconds_min", 900))
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
                interval = 900
            elif (
                volatility_ratio is not None and volatility_ratio >= 1.2
            ) or alignment_score >= 0.5:
                interval = 600
            else:
                interval = 1800

            # P3：窄幅震荡 / 状态切换期强制拉满节流
            if regime in ("ranging", "transitional"):
                interval = max(interval, 1800)

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

    @staticmethod
    def _collect_cached_plan_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        把 signals 表行里的"结构化交易计划"列规整成 TradingSignal 构造 kwargs
        --------------------------------------------------------------
        P0 Quant 修复 #3 的辅助函数。

        类型转换说明：
            - NUMERIC 列经 asyncpg 默认返回 ``decimal.Decimal``；TradingSignal
              的字段是 float，这里统一转 float（精度损失对交易计划无影响）。
            - JSONB 列在 database._init_connection 里已经注册了 codec，
              直接拿到 list / dict / None。
            - 任何字段为 None / 空 / 类型不符时，原样不放进 kwargs，
              让 TradingSignal 走默认值；如果 bias != neutral 又凑不齐
              entry_zone / stop_loss / 2 档 take_profit，model_validator
              会抛 ValueError，调用方 catch 后会让本轮走真 LLM 调用。

        参数：
            row: fetch_latest_signal_judgment 返回的 dict
        返回：
            可直接 **kwargs 解包给 TradingSignal(...) 的字典；
            缺什么字段就不放什么键，不会硬塞 None 进去。
        """

        def _to_float(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        kwargs: Dict[str, Any] = {}

        # entry_zone：JSONB 存成 [low, high]，TradingSignal 期望 tuple
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
            # 仅保留 value 是 str 的项，避免脏数据破坏 schema
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

        # P0 Quant 修复 #3：完整重建 TradingSignal（含结构化交易计划）
        # ------------------------------------------------------------
        # 旧版逻辑：先用 bias="neutral" 构造让 model_validator 通过，
        # 再 object.__setattr__ 覆盖 bias 回真实值。这绕过了 schema 强约束，
        # 会向上游透出"bias=long、entry_zone=None、take_profit=[]"的脏信号
        # （前端 service.generate 里 final.model_dump() 是无差别透出的）。
        #
        # 新版逻辑：从 row 里把 P0 升级新增的 7 个结构化列一并取出来，
        # 完整构造 TradingSignal 让 model_validator 真实通过。
        # 副作用：
        #   - 历史脏行（旧版 LLM 没产 plan 就入库的）会触发 ValueError，
        #     我们 catch 后返回 None，让本轮去真打一次 LLM 重新建判断；
        #     这恰恰是我们想要的行为：宁可多调一次 LLM，也不能透出脏 plan。
        #   - schema 里 RR < 1.5 会强制降级为 neutral 并清空 plan，
        #     与"真正调用 LLM 时的语义"保持一致。
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
            # 历史脏数据 / schema 不匹配 / 几何不合规：不强行卡死节流通道；
            # 记日志后当作没有缓存处理，让本轮去真打一次 LLM 重建判断。
            logger.warning(
                "无法从 signals 表行重建 %s 的 TradingSignal（多半是历史脏行/"
                "结构化字段缺失）；将按未命中缓存重新调用 LLM",
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
        evaluation_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        把多周期因子矩阵渲染成紧凑中文表格，作为 ChatPromptTemplate 输入
        ---------------------------------------------------------------
        参数：
            symbol / factors / rule_signal / rule_score / rule_contributions:
                同 analyze() 形参
            recent_settled:
                近 N 条已结算 lifecycle 行（可选）。**注意**：这里不再接收
                "上一条未结算"作为输入——本系统只产建议、不掌握用户是否
                实际下单，把 lifecycle 的 open 行当成"持仓"是概念错位，
                会导致 LLM 被自己 30 分钟前的旧判断绑架。方向冲突 / 净敞口
                控制属于后续持仓管理层职责，已从 prompt 中彻底移除。
            evaluation_summary:
                P3 升级新增。``signal_evaluation`` 表中近 24h 那一行
                （window_minutes=1440）。用于渲染"系统级评估"段，把胜率/
                翻转率/Sharpe/Brier 等宏观指标灌给 LLM 做自适应放慢决策。
                None 时段落显示"无系统级评估数据"，behavior 与 P2 完全一致。
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
        # 仅当 enable_llm_self_feedback=True 时渲染；
        # recent_settled 为空时 _render_self_feedback 自身会输出"样本不足"段。
        # open_lifecycle 注入链路已彻底移除（详见 _build_prompt_inputs docstring）。
        if bool(getattr(self.settings, "enable_llm_self_feedback", False)):
            self_feedback_block = self._render_self_feedback(
                symbol=symbol,
                recent_settled=recent_settled or [],
            )
        else:
            self_feedback_block = ""

        # P3：系统级评估段（注入到 self_feedback_block 之后）
        # ----------------------------------------------------------
        # 仅当 enable_signal_evaluation=True 且评估器写过表时才有内容；
        # 任一条件不满足都退化为"无数据"提示，避免 prompt 中出现噪声。
        if bool(getattr(self.settings, "enable_signal_evaluation", False)):
            evaluation_block = self._render_evaluation_block(
                symbol=symbol, evaluation=evaluation_summary
            )
        else:
            evaluation_block = ""

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
            "evaluation_block": evaluation_block,
        }

    @classmethod
    def _render_self_feedback(
        cls,
        symbol: str,
        recent_settled: List[Dict[str, Any]],
    ) -> str:
        """
        渲染近 N 次成绩单（P2 自我反馈，aggressive 修订版）
        ---------------------------------------------------------------
        参数：
            symbol         : 合约
            recent_settled : 近 N 条已结算 lifecycle（含 join 出来的 signals 字段）
        返回：
            紧凑中文表格 + 统计摘要；总长度严格控制在 ~1500 token 以内。

        修订点（相对历史版本）：
            1) 不再渲染"⚠️ 上一条信号 #N 仍未结算"段：信号引擎只产建议，
               不掌握用户实际持仓。方向冲突属于持仓管理层职责，不应在
               LLM prompt 里强制 neutral。
            2) 把成绩单按"是否真正入过场"拆成两段：
                 - 判断质量段（triggered_at IS NOT NULL）：仅统计
                   wins / (wins + losses)，expired-after-triggered（曾入场
                   但超时退出）单独列出，不污染胜率分母；
                 - 触发率段（fill rate）：triggered_count / total，反映
                   入场区间是否合理；fill_rate 偏低提示 reason 自我反思
                   "区间过窄/过远"，但不强制降置信度。
            3) 这样既保留"判断方向准不准"信号给 LLM，又不会把"价格没回到
               入场区间"误判成"判断错"，避免历史版本"5次全 expired→胜率0%
               →强压 confidence 至 0.3"的吸收态。
        """
        lines: List[str] = []
        n = len(recent_settled)
        lines.append(f"===== 你最近 {n} 次判断的成绩单（{symbol}）=====")

        if not recent_settled:
            lines.append("（样本不足：lifecycle 表里还没有该 symbol 的已结算记录）")
            lines.append("")
            return "\n".join(lines) + "\n"

        # 按"是否曾入场"切分两组
        triggered_rows: List[Dict[str, Any]] = []
        untriggered_rows: List[Dict[str, Any]] = []
        for row in recent_settled:
            if row.get("triggered_at") is not None:
                triggered_rows.append(row)
            else:
                untriggered_rows.append(row)

        # ----------------------------------------------------------------
        # 第 1 段：判断质量（仅曾入场的样本）
        # ----------------------------------------------------------------
        lines.append("")
        lines.append(
            f"【判断质量段】曾入场样本 = {len(triggered_rows)} / {n}"
        )
        wins = 0
        losses = 0
        expired_after_triggered = 0
        pnl_acc: List[float] = []
        if not triggered_rows:
            lines.append("（无曾入场样本：所有信号均未触发，无法评估方向准确性）")
        else:
            lines.append(
                "| ts | bias | conf | regime | 入场区间 | 触发价 | 终态 | PnL% | 最大有利% | 最大不利% |"
            )
            lines.append(
                "|----|------|------|--------|----------|--------|------|------|-----------|-----------|"
            )
            for row in triggered_rows:
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
                    expired_after_triggered += 1
                if pnl is not None:
                    pnl_acc.append(float(pnl))

            decided = wins + losses
            win_rate = (wins / decided) if decided > 0 else 0.0
            avg_pnl = sum(pnl_acc) / len(pnl_acc) if pnl_acc else 0.0
            lines.append(
                f"统计（仅曾入场）：胜率={win_rate:.0%}（{wins}赢 / {losses}输；"
                f"另有 {expired_after_triggered} 笔曾入场但超时未到 SL/TP，不计入分母）"
                f"  平均PnL={avg_pnl * 100:+.2f}%  样本={decided}"
            )

        # ----------------------------------------------------------------
        # 第 2 段：触发率（fill rate）—— 衡量入场区间设置是否合理
        # ----------------------------------------------------------------
        lines.append("")
        triggered_total = len(triggered_rows)
        fill_rate = triggered_total / n if n > 0 else 0.0
        lines.append(
            f"【触发率段】fill_rate={fill_rate:.0%}（{triggered_total}/{n} 触发入场）；"
            f"未触发 {len(untriggered_rows)} 笔（价格从未回到 entry_zone 即超时/作废）"
        )
        if untriggered_rows and fill_rate < 0.3:
            lines.append(
                "提示：fill_rate 偏低（< 30%）通常意味着入场区间过窄或方向偏移过早，"
                "建议在 reason 中反思入场设计，但不必单方面降低 confidence。"
            )

        lines.append("")
        return "\n".join(lines) + "\n"

    @classmethod
    def _render_evaluation_block(
        cls,
        symbol: str,
        evaluation: Optional[Dict[str, Any]],
    ) -> str:
        """
        渲染近 24h 系统级评估摘要（P3 升级新增）
        ---------------------------------------------------------------
        参数：
            symbol     : 合约（仅作标题展示）
            evaluation : ``signal_evaluation`` 表近 24h 那一行；None 表示
                         "评估器尚未跑过 / 无样本"。
        返回：
            一段紧凑中文文本，结尾带换行；保证 prompt 结构稳定。
        说明：
            - 任意字段缺失时显示 "-"，避免把 None / Decimal 直接拼进 prompt。
            - 与 P3 强约束段（SYSTEM_PROMPT 末尾）配合：LLM 看到
              direction_flip_rate > 30%、avg_pnl_pct < 0、brier_score > 0.30
              时会主动降低 confidence。
        """
        lines: List[str] = [
            f"===== 系统级评估（近 24h，{symbol}）====="
        ]
        if not evaluation:
            lines.append(
                "（暂无系统级评估数据：评估器尚未跑过或 24h 内无 LLM 信号）"
            )
            lines.append("")
            return "\n".join(lines) + "\n"

        def _pct(v: Any, digits: int = 2) -> str:
            """把 [0, 1] 比例 / [-1, 1] 收益率渲染成百分比字符串"""
            f = _to_float_safe(v)
            if f is None:
                return "-"
            return f"{f * 100:+.{digits}f}%"

        def _num(v: Any, digits: int = 4) -> str:
            f = _to_float_safe(v)
            if f is None:
                return "-"
            return f"{f:.{digits}f}"

        total = evaluation.get("total_signals") or 0
        triggered = evaluation.get("triggered_count") or 0
        wins = evaluation.get("wins")
        losses = evaluation.get("losses")
        decided = (int(wins or 0) + int(losses or 0)) if (wins is not None or losses is not None) else 0
        flip_rate = _to_float_safe(evaluation.get("direction_flip_rate"))
        avg_pnl = _to_float_safe(evaluation.get("avg_pnl_pct"))
        sharpe = _to_float_safe(evaluation.get("sharpe_estimated"))
        brier = _to_float_safe(evaluation.get("brier_score"))

        lines.append(
            f"信号总数={total}，触发入场={triggered}，"
            f"触发率={_pct(evaluation.get('fill_rate'), 1)}，"
            f"胜率={_pct(evaluation.get('win_rate'), 1)}（{decided} 笔决定性）"
        )
        lines.append(
            f"平均PnL={_pct(avg_pnl, 2)}，"
            f"累计PnL={_pct(evaluation.get('total_pnl_pct'), 2)}，"
            f"估算Sharpe={_num(sharpe, 3)}，Brier={_num(brier, 4)}"
        )
        lines.append(
            f"方向翻转={evaluation.get('direction_flip_count')} 次，"
            f"翻转率={_pct(flip_rate, 1)}"
        )

        # 自动写出主要预警信号（与 P3 强约束段配合）
        warnings: List[str] = []
        if flip_rate is not None and flip_rate > 0.30:
            warnings.append(
                f"方向翻转率 {flip_rate * 100:.0f}% 高于 30%，市场近 24h whipsaw"
            )
        if avg_pnl is not None and avg_pnl < 0:
            warnings.append(
                f"平均PnL {avg_pnl * 100:+.2f}% 为负，系统近 24h 净亏损"
            )
        if sharpe is not None and sharpe < 0:
            warnings.append(f"Sharpe {sharpe:.2f} < 0，风险调整收益为负")
        if brier is not None and brier > 0.30:
            warnings.append(
                f"Brier {brier:.3f} > 0.30，置信度严重失校"
            )
        if warnings:
            lines.append(
                "P3 预警：" + "；".join(warnings) + "（参见 SYSTEM_PROMPT P3 强约束段）"
            )

        lines.append("")
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
            recent_settled     ：近 N 条已结算 lifecycle（自我反馈用，可选）。
                                 注意：不再接收"上一条未结算"参数——见
                                 _build_prompt_inputs docstring。
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
                # P3：拉取近 24h 系统级评估摘要（无则 None，prompt 段降级为
                # "无数据"）。失败吞掉只记 debug：评估系统是"锦上添花"，
                # 不能因为它把整轮 LLM 调用打挂。
                evaluation_summary: Optional[Dict[str, Any]] = None
                if bool(getattr(self.settings, "enable_signal_evaluation", False)):
                    try:
                        evaluation_summary = (
                            await self.repos.fetch_latest_signal_evaluation(
                                symbol=symbol, window_minutes=1440
                            )
                        )
                    except Exception:
                        logger.debug(
                            "拉取 signal_evaluation 失败 symbol=%s（不影响主路径）",
                            symbol,
                            exc_info=True,
                        )
                        evaluation_summary = None
                prompt_inputs = self._build_prompt_inputs(
                    symbol=symbol,
                    factors=factors,
                    rule_signal=rule_signal,
                    rule_score=rule_score,
                    rule_contributions=rule_contributions,
                    recent_settled=recent_settled,
                    evaluation_summary=evaluation_summary,
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
