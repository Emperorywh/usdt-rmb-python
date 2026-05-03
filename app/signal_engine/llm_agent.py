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
from typing import Any, Dict, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.data_storage.repositories import Repositories
from app.logging_config import get_logger
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)


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
你是一名资深加密衍生品量化交易分析师。系统会给你一份从实时行情中
计算出的结构化因子快照，以及一个由确定性规则引擎给出的初步打分。

请基于这些信息输出**一个**简洁的交易偏置判断，并以严格符合给定
schema 的 JSON 对象返回。该信号仅作交易建议，不会被执行下单。

输出语言要求：
- reason / risk / suggestion 三个字段必须使用**简体中文**。
- bias 字段保持 long / short / neutral 三个英文枚举值之一，**不要翻译**。

判断要点：
- 综合考虑四类因子：资金流（capital_flow）、订单簿（orderbook）、
  衍生品（derivatives）、市场结构（market_structure）。
- 注意：当前没有链上数据与参与者画像，请勿引用任何 on-chain / whale /
  smart money 相关信息，也不要编造。
- 当各因子方向冲突时，优先输出 "neutral" 并给出较低的 confidence。
- "confidence" 必须落在 [0, 1] 区间，反映各因子方向的一致程度。
- "reason" 必须引用因子中最具代表性的具体数值（净流入金额、盘口失衡、
  资金费率、OI 变动百分比、趋势判断、关键支撑/阻力位等）。
- "risk" 必须列出明确的失效条件（例如：资金费率反转、关键价位被跌破、
  对侧大单墙形成等）。
- "suggestion" 必须给出**具体的入场区间、止损位、目标位**，并在文末
  注明"仅供参考，不构成交易指令"。
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
合约代码: {symbol}
时间戳: {ts}

规则引擎初判: bias={rule_bias} confidence={rule_confidence} score={rule_score}
规则引擎各因子贡献度: {rule_contributions}

因子快照 (JSON):
{factors_json}

请基于以上信息输出最终的 TradingSignal。"""


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
        """
        from langchain_openai import ChatOpenAI

        if not self.settings.deepseek_api_key:
            logger.warning(
                "未配置 DEEPSEEK_API_KEY，运行时将跳过 LLM 分析"
            )

        thinking_enabled = bool(self.settings.deepseek_thinking_enabled)

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

        llm = ChatOpenAI(**llm_kwargs)

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
        LLM 调用最小间隔（秒）。
        --------------------------------------------------------------
        从 settings 读取，方便单元测试通过 monkeypatch 修改 settings 调整。
        """
        return max(0, int(self.settings.llm_min_interval_seconds))

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
        self, symbol: str
    ) -> Optional[LLMAnalysisResult]:
        """
        若指定 symbol 在节流窗口内已有 LLM 判断（落在 signals 表里），
        则把它重建成 LLMAnalysisResult 返回；否则返回 None 让上层调用 LLM。
        --------------------------------------------------------------
        步骤：
            1) ``min_interval <= 0``：节流关闭，直接返回 None（每次都真调）。
            2) 查 signals 表里该 symbol 最近一条记录的 ts；
               表为空则视为未节流，让上层去打 LLM。
            3) 计算 ``now - ts``；
               - ``< min_interval``：节流命中，把那一行的 bias / confidence
                 / reason / risk / suggestion / reasoning_content 重建成
                 LLMAnalysisResult，并把 ``from_cache`` 置 True 返回。
                 service 层据此跳过本次入库，避免重复行。
               - ``>= min_interval``：节流过期，返回 None。
        异常处理：
            DB 查询失败时（如连接被 reset），不应阻塞 LLM 调用；记一行
            warning 后返回 None，让上层退化为"直接打 LLM"——成本上界仍是
            min_interval，最差也只是少一次节流命中。
        """
        if self.min_interval <= 0:
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
        if elapsed >= self.min_interval:
            return None

        try:
            cached_signal = TradingSignal(
                bias=row["bias"],
                confidence=float(row["confidence"]),
                reason=row.get("reason") or "",
                risk=row.get("risk") or "",
                suggestion=row.get("suggestion") or "",
            )
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

        remaining = self.min_interval - elapsed
        logger.info(
            "LLM 数据库节流命中 %s（已过 %.0fs，下次调用还需 %.0fs）",
            symbol,
            elapsed,
            remaining,
        )
        return LLMAnalysisResult(
            signal=cached_signal,
            reasoning_content=row.get("reasoning_content"),
            from_cache=True,
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
            candidates.append(additional.get("reasoning_content"))
        response_meta = getattr(raw_message, "response_metadata", None)
        if isinstance(response_meta, dict):
            candidates.append(response_meta.get("reasoning_content"))

        # dict 形态（旧版本或测试桩）
        if isinstance(raw_message, dict):
            candidates.append(raw_message.get("reasoning_content"))
            ak = raw_message.get("additional_kwargs")
            if isinstance(ak, dict):
                candidates.append(ak.get("reasoning_content"))

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

        # 快速路径：先在锁外查一次 DB，避免抢锁。
        cached = await self._load_recent_judgment(symbol)
        if cached is not None:
            return cached

        # 慢速路径：拿锁后再查一次（double-checked locking），
        # 防止同一 symbol 在节流刚过期的瞬间被并发调用打多次接口。
        async with self._get_lock(symbol):
            cached = await self._load_recent_judgment(symbol)
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
                    "LLM 发起调用 -> %s（最小间隔=%ss）", symbol, self.min_interval
                )
                result = await self._chain.ainvoke(
                    {
                        "symbol": symbol,
                        "ts": factors.get("computed_at"),
                        "rule_bias": rule_signal.bias,
                        "rule_confidence": rule_signal.confidence,
                        "rule_score": round(rule_score, 4),
                        "rule_contributions": json.dumps(rule_contributions),
                        "factors_json": json.dumps(
                            factors, ensure_ascii=False, default=str
                        ),
                    }
                )
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
