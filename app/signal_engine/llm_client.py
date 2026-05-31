"""LLM API 调用封装 + 延迟链构建 + 结果解析。

职责：
- 延迟构建 LangChain 调用链（思考/非思考两条路径）
- 调用 LLM 并返回原始结果
- 从结果中提取 reasoning_content / token_usage
- JSON 容错解析（去代码围栏）
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.logging_config import get_logger
from app.signal_engine.llm_prompts import (
    FEW_SHOT_EXAMPLES,
    HUMAN_PROMPT,
    SYSTEM_PROMPT,
    THINKING_OUTPUT_INSTRUCTIONS,
)
from app.signal_engine.schemas import TradingSignal

logger = get_logger(__name__)

# 防 prompt injection 的 symbol 白名单
_SYMBOL_PATTERN: re.Pattern[str] = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,31}$")


# ------------------------------------------------------------------
# DeepSeek 思考模式 reasoning_content 透传补丁
# ------------------------------------------------------------------
def _build_deepseek_chat_openai_class():
    """构造一个支持 DeepSeek reasoning_content 透传的 ChatOpenAI 子类。"""
    from langchain_openai import ChatOpenAI

    class _DeepSeekChatOpenAI(ChatOpenAI):
        """DeepSeek 思考模式专用 ChatOpenAI 子类。"""

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


class LLMClient:
    """封装 DeepSeek LLM API 调用。

    职责：
    - 延迟构建 LangChain 调用链（思考/非思考两条路径）
    - 调用 LLM 并返回原始结果
    - 从结果中提取 reasoning_content / token_usage
    - JSON 容错解析
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._chain = None
        self._chain_thinking_mode: bool = False

    def _build_chain(self):
        """构建 LangChain 调用链（首次 invoke 时触发）。"""
        from langchain_openai import ChatOpenAI

        if not self._settings.llm.deepseek_api_key:
            logger.warning("未配置 DEEPSEEK_API_KEY，运行时将跳过 LLM 分析")

        thinking_enabled = bool(self._settings.llm.deepseek_thinking_enabled)
        chat_openai_cls = (
            _build_deepseek_chat_openai_class() if thinking_enabled else ChatOpenAI
        )

        llm_kwargs: Dict[str, Any] = {
            "model": self._settings.llm.deepseek_model,
            "api_key": self._settings.llm.deepseek_api_key or "missing-key",
            "base_url": self._settings.llm.deepseek_base_url,
            "timeout": self._settings.llm.llm_timeout,
            "max_retries": 2,
        }

        if thinking_enabled:
            effort = (self._settings.llm.deepseek_reasoning_effort or "high").lower()
            if effort not in {"high", "max"}:
                logger.warning(
                    "未知的 reasoning_effort=%r，回退为 'high'", effort
                )
                effort = "high"
            llm_kwargs["reasoning_effort"] = effort
            llm_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            logger.info(
                "DeepSeek 思考模式已启用（model=%s，effort=%s，json_via=prompt）",
                self._settings.llm.deepseek_model,
                effort,
            )
        else:
            llm_kwargs["temperature"] = self._settings.llm.llm_temperature
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            logger.info(
                "DeepSeek 思考模式已禁用（model=%s，temperature=%.2f，json_via=function_calling）",
                self._settings.llm.deepseek_model,
                self._settings.llm.llm_temperature,
            )

        llm = chat_openai_cls(**llm_kwargs)

        # few-shot 示例以 message history 注入
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
        """是否启用 LLM。"""
        return bool(self._settings.llm.deepseek_api_key)

    @property
    def chain_thinking_mode(self) -> bool:
        """当前链是否为思考模式。"""
        return self._chain_thinking_mode

    async def invoke(self, prompt_inputs: dict) -> Any:
        """调用 LLM 并返回原始结果。链在首次调用时延迟构建。"""
        if self._chain is None:
            self._chain = self._build_chain()
        return await self._chain.ainvoke(prompt_inputs)

    # ------------------------------------------------------------------
    # 结果解析工具
    # ------------------------------------------------------------------
    @staticmethod
    def extract_reasoning(raw_message: Any) -> Optional[str]:
        """从 LangChain AIMessage / dict 中抽取 DeepSeek 思考模式的思维链原文。"""
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
    def extract_token_usage(raw_message: Any) -> Tuple[int, int, int]:
        """从 LangChain AIMessage 中提取 (input_tokens, output_tokens, total_tokens)。"""
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
    def parse_json_content(content: Any, symbol: str) -> Optional[Dict[str, Any]]:
        """从 LLM 原始 content 文本中抽出 JSON 对象（去代码围栏）。"""
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

    @staticmethod
    def validate_symbol(symbol: str) -> None:
        """校验 symbol 是否通过白名单。不通过时抛 ValueError。"""
        if not _SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                f"非法 symbol={symbol!r}：仅允许 [A-Z0-9_-]，"
                f"长度 3–32（首字符必须为字母或数字）"
            )
