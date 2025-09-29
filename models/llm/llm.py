import logging
import re
from collections.abc import Generator
from enum import Enum
from typing import Optional, Union

from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType, ParameterRule, ParameterType, ModelPropertyKey,
)
from dify_plugin.entities.model.llm import (
    LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool, AssistantPromptMessage, )
from dify_plugin.entities.provider_config import AppSelectorScope
from dify_plugin.errors.model import (
    CredentialsValidateFailedError, InvokeError, InvokeConnectionError, )
from httpx import HTTPStatusError
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CredentialParams(BaseModel):
    server_url: str
    api_key: Optional[str] = ''


class PanguThinkingToken(Enum):
    think_start = '[unused16]'
    think_end = '[unused17]'


class LightragLargeLanguageModel(LargeLanguageModel):
    """
    Model class for lightrag large language model.
    """

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [HTTPStatusError]
        }

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        print(f"stream: {stream}")
        params = CredentialParams(**credentials)
        client = OpenAI(
            api_key=params.api_key,
            base_url=params.server_url,
        )
        enable_reasoning = model_parameters.pop("enable_reasoning", True)
        print(prompt_messages)
        response = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            **model_parameters,
            stream=stream
        )
        if stream:
            return self._handle_stream_response(response, prompt_messages, enable_reasoning)
        return self._handle_sync_response(response, prompt_messages, enable_reasoning)

    def _warp_stream_thinking_content(self, content: str, enable_reasoning: bool) -> str:
        if not enable_reasoning:
            pattern = rf'^.*?\[unused17\]'
            return re.sub(pattern, '', content, flags=re.DOTALL)
        if PanguThinkingToken.think_start.value in content:
            return content.replace(PanguThinkingToken.think_start.value, '<think>\n')
        elif PanguThinkingToken.think_end.value in content:
            return content.replace(PanguThinkingToken.think_end.value, '\n</think>')
        return content

    def _warp_thinking_content(self, content: str, enable_reasoning: bool) -> str:
        wrap_content = content.replace(PanguThinkingToken.think_start.value, "<think>\n")
        wrap_content = wrap_content.replace(PanguThinkingToken.think_end.value, "\n</think>")
        if not enable_reasoning:
            wrap_content = re.sub(r"<think>.*?</think>", "", wrap_content, flags=re.DOTALL)
        return wrap_content

    def _handle_stream_response(self, response: Stream[ChatCompletionChunk],
                                prompt_messages: list[PromptMessage], enable_reasoning: bool) -> Generator:
        idx = 0
        has_end_thinking = False
        for chunk in response:
            detail = chunk.choices[0].delta
            if PanguThinkingToken.think_end.value in detail.content:
                has_end_thinking = True

            if not enable_reasoning and not has_end_thinking:
                continue

            finish_reason = detail.finish_reason if hasattr(detail, 'finish_reason') else None
            yield LLMResultChunk(
                model=chunk.model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=idx + 1,
                    message=AssistantPromptMessage(
                        content=self._warp_stream_thinking_content(detail.content, enable_reasoning),
                    ),
                    finish_reason=finish_reason,
                )

            )
            idx += 1

    def _handle_sync_response(self, response: ChatCompletion,
                              prompt_messages: list[PromptMessage], enable_reasoning: bool) -> LLMResult:
        return LLMResult(
            model=response.model,
            prompt_messages=prompt_messages,
            message=AssistantPromptMessage(
                content=self._warp_thinking_content(response.choices[0].message[0].content, enable_reasoning),
            ),
            usage=LLMUsage.empty_usage()
        )

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        return 0

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            params = CredentialParams(**credentials)
            client = OpenAI(
                api_key=params.api_key,
                base_url=params.server_url,
            )
            models = client.models.list()
            logger.info(f"connect pangu servers, models: {models}")
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def get_customizable_model_schema(
            self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        If your model supports fine-tuning, this method returns the schema of the base model
        but renamed to the fine-tuned model name.

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        rules = [
            ParameterRule(
                name="temperature", type=ParameterType.FLOAT,
                required=False,
                label=I18nObject(
                    en_US="Temperature", zh_Hans="温度"
                ),
                help=I18nObject(
                    en_US="Controls randomness: Lower values make output more deterministic, higher values make it more creative.",
                    zh_Hans="控制随机性：较低值使输出更确定，较高值使输出更具创造性。"
                )
            ),
            ParameterRule(
                name="top_p", type=ParameterType.FLOAT,
                required=False,
                label=I18nObject(
                    en_US="Top P", zh_Hans="核采样 Top P"
                ),
                help=I18nObject(
                    en_US="Nucleus sampling: model considers only tokens with top_p probability mass. (Set either temperature or top_p, not both).",
                    zh_Hans="核采样：模型只考虑累计概率质量在 top_p 内的候选词。（建议只设置 temperature 或 top_p 其中一个）。"
                )
            ),
            ParameterRule(
                name="max_tokens", type=ParameterType.INT,
                required=False,
                label=I18nObject(
                    en_US="Max Tokens", zh_Hans="最大生成 Token 数"
                ),
                help=I18nObject(
                    en_US="The maximum number of tokens to generate in the response.",
                    zh_Hans="生成回复的最大 token 数。"
                )
            ),
            ParameterRule(
                name="presence_penalty", type=ParameterType.FLOAT,
                required=False,
                label=I18nObject(
                    en_US="Presence Penalty", zh_Hans="出现惩罚"
                ),
                help=I18nObject(
                    en_US="Penalizes new tokens based on whether they appear in the text so far. Encourages introducing new topics.",
                    zh_Hans="根据新 token 是否已出现来惩罚，鼓励引入新话题。"
                )
            ),
            ParameterRule(
                name="frequency_penalty", type=ParameterType.FLOAT,
                required=False,
                label=I18nObject(
                    en_US="Frequency Penalty", zh_Hans="频率惩罚"
                ),
                help=I18nObject(
                    en_US="Penalizes new tokens based on their frequency in the text so far. Reduces repetition.",
                    zh_Hans="根据 token 出现频率惩罚，减少重复内容。"
                )
            ),
            ParameterRule(
                name="stop", type=ParameterType.STRING,
                required=False,
                label=I18nObject(
                    en_US="Stop Sequences", zh_Hans="停止序列"
                ),
                help=I18nObject(
                    en_US="A list of sequences where the model will stop generating further tokens.",
                    zh_Hans="当生成结果出现这些序列时，模型会停止输出。"
                )
            ),
            ParameterRule(
                name="enable_reasoning", type=ParameterType.BOOLEAN,
                required=False,
                label=I18nObject(
                    en_US="Enable Reasoning", zh_Hans="启用慢思考"
                ),
                help=I18nObject(
                    en_US="If true, the model will return reasoning traces (slow thinking) in <think>...</think> blocks.",
                    zh_Hans="如果为 true，模型将返回慢思考推理过程（以 <think>...</think> 包裹）。"
                )
            )
        ]

        entity = AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.MODE: AppSelectorScope.CHAT
            },
            parameter_rules=rules,
        )

        return entity
