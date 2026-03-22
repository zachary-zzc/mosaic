import os
import requests
import httpx
from typing import Optional, List, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from src.config_loader import get_api_key_and_base_url
from src.logger import setup_logger

_logger = setup_logger("llm")

ali_api_key, ali_base_url = get_api_key_and_base_url()

# OpenAI 客户端按 (key, base, timeout 配置) 复用，避免每次请求新建连接
_openai_clients: dict[tuple[str, str, float, float], OpenAI] = {}


def _dashscope_httpx_timeout() -> httpx.Timeout:
    """
    OpenAI SDK 默认 Timeout(connect=5.0, read=600, ...)；connect 过短时，
    经 HTTP(S) 代理做 TLS 握手易触发 ConnectTimeout（与 API Key 无关）。

    用环境变量可单独调大握手/读超时，避免误把 read 压成与 connect 相同的单值 float。
    """
    connect = float(os.environ.get("MOSAIC_HTTP_CONNECT_TIMEOUT", "120"))
    read = float(os.environ.get("MOSAIC_HTTP_READ_TIMEOUT", "600"))
    pool = float(os.environ.get("MOSAIC_HTTP_POOL_TIMEOUT", str(connect)))
    return httpx.Timeout(connect=connect, read=read, write=read, pool=pool)


def _get_dashscope_client(api_key: str, base_url: str) -> OpenAI:
    base = (base_url or "").strip().rstrip("/")
    timeout = _dashscope_httpx_timeout()
    cache_key = (api_key or "", base, timeout.connect, timeout.read)
    if cache_key not in _openai_clients:
        _openai_clients[cache_key] = OpenAI(
            api_key=api_key,
            base_url=base,
            timeout=timeout,
        )
    return _openai_clients[cache_key]

class CustomChatModel(BaseChatModel):
    """Custom Chat Model that interfaces with a custom server."""

    model_url: str  # URL of your custom model server
    model_name: str = "/home/dalhxwlyjsuo/criait_zhaotf/32B_RL_SFT"  # Default model name
    temperature: float = 0.0
    max_tokens: int = 32768

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Call the custom model server."""

        def convert_message(msg: BaseMessage):
            if isinstance(msg, HumanMessage):
                return {"role": "user", "content": msg.content}
            elif isinstance(msg, AIMessage):
                return {"role": "assistant", "content": msg.content}
            elif isinstance(msg, SystemMessage):
                return {"role": "system", "content": msg.content}
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [convert_message(msg) for msg in messages],
        }

        try:
            response = requests.post(self.model_url, json=payload, stream=False)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 透传 LangChain invoke(..., response_format=...) 等参数；兼容自定义服务端未实现的字段
        kwargs.pop("response_format", None)
        content = self._call(messages, stop=stop, **kwargs)
        generation = ChatGeneration(message=AIMessage(content=content))
        if run_manager:
            run_manager.on_llm_new_token(generation.message.content)
        return ChatResult(generations=[generation])

class QwenChatModel(BaseChatModel):
    """Qwen Chat Model that interfaces with the Qwen API."""

    model_name: str = "qwen3.5-plus"  # Default model name（与 fetch_default_llm_model 一致）
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    # DashScope 兼容模式对多数模型单次输出上限为 8192；超过会 400 invalid_request_error
    max_tokens: int = 8192

    @property
    def _llm_type(self) -> str:
        return "ali_api"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        def convert_message(msg: BaseMessage):
            if isinstance(msg, HumanMessage):
                return {"role": "user", "content": msg.content}
            elif isinstance(msg, AIMessage):
                return {"role": "assistant", "content": msg.content}
            elif isinstance(msg, SystemMessage):
                return {"role": "system", "content": msg.content}
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")

        if not (self.api_key or "").strip():
            raise ValueError(
                "DashScope API Key 为空：请在 mosaic/config/config.cfg 配置 [API_KEYS] ali_api_key，"
                "或设置环境变量 DASHSCOPE_API_KEY。"
            )
        if not (self.base_url or "").strip():
            raise ValueError(
                "DashScope base_url 为空：请配置 [API_KEYS] ali_base_url 或环境变量 DASHSCOPE_API_BASE。"
            )

        client = _get_dashscope_client(self.api_key, self.base_url)
        # 兼容模式单次 max_tokens 上限 8192（以接口报错为准）
        cap = 8192
        req_max = self.max_tokens if self.max_tokens and self.max_tokens > 0 else cap
        req_max = min(req_max, cap)
        # LangChain invoke(..., response_format=...) 等；仅透传 OpenAI 兼容字段
        response_format = kwargs.pop("response_format", None)
        create_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [convert_message(msg) for msg in messages],
            "temperature": self.temperature,
            "max_tokens": req_max,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        try:
            completion = client.chat.completions.create(**create_kwargs)
        except APIError as e:
            body = getattr(e, "body", None)
            _logger.error(
                "DashScope API 错误: status=%s message=%s body=%s base_url=%s model=%s",
                getattr(e, "status_code", None),
                str(e),
                body,
                (self.base_url or "").rstrip("/"),
                self.model_name,
            )
            raise
        except (APIConnectionError, APITimeoutError) as e:
            _logger.error(
                "DashScope 网络/超时: %s base_url=%s model=%s "
                "(若经代理握手慢可调 MOSAIC_HTTP_CONNECT_TIMEOUT；直连阿里云可对 dashscope 域名设 NO_PROXY)",
                e,
                (self.base_url or "").rstrip("/"),
                self.model_name,
            )
            raise

        choice = completion.choices[0].message
        if choice.content is None:
            _logger.warning("DashScope 返回 content 为空，model=%s", self.model_name)
        return choice.content or ""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._call(messages, stop=stop, **kwargs)
        generation = ChatGeneration(message=AIMessage(content=content))
        if run_manager:
            run_manager.on_llm_new_token(generation.message.content)
        return ChatResult(generations=[generation])

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model by its fully specified name."""
    model = provider = custom_model_url = ""
    if "|" in fully_specified_name:
        tokens = fully_specified_name.split("|")
        if len(tokens) == 2:
            provider, model = tokens
        elif len(tokens) == 3:
            provider, model, custom_model_url = tokens
            assert(provider == "custom")
        else:
            raise ValueError(f"Invalid format for chat model: {fully_specified_name}")
        # print(provider)
        # print(model)
        # print(custom_model_url)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "custom":
        assert(custom_model_url != "")
        print(f"Using CustomChatModel with URL: {custom_model_url}")
        return CustomChatModel(model_url=custom_model_url, model_name=model)
    if provider == "ali_api":
        #print(f"Using qwen model api, load model: {model}")
        return QwenChatModel(model_name=model, api_key=ali_api_key, base_url=ali_base_url)
    else:
        print(f"Using init_chat_model for provider: {provider}, model: {model}")
        return init_chat_model(model, model_provider=provider)



if __name__ == "__main__":
    llm = load_chat_model("ali_api|qwen3.5-plus")
    #llm = load_chat_model("ali_api|deepseek-r1")
    prompt = "你是谁"
    response = llm.invoke(prompt)
    print(response.content)
