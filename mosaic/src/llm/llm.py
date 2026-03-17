import requests
from typing import Optional, List, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage, ChatResult
from langchain_core.outputs import ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
import configparser

# Read API keys from config file
config = configparser.ConfigParser()
config.read('D:/model/conv/GraphConv/oop_graph/config/config.cfg')
ali_api_key = config.get('API_KEYS', 'ali_api_key')
ali_base_url = config.get('API_KEYS', 'ali_base_url')

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
        content = self._call(messages, stop=stop, **kwargs)
        generation = ChatGeneration(message=AIMessage(content=content))
        if run_manager:
            run_manager.on_llm_new_token(generation.message.content)
        return ChatResult(generations=[generation])

class QwenChatModel(BaseChatModel):
    """Qwen Chat Model that interfaces with the Qwen API."""

    model_name: str = "qwen-plus"  # Default model name
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 32768

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

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[convert_message(msg) for msg in messages],
        )
        return completion.choices[0].message.content

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
    llm = load_chat_model("ali_api|qwen-max-latest")
    #llm = load_chat_model("ali_api|deepseek-r1")
    prompt = "你是谁"
    response = llm.invoke(prompt)
    print(response.content)
