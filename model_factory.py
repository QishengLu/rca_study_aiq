"""
model_factory.py — 统一模型工厂（thinkdepthai / deerflow / aiq 共用）

所有模型通过同一个 shubiaobiao proxy 调用，同一个 API key。
仅 Claude 系列因 shubiaobiao client 校验必须走 ChatAnthropic SDK，
其余（gemini/gpt/qwen/kimi/deepseek/grok/glm）统一走 ChatOpenAI SDK。

用法：
    from model_factory import create_model
    model = create_model("claude-sonnet-4-6")       # → ChatAnthropic
    model = create_model("gemini-3.1-pro-preview")  # → ChatOpenAI
    model = create_model("gpt-5.3-chat")            # → ChatOpenAI
    model = create_model("qwen3.5-plus")            # → ChatOpenAI
"""

import os
from langchain_core.language_models import BaseChatModel


def create_model(model_name: str, *, max_tokens: int = 32768) -> BaseChatModel:
    """根据 model_name 创建 LangChain ChatModel。

    Claude → ChatAnthropic（shubiaobiao 要求 Anthropic SDK）
    其余 → ChatOpenAI（shubiaobiao OpenAI-compatible）
    """
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.shubiaobiao.cn/v1")

    if model_name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        # Anthropic SDK 要求 base_url 不含 /v1
        anthropic_url = base_url.rstrip("/")
        if anthropic_url.endswith("/v1"):
            anthropic_url = anthropic_url[:-3]

        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            base_url=anthropic_url,
            max_tokens=max_tokens,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=0,
        )
