"""
LLM 客户端 —— 文本 + 多模态

后端：OpenAI 兼容 API（OpenAI / 通义千问 DashScope / DeepSeek / 任何 openai 协议端点）
支持发送图片（base64 编码的 PNG）给 VLM 做 grounding。

接口：
    chat(prompt: str, *, image: Optional[bytes] = None) -> str
"""

from __future__ import annotations

import base64
import os
from typing import Optional


class BaseLLM:
    def chat(self, prompt: str, *, image: Optional[bytes] = None) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.0, max_tokens: int = 800,
                 timeout: float = 60.0,
                 image_mime: str = "image/png"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.image_mime = image_mime

    def chat(self, prompt: str, *, image: Optional[bytes] = None) -> str:
        import requests

        if image is not None:
            b64 = base64.b64encode(image).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{self.image_mime};base64,{b64}"}},
            ]
        else:
            content = prompt

        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def build_llm_from_env(prefer: str = "auto", *, vision: bool = False) -> BaseLLM:
    """根据环境变量构造 LLM：
       prefer = 'openai' | 'qwen' | 'auto'
       vision=True：自动选会看图的模型默认值
       auto: OPENAI_API_KEY > DASHSCOPE_API_KEY
       缺失时抛错。
    """
    default_openai_model = (
        os.environ.get("OPENAI_CHAT_MODEL")
        or ("gpt-4o" if vision else "gpt-4o-mini")
    )
    default_qwen_model = (
        os.environ.get("QWEN_CHAT_MODEL")
        or ("qwen-vl-max" if vision else "qwen-plus")
    )

    if prefer in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleLLM(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ["OPENAI_API_KEY"],
            model=default_openai_model,
        )
    if prefer in ("auto", "qwen") and os.environ.get("DASHSCOPE_API_KEY"):
        return OpenAICompatibleLLM(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model=default_qwen_model,
        )
    raise RuntimeError(
        "未配置 LLM。请设置环境变量 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。"
    )
