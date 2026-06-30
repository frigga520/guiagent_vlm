"""
Vision-grounded Android GUI Agent.

主要 API：
    GUIAgent       —— vision-grounded 主循环（观察→规划→执行→自检）
    AdbExecutor    —— 真机 ADB 控制
    build_llm_from_env(vision=True) —— 一键拿 VLM 客户端

可选子模块 .skills：参数化技能库，加速重复任务（不接也不影响 agent 跑通）
"""

from mobilerun.agent import GUIAgent, GUIAgentResult, GUIAgentStep
from mobilerun.executor import AdbExecutor
from mobilerun.llm import BaseLLM, OpenAICompatibleLLM, build_llm_from_env

__all__ = [
    "GUIAgent",
    "GUIAgentResult",
    "GUIAgentStep",
    "AdbExecutor",
    "BaseLLM",
    "OpenAICompatibleLLM",
    "build_llm_from_env",
]

__version__ = "0.1.0"
