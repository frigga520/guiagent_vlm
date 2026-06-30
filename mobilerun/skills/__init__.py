"""
Skill Library —— GUIAgent 的可选加速子模块

技能库本身不能独立完成任务，它只是给 GUIAgent 加一条"快路径"：
当用户的新任务和库里某条技能语义相近、参数有差异时，跳过 agent
主循环里的 VLM 推理，直接用老步骤+新参数完成。

不接技能库的话，GUIAgent 也能跑通（只是每次都从零观察+规划）。

公开 API：
    Skill / SkillStep / Slot / SkillExecutionResult     数据结构
    SkillStore        —— JSON 持久化
    SkillRetriever    —— 语义召回（OpenAI/Qwen embedding）
    SkillExtractor    —— 把 GUIAgent 跑成功的 trace 蒸馏成参数化技能
    SkillFiller       —— 用户任务 → 填参后的步骤
    SkillSelfHealer   —— 步骤失效时重定位（兜底再走 UI 树文本相似度）
    SkillRunner       —— 端到端编排：召回 → 填参 → 执行 → 自愈
"""

from mobilerun.schema import (
    Skill, SkillExecutionResult, SkillStep, Slot,
)
from mobilerun.self_heal import ScreenElement, SkillSelfHealer
from mobilerun.skills.extractor import SkillExtractor
from mobilerun.skills.filler import SkillFiller, SlotMissingError
from mobilerun.skills.retriever import (
    BaseEmbedder, OpenAICompatibleEmbedder, SkillRetriever,
    build_embedder_from_env,
)
from mobilerun.skills.runner import NoMatchingSkillError, SkillRunner
from mobilerun.skills.store import SkillStore

__all__ = [
    "Skill", "SkillStep", "Slot", "SkillExecutionResult",
    "ScreenElement",
    "SkillStore", "SkillRetriever",
    "BaseEmbedder", "OpenAICompatibleEmbedder", "build_embedder_from_env",
    "SkillExtractor", "SkillFiller", "SkillSelfHealer", "SkillRunner",
    "SlotMissingError", "NoMatchingSkillError",
]
