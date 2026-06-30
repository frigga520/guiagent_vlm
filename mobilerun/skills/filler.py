"""
SkillFiller：用户任务 + 参数化技能 → 已填参的 SkillStep 列表
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from mobilerun.llm import BaseLLM
from mobilerun.schema import Skill, SkillStep, Slot


_FILL_PROMPT = """你是一个 GUI Agent 的"参数提取器"。
用户当前任务：<user_task>{task}</user_task>
我已经有一个可复用的技能，它需要这些参数：
<slot_spec>{slots_json}</slot_spec>

请从用户任务里提取出每个 slot 的值。只返回一个 JSON：
{{
  "<slot_name>": "<提取的值>",
  ...
}}
- 如果某个 slot 没法从任务里提取，对应键值留空字符串 ""
- 不要任何额外解释、不要 Markdown
"""


_SLOT_PAT = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class SlotMissingError(ValueError):
    def __init__(self, slot_name: str):
        super().__init__(f"required slot '{slot_name}' not provided")
        self.slot_name = slot_name


class SkillFiller:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # ------------------------------------------------------------------
    # 入口 1：自动从用户任务提取 slot 值
    # ------------------------------------------------------------------
    def fill_from_task(self, skill: Skill, user_task: str) -> Dict[str, str]:
        if not skill.slots:
            return {}
        prompt = _FILL_PROMPT.format(
            task=user_task,
            slots_json=json.dumps([s.model_dump() for s in skill.slots],
                                  ensure_ascii=False),
        )
        rsp = self.llm.chat(prompt)
        try:
            data = json.loads(rsp.strip().strip("`"))
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", rsp, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        # 只保留 spec 里声明过的 slot
        names = {s.name for s in skill.slots}
        return {k: v for k, v in data.items() if k in names and isinstance(v, str)}

    # ------------------------------------------------------------------
    # 入口 2：用给定的 slot_values 渲染步骤
    # ------------------------------------------------------------------
    def render_steps(self, skill: Skill, slot_values: Dict[str, str]) -> List[SkillStep]:
        # 必填校验
        for s in skill.slots:
            if s.required and not slot_values.get(s.name):
                raise SlotMissingError(s.name)

        rendered: List[SkillStep] = []
        for step in skill.steps:
            new = step.model_copy(deep=True)
            new.target_text = _apply(step.target_text, slot_values)
            new.text = _apply(step.text, slot_values)
            new.anchor_description = _apply(step.anchor_description, slot_values)
            new.expected_post_text = _apply(step.expected_post_text, slot_values)
            rendered.append(new)
        return rendered

    # ------------------------------------------------------------------
    # 一步到位
    # ------------------------------------------------------------------
    def fill(self, skill: Skill, user_task: str,
             overrides: Optional[Dict[str, str]] = None
             ) -> tuple[List[SkillStep], Dict[str, str]]:
        values = self.fill_from_task(skill, user_task)
        if overrides:
            values.update({k: v for k, v in overrides.items() if v})
        return self.render_steps(skill, values), values


def _apply(template: Optional[str], values: Dict[str, str]) -> Optional[str]:
    if not template:
        return template

    def repl(m):
        name = m.group(1)
        return values.get(name, m.group(0))

    return _SLOT_PAT.sub(repl, template)
