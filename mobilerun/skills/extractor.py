"""
SkillExtractor：把一次成功执行的"轨迹"蒸馏为参数化技能

输入：
- task_description: 用户当时说的话
- recorded_steps: 一组动作（list of dict，至少含 action_type / target_text / text）
- app_package: 可选

流程：
1. 让 LLM 判断哪些参数是"会变的"（联系人、消息内容、备忘录正文等）
2. LLM 用 {{slot}} 模板替换原始 text/target_text，并返回 slots 列表
3. 生成 Skill 对象（这里不写库，由调用方决定 store.put）
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

from mobilerun.llm import BaseLLM
from mobilerun.schema import Skill, SkillStep, Slot


_EXTRACT_PROMPT = """你是一个 GUI 自动化助手的"技能蒸馏器"。
用户的任务是：<task>{task}</task>
我把他完成这个任务时的动作录下来了：
<steps>{steps_json}</steps>

请从这些步骤里识别出"会变化的参数（slot）"。
- 不要把界面固有的按钮文字（如「设置」「确定」「显示」「保存」）当 slot
- 联系人姓名、消息正文、备忘录内容、搜索关键词、金额等才是 slot
- 把对应步骤的 target_text 或 text 改成 {{{{slot_name}}}} 占位

只返回一个 JSON：
{{
  "slots": [
    {{"name":"contact","type":"text","description":"联系人姓名","example":"张三"}},
    ...
  ],
  "steps": [<改写后的步骤数组，结构和原 steps 一致，只是把可变值替换成 {{{{slot}}}}>]
}}
不要任何额外解释、不要 Markdown。
"""


class SkillExtractor:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def extract(
        self,
        *,
        task_description: str,
        recorded_steps: List[Dict],
        app_package: Optional[str] = None,
        name_hint: Optional[str] = None,
    ) -> Skill:
        prompt = _EXTRACT_PROMPT.format(
            task=task_description,
            steps_json=json.dumps(recorded_steps, ensure_ascii=False),
        )
        rsp = self.llm.chat(prompt)
        payload = _extract_json(rsp) or {"slots": [], "steps": recorded_steps}

        slots = [Slot(**s) for s in payload.get("slots", []) if isinstance(s, dict)]
        steps_raw = payload.get("steps", recorded_steps)
        steps = [_to_skill_step(s, slots) for s in steps_raw]

        sid = _slug(name_hint or task_description)
        return Skill(
            id=sid,
            name=name_hint or task_description,
            description=task_description,
            app=app_package,
            slots=slots,
            steps=steps,
            origin="first_record",
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
_SLOT_PAT = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def _to_skill_step(raw: Dict, slots: List[Slot]) -> SkillStep:
    slot_names = {s.name for s in slots}
    # 自动给步骤里出现的 {{slot}} 一个 anchor description
    anchor = ""
    for field in ("target_text", "text"):
        v = raw.get(field) or ""
        for m in _SLOT_PAT.finditer(v):
            if m.group(1) in slot_names:
                anchor = anchor or f"包含 {field} = {v}"
    coords = raw.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        coords = [int(coords[0]), int(coords[1])]
    else:
        coords = None
    return SkillStep(
        action_type=raw.get("action_type") or raw.get("action") or "click",
        target_text=raw.get("target_text"),
        target_index=raw.get("target_index"),
        text=raw.get("text"),
        coordinates=coords,
        direction=raw.get("direction"),
        package=raw.get("package"),
        extra={k: v for k, v in raw.items()
               if k not in {"action_type", "action", "target_text", "target_index",
                            "text", "coordinates", "direction", "package"}},
        anchor_description=anchor or (raw.get("target_text") or ""),
        expected_post_text=raw.get("expected_post_text", ""),
    )


def _slug(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w一-龥]+", "_", s)
    return s.lower()[:60] or f"skill_{int(time.time())}"


def _extract_json(text: str) -> Optional[Dict]:
    """从 LLM 输出里抢一个 JSON 对象，容错 Markdown 包裹。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*", "", t).rstrip("`").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
