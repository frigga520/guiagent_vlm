"""
SkillSelfHealer：技能某一步执行失败时，借助当前屏幕状态把它"修好"

触发场景：
- 旧 skill 录制的 target_text="显示"，但 App 改版后变成了"显示设置"
- 旧的 coordinates 失效（屏幕分辨率不同 / 控件移位）

修复策略（轻到重）：
1. **text 相似度**：把当前屏幕里所有可点击元素的文字和 step.anchor_description 比相似度，挑最相似的替换 target_text + 重算坐标
2. **LLM 重定位**：把屏幕文字 + anchor 喂给 LLM，让它选一个 target_id
3. **VLM 视觉重定位**：给截图 + anchor，VLM 返回新坐标（留接口，本作业不强制实现）

修好后：
- 更新 step.target_text / coordinates
- step.repaired_count += 1
- Skill.last_repaired_at = now
- 调用方负责把更新后的 Skill 写回 store
"""

from __future__ import annotations

import difflib
import json
import re
import time
from typing import Dict, List, Optional

from mobilerun.llm import BaseLLM
from mobilerun.schema import Skill, SkillStep


class ScreenElement:
    """统一描述当前屏幕一个交互元素，自愈用。
    可以从 mobilerun.tools 的 UI 树适配过来。
    """
    __slots__ = ("text", "content_desc", "bounds", "clickable")

    def __init__(self, text: str = "", content_desc: str = "",
                 bounds=((0, 0), (0, 0)), clickable: bool = False):
        self.text = text or ""
        self.content_desc = content_desc or ""
        self.bounds = bounds
        self.clickable = clickable

    @property
    def label(self) -> str:
        return self.text or self.content_desc

    @property
    def center(self):
        (x1, y1), (x2, y2) = self.bounds
        return (x1 + x2) // 2, (y1 + y2) // 2


class HealResult:
    def __init__(self, ok: bool, step: Optional[SkillStep] = None,
                 reason: str = "", strategy: str = ""):
        self.ok = ok
        self.step = step
        self.reason = reason
        self.strategy = strategy

    def __repr__(self):
        return f"HealResult(ok={self.ok}, strategy={self.strategy}, reason={self.reason!r})"


# ----------------------------------------------------------------------
# Healer
# ----------------------------------------------------------------------
_LLM_HEAL_PROMPT = """你是一个 GUI Agent 的"自愈助手"。
某一步动作描述：<anchor>{anchor}</anchor>
原本它想点击的目标文字是：<old_target>{old_target}</old_target>
但是当前屏幕上找不到这个文字。下面是屏幕上所有可点击元素：

<elements>{elements_json}</elements>

请在元素列表里挑出最有可能对应『{anchor}』的那一个。只返回 JSON：
{{"chosen_id": <数字编号，从 1 开始>, "reason":"..."}}
找不到合适的就返回 {{"chosen_id": null, "reason":"..."}}
不要 Markdown，不要额外解释。
"""


class SkillSelfHealer:
    def __init__(self, llm: BaseLLM,
                 sim_threshold: float = 0.55,
                 use_llm_fallback: bool = True):
        self.llm = llm
        self.sim_threshold = sim_threshold
        self.use_llm_fallback = use_llm_fallback

    # ------------------------------------------------------------------
    def heal(self, skill: Skill, step: SkillStep,
             elements: List[ScreenElement]) -> HealResult:
        if step.action_type not in ("click", "long_press", "type"):
            return HealResult(False, reason="non-locatable action")
        if not elements:
            return HealResult(False, reason="empty screen")

        # 1) text 相似度
        winner = self._best_text_match(step, elements)
        if winner:
            elem, sim = winner
            new_step = self._apply(step, elem)
            new_step.repaired_count += 1
            skill.last_repaired_at = _now()
            return HealResult(True, step=new_step,
                              reason=f"text similarity={sim:.2f} → '{elem.label}'",
                              strategy="text_sim")

        # 2) LLM 重定位
        if self.use_llm_fallback:
            elem = self._llm_pick(step, elements)
            if elem:
                new_step = self._apply(step, elem)
                new_step.repaired_count += 1
                skill.last_repaired_at = _now()
                return HealResult(True, step=new_step,
                                  reason=f"LLM picked '{elem.label}'",
                                  strategy="llm")

        return HealResult(False, reason="no candidate matched")

    # ------------------------------------------------------------------
    # 策略 1：文本相似度
    # ------------------------------------------------------------------
    def _best_text_match(self, step: SkillStep,
                         elements: List[ScreenElement]
                         ) -> Optional[tuple[ScreenElement, float]]:
        query = (step.target_text or step.anchor_description or "").strip()
        if not query:
            return None
        best, best_sim = None, 0.0
        for e in elements:
            if not e.clickable:
                continue
            label = e.label.strip()
            if not label:
                continue
            sim = _string_sim(query, label)
            # 包含关系直接加大
            if query in label or label in query:
                sim = max(sim, 0.8)
            if sim > best_sim:
                best, best_sim = e, sim
        if best and best_sim >= self.sim_threshold:
            return best, best_sim
        return None

    # ------------------------------------------------------------------
    # 策略 2：LLM 重定位
    # ------------------------------------------------------------------
    def _llm_pick(self, step: SkillStep,
                  elements: List[ScreenElement]) -> Optional[ScreenElement]:
        anchor = step.anchor_description or step.target_text or ""
        if not anchor:
            return None
        payload = [
            {"id": i + 1, "text": e.label, "clickable": e.clickable}
            for i, e in enumerate(elements)
        ]
        prompt = _LLM_HEAL_PROMPT.format(
            anchor=anchor,
            old_target=step.target_text or "",
            elements_json=json.dumps(payload, ensure_ascii=False),
        )
        rsp = self.llm.chat(prompt)
        try:
            data = json.loads(rsp.strip().strip("`"))
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", rsp, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        cid = data.get("chosen_id")
        if isinstance(cid, int) and 1 <= cid <= len(elements):
            return elements[cid - 1]
        return None

    # ------------------------------------------------------------------
    def _apply(self, step: SkillStep, elem: ScreenElement) -> SkillStep:
        new = step.model_copy(deep=True)
        new.target_text = elem.label
        cx, cy = elem.center
        new.coordinates = [int(cx), int(cy)]
        return new


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _string_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
