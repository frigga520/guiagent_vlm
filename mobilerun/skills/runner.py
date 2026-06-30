"""
SkillRunner —— 端到端编排

流程：
    user_task
        ↓ SkillRetriever.best
    Skill (best match)
        ↓ SkillFiller.fill (slot extraction)
    rendered steps (slot already filled)
        ↓ DeviceExecutor.act per step
        ↓ on failure: SkillSelfHealer.heal → DeviceExecutor.act(new step)
    SkillExecutionResult
"""

from __future__ import annotations

import time
from typing import List, Optional, Protocol

from mobilerun.skills.filler import SkillFiller, SlotMissingError
from mobilerun.skills.retriever import SkillRetriever
from mobilerun.schema import Skill, SkillExecutionResult, SkillStep
from mobilerun.self_heal import ScreenElement, SkillSelfHealer
from mobilerun.skills.store import SkillStore


class DeviceExecutor(Protocol):
    """对外约定：runner 用这俩方法。
    由 AdbExecutor 实现。
    """

    def act(self, step: SkillStep) -> bool:
        ...

    def screen(self) -> List[ScreenElement]:
        ...


class NoMatchingSkillError(LookupError):
    pass


class SkillRunner:
    def __init__(
        self,
        store: SkillStore,
        retriever: SkillRetriever,
        filler: SkillFiller,
        healer: SkillSelfHealer,
        executor: DeviceExecutor,
        *,
        max_heal_attempts_per_step: int = 1,
    ):
        self.store = store
        self.retriever = retriever
        self.filler = filler
        self.healer = healer
        self.executor = executor
        self.max_heal_attempts_per_step = max_heal_attempts_per_step

    # ------------------------------------------------------------------
    def run(
        self,
        user_task: str,
        *,
        slot_overrides: Optional[dict] = None,
        log_callback=None,
    ) -> SkillExecutionResult:
        log = log_callback or (lambda msg: None)

        # 1) 召回
        best = self.retriever.best(user_task)
        if not best:
            raise NoMatchingSkillError(f"no skill matches: {user_task}")
        skill, score = best
        log(f"[Retrieve] hit {skill.id!r} (score={score:.2f})")

        # 2) 填参
        try:
            steps, slot_values = self.filler.fill(skill, user_task, overrides=slot_overrides)
        except SlotMissingError as e:
            skill.mark_used(success=False)
            self.store.put(skill, save=True)
            return SkillExecutionResult(
                skill_id=skill.id, success=False,
                steps_run=0, steps_total=len(skill.steps),
                slot_values={}, failed_reason=f"missing slot: {e.slot_name}",
            )
        log(f"[Fill] slots={slot_values}")

        # 3) 执行 + 自愈
        healed = 0
        for i, step in enumerate(steps, start=1):
            ok, healed_step = self._execute_with_heal(skill, step, log=log)
            if not ok:
                skill.mark_used(success=False)
                self.store.put(skill, save=True)
                return SkillExecutionResult(
                    skill_id=skill.id, success=False,
                    steps_run=i - 1, steps_total=len(steps),
                    healed_steps=healed,
                    slot_values=slot_values,
                    failed_reason=f"step {i} failed even after healing",
                )
            if healed_step is not None:
                healed += 1
                # 把修好的步骤写回 skill 模板（注意：渲染后的 step 是 deep_copy，
                # 我们要把修复结果反向同步到 skill.steps 的对应位置）
                self._sync_repair_to_template(skill, i - 1, healed_step)

        skill.mark_used(success=True)
        self.store.put(skill, save=True)
        return SkillExecutionResult(
            skill_id=skill.id, success=True,
            steps_run=len(steps), steps_total=len(steps),
            healed_steps=healed, slot_values=slot_values,
        )

    # ------------------------------------------------------------------
    def _execute_with_heal(self, skill: Skill, step: SkillStep, log):
        """返回 (ok, healed_step_or_None)。
        healed_step 非 None 表示这步是被自愈过的，调用方应回写模板。"""
        attempt = 0
        cur = step
        last_healed: Optional[SkillStep] = None
        while True:
            ok = self.executor.act(cur)
            if ok:
                log(f"[Act] OK  {cur.action_type} target={cur.target_text!r}")
                return True, last_healed
            log(f"[Act] FAIL {cur.action_type} target={cur.target_text!r}")
            if attempt >= self.max_heal_attempts_per_step:
                return False, last_healed
            attempt += 1
            elems = self.executor.screen()
            res = self.healer.heal(skill, cur, elems)
            log(f"[Heal] {res}")
            if not res.ok or not res.step:
                return False, last_healed
            cur = res.step
            last_healed = cur
            time.sleep(0.2)

    # ------------------------------------------------------------------
    def _sync_repair_to_template(self, skill: Skill, idx: int, repaired: SkillStep):
        """把成功修复的 target_text/coordinates 写回 skill 模板。
        要避开 slot 占位符位置（不要把 {{contact}} 替换成具体值！）"""
        tmpl = skill.steps[idx]
        # 只在原模板里没有 slot 占位符的情况下，才同步具体值
        if tmpl.target_text and "{{" not in tmpl.target_text:
            tmpl.target_text = repaired.target_text
        if repaired.coordinates:
            tmpl.coordinates = repaired.coordinates
        tmpl.repaired_count = repaired.repaired_count
