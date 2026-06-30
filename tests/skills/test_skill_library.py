"""
技能库测试

测试策略：
- 主线测试：不调真 LLM、不连真机，但**也不引入 src 里的 mock**。
  用本文件内定义的 _StubLLM 模拟少量 LLM 调用，验证流水线串得通 + 数据结构正确。
- 真机集成测试：仅在设置 RUN_DEVICE_TESTS=1 + 连了设备时才跑。

跑法：
    python tests/skills/test_skill_library.py
    RUN_DEVICE_TESTS=1 python tests/skills/test_skill_library.py   # 含真机
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from mobilerun.llm import BaseLLM
from mobilerun.skills import (
    Skill, SkillStep, Slot, SkillStore,
    SkillExtractor, SkillFiller, SlotMissingError,
    SkillSelfHealer, ScreenElement,
    BaseEmbedder, SkillRetriever,
)


# ============================================================
# 测试用最简 stub（不进生产包）
# ============================================================
class _StubLLM(BaseLLM):
    """根据 prompt 关键词返回预设 JSON。仅用于测试，绝不上生产。"""

    def __init__(self, fixed: dict):
        self.fixed = fixed

    def chat(self, prompt: str) -> str:
        import json
        return json.dumps(self.fixed, ensure_ascii=False)


class _StubEmbedder(BaseEmbedder):
    """根据字符串长度模拟向量，长度相近的得分高（仅用于测试）。"""
    dim = 8

    def embed(self, text: str):
        # 取 8 个特征：前 8 字符的 unicode codepoint mod 100
        v = [0.0] * self.dim
        for i, ch in enumerate(text[:self.dim]):
            v[i] = (ord(ch) % 100) / 100.0
        # L2 归一化
        import math
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


# ============================================================
# 纯逻辑单元测试
# ============================================================
def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "skills.json")
        store = SkillStore(path)
        skill = Skill(
            id="t1", name="测试", description="测试技能",
            steps=[SkillStep(action_type="click", target_text="X")],
        )
        store.put(skill)
        assert "t1" in store

        store2 = SkillStore(path)
        loaded = store2.get("t1")
        assert loaded.name == "测试"
        assert loaded.steps[0].target_text == "X"


def test_filler_raises_when_slot_missing():
    skill = Skill(
        id="t", name="x", description="x",
        slots=[Slot(name="contact", required=True)],
        steps=[SkillStep(action_type="click", target_text="{{contact}}")],
    )
    filler = SkillFiller(llm=_StubLLM({}))
    try:
        filler.render_steps(skill, {})
    except SlotMissingError as e:
        assert e.slot_name == "contact"
    else:
        raise AssertionError("应抛 SlotMissingError")


def test_filler_renders_slots():
    skill = Skill(
        id="t", name="x", description="x",
        slots=[Slot(name="contact"), Slot(name="message")],
        steps=[
            SkillStep(action_type="click", target_text="{{contact}}"),
            SkillStep(action_type="type", text="{{message}}"),
        ],
    )
    filler = SkillFiller(llm=_StubLLM({"contact": "李四", "message": "我吃完了"}))
    steps, values = filler.fill(skill, "随便什么任务文本")
    assert values["contact"] == "李四"
    assert steps[0].target_text == "李四"
    assert steps[1].text == "我吃完了"


def test_extractor_uses_llm_output():
    extractor = SkillExtractor(llm=_StubLLM({
        "slots": [{"name": "msg", "description": "消息"}],
        "steps": [
            {"action_type": "click", "target_text": "微信"},
            {"action_type": "type", "text": "{{msg}}"},
        ],
    }))
    skill = extractor.extract(
        task_description="发消息",
        recorded_steps=[
            {"action_type": "click", "target_text": "微信"},
            {"action_type": "type", "text": "原始消息"},
        ],
    )
    assert {s.name for s in skill.slots} == {"msg"}
    assert skill.steps[1].text == "{{msg}}"


def test_healer_text_similarity():
    # 用一个永远不被调用的 stub，因为 text_sim 命中后不会落 LLM
    healer = SkillSelfHealer(llm=_StubLLM({}), sim_threshold=0.55)
    skill = Skill(id="x", name="x", description="x",
                  steps=[SkillStep(action_type="click", target_text="深色模式")])
    elems = [
        ScreenElement(text="亮度", clickable=True, bounds=((0, 0), (200, 100))),
        ScreenElement(text="深色模式", clickable=True, bounds=((0, 200), (200, 300))),
        ScreenElement(text="字体", clickable=True, bounds=((0, 400), (200, 500))),
    ]
    res = healer.heal(skill, skill.steps[0], elems)
    assert res.ok and res.strategy == "text_sim"
    assert res.step.target_text == "深色模式"


def test_healer_llm_pick():
    # text_sim 阈值卡死，强迫走 LLM；stub 返回 chosen_id=1（即查找）
    healer = SkillSelfHealer(
        llm=_StubLLM({"chosen_id": 1, "reason": "stub"}),
        sim_threshold=0.99,
    )
    skill = Skill(id="x", name="x", description="x",
                  steps=[SkillStep(action_type="click", target_text="搜索",
                                   anchor_description="搜索框")])
    elems = [
        ScreenElement(text="查找", clickable=True, bounds=((0, 0), (200, 100))),
        ScreenElement(text="通讯录", clickable=True, bounds=((0, 200), (200, 300))),
    ]
    res = healer.heal(skill, skill.steps[0], elems)
    assert res.ok and res.strategy == "llm", res
    assert res.step.target_text == "查找"


def test_retriever_indexes_and_returns_best():
    with tempfile.TemporaryDirectory() as td:
        store = SkillStore(os.path.join(td, "s.json"))
        embedder = _StubEmbedder()
        ret = SkillRetriever(store, embedder, score_threshold=0.0)
        for sid, desc in [("a", "发微信消息"), ("b", "调亮度")]:
            s = Skill(id=sid, name=sid, description=desc,
                      steps=[SkillStep(action_type="click", target_text="x")])
            ret.index_skill(s)
            store.put(s)
        hit = ret.best("发个微信消息")
        assert hit is not None
        assert hit[0].id == "a"


# ============================================================
# 真机集成测试（默认跳过）
# ============================================================
def test_real_device_screen_works():
    if os.environ.get("RUN_DEVICE_TESTS") != "1":
        print("   skipped (RUN_DEVICE_TESTS != 1)")
        return
    from mobilerun.skills import AdbExecutor
    exe = AdbExecutor()
    elems = exe.screen()
    assert isinstance(elems, list)
    assert len(elems) > 0, "屏幕上没抓到任何控件？"
    print(f"   抓到 {len(elems)} 个控件")


# ============================================================
ALL_TESTS = [
    test_store_roundtrip,
    test_filler_raises_when_slot_missing,
    test_filler_renders_slots,
    test_extractor_uses_llm_output,
    test_healer_text_similarity,
    test_healer_llm_pick,
    test_retriever_indexes_and_returns_best,
    test_real_device_screen_works,
]


def main():
    fails = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"[OK]   {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception:
            fails += 1
            print(f"[ERR]  {fn.__name__}")
            traceback.print_exc()
    total = len(ALL_TESTS)
    print(f"\nSummary: {total - fails}/{total} passed")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
