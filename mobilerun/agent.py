"""
GUIAgent —— Vision-grounded Android GUI Agent

主循环：
    user_task
        ↓
    for step in range(max_steps):
        screenshot = adb.screencap        # 真机截屏
        decision = vlm.chat(prompt, image=screenshot)
                                          # VLM 直接吐坐标
                                          # {"action":"tap","x":540,"y":1200,...}
        executor.do(decision)             # adb 物理像素 tap/type/swipe
        if decision.done: break
    return GUIAgentResult(trace, success)

设计要点：
- 截图缩放到 ≤1280 宽再发给模型（省 token），坐标按比例缩放回设备分辨率
- VLM 看到的就是用户看到的，不再走 UI 树（UI 树只留给 skill 自愈兜底）
- 输出严格 JSON：{thought, action, x, y, text?, direction?, package?, done}
- 失败、坐标越界都按"动作失败"处理，下一步让 LLM 自己看新屏幕重决策
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mobilerun.executor import AdbExecutor
from mobilerun.llm import BaseLLM


_SYSTEM = """你是一个操作 Android 手机的 GUI Agent。看屏幕截图 + 任务 → 输出**下一步**动作。

# 输出规范（最重要，违反一次任务就挂）
**只输出一个 JSON 对象**。不要 Markdown 代码块、不要 ```json、不要前后任何说明文字。
JSON 必须 100% 合法：
- 字符串两端必须配对的双引号，数字两侧**禁止**出现引号
  错：{{"y": 1105"}}    对：{{"y": 1105}}
- tap / long_press 必须**同时**给 "x" 和 "y" 两个键，**不能**写成 {{"x": 405, 1107}}
- 不能有末尾逗号、不能用单引号、不能用 Python 的 True/False/None

正确示例（你要照这个格式输出，但内容根据屏幕决定；x/y 在 [0,1000] 区间）：
{{"action": "tap", "x": 900, "y": 70, "thought": "点击右上角搜索图标", "done": false}}

# 可用动作（每次只能选一个）
- {{"action": "tap",       "x": <int>, "y": <int>, "thought": "...", "done": false}}
- {{"action": "long_press","x": <int>, "y": <int>, "thought": "...", "done": false}}
- {{"action": "type",      "text": "<要输入的文字>", "thought": "...", "done": false}}
       （type 前先确保焦点在输入框，必要时先 tap 输入框）
- {{"action": "swipe",     "direction": "up|down|left|right", "thought": "...", "done": false}}
- {{"action": "back",      "thought": "...", "done": false}}
- {{"action": "open_app",  "package": "com.tencent.mm", "thought": "...", "done": false}}
- {{"action": "wait",      "thought": "等页面加载", "done": false}}
- {{"action": "finish",    "thought": "任务完成", "done": true}}

# 坐标说明（极重要，写错就点不到）
- x, y 一律使用 **[0, 1000] 归一化坐标**（图像左上角 = (0,0)，右下角 = (1000,1000)）
  例：屏幕右上角的搜索图标，x ≈ 900~950，y ≈ 50~100
  例：屏幕正中间的元素，x ≈ 500，y ≈ 500
  例：屏幕底部 tab 栏的图标，y ≈ 950
- **不要**输出原始像素坐标，不要超过 1000
- 当前截图原始尺寸供你参考：宽 {img_w} × 高 {img_h}（**只用于估算比例**，输出仍是 [0,1000]）
- 系统会自动把归一化坐标缩放到设备物理像素并 tap

# 决策原则（避免常见坑）
1. **先看清当前屏幕在哪个 app / 哪个页面**（看顶部标题、底部 tab、键盘有没有弹出）
2. **若已经在目标页面或目标元素已可见，直接 tap / type，不要乱按 back**
   —— 例如打开微信后默认在"聊天"列表，要找联系人就**用搜索**，不要按 back 退出微信
3. **找联系人/应用/设置项的标准流程：用搜索，不要猜列表位置**
   微信：打开微信 → tap 右上角放大镜图标（≈ 截图右上角，y 大约在顶栏内）→
        输入名字 → tap 搜索结果里"联系人"分区的目标条目 → 进入聊天页 →
        tap 底部输入框 → type 消息 → tap"发送"按钮
   **不要在聊天列表里凭印象 tap 第几条**——你看不清谁是谁的时候必须走搜索
4. tap 的坐标要指向元素的**中心**，避免误点边缘或临近控件
5. 输入文字前确认焦点已在输入框（光标可见或键盘已弹出）
6. **如果上一步动作执行后页面没有变化或进入了错的页面**（参考"已经做过的步骤"），
   说明刚才的坐标错了 —— **换完全不同的策略**（比如改走搜索），**不要重复同样的 tap**
7. 只要任务**已经完成**（消息已发送 / 设置已开启 / 备忘录已保存），立刻输出 finish + done=true
"""


_USER_TEMPLATE = """## 用户任务
{task}

## 已经做过的步骤
{history}

请观察截图，输出下一步的 JSON。
"""


@dataclass
class GUIAgentStep:
    step_index: int
    thought: str
    action: Dict[str, Any]
    ok: bool
    error: str = ""


@dataclass
class GUIAgentResult:
    task: str
    success: bool
    steps: List[GUIAgentStep] = field(default_factory=list)
    stop_reason: str = ""

    def trace_for_skill(self) -> List[Dict[str, Any]]:
        """成功路径上的步骤序列（去掉失败步和 finish），可喂给 SkillExtractor"""
        out = []
        for s in self.steps:
            if not s.ok:
                continue
            a = s.action
            t = a.get("action")
            if t == "finish":
                continue
            entry = {"action_type": t}
            for k in ("text", "direction", "package"):
                if a.get(k):
                    entry[k] = a[k]
            if "x" in a and "y" in a:
                entry["coordinates"] = [int(a["x"]), int(a["y"])]
            out.append(entry)
        return out


class GUIAgent:
    """Vision-grounded GUI Agent.

    用法：
        from mobilerun.executor import AdbExecutor
        from mobilerun.llm import build_llm_from_env
        from mobilerun.agent import GUIAgent

        agent = GUIAgent(AdbExecutor(), build_llm_from_env(vision=True))
        result = agent.run("打开微信，给张三发『我到了』")
    """

    def __init__(
        self,
        executor: AdbExecutor,
        llm: BaseLLM,
        *,
        max_steps: int = 20,
        screenshot_max_width: int = 1280,
        loop_threshold: int = 3,
        screens_dir: Optional[str] = None,
        log=None,
    ):
        self.executor = executor
        self.llm = llm
        self.max_steps = max_steps
        self.screenshot_max_width = screenshot_max_width
        self.loop_threshold = loop_threshold
        self.screens_dir = Path(screens_dir) if screens_dir else None
        if self.screens_dir:
            self.screens_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or print

    # ------------------------------------------------------------------
    def run(self, task: str) -> GUIAgentResult:
        self.log(f"\n=== Task: {task} ===")
        result = GUIAgentResult(task=task, success=False)
        history: List[str] = []
        recent_actions: List[str] = []

        for step_i in range(1, self.max_steps + 1):
            self.log(f"\n--- Step {step_i} ---")

            # 1) 观察：截图 + 缩放
            try:
                raw_png = self.executor.screenshot()
            except Exception as e:
                result.stop_reason = f"screenshot failed: {e}"
                return result

            small_png, (img_w, img_h), scale = self._resize_png(raw_png)
            self.log(f"[Observe] image {img_w}x{img_h}  scale={scale:.3f}")

            if self.screens_dir:
                p = self.screens_dir / f"step_{step_i:02d}.png"
                p.write_bytes(small_png)
                self.log(f"[Observe] 已保存截图 {p}")

            # 2) 问 VLM（允许一次"只返回 JSON"的重试）
            prompt = self._build_prompt(task, history, img_w, img_h)
            decision = None
            rsp = ""
            for attempt in range(2):
                try:
                    p = prompt if attempt == 0 else (
                        prompt + "\n\n上次回复不是合法 JSON，请**只**输出一个 JSON 对象，"
                        "不要任何代码块/前后文字。"
                    )
                    rsp = self.llm.chat(p, image=small_png)
                except Exception as e:
                    step = GUIAgentStep(step_i, "", {}, False, f"LLM error: {e}")
                    result.steps.append(step)
                    result.stop_reason = f"llm error: {e}"
                    return result
                decision = self._parse(rsp)
                if decision is not None:
                    break
                self.log(f"[Warn]    解析失败，原文: {rsp[:200]!r} -- 重试")

            if decision is None:
                self.log(f"[Fail]    模型连续返回非 JSON。最后原文:\n{rsp[:400]}")
                step = GUIAgentStep(step_i, "", {}, False,
                                    f"non-JSON: {rsp[:200]}")
                result.steps.append(step)
                result.stop_reason = "non-JSON response"
                return result

            thought = decision.get("thought", "")
            action = self._normalize_action(decision)
            self.log(f"[Plan]    {thought[:80]}")
            self.log(f"[Plan]    {action}")

            # 3) 结束判定
            if action.get("action") == "finish" or decision.get("done"):
                step = GUIAgentStep(step_i, thought, action, True, "")
                result.steps.append(step)
                result.success = True
                result.stop_reason = "done"
                self.log(f"=== Done after {step_i} steps ===")
                return result

            # 4) 执行
            ok, err = self._dispatch(action, scale)
            self.log(f"[Act]     {'OK' if ok else 'FAIL'} {err}")
            step = GUIAgentStep(step_i, thought, action, ok, err)
            result.steps.append(step)
            history.append(self._summarize_step(action, ok))
            history = history[-10:]

            # 5) 死循环检测：只对 tap / long_press 起作用
            #    （back/swipe/wait 这种导航动作允许合法连击）
            if action.get("action") in ("tap", "long_press"):
                sig = self._action_signature(action)
                recent_actions.append(sig)
                recent_actions = recent_actions[-self.loop_threshold:]
                if (len(recent_actions) >= self.loop_threshold
                        and len(set(recent_actions)) == 1):
                    result.stop_reason = (
                        f"detected loop: 同一坐标连续 tap {self.loop_threshold} 次 ({sig})"
                    )
                    self.log(f"=== Stop: {result.stop_reason} ===")
                    self.log("    提示：模型反复点同一处但页面不前进，"
                             "说明 grounding 错了或目标不在当前屏。看 data/screens/ 对照。")
                    return result
            else:
                recent_actions.clear()  # 非 tap 动作打断了 tap 重复链

            time.sleep(0.4)

        result.stop_reason = f"reached max_steps={self.max_steps}"
        self.log(f"=== Stop: {result.stop_reason} ===")
        return result

    # ------------------------------------------------------------------
    def _resize_png(self, raw: bytes) -> Tuple[bytes, Tuple[int, int], float]:
        """缩到 ≤ screenshot_max_width 宽。
        返回 (new_png, (w,h), scale)。scale = device_pixel / image_pixel。"""
        try:
            from PIL import Image
        except ImportError:
            dev_w, dev_h = self.executor.width, self.executor.height
            return raw, (dev_w, dev_h), 1.0

        img = Image.open(io.BytesIO(raw))
        ow, oh = img.size
        if ow <= self.screenshot_max_width:
            new_img = img
            new_w, new_h = ow, oh
        else:
            ratio = self.screenshot_max_width / ow
            new_w = self.screenshot_max_width
            new_h = int(oh * ratio)
            new_img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        new_img.save(buf, format="PNG")
        dev_w, _ = self.executor.width, self.executor.height
        scale = dev_w / new_w if new_w else 1.0
        return buf.getvalue(), (new_w, new_h), scale

    # ------------------------------------------------------------------
    def _build_prompt(self, task: str, history: List[str],
                      img_w: int, img_h: int) -> str:
        hist = "\n".join(history) or "(无)"
        return (
            _SYSTEM.format(img_w=img_w, img_h=img_h)
            + "\n"
            + _USER_TEMPLATE.format(task=task, history=hist)
        )

    @staticmethod
    def _parse(rsp: str) -> Optional[Dict]:
        t = (rsp or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*", "", t).rstrip("`").strip()

        candidates = [t]
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            candidates.append(m.group(0))

        for c in candidates:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(GUIAgent._repair_json(c))
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _repair_json(s: str) -> str:
        """容错修复 VLM 输出里的常见 JSON 小毛病：
           - 数字后多余引号：  "y": 1105"   ->  "y": 1105
           - 漏掉键名：       "x": 405, 1107  ->  "x": 405, "y": 1107
           - 末尾逗号：       , }            ->  }
        """
        s = re.sub(r'(-?\d+(?:\.\d+)?)\s*"(\s*[,}\]])', r'\1\2', s)
        if '"y"' not in s:
            s = re.sub(
                r'("x"\s*:\s*-?\d+(?:\.\d+)?\s*,\s*)(-?\d+(?:\.\d+)?)',
                r'\1"y": \2', s,
            )
        s = re.sub(r',(\s*[}\]])', r'\1', s)
        return s

    @staticmethod
    def _normalize_action(decision: Dict) -> Dict[str, Any]:
        action: Dict[str, Any] = {}
        a = decision.get("action") or decision.get("action_type")
        if isinstance(a, dict):
            action.update(a)
            a = a.get("type") or a.get("action") or a.get("action_type")
        action["action"] = a
        for k in ("x", "y", "text", "direction", "package"):
            if k in decision and decision[k] is not None:
                action[k] = decision[k]
        return action

    # ------------------------------------------------------------------
    def _dispatch(self, action: Dict[str, Any], scale: float
                  ) -> Tuple[bool, str]:
        a = action.get("action")
        try:
            if a == "tap":
                x, y = self._to_device_px(action)
                if x is None:
                    return False, "tap 缺少 x/y"
                self.executor.tap(x, y)
                action["_device_xy"] = (x, y)
                return True, f"tap ({x},{y})"
            if a == "long_press":
                x, y = self._to_device_px(action)
                if x is None:
                    return False, "long_press 缺少 x/y"
                from mobilerun.executor import _adb_raw
                _adb_raw(["-s", self.executor.serial, "shell", "input",
                          "swipe", str(x), str(y), str(x), str(y), "1000"])
                time.sleep(self.executor.action_pause)
                return True, f"long_press ({x},{y})"
            if a == "type":
                text = action.get("text", "")
                if not text:
                    return False, "type 缺少 text"
                self.executor.type_text(text)
                return True, f"type {text!r}"
            if a == "swipe":
                direction = action.get("direction", "up")
                from mobilerun.schema import SkillStep
                step = SkillStep(action_type="swipe", direction=direction)
                ok = self.executor.act(step)
                return ok, f"swipe {direction}"
            if a == "back":
                self.executor.back()
                return True, "back"
            if a == "wait":
                time.sleep(1.0)
                return True, "wait"
            if a == "open_app":
                pkg = action.get("package")
                if not pkg:
                    return False, "open_app 缺少 package"
                self.executor.open_app(pkg)
                return True, f"open {pkg}"
            return False, f"unknown action: {a}"
        except Exception as e:
            return False, f"dispatch error: {e}"

    @staticmethod
    def _scale_xy(action: Dict, scale: float
                  ) -> Tuple[Optional[int], Optional[int]]:
        x = action.get("x")
        y = action.get("y")
        if x is None or y is None:
            return None, None
        return int(round(float(x) * scale)), int(round(float(y) * scale))

    def _to_device_px(self, action: Dict) -> Tuple[Optional[int], Optional[int]]:
        """模型坐标 → 设备像素坐标。

        约定：模型输出 [0, 1000] 归一化坐标（Qwen2-VL/3-VL 标准）。
        若任一坐标 > 1000，则认为模型输出的是图像绝对像素，做兼容处理。
        """
        x = action.get("x")
        y = action.get("y")
        if x is None or y is None:
            return None, None
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            return None, None
        dev_w, dev_h = self.executor.width, self.executor.height
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            px = int(round(x * dev_w / 1000.0))
            py = int(round(y * dev_h / 1000.0))
        else:
            px, py = int(round(x)), int(round(y))
        px = max(0, min(px, dev_w - 1))
        py = max(0, min(py, dev_h - 1))
        return px, py

    @staticmethod
    def _action_signature(action: Dict) -> str:
        """循环检测用的动作签名：动作类型 + 关键参数。"""
        a = action.get("action", "?")
        bits = [a]
        if "x" in action and "y" in action:
            bits.append(f"{action['x']},{action['y']}")
        for k in ("text", "direction", "package"):
            if action.get(k):
                bits.append(f"{k}={action[k]}")
        return "|".join(bits)

    @staticmethod
    def _summarize_step(action: Dict, ok: bool) -> str:
        a = action.get("action")
        bits = [a or "?"]
        if "x" in action and "y" in action:
            bits.append(f"({action['x']},{action['y']})")
        if action.get("text"):
            bits.append(f"text={action['text']!r}")
        if action.get("direction"):
            bits.append(action["direction"])
        if action.get("package"):
            bits.append(action["package"])
        bits.append("OK" if ok else "FAIL")
        return " ".join(bits)
