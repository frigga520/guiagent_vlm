"""
AdbExecutor —— 真机执行器

只用 Python 标准库 + adb 命令行，不需要在手机上装任何额外控制 APK。

约束：
- adb 必须在 PATH 里
- 手机连 USB 或同网段 adb tcpip 连接，开发者模式开 USB 调试
- adb shell input text 默认只支持 ASCII；中文输入需在手机装 ADBKeyBoard
  （详见 README 的"真机演示"小节）

提供两个核心方法以满足 SkillRunner 的 DeviceExecutor 协议：
    screen() -> List[ScreenElement]   # 解析当前 UI 树
    act(step: SkillStep) -> bool      # 执行 click / type / swipe / back / open_app
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

from mobilerun.schema import SkillStep
from mobilerun.self_heal import ScreenElement


class AdbError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# AdbExecutor
# ----------------------------------------------------------------------
class AdbExecutor:
    def __init__(self, serial: str = "", *, action_pause: float = 0.6,
                 wait_after_open: float = 2.0):
        """
        Args:
            serial: 指定设备 id（adb devices 出来的那个）。空表示用第一个设备。
            action_pause: 每个动作执行后暂停（让 UI 反应）
            wait_after_open: open_app 后多等一会儿等首屏起来
        """
        self.serial = serial or self._auto_pick_serial()
        self.action_pause = action_pause
        self.wait_after_open = wait_after_open
        self.width, self.height = self._device_size()

    # ------------------------------------------------------------------
    # 工厂：批量获取所有连接的设备
    # ------------------------------------------------------------------
    @staticmethod
    def list_devices() -> List[str]:
        out = _adb_raw(["devices"])
        devs = []
        for line in out.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devs.append(parts[0])
        return devs

    def _auto_pick_serial(self) -> str:
        devs = self.list_devices()
        if not devs:
            raise AdbError("没找到任何 adb 设备。请插 USB / 开 USB 调试 / 信任本电脑。")
        return devs[0]

    # ------------------------------------------------------------------
    # 设备元信息
    # ------------------------------------------------------------------
    def _adb(self, *args: str, timeout: float = 30.0) -> str:
        return _adb_raw(["-s", self.serial, *args], timeout=timeout)

    def _device_size(self) -> Tuple[int, int]:
        out = self._adb("shell", "wm", "size")
        m = re.search(r"(\d+)x(\d+)", out)
        if not m:
            raise AdbError(f"无法解析屏幕尺寸: {out!r}")
        return int(m.group(1)), int(m.group(2))

    # ------------------------------------------------------------------
    # screenshot()：抓屏幕图（PNG bytes + 设备分辨率）
    # 这是 vision-grounded agent 的主要观察接口
    # ------------------------------------------------------------------
    def screenshot(self) -> bytes:
        """直接拉取一张 PNG。adb exec-out 比 shell screencap 然后 pull 快很多。"""
        try:
            r = subprocess.run(
                ["adb", "-s", self.serial, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=15.0,
            )
        except subprocess.TimeoutExpired as e:
            raise AdbError("截图超时") from e
        if r.returncode != 0:
            raise AdbError(f"截图失败: {r.stderr.decode(errors='ignore').strip()}")
        return r.stdout

    def tap(self, x: int, y: int):
        """直接按坐标点击（vision agent 用）"""
        self._adb("shell", "input", "tap", str(x), str(y))
        self._pause()

    def type_text(self, text: str):
        """直接输入文字。中文走 ADBKeyBoard，ASCII 走原生。"""
        if _is_ascii(text):
            escaped = text.replace(" ", "%s").replace("'", "")
            self._adb("shell", "input", "text", escaped)
        else:
            ok = self._type_via_adbkeyboard(text)
            if not ok:
                self._adb("shell", "input", "text", text)
        self._pause()

    def back(self):
        self._adb("shell", "input", "keyevent", "KEYCODE_BACK")
        self._pause()

    def open_app(self, package: str):
        self._adb("shell", "monkey", "-p", package,
                  "-c", "android.intent.category.LAUNCHER", "1")
        time.sleep(self.wait_after_open)

    # ------------------------------------------------------------------
    # screen()：抓 UI 树（保留给 skill self-heal 兜底用，agent 主循环不再依赖）
    # ------------------------------------------------------------------
    def screen(self) -> List[ScreenElement]:
        xml_path = self._pull_uiautomator_dump()
        if not xml_path:
            return []
        try:
            return _parse_ui_xml(xml_path)
        finally:
            try:
                os.unlink(xml_path)
            except OSError:
                pass

    def _pull_uiautomator_dump(self) -> Optional[str]:
        device_path = "/sdcard/uia_dump.xml"
        # 让 UI Automator 把当前 UI 树 dump 到 sdcard
        self._adb("shell", "uiautomator", "dump", device_path)
        # 拉回本地临时文件
        local = tempfile.NamedTemporaryFile(
            prefix="uia_", suffix=".xml", delete=False)
        local.close()
        try:
            self._adb("pull", device_path, local.name)
        except AdbError:
            try:
                os.unlink(local.name)
            except OSError:
                pass
            return None
        return local.name

    # ------------------------------------------------------------------
    # act()：执行动作
    # ------------------------------------------------------------------
    def act(self, step: SkillStep) -> bool:
        a = step.action_type
        try:
            if a == "click":
                return self._do_click(step)
            if a == "long_press":
                return self._do_long_press(step)
            if a == "type":
                return self._do_type(step)
            if a == "swipe":
                return self._do_swipe(step)
            if a == "back":
                self._adb("shell", "input", "keyevent", "KEYCODE_BACK")
                self._pause()
                return True
            if a == "wait":
                time.sleep(1.0)
                return True
            if a == "open_app":
                return self._do_open_app(step)
            return False
        except AdbError:
            return False

    # ------------------------------------------------------------------
    # 具体动作
    # ------------------------------------------------------------------
    def _do_click(self, step: SkillStep) -> bool:
        x, y = self._resolve_xy(step)
        if x is None:
            return False
        self._adb("shell", "input", "tap", str(x), str(y))
        self._pause()
        return True

    def _do_long_press(self, step: SkillStep) -> bool:
        x, y = self._resolve_xy(step)
        if x is None:
            return False
        # 用 swipe 同点制造长按
        self._adb("shell", "input", "swipe",
                  str(x), str(y), str(x), str(y), "1000")
        self._pause()
        return True

    def _do_type(self, step: SkillStep) -> bool:
        text = step.text or ""
        if not text:
            return False
        # 如果指定了目标控件，先点一下让焦点进去
        if step.target_text or step.coordinates:
            x, y = self._resolve_xy(step)
            if x is not None:
                self._adb("shell", "input", "tap", str(x), str(y))
                time.sleep(0.3)
        # 用 ADBKeyBoard 输中文（如果装了），否则走原生 input text（ASCII）
        if _is_ascii(text):
            escaped = text.replace(" ", "%s").replace("'", "")
            self._adb("shell", "input", "text", escaped)
        else:
            ok = self._type_via_adbkeyboard(text)
            if not ok:
                # 兜底：警告并尝试直接发（多数 ROM 不支持中文，会得到乱码或空）
                self._adb("shell", "input", "text", text)
        self._pause()
        return True

    def _type_via_adbkeyboard(self, text: str) -> bool:
        # 通过广播给 ADBKeyBoard，它会把文字注入到当前焦点
        # APK: https://github.com/senzhk/ADBKeyBoard
        safe = text.replace('"', '\\"')
        try:
            self._adb(
                "shell",
                "am", "broadcast",
                "-a", "ADB_INPUT_TEXT",
                "--es", "msg", f'"{safe}"',
            )
            return True
        except AdbError:
            return False

    def _do_swipe(self, step: SkillStep) -> bool:
        direction = step.direction or "up"
        cx, cy = self.width // 2, self.height // 2
        delta = self.height // 4
        if direction == "up":
            ex, ey = cx, cy - delta
        elif direction == "down":
            ex, ey = cx, cy + delta
        elif direction == "left":
            ex, ey = cx - delta, cy
        elif direction == "right":
            ex, ey = cx + delta, cy
        else:
            return False
        self._adb("shell", "input", "swipe",
                  str(cx), str(cy), str(ex), str(ey), "400")
        self._pause()
        return True

    def _do_open_app(self, step: SkillStep) -> bool:
        pkg = step.package or step.target_text
        if not pkg:
            return False
        try:
            self._adb("shell", "monkey", "-p", pkg,
                      "-c", "android.intent.category.LAUNCHER", "1")
        except AdbError:
            return False
        time.sleep(self.wait_after_open)
        return True

    # ------------------------------------------------------------------
    # 坐标解析：优先用 step.coordinates；没坐标就靠 target_text 在 UI 树里查
    # ------------------------------------------------------------------
    def _resolve_xy(self, step: SkillStep) -> Tuple[Optional[int], Optional[int]]:
        if step.coordinates and len(step.coordinates) == 2:
            return step.coordinates[0], step.coordinates[1]
        target = (step.target_text or "").strip()
        if not target:
            return None, None
        for e in self.screen():
            if e.text == target or e.content_desc == target:
                if e.clickable:
                    return e.center
        # 退化：包含匹配
        for e in self.screen():
            if target and (target in e.text or target in e.content_desc):
                return e.center
        return None, None

    def _pause(self):
        time.sleep(self.action_pause)


# ----------------------------------------------------------------------
# adb 进程封装
# ----------------------------------------------------------------------
def _adb_raw(args: List[str], timeout: float = 30.0) -> str:
    try:
        r = subprocess.run(
            ["adb", *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as e:
        raise AdbError(
            "找不到 adb 命令。请装 platform-tools 并加入 PATH。"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise AdbError(f"adb 命令超时: {' '.join(args)}") from e
    if r.returncode != 0:
        raise AdbError(
            f"adb 失败 (returncode={r.returncode}): "
            f"args={' '.join(args)}, stderr={r.stderr.strip()}"
        )
    return r.stdout.strip()


# ----------------------------------------------------------------------
# UI 树解析
# ----------------------------------------------------------------------
def _parse_ui_xml(xml_path: str) -> List[ScreenElement]:
    elems: List[ScreenElement] = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return elems

    seen_centers: List[Tuple[int, int]] = []
    for node in tree.iter("node"):
        a = node.attrib
        clickable = a.get("clickable") == "true"
        text = (a.get("text") or "").strip()
        desc = (a.get("content-desc") or "").strip()
        if not (clickable or text or desc):
            continue
        bounds = _parse_bounds(a.get("bounds", ""))
        if not bounds:
            continue
        (x1, y1), (x2, y2) = bounds
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # 去重：中心点距 < 20px 的视为同一控件
        close = False
        for sx, sy in seen_centers:
            if (cx - sx) ** 2 + (cy - sy) ** 2 < 400:
                close = True
                break
        if close:
            continue
        seen_centers.append((cx, cy))
        elems.append(ScreenElement(
            text=text, content_desc=desc,
            bounds=bounds, clickable=clickable,
        ))
    return elems


def _parse_bounds(s: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    # "[x1,y1][x2,y2]"
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2))), (int(m.group(3)), int(m.group(4)))


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False
