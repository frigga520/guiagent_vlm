"""
项目主入口：vision-grounded GUI Agent

用法：
    python run_agent.py "打开微信，给张三发『我到了』"
    python run_agent.py "打开设置，开启深色模式"
    python run_agent.py "在备忘录里新建一条：明天交作业"

前置条件：
    1. 手机插好 USB 调试，adb devices 能看到
    2. 设环境变量 DASHSCOPE_API_KEY（推荐，通义 VL-Max 便宜）
       或 OPENAI_API_KEY（要用 gpt-4o 这种带视觉的）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(Path(__file__).resolve().parent / ".env")

from mobilerun.agent import GUIAgent
from mobilerun.executor import AdbExecutor
from mobilerun.llm import build_llm_from_env


def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("用法: python run_agent.py \"你的任务\"")
        sys.exit(2)
    task = " ".join(sys.argv[1:]).strip()

    executor = AdbExecutor()
    print(f"设备: {executor.serial}  分辨率: {executor.width}x{executor.height}")

    llm = build_llm_from_env(vision=True)
    print(f"模型: {llm.__class__.__name__}  ({llm.model})")

    max_steps = int(os.environ.get("MAX_STEPS", "20"))
    screens_dir = os.environ.get("SCREENS_DIR", "data/screens")
    agent = GUIAgent(
        executor, llm,
        max_steps=max_steps,
        screens_dir=screens_dir,
    )
    result = agent.run(task)

    print("\n========== 结果 ==========")
    print(f"  success     : {result.success}")
    print(f"  steps       : {len(result.steps)}")
    print(f"  stop_reason : {result.stop_reason}")
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
