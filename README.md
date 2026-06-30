# Mobile GUI Agent

一个**操作真实 Android 手机的 vision-grounded GUI agent**：用一句自然语言描述任务，
agent 自己看屏幕截图、用视觉大模型（VLM）推理"该点哪 / 输什么"、用 adb 控制手机。

```bash
python run_agent.py "打开微信，给张三发『我到了』"
python run_agent.py "打开设置，开启深色模式"
python run_agent.py "打开备忘录，新建一条：明天交作业"
```

---

## 1. 架构

```
        用户任务（自然语言）
                │
                ▼
  ┌─────────────────────────────────────┐
  │   GUIAgent.run() 主循环               │
  │                                       │
  │   for step in range(max_steps):       │
  │       img = adb screencap              │← 真机截图
  │       decision = vlm(prompt, img)      │← VLM 推理（出归一化坐标 [0,1000]）
  │       executor.do(decision)            │← adb input tap/text/swipe
  │       if decision.done: break          │
  └──────────┬───────────┬──────────────┘
             │           │
             ▼           ▼
        AdbExecutor    VLM (Qwen-VL / GPT-4o / Claude)
        - screenshot   - 看图 + 任务 → JSON 决策
        - tap(x,y)       {"action":"tap","x":540,"y":1200,"done":false}
        - type_text
        - swipe / back / open_app
```

**核心代码** ~600 行：
- `mobilerun/agent.py` (~360 行)  vision-grounded 主循环 + 截图缩放 + JSON 解析/容错 +
                                  归一化坐标转换 + 死循环检测 + 截图落盘
- `mobilerun/executor.py` (~270 行) 自研 ADB 真机执行器（subprocess + adb）
- `mobilerun/llm.py` (~90 行)     OpenAI 兼容多模态客户端

**可选加速模块** `mobilerun/skills/`（参数化技能库）：
- 跑成功的 trace 蒸馏成参数化技能（contact / message 等 slot）
- 语义召回：embedding + cosine
- 自愈：UI 改版时文本相似度 / LLM 同义词重定位
- 接不接都不影响 agent 跑通，是单纯的"重复任务加速器"

---

## 2. 跑起来

### 2.1 装

```powershell
cd D:\gui_agent\mobilerun
pip install -e .
adb version       # 确认 platform-tools 在 PATH
```

依赖只有 `pydantic + requests + pillow`。

### 2.2 配 VLM

在项目根目录创建 `.env`：

```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx
QWEN_CHAT_MODEL=qwen3-vl-plus
```

推荐 `qwen3-vl-plus`（Qwen3-VL 系列，原生用 `[0, 1000]` 归一化坐标，UI grounding 准）。
也支持 OpenAI（`OPENAI_API_KEY` + 任意 vision-capable 模型如 `gpt-4o`）。

`.env` 加载逻辑写在 `run_agent.py` 顶部，无需 `python-dotenv`。

### 2.3 手机准备

1. 开发者选项 → 打开 USB 调试
2. 数据线插 PC，弹"允许 USB 调试吗"勾允许
3. `adb devices` 能看到设备且状态 `device`
4. （可选，中文输入需要）装 [ADBKeyBoard.apk](https://github.com/senzhk/ADBKeyBoard)：
   ```powershell
   adb install ADBKeyboard.apk
   adb shell ime enable com.android.adbkeyboard/.AdbIME
   adb shell ime set com.android.adbkeyboard/.AdbIME
   ```

### 2.4 跑

```powershell
python run_agent.py "打开设置，开启深色模式"
```

日志大致是：

```
设备: 7QH7N20A26000170  分辨率: 1080x2400
模型: OpenAICompatibleLLM  (qwen3-vl-plus)
=== Task: 打开设置，开启深色模式 ===
--- Step 1 ---
[Observe] image 1080x2400  scale=1.000
[Observe] 已保存截图 data/screens/step_01.png
[Plan]    我看到桌面，需要先打开设置 app
[Plan]    {'action': 'open_app', 'package': 'com.android.settings'}
[Act]     OK open com.android.settings
--- Step 2 ---
[Observe] image 1080x2400  scale=1.000
[Plan]    现在在设置首页，找到"显示"
[Plan]    {'action': 'tap', 'x': 500, 'y': 410}
[Act]     OK tap (540,984)
...
=== Done after 5 steps ===
========== 结果 ==========
  success     : True
  steps       : 5
  stop_reason : done
```

每步的截图保存在 `data/screens/step_NN.png`，方便对照模型说的坐标和实际页面。

---

## 3. 目录结构

```
mobile-gui-agent/
├── run_agent.py                   ← 主入口
├── README.md
├── pyproject.toml                 deps: pydantic + requests + pillow
├── .env                           本地 API key（.gitignore 已忽略）
├── mobilerun/
│   ├── __init__.py                exports GUIAgent / AdbExecutor / build_llm_from_env
│   ├── agent.py                   GUIAgent vision-grounded 主循环
│   ├── executor.py                AdbExecutor (screencap / tap / type / swipe)
│   ├── llm.py                     OpenAI 兼容多模态客户端
│   ├── schema.py                  SkillStep / Slot 等数据结构
│   ├── self_heal.py               (skill 自愈用，agent 主路径不依赖)
│   └── skills/                    （可选）参数化技能库
│       ├── store.py
│       ├── retriever.py
│       ├── extractor.py
│       ├── filler.py
│       └── runner.py
├── tests/skills/test_skill_library.py   8 个单元测试
└── data/                          运行时产生（截图、技能、状态图）
```

---

## 4. 关键实现要点

### 4.1 归一化坐标 [0, 1000]
Qwen3-VL / Qwen2-VL 等现代 VLM 用 `[0, 1000]` 归一化坐标系（图像左上 = (0,0)，
右下 = (1000,1000)）。`GUIAgent._to_device_px()` 自动把模型输出乘以设备分辨率比例
得到真实像素坐标。对老模型（如 `qwen-vl-max` 用绝对像素）会自动退到兼容模式。

### 4.2 JSON 容错
VLM 偶尔会出小毛病的 JSON（数字后多余引号、漏键名、末尾逗号、外面包代码块）。
`GUIAgent._repair_json()` 自动修复这几类常见错误后再交给严格解析；首次失败还会附一
条"请只返回 JSON"的重试。

### 4.3 死循环检测
连续 3 次完全相同的 `tap`（含坐标）就 halt，避免模型反复点同一处而页面不前进。
导航类动作（back / swipe / wait）允许合法连击不触发。

### 4.4 中文输入
`adb shell input text` 原生只吃 ASCII。装了 ADBKeyBoard 后，`executor.type_text()`
自动识别中文 / ASCII 走对应路径，中文通过 broadcast intent 注入。

---

## 5. FAQ

**Q: 坐标怎么换算的？**
A: 模型输出 `[0, 1000]` 归一化坐标 → `device_x = x * device_width / 1000`。
   截图缩放与设备像素是两套独立坐标系，由 `_resize_png()` 和 `_to_device_px()`
   分别管，互不干扰。

**Q: vision-grounded 比 UI 树 + 编号好在哪？**
A: 不依赖 accessibility 树 —— Canvas、WebView、游戏内嵌都能看见。坐标精度看 VLM；
   `qwen3-vl-plus` 在 ~1080p 截图上的 grounding 准度可用。

**Q: 中文输入怎么办？**
A: 见 4.4。装 ADBKeyBoard，executor 自动判断走 broadcast 注入。

**Q: 跑一次大概多少 token？**
A: 截图缩到 1280 宽再发，`qwen3-vl-plus` 每步约 1.5-2k input + 100 output。
   10 步任务 ≈ 20k tokens。

**Q: 不止微信，其他 app 也能跑？**
A: 能。Agent 直接看截图 + 操控 adb，对任何 app 都通用。`open_app` 需要知道包名
   （如 `com.android.settings`），也可以让模型先 tap 桌面图标进入。
