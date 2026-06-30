"""
技能数据结构（Pydantic）

设计要点：
- Slot 单独抽出来，便于 LLM 抽取和填充
- SkillStep 同时保留原始 action（tap / type / swipe ...）和 slot 引用
- Skill 自带 embedding，用于语义召回
- 自愈相关字段：anchor_text / expected_post_text / last_repaired_at
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Slot：可参数化的槽位
# ----------------------------------------------------------------------
class Slot(BaseModel):
    name: str = Field(..., description="槽位标识符，如 contact / message")
    type: str = Field("text", description="text | number | choice")
    description: str = Field("", description="自然语言说明，喂给 LLM 抽取")
    example: str = Field("", description="示例值，便于 LLM 理解")
    required: bool = True


# ----------------------------------------------------------------------
# SkillStep：一步动作
# ----------------------------------------------------------------------
class SkillStep(BaseModel):
    """单步动作。保留 mobilerun macro 的 action_type + 关键参数，
    并加入 slot 占位符（如 target_text = "{{contact}}"）便于填充。"""

    action_type: str = Field(..., description="click | type | swipe | system_button | open_app | wait")
    target_text: Optional[str] = Field(None, description="目标控件文字，可含 {{slot}} 模板")
    target_index: Optional[int] = Field(None, description="按编号点击时用的 index")
    text: Optional[str] = Field(None, description="type 动作要输入的文字，可含 {{slot}} 模板")
    coordinates: Optional[List[int]] = Field(None, description="[x, y] 兜底坐标，自愈时会更新")
    direction: Optional[str] = Field(None, description="swipe 方向")
    package: Optional[str] = Field(None, description="open_app 包名")
    extra: Dict[str, Any] = Field(default_factory=dict)

    # 自愈：用于 VLM 重定位时的语义锚点
    anchor_description: str = Field("", description="本步目标的自然语言描述，自愈时给 VLM 找位置用")
    expected_post_text: str = Field("", description="执行后屏幕应该出现的关键词，用于验证")

    # 健康度
    fail_count: int = 0
    repaired_count: int = 0


# ----------------------------------------------------------------------
# Skill：参数化技能模板
# ----------------------------------------------------------------------
class Skill(BaseModel):
    id: str = Field(..., description="kebab-case 唯一标识")
    name: str = Field(..., description="人类可读名称，如『微信发消息』")
    description: str = Field(..., description="一句话描述，用于语义召回")
    app: Optional[str] = Field(None, description="目标 App 包名（若有）")
    slots: List[Slot] = Field(default_factory=list)
    steps: List[SkillStep]

    # 召回
    embedding: Optional[List[float]] = Field(None, description="description 的向量")

    # 元数据
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    last_used_at: Optional[str] = None
    last_repaired_at: Optional[str] = None

    use_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    # 来源：first_record（首次录制）| distilled（从轨迹蒸馏）| manual
    origin: str = "first_record"

    def touch(self):
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def mark_used(self, success: bool):
        self.use_count += 1
        self.last_used_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

    def success_rate(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count


# ----------------------------------------------------------------------
# 执行结果
# ----------------------------------------------------------------------
class SkillExecutionResult(BaseModel):
    skill_id: str
    success: bool
    steps_run: int
    steps_total: int
    healed_steps: int = 0
    slot_values: Dict[str, str] = Field(default_factory=dict)
    failed_reason: Optional[str] = None
    log_path: Optional[str] = None
