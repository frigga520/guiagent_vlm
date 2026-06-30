"""
SkillStore：技能的 JSON 持久化 + 增删改查

文件格式（单文件）：
{
  "version": 1,
  "skills": {
    "<skill_id>": {<Skill 序列化>}
  }
}
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from mobilerun.schema import Skill


_STORE_VERSION = 1


class SkillStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._skills: Dict[str, Skill] = {}
        self._load()

    # ---------- IO ----------
    def _load(self):
        if not os.path.exists(self.path):
            self._skills = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._skills = {}
            return
        raw = data.get("skills", {}) if isinstance(data, dict) else {}
        self._skills = {
            sid: Skill(**body) for sid, body in raw.items() if isinstance(body, dict)
        }

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._lock:
            payload = {
                "version": _STORE_VERSION,
                "skills": {sid: s.model_dump() for sid, s in self._skills.items()},
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    # ---------- CRUD ----------
    def put(self, skill: Skill, *, save: bool = True) -> Skill:
        with self._lock:
            skill.touch()
            self._skills[skill.id] = skill
        if save:
            self.save()
        return skill

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def remove(self, skill_id: str, *, save: bool = True) -> bool:
        with self._lock:
            existed = skill_id in self._skills
            self._skills.pop(skill_id, None)
        if existed and save:
            self.save()
        return existed

    def list_all(self) -> List[Skill]:
        return list(self._skills.values())

    def __len__(self):
        return len(self._skills)

    def __contains__(self, skill_id: str):
        return skill_id in self._skills

    # ---------- 便利查询 ----------
    def by_app(self, app: str) -> List[Skill]:
        return [s for s in self._skills.values() if s.app == app]

    def keyword_search(self, query: str, limit: int = 5) -> List[Skill]:
        """没装 embedding 时的退化方案：子串 + 出现次数排序。"""
        q = query.lower()
        scored = []
        for s in self._skills.values():
            hay = (s.name + " " + s.description).lower()
            score = sum(hay.count(tok) for tok in q.split() if tok)
            if score > 0 or q in hay:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:limit]]
