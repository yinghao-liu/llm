"""VLM 描述生成：OpenAI 兼容接口（Qwen-VL / GLM-4V / GPT-4o 等通用）。"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import cv2
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------- 提示词（影视剧镜头描述，防幻觉是第一要求） ----------
SYSTEM_PROMPT = """你是影视内容标注员，负责为视频检索系统生成结构化镜头描述。
规则：
1. 只描述画面中确定可见的内容，禁止推测剧情、猜测人物姓名或演员名。
2. 人物一律用代称："男主""女主""男子A""女子B""老人""孩童"等；无法判断关系时用中性代称。
3. caption 用 1-2 句现在时中文，覆盖：地点/时间/天气、人物动作、情绪、关键道具、镜头语言。
4. dialogue 原样抄写图中台词/字幕，没有则为空数组。
5. 严格输出 JSON，不要任何额外文字。"""

USER_PROMPT_TEMPLATE = """这是同一个镜头按时间顺序的 {n} 张画面（第1张最早）。
镜头所在作品的已知角色表（如为空则忽略，人物用代称）：
{cast}

该镜头时间段内的台词（来自字幕/ASR，供参考，不要重复写进 caption）：
{dialogue}

请输出 JSON，字段如下：
{{
  "caption": "1-2句中文描述",
  "characters": ["出现的人物代称或角色名"],
  "location": "室内/室外 + 场景 + 时间(白天/夜晚) + 天气",
  "action": "核心动作短语",
  "mood": "情绪氛围",
  "objects": ["关键道具"],
  "dialogue": ["图中可见或给定的台词"],
  "visual_style": "镜头语言/色调/景别",
  "is_flashback": false
}}"""


@dataclass
class ShotAnnotation:
    shot_id: int
    caption: str = ""
    characters: Optional[List[str]] = None
    location: str = ""
    action: str = ""
    mood: str = ""
    objects: Optional[List[str]] = None
    dialogue: Optional[List[str]] = None
    visual_style: str = ""
    is_flashback: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("characters", "objects", "dialogue"):
            d[k] = d[k] or []
        return d


def _encode_jpg(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("ascii")


class VLMClient:
    def __init__(self, model: str, base_url: str, api_key: str,
                 timeout: float = 60.0, max_retries: int = 3):
        if OpenAI is None:
            raise ImportError("pip install openai")
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             timeout=timeout, max_retries=max_retries)
        self.model = model

    def describe_shot(self, shot_id: int,
                      frames: List[np.ndarray],
                      dialogue: List[str],
                      cast: Optional[List[str]] = None) -> ShotAnnotation:
        content = []
        for f in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_jpg(f)}"},
            })
        content.append({
            "type": "text",
            "text": USER_PROMPT_TEMPLATE.format(
                n=len(frames),
                cast="、".join(cast) if cast else "（无）",
                dialogue="；".join(dialogue) if dialogue else "（无字幕）",
            ),
        })
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        raw = resp.choices[0].message.content
        return self._parse(shot_id, raw, dialogue)

    @staticmethod
    def _parse(shot_id: int, raw: str, dialogue: List[str]) -> ShotAnnotation:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:                # 兜底：截取首个 {...}
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
        ann = ShotAnnotation(shot_id=shot_id)
        for k in ("caption", "location", "action", "mood", "visual_style"):
            setattr(ann, k, str(data.get(k, "") or ""))
        for k in ("characters", "objects"):
            v = data.get(k) or []
            setattr(ann, k, [str(x) for x in v])
        ann.dialogue = [str(x) for x in (data.get("dialogue") or [])] or dialogue
        ann.is_flashback = bool(data.get("is_flashback", False))
        return ann