"""video_rag 共享数据模型（契约层）。

依赖规则：
    common 不依赖任何业务包；
    shot_seg / annotate / indexer / search 只依赖 common；
    仅 jobs（编排层）可以同时 import 所有包。

各阶段产物落盘格式（包间交接契约）：
    shots.json        镜头表          （shot_seg 产出）
    keyframes/        关键帧          （shot_seg 产出）
    annotations/      每镜头标注缓存  （annotate 产出，支持断点续跑）
    documents.json    检索文档        （annotate 产出，indexer 消费）
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type, TypeVar

# ---------------- 转场类型（跨层契约，全项目统一约定） ----------------
TRANS_START = "start"       # 片头首镜头
TRANS_CUT = "cut"           # 硬切
TRANS_GRADUAL = "gradual"   # 叠化/渐隐渐显/划像（边界取渐变区间中点）
TRANS_BLACK = "black"       # 黑场/白场（边界取画面恢复的首帧）
TRANS_END = "end"           # 哨兵

TRANSITION_TYPES = (TRANS_START, TRANS_CUT, TRANS_GRADUAL, TRANS_BLACK)

T = TypeVar("T", bound="_Base")


# ---------------- 工具函数 ----------------
def format_timecode(seconds: float, msec: bool = True) -> str:
    """秒 -> 'HH:MM:SS.mmm'，用于检索结果的时间段引用。"""
    ms_total = int(round(max(0.0, float(seconds)) * 1000))
    h, ms_total = divmod(ms_total, 3600_000)
    m, ms_total = divmod(ms_total, 60_000)
    s, ms = divmod(ms_total, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if msec else f"{h:02d}:{m:02d}:{s:02d}"


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """按字段名过滤后构造（忽略 to_dict 里带的计算字段）。"""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


# ---------------- 1. 镜头表（shot_seg 产出） ----------------
@dataclass
class Shot:
    shot_id: int
    start_frame: int
    end_frame: int          # 开区间
    fps: float
    transition_in: str = TRANS_START
    keyframes: List[str] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.start_frame / self.fps

    @property
    def end(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def id(self) -> str:
        return f"shot_{self.shot_id:06d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "fps": self.fps,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "start_tc": format_timecode(self.start),
            "end_tc": format_timecode(self.end),
            "transition_in": self.transition_in,
            "keyframes": list(self.keyframes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Shot":
        return _from_dict(cls, data)


# ---------------- 2. 镜头标注（annotate 产出） ----------------
@dataclass
class ShotAnnotation:
    shot_id: int
    caption: str = ""
    characters: List[str] = field(default_factory=list)
    location: str = ""
    action: str = ""
    mood: str = ""
    objects: List[str] = field(default_factory=list)
    dialogue: List[str] = field(default_factory=list)
    visual_style: str = ""
    is_flashback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShotAnnotation":
        return _from_dict(cls, data)


# ---------------- 3. 检索文档（indexer 消费） ----------------
@dataclass
class Document:
    id: str
    text: str                       # 向量化用的主文本
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_shot(cls, shot: Shot, ann: ShotAnnotation,
                  video_id: str = "") -> "Document":
        """镜头 + 标注 -> 检索文档：caption/动作/人物/台词拼成主文本。"""
        parts = [
            ann.caption,
            ann.action,
            ann.location,
            ann.mood,
            ("人物:" + "、".join(ann.characters)) if ann.characters else "",
            ("台词:" + "；".join(ann.dialogue)) if ann.dialogue else "",
        ]
        meta = ann.to_dict()
        meta.update({
            "video": video_id,
            "shot_id": shot.shot_id,
            "start": round(shot.start, 3),
            "end": round(shot.end, 3),
            "start_tc": format_timecode(shot.start),
            "end_tc": format_timecode(shot.end),
            "transition_in": shot.transition_in,
            "keyframes": list(shot.keyframes),
        })
        return cls(id=shot.id,
                   text="。".join(p for p in parts if p),
                   meta=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(id=data["id"], text=data.get("text", ""),
                   meta=data.get("meta", {}))


# ---------------- JSON 读写（原子写，防止半截文件） ----------------
def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def dump_jsonl(path: str | Path, docs: Iterable[Document]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path: str | Path) -> List[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document.from_dict(json.loads(line)))
    return docs