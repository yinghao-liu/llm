"""SRT/ASR 与镜头时间段对齐：影视检索中台词是第一检索信号。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .core import Shot


@dataclass
class SubLine:
    start: float   # 秒
    end: float
    text: str


def parse_srt(path: str) -> List[SubLine]:
    """轻量 SRT 解析（兼容 '00:01:02,500 --> 00:01:04,000'）。"""
    ts = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    lines: List[SubLine] = []
    with open(path, encoding="utf-8-sig") as f:
        cur = None
        for raw in f:
            m = ts.search(raw)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                cur = SubLine(start, end, "")
                lines.append(cur)
            elif cur is not None and raw.strip():
                cur.text = (cur.text + " " + raw.strip()).strip()
    return lines


def align_dialogue(subs: List[SubLine], shot: Shot,
                   overlap: float = 0.5) -> List[str]:
    """取与镜头时间区间有重叠的台词行（按台词时长算重叠比例）。"""
    out = []
    for s in subs:
        inter = max(0.0, min(s.end, shot.end) - max(s.start, shot.start))
        if inter / max(1e-6, s.end - s.start) >= overlap:
            out.append(s.text)
    return out