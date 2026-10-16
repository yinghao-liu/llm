"""解码与帧迭代：统一 OpenCV，按需缩放，帧号保持与原视频一致。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {path}")
    meta = VideoMeta(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


def iter_frames(path: str,
                width: int | None = None) -> Iterator[Tuple[int, np.ndarray]]:
    """顺序 yield (原始帧号, BGR帧)。width 非空时等比缩小（高取偶数）。"""
    cap = cv2.VideoCapture(path)
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if width and frame.shape[1] != width:
                h = round(frame.shape[0] * width / frame.shape[1] / 2) * 2
                frame = cv2.resize(frame, (width, h),
                                   interpolation=cv2.INTER_AREA)
            yield idx, frame
    finally:
        cap.release()


def read_frames_at(path: str,
                   frame_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    """随机读取若干帧（关键帧抽取用，seek 实现）。"""
    cap = cv2.VideoCapture(path)
    out: Dict[int, np.ndarray] = {}
    try:
        for fid in sorted(set(frame_ids)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if ok:
                out[int(fid)] = frame
    finally:
        cap.release()
    return out