"""每镜头抽 1~3 帧关键帧，候选位置附近择最清晰帧（拉普拉斯方差）。"""
from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from .core import Shot
from .video_io import read_frames_at


def _blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(path: str, shot: Shot, out_dir: str,
                      n: int = 2, refine: int = 2) -> List[str]:
    length = shot.end_frame - shot.start_frame
    if n <= 1:
        bases = [shot.start_frame + length // 2]
    else:
        bases = [shot.start_frame + max(1, length // 4),
                 shot.start_frame + length // 2]
        if n >= 3:
            bases.append(shot.end_frame - max(1, length // 4))

    # 候选帧集合：每个基准位置 ± refine 帧（不越出镜头范围）
    candidates = set()
    for b in bases:
        lo = max(shot.start_frame, b - refine)
        hi = min(shot.end_frame - 1, b + refine)
        candidates.update(range(lo, hi + 1))
    frames = read_frames_at(path, candidates)

    os.makedirs(out_dir, exist_ok=True)
    picked: List[str] = []
    for b in bases:
        pool = [f for f in candidates
                if abs(f - b) <= refine and f in frames]
        best = max(pool, key=lambda f: _blur_score(frames[f]))
        name = f"shot_{shot.shot_id:06d}_f{best:07d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frames[best])
        picked.append(name)
    return picked