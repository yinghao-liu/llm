"""后处理：去重 -> 黑场规整 -> 渐变规整 -> 去毛刺 -> 最短镜头合并。"""
from __future__ import annotations

from typing import List

from .core import (CutPoint, Shot, ShotConfig,
                   TRANS_BLACK, TRANS_GRADUAL)
from .detectors import FrameStats, detect_black_runs, detect_gradual_runs


def refine_cuts(cuts: List[CutPoint], stats: FrameStats,
                cfg: ShotConfig) -> List[CutPoint]:
    # 1) 同帧去重：保留分数高者
    uniq = {}
    for c in cuts:
        if not (0 < c.frame < len(stats)):
            continue
        if c.frame not in uniq or c.score > uniq[c.frame].score:
            uniq[c.frame] = c
    cuts = list(uniq.values())

    # 2) 黑场规整：黑场区间内的所有切点 -> 合并为"区间末端"一个黑场切点
    #    （新镜头从画面重新亮起的第一帧开始，全项目统一该约定）
    for b0, b1 in detect_black_runs(stats, cfg):
        inside = [c for c in cuts if b0 <= c.frame < b1]
        if inside or True:
            cuts = [c for c in cuts if not (b0 <= c.frame < b1)]
            cuts.append(CutPoint(frame=b1, kind=TRANS_BLACK, score=1.0,
                                 grad_start=b0, grad_end=b1))

    # 3) 渐变规整：渐变区间附近的切点 -> 塌缩为区间中点一个切点
    for g0, g1, mean in detect_gradual_runs(stats, cfg):
        near = [c for c in cuts if g0 - 2 <= c.frame < g1 + 2]
        cuts = [c for c in cuts if not (g0 - 2 <= c.frame < g1 + 2)]
        score = max([c.score for c in near], default=mean)
        cuts.append(CutPoint(frame=g0 + (g1 - g0) // 2,
                             kind=TRANS_GRADUAL, score=score,
                             grad_start=g0, grad_end=g1))

    # 4) 去毛刺：间隔 < despike_gap 的相邻切点只留高分者
    cuts.sort(key=lambda c: c.frame)
    despiked: List[CutPoint] = []
    for c in cuts:
        if despiked and c.frame - despiked[-1].frame < cfg.despike_gap:
            if c.score > despiked[-1].score:
                despiked[-1] = c
        else:
            despiked.append(c)
    return despiked


def shots_from_cuts(cuts: List[CutPoint], total_frames: int,
                    fps: float, cfg: ShotConfig) -> List[Shot]:
    """切点 -> 镜头列表，并迭代合并过短镜头（防闪白误切、碎切）。"""
    pts = [CutPoint(frame=0, kind="start")] + \
          sorted([c for c in cuts if 0 < c.frame < total_frames],
                 key=lambda c: c.frame) + \
          [CutPoint(frame=total_frames, kind="end")]

    while len(pts) > 2:
        lengths = [pts[i + 1].frame - pts[i].frame for i in range(len(pts) - 1)]
        short = next((i for i, L in enumerate(lengths)
                      if L < cfg.min_shot_len), None)
        if short is None:
            break
        i = short
        if i == 0:                       # 首镜头过短 -> 并入右邻
            del pts[1]
        elif i == len(pts) - 2:          # 末镜头过短 -> 并入左邻
            del pts[i]
        else:                            # 并入较长的一侧
            left = pts[i].frame - pts[i - 1].frame
            right = pts[i + 2].frame - pts[i + 1].frame
            if left >= right:
                del pts[i]        # 左邻更长，删掉本镜头起点（并入左边）
            else:
                del pts[i + 1]    # 右邻更长，删掉本镜头终点（并入右边）

    shots = []
    for i in range(len(pts) - 1):
        shots.append(Shot(shot_id=i,
                          start_frame=pts[i].frame,
                          end_frame=pts[i + 1].frame,
                          fps=fps,
                          transition_in=pts[i].kind))
    return shots