# shot_seg/core.py
"""镜头切分内部数据结构与配置。

数据模型（Shot / 转场常量）已上移至 common.models，本文件只保留
切分内部使用的 CutPoint 与 ShotConfig，并 re-export 公共符号供子模块使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 公共数据模型：单一事实来源在 common.models，此处 re-export
# （detectors.py / postprocess.py 仍可 from .core import Shot, TRANS_CUT ...）
from common.models import (  # noqa: F401
    TRANS_BLACK,
    TRANS_CUT,
    TRANS_GRADUAL,
    TRANS_START,
    Shot,
)

__all__ = ["Shot", "CutPoint", "ShotConfig",
           "TRANS_START", "TRANS_CUT", "TRANS_GRADUAL", "TRANS_BLACK"]


@dataclass
class CutPoint:
    """切点 = 新镜头的起始帧（帧号均为原视频帧号）。"""
    frame: int
    kind: str = TRANS_CUT
    score: float = 0.0
    grad_start: Optional[int] = None   # 渐变/黑场区间 [grad_start, grad_end)
    grad_end: Optional[int] = None


@dataclass
class ShotConfig:
    # ---- 解码/采样 ----
    detect_width: int = 320            # 检测时缩放宽度（提速，不改帧率）
    # ---- 硬切检测（PySceneDetect HSV 内容差，默认阈值 27.0）----
    content_threshold: float = 27.0    # 影视标准剪辑建议 25~30，越小越敏感
    min_scene_len: int = 15
    # ---- 渐变检测（自研帧差游程）----
    gradual_enabled: bool = True
    gradual_min_len: int = 8           # 渐变最短持续帧数（≈0.3s@24fps）
    gradual_max_len: int = 75
    gradual_low: float = 0.20          # 归一化帧差下限（持续高于才算渐变）
    gradual_high: float = 0.85         # 平均值高于此视为快切串而非渐变
    # ---- 黑场检测 ----
    black_luma: float = 20.0
    black_min_len: int = 4
    # ---- 后处理 ----
    min_shot_len: int = 12             # 最短镜头帧数，过短合并（防闪白/碎切）
    despike_gap: int = 4               # 两切点间隔小于此值时合并去毛刺