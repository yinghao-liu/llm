"""切点检测后端（可插拔）+ 逐帧统计（渐变/黑场分析共用）。

- scenedetect : PySceneDetect ContentDetector，硬切主力，CPU 即可
- transnet    : TransNet V2（官方 inference-pytorch 权重），精度最高，含渐变
- diff        : 纯 OpenCV 直方图帧差兜底
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .core import (CutPoint, ShotConfig, TRANS_BLACK, TRANS_CUT, TRANS_GRADUAL)
from .video_io import iter_frames


# ---------------------------------------------------------------- 帧统计
@dataclass
class FrameStats:
    diff: np.ndarray   # diff[i]: 帧i 与帧i-1 的归一化差异（diff[0]=0）
    luma: np.ndarray   # 每帧平均亮度

    def __len__(self) -> int:
        return len(self.diff)


def analyze_frame_stats(path: str, cfg: ShotConfig) -> FrameStats:
    """一遍扫描：HSV 直方图相关性帧差 + 平均亮度。全帧率、缩放后计算。"""
    diffs, lumas = [], []
    prev_hist = None
    for _, frame in iter_frames(path, width=cfg.detect_width):
        lumas.append(float(frame.mean()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is None:
            diffs.append(0.0)
        else:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diffs.append(float(1.0 - corr))
        prev_hist = hist
    return FrameStats(diff=np.asarray(diffs, dtype=np.float32),
                      luma=np.asarray(lumas, dtype=np.float32))


def detect_black_runs(stats: FrameStats,
                      cfg: ShotConfig) -> List[Tuple[int, int]]:
    """返回连续黑帧区间 [b0, b1)。"""
    runs, i, n = [], 0, len(stats)
    while i < n:
        if stats.luma[i] <= cfg.black_luma:
            j = i
            while j < n and stats.luma[j] <= cfg.black_luma:
                j += 1
            if j - i >= cfg.black_min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def detect_gradual_runs(stats: FrameStats,
                        cfg: ShotConfig) -> List[Tuple[int, int, float]]:
    """返回疑似渐变区间 [g0, g1) 及其平均帧差。

    原理：叠化/渐隐是"持续多帧的中等帧差"，与硬切（单帧高脉冲）、
    正常运动（低帧差）在游程长度上可分。
    """
    runs, i, n = [], 0, len(stats)
    diff = stats.diff
    while i < n:
        if diff[i] >= cfg.gradual_low:
            j, total = i, 0.0
            while j < n and diff[j] >= cfg.gradual_low:
                total += float(diff[j])
                j += 1
            run_len = j - i
            mean = total / run_len
            if (cfg.gradual_min_len <= run_len <= cfg.gradual_max_len
                    and mean < cfg.gradual_high):
                runs.append((i, j, mean))
            i = j
        else:
            i += 1
    return runs


# ---------------------------------------------------------------- 后端
def detect_cuts_scenedetect(path: str, cfg: ShotConfig) -> List[CutPoint]:
    """PySceneDetect ContentDetector：HSV 空间像素变化加权平均做硬切检测。"""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    def _make():
        try:
            return ContentDetector(threshold=cfg.content_threshold,
                                   min_scene_len=cfg.min_scene_len)
        except TypeError:  # 兼容旧版本签名
            return ContentDetector(threshold=cfg.content_threshold)

    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(_make())
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    # 场景列表 -> 切点（每个场景起点即新镜头首帧）
    return [CutPoint(frame=int(start.get_frames()), kind=TRANS_CUT)
            for start, _ in scenes[1:]]


def detect_cuts_diff(stats: FrameStats, cfg: ShotConfig) -> List[CutPoint]:
    """纯 OpenCV 兜底：归一化帧差高脉冲 + 局部极大 = 硬切。"""
    diff = stats.diff
    cuts: List[CutPoint] = []
    for i in range(1, len(diff)):
        if diff[i] < 0.45:
            continue
        lo, hi = max(0, i - 2), min(len(diff), i + 3)
        if diff[i] >= diff[lo:hi].max():          # 局部极大，抑制毛刺
            cuts.append(CutPoint(frame=i, kind=TRANS_CUT, score=float(diff[i])))
    return cuts


class TransNetV2Detector:
    """TransNet V2（官方 inference-pytorch）。

    模型输入 (batch, T, 27, 48, 3) 的 RGB uint8；
    输出 single_frame_pred（逐帧转场概率）与 all_frame_pred["many_hot"]。
    """

    def __init__(self, repo_dir: str, weights: str,
                 device: str = "cuda", window: int = 100):
        import torch
        sys.path.insert(0, repo_dir)                       # TransNetV2/inference-pytorch
        from transnetv2_pytorch import TransNetV2

        self.torch = torch
        self.window = window
        self.device = device
        self.model = TransNetV2()
        state = torch.load(weights, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval().to(device)

    def _forward(self, frames_rgb: List[np.ndarray]) -> List[float]:
        """frames_rgb: T 个 (27,48,3) uint8 -> T 个转场概率。"""
        video = np.stack(frames_rgb, axis=0)[None]         # (1,T,27,48,3)
        x = self.torch.from_numpy(video).to(self.device)
        with self.torch.no_grad():
            single, _ = self.model(x)
        p = self.torch.sigmoid(single)[0].reshape(-1).cpu().numpy()
        return [float(v) for v in p[:len(frames_rgb)]]

    def detect(self, path: str, threshold: float = 0.5) -> List[CutPoint]:
        probs: List[float] = []
        buf: List[np.ndarray] = []
        for _, frame in iter_frames(path):
            small = cv2.resize(frame, (48, 27))[:, :, ::-1]  # BGR->RGB
            buf.append(small)
            if len(buf) == self.window:
                probs.extend(self._forward(buf))
                buf = []
        if buf:                                             # 末尾补齐再截断
            pad = [buf[-1]] * (self.window - len(buf))
            probs.extend(self._forward(buf + pad)[:len(buf)])

        trans = np.asarray(probs, dtype=np.float32)
        cuts: List[CutPoint] = []
        i, n = 0, len(trans)
        while i < n:
            if trans[i] >= threshold:
                j = i
                while j < n and trans[j] >= threshold:
                    j += 1
                mid = i + (j - i) // 2
                kind = TRANS_CUT if j - i <= 2 else TRANS_GRADUAL
                cuts.append(CutPoint(
                    frame=mid, kind=kind,
                    score=float(trans[i:j].max()),
                    grad_start=i if kind == TRANS_GRADUAL else None,
                    grad_end=j if kind == TRANS_GRADUAL else None))
                i = j
            else:
                i += 1
        return cuts