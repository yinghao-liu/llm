"""检测 -> 后处理 -> 镜头表 -> 导出 JSON/CSV/关键帧。"""
from __future__ import annotations

import csv
import json
import os
from typing import List, Optional

from .core import Shot, ShotConfig
from .detectors import (TransNetV2Detector, analyze_frame_stats,
                        detect_cuts_diff, detect_cuts_scenedetect)
from .keyframes import extract_keyframes
from .postprocess import refine_cuts, shots_from_cuts
from .video_io import probe


class ShotSegmenter:
    def __init__(self, cfg: Optional[ShotConfig] = None,
                 backend: str = "scenedetect",
                 transnet_repo: Optional[str] = None,
                 transnet_weights: Optional[str] = None,
                 device: str = "cuda"):
        self.cfg = cfg or ShotConfig()
        self.backend = backend
        self.transnet = None
        if backend == "transnet":
            self.transnet = TransNetV2Detector(
                repo_dir=transnet_repo, weights=transnet_weights, device=device)

    def _hard_cuts(self, path, stats):
        if self.backend == "transnet":
            return self.transnet.detect(path)
        if self.backend == "diff":
            return detect_cuts_diff(stats, self.cfg)
        return detect_cuts_scenedetect(path, self.cfg)

    def segment(self, video_path: str) -> List[Shot]:
        meta = probe(video_path)
        stats = analyze_frame_stats(video_path, self.cfg)
        total_frames = len(stats)                    # 以实际解码帧数为准

        cuts = self._hard_cuts(video_path, stats)    # 1) 硬切（主后端）
        cuts = refine_cuts(cuts, stats, self.cfg)    # 2) 黑场/渐变/去毛刺
        shots = shots_from_cuts(cuts, total_frames,  # 3) 合并过短镜头
                                meta.fps, self.cfg)
        return shots

    def segment_and_export(self, video_path: str, out_dir: str,
                           keyframes_per_shot: int = 2) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        meta = probe(video_path)
        shots = self.segment(video_path)

        if keyframes_per_shot > 0:
            kf_dir = os.path.join(out_dir, "keyframes")
            for s in shots:
                s.keyframes = extract_keyframes(
                    video_path, s, kf_dir, n=keyframes_per_shot)

        data = {
            "video": os.path.basename(video_path),
            "fps": meta.fps,
            "duration": round(meta.duration, 3),
            "shot_count": len(shots),
            "backend": self.backend,
            "shots": [s.to_dict() for s in shots],
        }
        with open(os.path.join(out_dir, "shots.json"), "w",
                  encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, "shots.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shot_id", "start", "end", "start_frame",
                        "end_frame", "transition_in", "keyframes"])
            for s in shots:
                w.writerow([s.shot_id, round(s.start, 3), round(s.end, 3),
                            s.start_frame, s.end_frame, s.transition_in,
                            ";".join(s.keyframes)])
        return data