"""批量标注编排：镜头 + 关键帧 + 台词 -> VLM -> 检索文档。

职责边界（勿越界）：
- 本文件：并发调度、镜头级缓存、失败重试语义、组装 Document
- jobs/build_episode.py：阶段编排、参数指纹、CLI
- vlm.py：单镜头怎么描述（提示词/解析）
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2

from common.models import (Document, Shot, ShotAnnotation,
                           atomic_write_json, read_json)
from .subtitle import SubLine, align_dialogue, parse_srt
from .vlm import VLMClient

log = logging.getLogger("annotate")


# ---------------------------------------------------------------- 内部
def _load_keyframes(keyframe_dir: Path, names: Sequence[str]):
    frames = []
    for name in names:
        img = cv2.imread(str(keyframe_dir / name))
        if img is not None:
            frames.append(img)
    return frames


def _annotate_one(shot: Shot,
                  keyframe_dir: Path,
                  vlm: VLMClient,
                  subs: Sequence[SubLine],
                  cache_dir: Optional[Path],
                  cast: Optional[Sequence[str]],
                  force: bool,
                  video_id: str) -> Tuple[Document, bool]:
    """标注单个镜头。返回 (Document, 是否命中缓存)。

    命中缓存直接复用；未命中则调 VLM 并落缓存（原子写）。
    异常向上抛，由调用方按镜头隔离处理——失败的镜头不落缓存，下次自动重试。
    """
    cache = cache_dir / f"{shot.id}.json" if cache_dir else None
    if cache is not None and cache.exists() and not force:
        ann = ShotAnnotation.from_dict(read_json(cache))
        return Document.from_shot(shot, ann, video_id=video_id), True

    frames = _load_keyframes(keyframe_dir, shot.keyframes)
    if not frames:
        raise RuntimeError(f"关键帧缺失: {shot.id} {shot.keyframes}")

    dialogue = align_dialogue(list(subs), shot)
    ann = vlm.describe_shot(shot.shot_id, frames, dialogue,
                            cast=list(cast) if cast else None)
    if cache is not None:
        atomic_write_json(cache, ann.to_dict())
    return Document.from_shot(shot, ann, video_id=video_id), False


# ---------------------------------------------------------------- 主 API
def annotate_shots(shots: Sequence[Shot],
                   keyframe_dir: Union[str, Path],
                   vlm: VLMClient,
                   subs: Sequence[SubLine] = (),
                   cache_dir: Optional[Union[str, Path]] = None,
                   cast: Optional[Sequence[str]] = None,
                   workers: int = 4,
                   force: bool = False,
                   video_id: str = "") -> List[Document]:
    """批量标注，返回与 shots 同序的 Document 列表。

    - cache_dir 提供时启用镜头级缓存（断点续跑/幂等）
    - 单镜头失败只记日志跳过，其余继续；全部失败时抛 RuntimeError
    """
    keyframe_dir = Path(keyframe_dir)
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    docs_by_id: Dict[int, Document] = {}
    cached = failed = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(_annotate_one, s, keyframe_dir, vlm, subs,
                        cache_dir, cast, force, video_id): s
            for s in shots
        }
        for fut in as_completed(futs):
            shot = futs[fut]
            try:
                doc, from_cache = fut.result()
                docs_by_id[shot.shot_id] = doc
                cached += int(from_cache)
            except Exception:
                failed += 1
                log.exception("[annotate] 失败: %s", shot.id)
            done = cached + failed
            if done % 20 == 0 and done:
                log.info("[annotate] 进度 %d/%d", done, len(shots))

    log.info("[annotate] 完成 %d（缓存命中 %d），失败 %d（下次自动重试）",
             len(docs_by_id), cached, failed)
    if shots and not docs_by_id:
        raise RuntimeError("标注全部失败，请检查网络 / API Key / 模型名")

    # 保证输出顺序与输入一致（线程池返回是乱序的）
    return [docs_by_id[s.shot_id] for s in shots if s.shot_id in docs_by_id]


def annotate_video(shots: Union[str, Path, dict, Sequence[Shot]],
                   keyframe_dir: Union[str, Path],
                   vlm: VLMClient,
                   srt: Optional[Union[str, Path]] = None,
                   cache_dir: Optional[Union[str, Path]] = None,
                   cast: Optional[Sequence[str]] = None,
                   workers: int = 4,
                   force: bool = False,
                   video_id: str = "") -> List[Document]:
    """兼容旧调用姿势的包装：shots 可传 shots.json 路径 / dict / List[Shot]。"""
    if isinstance(shots, (str, Path)):
        data = read_json(shots)
        shots_list = [Shot.from_dict(s) for s in data["shots"]]
        video_id = video_id or Path(shots).stem
    elif isinstance(shots, dict):
        shots_list = [Shot.from_dict(s) for s in shots["shots"]]
        video_id = video_id or str(shots.get("video", "")).rsplit(".", 1)[0]
    else:
        shots_list = list(shots)

    subs = parse_srt(str(srt)) if srt else []
    if cache_dir is None:
        cache_dir = Path(keyframe_dir).parent / "annotations"

    return annotate_shots(shots_list, keyframe_dir, vlm, subs=subs,
                          cache_dir=cache_dir, cast=cast,
                          workers=workers, force=force, video_id=video_id)