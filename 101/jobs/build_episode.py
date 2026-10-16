"""单集/单片全流程编排：镜头切分 -> 关键帧 -> VLM 标注 -> (可选)入库。

特性：
- 阶段级断点续跑：_state.json 记录每阶段参数指纹，参数没变且产物存在即跳过
- 镜头级标注缓存：annotations/shot_xxxxxx.json，单镜头失败只重试该镜头
- 原子写产物；--force 全量重跑
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 允许直接运行

from common.models import (Document, Shot, ShotAnnotation,
                           atomic_write_json, read_json)
from shot_seg.core import ShotConfig
from shot_seg.pipeline import ShotSegmenter
from annotate.subtitle import align_dialogue, parse_srt
from annotate.vlm import VLMClient

log = logging.getLogger("build_episode")


# ---------------------------------------------------------------- 任务配置
@dataclass
class JobSpec:
    video: Path
    outdir: Path
    subs: Optional[Path] = None

    # 切分参数
    backend: str = "scenedetect"          # scenedetect | transnet | diff
    transnet_repo: Optional[str] = None
    transnet_weights: Optional[str] = None
    device: str = "cuda"
    keyframes_per_shot: int = 2
    min_shot_len: int = 12
    content_threshold: float = 27.0

    # 标注参数
    vlm_model: Optional[str] = None
    vlm_base_url: Optional[str] = None
    vlm_api_key_env: str = "OPENAI_API_KEY"
    cast: List[str] = field(default_factory=list)   # 角色表
    workers: int = 4
    skip_annotate: bool = False

    # 其他
    force: bool = False
    with_index: bool = False


# ---------------------------------------------------------------- 编排器
class EpisodePipeline:
    def __init__(self, spec: JobSpec):
        self.spec = spec
        self.outdir = Path(spec.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.outdir / "_state.json"
        self.state: Dict[str, dict] = (read_json(self.state_path)
                                       if self.state_path.exists() else {})
        self.video_id = Path(spec.video).stem

    # ---- 阶段状态管理 ----
    @staticmethod
    def _hash(params: dict) -> str:
        blob = json.dumps(params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(blob.encode()).hexdigest()[:12]

    def _stage_done(self, stage: str, params_hash: str, artifact: Path) -> bool:
        rec = self.state.get(stage)
        return bool(rec and rec.get("params_hash") == params_hash
                    and Path(rec.get("artifact", "")).exists()
                    and artifact.exists())

    def _mark(self, stage: str, params_hash: str, artifact: Path) -> None:
        self.state[stage] = {
            "params_hash": params_hash,
            "artifact": str(artifact),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(self.state_path, self.state)

    def _video_fingerprint(self) -> dict:
        st = self.spec.video.stat()
        return {"size": st.st_size, "mtime": int(st.st_mtime)}

    # ---- 阶段 1：镜头切分 ----
    def stage_segment(self) -> dict:
        params = {
            "video": self._video_fingerprint(),
            "backend": self.spec.backend,
            "min_shot_len": self.spec.min_shot_len,
            "content_threshold": self.spec.content_threshold,
            "keyframes_per_shot": self.spec.keyframes_per_shot,
        }
        h = self._hash(params)
        artifact = self.outdir / "shots.json"
        if not self.spec.force and self._stage_done("segment", h, artifact):
            log.info("[segment] 参数未变且产物存在，跳过 -> %s", artifact)
            return read_json(artifact)

        log.info("[segment] 开始切分: %s (backend=%s)",
                 self.spec.video, self.spec.backend)
        cfg = ShotConfig(min_shot_len=self.spec.min_shot_len,
                         content_threshold=self.spec.content_threshold)
        seg = ShotSegmenter(
            cfg, backend=self.spec.backend,
            transnet_repo=self.spec.transnet_repo,
            transnet_weights=self.spec.transnet_weights,
            device=self.spec.device)
        data = seg.segment_and_export(str(self.spec.video), str(self.outdir),
                                      keyframes_per_shot=self.spec.keyframes_per_shot)
        self._mark("segment", h, artifact)
        log.info("[segment] 完成：%d 个镜头", data["shot_count"])
        return data

    # ---- 阶段 2：VLM 标注（委托 annotate 包，本层只管参数指纹/落盘） ----
    def stage_annotate(self, shots_data: dict) -> List[Document]:
        artifact = self.outdir / "documents.json"
        params = {
            "vlm_model": self.spec.vlm_model,
            "cast": self.spec.cast,
            "subs": bool(self.spec.subs and self.spec.subs.exists()),
            "keyframes_per_shot": self.spec.keyframes_per_shot,
        }
        h = self._hash(params)
        if not self.spec.force and self._stage_done("annotate", h, artifact):
            log.info("[annotate] 参数未变且产物存在，跳过 -> %s", artifact)
            return [Document.from_dict(d) for d in read_json(artifact)]
        if self.spec.skip_annotate:
            log.info("[annotate] --skip-annotate 已指定，跳过")
            return []
        if not self.spec.vlm_model:
            raise SystemExit("[annotate] 需要 --vlm-model（或加 --skip-annotate）")
        shots = [Shot.from_dict(s) for s in shots_data["shots"]]
        subs = (parse_srt(str(self.spec.subs))
                if self.spec.subs and self.spec.subs.exists() else [])
        vlm = VLMClient(
            model=self.spec.vlm_model,
            base_url=self.spec.vlm_base_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.environ.get(self.spec.vlm_api_key_env, ""),
        )
        docs = annotate_shots(                       # ← 唯一变化点
            shots,
            keyframe_dir=self.outdir / "keyframes",
            vlm=vlm,
            subs=subs,
            cache_dir=self.outdir / "annotations",
            cast=self.spec.cast or None,
            workers=self.spec.workers,
            force=self.spec.force,
            video_id=self.video_id,
        )
        atomic_write_json(artifact, [d.to_dict() for d in docs])
        self._mark("annotate", h, artifact)
        log.info("[annotate] 完成：%d 篇文档 -> %s", len(docs), artifact)
        return docs

    # ---- 阶段 3：入库（预留钩子，indexer 包就绪后生效） ----
    def stage_index(self, docs: List[Document]) -> None:
        if not self.spec.with_index or not docs:
            return
        try:
            from indexer.store import IndexStore        # noqa: WPS433
        except ImportError:
            log.warning("[index] 未找到 indexer 包，跳过入库")
            return
        store = IndexStore()
        store.upsert([d.to_dict() for d in docs])
        log.info("[index] 入库完成：%d 条", len(docs))

    # ---- 总入口 ----
    def run(self) -> None:
        t0 = time.time()
        shots_data = self.stage_segment()
        docs = self.stage_annotate(shots_data)
        self.stage_index(docs)

        log.info("=" * 60)
        log.info("视频     : %s", self.spec.video)
        log.info("镜头数   : %s", shots_data.get("shot_count"))
        log.info("文档数   : %d", len(docs))
        log.info("产物目录 : %s", self.outdir)
        for name in ("shots.json", "shots.csv", "documents.json", "_state.json"):
            p = self.outdir / name
            if p.exists():
                log.info("  - %s (%.1f KB)", name, p.stat().st_size / 1024)
        log.info("耗时     : %.1f s", time.time() - t0)


# ---------------------------------------------------------------- CLI
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="影视剧视频检索：单集索引构建")
    p.add_argument("video")
    p.add_argument("--outdir", required=True)
    p.add_argument("--subs", default=None, help="SRT 字幕文件（强烈建议提供）")

    p.add_argument("--backend", default="scenedetect",
                   choices=["scenedetect", "transnet", "diff"])
    p.add_argument("--transnet-repo", default=None)
    p.add_argument("--transnet-weights", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keyframes", type=int, default=2)
    p.add_argument("--min-shot-len", type=int, default=12)
    p.add_argument("--content-threshold", type=float, default=27.0)

    p.add_argument("--vlm-model", default=None,
                   help="如 qwen2.5-vl-72b-instruct / glm-4v / gpt-4o")
    p.add_argument("--vlm-base-url", default=None)
    p.add_argument("--vlm-api-key-env", default="OPENAI_API_KEY",
                   help="API Key 所在的环境变量名")
    p.add_argument("--cast", nargs="*", default=[],
                   help='角色表，如 "沈砚-男主，短发黑色风衣" "林晚-女主"')
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--skip-annotate", action="store_true")
    p.add_argument("--with-index", action="store_true")
    p.add_argument("--force", action="store_true", help="忽略缓存全量重跑")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    a = build_argparser().parse_args()

    spec = JobSpec(
        video=Path(a.video), outdir=Path(a.outdir),
        subs=Path(a.subs) if a.subs else None,
        backend=a.backend,
        transnet_repo=a.transnet_repo, transnet_weights=a.transnet_weights,
        device=a.device, keyframes_per_shot=a.keyframes,
        min_shot_len=a.min_shot_len, content_threshold=a.content_threshold,
        vlm_model=a.vlm_model, vlm_base_url=a.vlm_base_url,
        vlm_api_key_env=a.vlm_api_key_env, cast=a.cast,
        workers=a.workers, skip_annotate=a.skip_annotate,
        force=a.force, with_index=a.with_index,
    )
    if not spec.video.exists():
        raise SystemExit(f"视频不存在: {spec.video}")
    EpisodePipeline(spec).run()


if __name__ == "__main__":
    main()