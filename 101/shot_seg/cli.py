"""命令行入口。"""
import argparse
from .core import ShotConfig
from .pipeline import ShotSegmenter


def main():
    p = argparse.ArgumentParser(description="影视剧镜头切分")
    p.add_argument("video")
    p.add_argument("--backend", default="scenedetect",
                   choices=["scenedetect", "transnet", "diff"])
    p.add_argument("--out", default="./shot_out")
    p.add_argument("--keyframes", type=int, default=2)
    p.add_argument("--min-shot-len", type=int, default=12)
    p.add_argument("--content-threshold", type=float, default=27.0)
    p.add_argument("--transnet-repo", default=None,
                   help="TransNetV2/inference-pytorch 目录")
    p.add_argument("--transnet-weights", default=None,
                   help="transnetv2-pytorch-weights.pth")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    cfg = ShotConfig(min_shot_len=a.min_shot_len,
                     content_threshold=a.content_threshold)
    seg = ShotSegmenter(cfg, backend=a.backend,
                        transnet_repo=a.transnet_repo,
                        transnet_weights=a.transnet_weights,
                        device=a.device)
    data = seg.segment_and_export(a.video, a.out,
                                  keyframes_per_shot=a.keyframes)
    print(f"共 {data['shot_count']} 个镜头 -> {a.out}/shots.json")
    for s in data["shots"][:10]:
        print(f"  #{s['shot_id']:>4}  {s['start']:>8.2f}s ~ {s['end']:<8.2f}s "
              f"[{s['transition_in']}]")


if __name__ == "__main__":
    main()