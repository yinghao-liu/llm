#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ---------------- 模型 ----------------
model = SentenceTransformer("BAAI/bge-m3")   # 首次运行自动下载(约2GB)

# ---------------- 示例数据 ----------------
captions = [
    {"video_id": "v1", "t_start": 5,   "t_end": 10,
     "text": "一名穿红色上衣的长发女子推门进入店内，站在货架前"},
    {"video_id": "v1", "t_start": 10,  "t_end": 15,
     "text": "该女子从包里拿出手机，低头看屏幕"},
    {"video_id": "v1", "t_start": 20,  "t_end": 25,
     "text": "一名男子在窗边点燃香烟，吸了一口并看向窗外"},
    {"video_id": "v1", "t_start": 45,  "t_end": 50,
     "text": "女子将货架上的商品放入自己的包中，随后空手离开"},
    {"video_id": "v1", "t_start": 80,  "t_end": 85,
     "text": "店员在收银台整理货物，店内没有其他顾客"},
    {"video_id": "v2", "t_start": 0,   "t_end": 5,
     "text": "两个孩子在公园的草坪上追逐玩耍"},
    {"video_id": "v2", "t_start": 30,  "t_end": 35,
     "text": "一名穿蓝色制服的保安从大门走过"},
]

# ---------------- 建索引 ----------------
def build_index(caps: list[dict]) -> faiss.Index:
    texts = [c["text"] for c in caps]
    vecs = model.encode(texts,
                        normalize_embeddings=True,   # 归一化 → 余弦=点积
                        batch_size=64,
                        show_progress_bar=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.asarray(vecs, dtype="float32"))
    return index

# ---------------- 检索 ----------------
def search_vector(query: str, index: faiss.Index, caps: list[dict], top_k=5):
    # bge-m3 不加指令前缀；若换成 bge-large-zh-v1.5 再加：
    # query = "为这个句子生成表示以用于检索相关文章：" + query
    q_vec = model.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.asarray(q_vec, dtype="float32"), top_k)
    return [{**caps[i], "score": float(s)} for i, s in zip(idxs[0], scores[0])]

# ---------------- 工具：秒 → mm:ss ----------------
def fmt(t: float) -> str:
    return f"{int(t)//60:02d}:{int(t)%60:02d}"

def show(query: str, results: list[dict]):
    print(f"\n问句：{query}")
    print("-" * 72)
    for r in results:
        print(f"  [{fmt(r['t_start'])} - {fmt(r['t_end'])}]  "
              f"(相似度 {r['score']:.3f}, 视频 {r['video_id']})")
        print(f"    {r['text']}")
    if results:
        best = results[0]
        print(f"\n  ✔ 最匹配位置：{best['video_id']} 的 "
              f"{fmt(best['t_start'])} ~ {fmt(best['t_end'])}")

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    index = build_index(captions)
    print(f"\n索引完成：{len(captions)} 条记录，向量维度 {index.d}")

    # 演示查询
    for q in ["有人抽烟吗", "穿红衣服的女人什么时候出现",
              "有人偷东西吗", "小孩在玩"]:
        show(q, search_vector(q, index, captions, top_k=3))

    # 交互式查询
    print("\n" + "=" * 72)
    while True:
        q = input("\n输入检索问句(回车退出): ").strip()
        if not q:
            break
        show(q, search_vector(q, index, captions, top_k=5))