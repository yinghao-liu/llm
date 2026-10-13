"""
Caption-based Image RAG
索引: 图片 -> GPT-4o-mini 生成结构化 caption -> 文本向量 -> 向量索引
检索: query -> 文本向量 -> 相似度 TopK -> (可选 rerank) -> GPT-4o 读图回答
"""
# import os
import base64
import io
import json
from pathlib import Path

import numpy as np
from openai import OpenAI
from PIL import Image, ImageTk
import tkinter as tk

import jieba
from rank_bm25 import BM25Okapi
# from sentence_transformers import CrossEncoder



# ---------- 模型配置 ----------
MODEL_CAPTION = "qwen/qwen3-vl-4b"      # 生成 caption
MODEL_EMBED = "text-embedding-bge-m3"  # caption/query 向量化
MODEL_ANSWER = "qwen/qwen3-vl-4b"       # 最终读图回答
# MODEL_RERANK = "BAAI/bge-reranker-v2-m3"    # rerank

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


# ---------- 工具函数 ----------
def encode_image(path: Path, max_side: int = 1024) -> str:
    """读取图片，过大则缩小，返回 base64 data URL（省 token）"""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

def show_image(path, max_scale=0.9):
    root = tk.Tk()
    root.title("图片预览")
    # 1. 获取屏幕可用区域（自动去掉任务栏）
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    # 2. 打开图片并按比例缩放到屏幕的 max_scale 倍以内
    img = Image.open(path)
    ratio = min(screen_w * max_scale / img.width,
                screen_h * max_scale / img.height, 1.0)  # 不放大
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=photo)
    label.pack()
    # 3. 等待用户操作：点击或按任意键 / Esc 关闭
    root.bind("<Button>", lambda e: root.destroy())
    root.bind("<Key>", lambda e: root.destroy())
    root.focus_set()
    root.mainloop()   # 阻塞在这里，直到窗口关闭


STOPWORDS = set("的 了 在 是 和 有 与 及 请 为 这 那 上 一只 一个".split())
def tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text)
            if w.strip() and w not in STOPWORDS and len(w.strip()) > 1]

CAPTION_SYSTEM = """你是图像检索系统的标注员。请为图片生成用于"以文搜图"的结构化描述。
要求：
- caption: 80字以内的中文描述，覆盖主体、外观属性(颜色/材质/形状/数量)、场景背景、动作状态
- ocr: 图片中出现的所有文字（招牌/标签/水印/屏幕文字），没有则为空字符串
- tags: 8个以内的中文关键词
只输出 JSON：{"caption": "...", "ocr": "...", "tags": ["...", ...]}"""


def generate_caption(client: OpenAI, image_url: str) -> dict:
    """调用多模态模型生成结构化 caption"""
    resp = client.chat.completions.create(
        model=MODEL_CAPTION,
        temperature=0,
        # response_format={"type": "json_object"},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "caption",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["caption", "tags"],
                    "additionalProperties": False,
                },
            },
        },
        messages=[
            {"role": "system", "content": CAPTION_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "请为这张图片生成结构化描述。"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "caption": (data.get("caption") or "").strip(),
        "ocr": (data.get("ocr") or "").strip(),
        "tags": [t.strip() for t in data.get("tags", []) if t.strip()],
    }


def to_description(des: dict) -> str:
    """把结构化 caption 拼成一段用于检索的文本"""
    parts = [des["caption"]]
    if des["tags"]:
        parts.append("标签: " + "、".join(des["tags"]))
    if des["ocr"]:
        parts.append("图片文字: " + des["ocr"])
    return "\n".join(parts)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


# ---------- 主体 ----------
class ImageRAG:
    def __init__(self, image_dir: str, cache_file: str = "caption_cache.json"):
        self.client = OpenAI(api_key="sk-xxx", base_url="http://127.0.0.1:1234/v1")
        self.image_dir = Path(image_dir)
        self.cache_file = Path(cache_file)
        self.cache: dict = {}
        if self.cache_file.exists():
            self.cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
        self.paths: list[Path] = []
        self.docs: list[str] = []
        self.vecs: np.ndarray | None = None
        self.bm25 = None
        # self.rerank = CrossEncoder(MODEL_RERANK, max_length=512)


    # ---- 索引阶段 ----
    def build(self):
        """扫描图片 -> 生成/复用 caption -> 向量化。带缓存"""
        paths = sorted(p for p in self.image_dir.rglob("*")
                       if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise ValueError(f"{self.image_dir} 下没有图片")

        kept, caps, tags = [], [], []
        for p in paths:
            key = str(p.relative_to(self.image_dir))
            mtime = p.stat().st_mtime
            cached = self.cache.get(key)

            if cached is None or cached.get("mtime") != mtime:
                try:
                    cap = generate_caption(self.client, encode_image(p))
                except Exception as e:                      # 单张失败不中断整体
                    print(f"[warn] caption 失败 {p}: {e}")
                    continue
                cached = {"mtime": mtime, "description": cap}
                self.cache[key] = cached
                self._save_cache()                          # 每张存一次，崩溃也不丢
                print(f"[caption] {key}: {cap['caption'][:30]}...")

            kept.append(p)
            # docs.append(to_document(cached["description"]))
            caps.append(cached["description"]["caption"])
            tags.append(cached["description"]["tags"])

        # 向量索引
        self.paths, self.docs = kept, caps
        self.vecs = l2_normalize(self._embed(caps))
        print(f"embedding 索引完成: {len(kept)} 张图片")

        # bm25关键词索引 TODO tags + ocr
        self.bm25 = BM25Okapi(tags)
        print(f"BM25 索引完成: {len(tags)} 张图片")


    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):                  # 批量请求
            batch = texts[i:i + 64]
            r = self.client.embeddings.create(model=MODEL_EMBED, input=batch)
            out.extend(d.embedding for d in sorted(r.data, key=lambda x: x.index))
        return np.array(out, dtype="float32")

    def _save_cache(self):
        self.cache_file.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 检索阶段 ----
    def search(self, query: str, topk: int = 5) -> list[dict]:
        """以文搜图：返回 [{path, score, caption}]"""
        if self.vecs is None:
            raise RuntimeError("请先调用 build()")
        q = l2_normalize(self._embed([query]))
        scores = (self.vecs @ q[0])                          # 余弦相似度(已归一化)
        idx = np.argsort(-scores)[:topk]
        return [{"path": str(self.paths[i]),
                 "score": float(scores[i]),
                 "caption": self.docs[i]} for i in idx]

    def llm_rerank(self, query: str, hits: list[dict], keep: int = 3) -> list[dict]:
        """可选：用 LLM 只看 caption 文本做精排（比逐图打分便宜）"""
        lines = "\n".join(f"[{i}] {h['caption']}" for i, h in enumerate(hits))
        resp = self.client.chat.completions.create(
            model=MODEL_CAPTION, temperature=0,
            messages=[{"role": "user", "content":
                f"用户检索词：{query}\n候选图片描述：\n{lines}\n\n"
                f"按相关度从高到低输出最相关的 {keep} 个编号，JSON 数组格式，如 [2,0,1]"}],
        )
        try:
            order = json.loads(resp.choices[0].message.content)
            return [hits[i] for i in order[:keep] if i < len(hits)]
        except Exception:
            return hits[:keep]

    # ---- 生成回答 ----
    def answer(self, query: str, topk: int = 3, use_rerank: bool = False) -> str:
        """Image RAG：召回 -> (精排) -> 把图片+caption 交给多模态大模型回答"""
        hits = self.search(query, topk=max(topk, 10) if use_rerank else topk)
        if use_rerank:
            hits = self.llm_rerank(query, hits, keep=topk)

        content = [{"type": "text",
                    "text": f"用户问题：{query}\n\n以下是检索到的图片及其描述："}]
        for i, h in enumerate(hits):
            content.append({"type": "text",
                            "text": f"\n图片{i + 1}（描述：{h['caption']}）："})
            content.append({"type": "image_url",
                            "image_url": {"url": encode_image(Path(h["path"]))}})

        resp = self.client.chat.completions.create(
            model=MODEL_ANSWER, temperature=0.3,
            messages=[
                {"role": "system", "content":
                 "你是图片库问答助手。根据检索到的图片回答问题，"
                 "如果图片不足以回答请明确说明，不要编造。"},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content



    # ---- 检索：向量 + BM25 两路召回，RRF 融合 ----
    def search_hybrid(self, query: str, topk: int = 5, k: int = 20) -> list[dict]:
        # 第一路：稠密向量（语义）
        vec_hits = self.search(query, topk=10)
        vec_rank = {h["path"]:h["score"] for h in vec_hits}
        # print(caption_map)
        print(f"--vec_rank-- {vec_rank}")


        # 第二路：BM25（词法）
        kw_scores = self.bm25.get_scores(tokenize(query))
        # kw_scores = 1 / (1 + np.exp(-scores))
        # print(f"--scores-- {scores}")
        # print(f"--kw_scores-- {kw_scores}")
        kw_rank = {str(self.paths[i]):float(kw_scores[i])
                    for i in np.argsort(-kw_scores)[:10] if kw_scores[i] > 0}
        print(f"--kw_rank-- {kw_rank}")

        return [{"path": path, "score": vec_rank[path] + kw_rank.get(path, 0)} for path in vec_rank]


    def search_hybrid_bak(self, query: str, topk: int = 5, k: int = 20) -> list[dict]:
        
        # 第一路：稠密向量（语义）
        vec_hits = self.search(query, topk=10)
        vec_rank = [h["path"] for h in vec_hits]
        # print(caption_map)
        print(f"--vec_rank-- {vec_rank}")


        # 第二路：BM25（词法）
        kw_scores = self.bm25.get_scores(tokenize(query))
        kw_rank = [str(self.paths[i])
                   for i in np.argsort(-kw_scores)[:10] if kw_scores[i] > 0]
        print(f"--kw_rank-- {kw_rank}")

        # RRF 融合：score = Σ 1/(k + rank)，对排名融合而非对分数融合
        # rrf = {}
        # for rank, path in enumerate(vec_rank):
        #     rrf[path] = rrf.get(path, 0) + 1 / (k + rank + 1)
        # for rank, path in enumerate(kw_rank):
        #     rrf[path] = rrf.get(path, 0) + 1 / (k + rank + 1)
        # top = sorted(rrf, key=rrf.get, reverse=True)[:topk]
        # print(f"rrf:{rrf}")
        # print(top)

        # rerank
        # pairs = [('什么是重排序？', '重排序是对候选文档重新排序的过程'),
        #         ('什么是重排序？', '今天天气晴朗')]
        # description_map = {}
        # for i in top:
        #     name = os.path.basename(i)
        #     description_map[i] = to_description(rag.cache[name]["description"])
        # caption_map = {h["path"]:h["caption"] for h in vec_hits}

        # pairs = [(query, description_map[i]) for i in top]
        # print(f"pairs:{pairs}")
        # scores = self.rerank.predict(pairs)                  # numpy array，logits
        # print(f"scores:{scores}")
        # # 转 0~1 概率
        # probs = 1 / (1 + np.exp(-scores))
        # print(probs)
        # return [{"path": path, "score": probs[index]} for index,path in enumerate(top)]




# ---------- 演示入口 ----------
if __name__ == "__main__":
    rag = ImageRAG("../../resource/image/")     # 图片目录
    rag.build()                    # 首次会调用 API 生成 caption，之后走缓存

    # print("\n以文搜图 Top3:")
    # for h in rag.search("天上的云彩", topk=3):
    #     print(f"  {h['score']:.3f}  {h['path']}")
    # print(rag.cache)
    # 交互式查询
    print("\n" + "=" * 72)
    while True:
        q = input("\n输入检索问句(回车退出): ").strip()
        if not q:
            break
        for h in rag.search_hybrid(q, topk=3):
            print(f"  {h['score']:.3f}  {h['path']}")
            if h['score']  < 0.6:
                continue
            show_image(h['path'])

    # print("\nImage RAG 回答:")
    # print(rag.answer("小狗奔跑的图片", topk=3, use_rerank=True))