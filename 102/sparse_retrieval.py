import jieba
from rank_bm25 import BM25Okapi

STOPWORDS = set("的 了 在 是 和 有 与 及 请 为 这 那 上 一只 一个".split())
def tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text)
            if w.strip() and w not in STOPWORDS and len(w.strip()) > 1]
# tokenize test
jieba.add_word("杰克罗素梗")
print(tokenize("杰克罗素梗在奔跑"))   # ['杰克罗素梗', '在', '奔跑'] ✔️
# 建索引时：tags + OCR 文本 → 分词 → BM25
# corpus = [tokenize("小狗 奔跑 梗犬 草地"), tokenize("SALE 五折 促销 红色海报")]
# bm25 = BM25Okapi(corpus)
# 检索时：query 同样分词
# scores = bm25.get_scores(tokenize("小狗奔跑"))


# 1. 语料：每张图的 tags + OCR 文本
docs = [
    "小狗 奔跑 梗犬 草地",          # 图1
    "SALE 五折 促销海报 红色",       # 图2
    "小狗 趴着 窗台 橘猫",           # 图3
]


corpus = [tokenize(d) for d in docs]
bm25 = BM25Okapi(corpus)              # 建索引（内部算好 IDF、文档长度）
# 3. 检索
scores = bm25.get_scores(tokenize("小狗奔跑"))
print(scores)
# scores = [1.52, 0.0, 0.31]  → 图1 排第一
ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
print(ranked)