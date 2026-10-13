from sentence_transformers import CrossEncoder
model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
pairs = [('什么是重排序？', '重排序是对候选文档重新排序的过程'),
         ('什么是重排序？', '今天天气晴朗')]
scores = model.predict(pairs)                  # numpy array，logits
print(scores)
# 转 0~1 概率
import numpy as np
probs = 1 / (1 + np.exp(-scores))
print(probs)
