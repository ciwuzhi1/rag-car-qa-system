# -*- coding: utf-8 -*-
"""最简 RAG 检索：使用本地 bge-m3 嵌入模型对知识库做语义向量匹配。

用法：
    import rag
    if rag.is_available():
        hits = rag.search(rows, question)   # [(row, score), ...] 按相似度降序
"""
import os

# 本地 bge-m3 模型目录（如路径变动，修改此处或设置为环境变量 BGE_M3_PATH）
MODEL_PATH = os.environ.get(
    'BGE_M3_PATH', r'D:\尚硅谷day01\资料\LangChain_Day01\2.资料\bge-m3')

_model = None
_vector_cache = {}   # knowledge_id -> (内容签名, 归一化向量)


def is_available():
    return os.path.isdir(MODEL_PATH)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_PATH, device='cpu')
    return _model


def _signature(row):
    return (row.get('question') or '') + '|' + (row.get('keywords') or '')


def search(rows, question, top_k=3):
    """在知识条目中检索与问题语义最相似的条目。

    rows: [{'id', 'question', 'answer', 'keywords'}, ...]
    返回 [(row, score), ...]，score 为余弦相似度（0~1），按降序排列。
    内容未变化的条目直接复用缓存向量，只有新增/修改过的条目才重新编码。
    """
    if not rows:
        return []
    model = _get_model()
    import numpy as np

    # 只对内容有变化的知识重新编码
    changed, vectors = [], {}
    for r in rows:
        sig = _signature(r)
        cached = _vector_cache.get(r['id'])
        if cached and cached[0] == sig:
            vectors[r['id']] = cached[1]
        else:
            changed.append((r, sig))
    if changed:
        embs = model.encode([_signature(r) for r, _ in changed],
                            normalize_embeddings=True)
        for (r, sig), emb in zip(changed, embs):
            _vector_cache[r['id']] = (sig, emb)
            vectors[r['id']] = emb

    q = model.encode([question], normalize_embeddings=True)[0]
    ids = [r['id'] for r in rows]
    mat = np.stack([vectors[i] for i in ids])
    scores = mat @ q                      # 归一化向量的点积即余弦相似度
    order = np.argsort(-scores)[:top_k]
    return [(rows[i], float(scores[i])) for i in order]
