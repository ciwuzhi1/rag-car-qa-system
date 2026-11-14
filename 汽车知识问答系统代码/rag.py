# -*- coding: utf-8 -*-
"""最简 RAG 检索：使用本地 bge-m3 嵌入模型对知识库做语义向量匹配。

向量持久化在 SQLite 的 knowledge_vectors 表中（knowledge_id + 内容签名 + 向量 BLOB）：
- 服务重启后直接复用已存向量，无需重新编码；
- 只有新增或修改过的知识条目（内容签名变化）才会重新编码并写回数据库；
- 已删除知识条目对应的向量会在检索时自动清理。

用法：
    import rag
    if rag.is_available():
        hits = rag.search(rows, question)   # [(row, score), ...] 按相似度降序
"""
import os
import sqlite3

# 本地 bge-m3 模型目录（如路径变动，修改此处或设置为环境变量 BGE_M3_PATH）
MODEL_PATH = os.environ.get(
    'BGE_M3_PATH', r'D:\models\bge-m3')

# 向量存储库与应用主库为同一 SQLite 文件
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'knowledge_qa.db')

_model = None


def is_available():
    return os.path.isdir(MODEL_PATH)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_PATH, device='cpu')
    return _model


def _db():
    conn = sqlite3.connect(DB_PATH)
    return conn


def _signature(row):
    return (row.get('question') or '') + '|' + (row.get('keywords') or '')


def _load_cached_vectors(rows):
    """从 SQLite 加载与当前内容签名一致的向量，返回 {id: numpy 向量}"""
    import numpy as np
    wanted = {r['id']: _signature(r) for r in rows}
    vectors = {}
    conn = _db()
    try:
        cur = conn.execute(
            'SELECT knowledge_id, signature, embedding FROM knowledge_vectors')
        for kid, sig, blob in cur.fetchall():
            if wanted.get(kid) == sig:
                vectors[kid] = np.frombuffer(blob, dtype=np.float32)
    finally:
        conn.close()
    return vectors


def _save_vectors(items):
    """写入/更新向量，items: [(knowledge_id, signature, numpy 向量)]"""
    conn = _db()
    try:
        conn.executemany(
            'INSERT OR REPLACE INTO knowledge_vectors '
            '(knowledge_id, signature, embedding) VALUES (?, ?, ?)',
            [(kid, sig, vec.astype('float32').tobytes())
             for kid, sig, vec in items])
        conn.commit()
    finally:
        conn.close()


def _delete_stale_vectors(valid_ids):
    """清理已不存在（或内容已变化但 id 保留）的向量由签名比对兜底，这里只删 id 失效的"""
    conn = _db()
    try:
        if valid_ids:
            marks = ','.join('?' * len(valid_ids))
            conn.execute(f'DELETE FROM knowledge_vectors '
                         f'WHERE knowledge_id NOT IN ({marks})', valid_ids)
        else:
            conn.execute('DELETE FROM knowledge_vectors')
        conn.commit()
    finally:
        conn.close()


def search(rows, question, top_k=3):
    """在知识条目中检索与问题语义最相似的条目。

    rows: [{'id', 'question', 'answer', 'keywords'}, ...]
    返回 [(row, score), ...]，score 为余弦相似度（0~1），按降序排列。
    """
    if not rows:
        return []
    model = _get_model()
    import numpy as np

    vectors = _load_cached_vectors(rows)

    # 只对新增/修改过的条目编码并持久化
    changed = [(r, _signature(r)) for r in rows if r['id'] not in vectors]
    if changed:
        embs = model.encode([_signature(r) for r, _ in changed],
                            normalize_embeddings=True)
        _save_vectors([(r['id'], sig, emb) for (r, sig), emb in zip(changed, embs)])
        for (r, sig), emb in zip(changed, embs):
            vectors[r['id']] = emb

    # 清理已被删除的知识条目残留向量
    _delete_stale_vectors([r['id'] for r in rows])

    q = model.encode([question], normalize_embeddings=True)[0]
    ids = [r['id'] for r in rows]
    mat = np.stack([vectors[i] for i in ids])
    scores = mat @ q                      # 归一化向量的点积即余弦相似度
    order = np.argsort(-scores)[:top_k]
    return [(rows[i], float(scores[i])) for i in order]
