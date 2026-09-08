import re
from typing import Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL_NAME, STOPWORDS
from .models import RetrievalConfig, RetrievedChunk
from .storage.base import VectorStorageBackend
from .embeddings import get_local_embedding_model
_reranker_model = None

def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model

def _tokenize(text):
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def build_bm25_index(storage):
    rows = storage.get_all_chunk_texts()
    if not rows:
        return None, []
    corpus_chunk_ids = [r[0] for r in rows]
    tokenized_corpus = [_tokenize(r[1]) for r in rows]
    return BM25Okapi(tokenized_corpus), corpus_chunk_ids

def _keyword_search(bm25, corpus_chunk_ids, query_text, top_k, min_score=1.0):
    if bm25 is None or top_k <= 0:
        return []
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(corpus_chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, score in ranked if score >= min_score][:top_k]

def _reciprocal_rank_fusion(ranked_lists, rrf_k=60):
    fused = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

def _rerank(query_text, chunk_ids, texts_by_chunk, top_n):
    candidates = [(cid, texts_by_chunk[cid]) for cid in chunk_ids if cid in texts_by_chunk]
    if not candidates:
        return []
    reranker = get_reranker_model()
    pairs = [(query_text, text) for _, text in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [cid for _, (cid, _) in ranked[:top_n]]

def retrieve(storage, query_text, cfg, bm25=None, corpus_chunk_ids = None):
    emb = storage.embed_query(query_text)

    from_q = storage.search_by_type(emb, 'question', cfg.question_distinct, overfetch_k=cfg.question_overfetch_k)
    from_s = storage.search_by_type(emb, 'summary', cfg.summary_distinct)
    from_r = storage.search_by_type(emb, 'raw_chunk', cfg.raw_chunk_distinct)

    semantic_merged = dict(from_q)
    for source in (from_s, from_r):
        for cid, score in source.items():
            if cid not in semantic_merged or score < semantic_merged[cid]:
                semantic_merged[cid] = score
    semantic_ranked = [cid for cid, _ in sorted(semantic_merged.items(), key=lambda x: x[1])]

    if cfg.use_keyword and bm25 is not None:
        keyword_ranked = _keyword_search(bm25, corpus_chunk_ids, query_text, cfg.keyword_k, cfg.keyword_min_score)
        fused = _reciprocal_rank_fusion([semantic_ranked, keyword_ranked])
        shortlisted = [cid for cid, _ in fused[:cfg.fused_top_n]]
    else:
        shortlisted = semantic_ranked[:cfg.fused_top_n]

    if not shortlisted:
        return []

    texts_by_chunk = storage.get_full_text_for_chunks(shortlisted)
    final_ids = _rerank(query_text, shortlisted, texts_by_chunk, cfg.top_k)
    return [RetrievedChunk(chunk_id=cid, text=texts_by_chunk[cid]) for cid in final_ids if cid in texts_by_chunk]
