from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class VectorStorageBackend(ABC):
    """source_type is one of: 'summary', 'question', 'raw_chunk'"""

    @abstractmethod
    def embed_texts(self, texts):
        """batch-embed texts for indexing. Backend-specific: local SentenceTransformer
        for turbovec+sqlite, Bedrock Titan for S3 Vectors. Callers never load an
        embedding model themselves - always go through the backend"""
        ...

    @abstractmethod
    def embed_query(self, text):
        """embed a single query string, in the SAME embedding space as embed_texts -
        returned pre-shaped as (1, dim) float32 contiguous, ready for search_by_type"""
        ...

    @abstractmethod
    def upsert_vectors(self, chunk_id, source_document, category, specs):
        """specs: list of (source_type, question_text_or_None, stored_text, embedding)"""
        ...

    @abstractmethod
    def chunk_already_indexed(self, chunk_id, document_name = None):
        """document_name scopes the check per-document where the backend supports it
        (S3 Vectors - identical chunk content in two different uploaded documents
        must NOT be treated as already-indexed for the second one). Backends that
        don't need document scoping (turbovec+sqlite) may ignore document_name"""
        ...

    @abstractmethod
    def search_by_type(self, query_embedding, source_type, target_distinct, overfetch_k = None, category = None):
        """returns {chunk_id: distance_score}, best (lowest) distance kept per chunk_id.
        category, when given, restricts results to that category ('asset' or
        'general') - mainly useful for callers who want to explicitly scope a
        query to asset-only or general-only content. Left None, no category
        constraint is applied (the normal case: source_type alone already
        distinguishes asset content, since only asset chunks produce
        'raw_chunk' rows)"""
        ...

    @abstractmethod
    def get_full_text_for_chunks(self, chunk_ids):
        ...

    @abstractmethod
    def get_all_chunk_texts(self):
        """for BM25 corpus construction"""
        ...

    @abstractmethod
    def finalize(self):
        """flush/write the index to durable storage (file write, S3 PUT, etc)"""
        ...

    @abstractmethod
    def close(self):
        ...

    def finalize_document(self, document_name, current_chunk_ids):
        """called once after ALL chunks of ONE document have been processed
        (new chunks upserted + unchanged chunks marked via mark_chunks_unchanged).
        Default no-op — turbovec+sqlite has no per-document versioning concept.
        S3VectorsBackend overrides this to push a version manifest, deactivate
        stale chunks no longer present in this version, and evict old versions
        past the retention limit"""
        pass

    def mark_chunks_unchanged(self, document_name, chunk_ids):
        """called for chunks that were already indexed and skipped re-extraction
        entirely (chunking.py never called the LLM for them), so this document's
        current re-index run still knows they belong to the CURRENT version.
        Default no-op. S3VectorsBackend overrides this to bump these chunks'
        existing vectors onto the current version/active=True without
        re-embedding or re-calling the LLM"""
        pass

    def get_and_reset_embedding_cost_usd(self):
        """cumulative embedding API cost incurred since the last call to this
        method, then resets the counter to zero. Default 0.0 - local embedding
        models (turbovec+sqlite) have no per-token API cost. S3VectorsBackend
        overrides this using Bedrock's reported input token counts"""
        return 0.0
