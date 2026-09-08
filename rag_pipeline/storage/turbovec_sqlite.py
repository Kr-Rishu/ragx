"""
Local storage backend: turbovec IdMapIndex for vectors, sqlite for
chunk_id/source_type/content metadata. Embeddings come from the local
SentenceTransformer model (see embeddings.py) — no API cost.

check_same_thread=False + an internal lock: Streamlit's rerun model can call
into this backend from different threads across button clicks while the
connection object (held in st.session_state) is reused, which raises
sqlite3's "created in a thread can only be used in that thread" error without
this. The lock prevents overlapping writes across reruns from corrupting the
sqlite/turbovec state.
"""
import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from turbovec import IdMapIndex

from ..config import EMBEDDING_DIMENSION, INDEX_BIT_WIDTH
from ..embeddings import get_local_embedding_model
from .base import VectorStorageBackend


class TurbovecSqliteBackend(VectorStorageBackend):
    def __init__(self, collection_dir: str, collection_name: str, force_rebuild: bool = False):
        self.dir = Path(collection_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / f"{collection_name}_store.sqlite"
        self.index_path = self.dir / f"{collection_name}_index.tvim"

        if force_rebuild:
            self.db_path.unlink(missing_ok=True)
            self.index_path.unlink(missing_ok=True)

        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT,
                source_type TEXT,
                question_text TEXT,
                source_document TEXT,
                category TEXT,
                content TEXT
            )
        """)
        self.conn.commit()

        if force_rebuild or not self.index_path.exists():
            self.index = IdMapIndex(dim=EMBEDDING_DIMENSION, bit_width=INDEX_BIT_WIDTH)
        else:
            self.index = IdMapIndex.load(str(self.index_path))

    def embed_texts(self, texts):
        model = get_local_embedding_model()
        embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        return [e.astype(np.float32) for e in embeddings]

    def embed_query(self, text):
        model = get_local_embedding_model()
        emb = np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32).reshape(1, -1)
        return np.ascontiguousarray(emb)

    def chunk_already_indexed(self, chunk_id, document_name = None):
        with self._lock:
            row = self.conn.execute("SELECT 1 FROM chunks WHERE chunk_id = ? LIMIT 1", (chunk_id,)).fetchone()
        return row is not None

    def upsert_vectors(self, chunk_id, source_document, category, specs):
        with self._lock:
            vectors, ids = [], []
            for source_type, question_text, stored_text, embedding in specs:
                cur = self.conn.execute(
                    "INSERT INTO chunks (chunk_id, source_type, question_text, source_document, category, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, source_type, question_text, source_document, category, stored_text),
                )
                self.conn.commit()
                vectors.append(embedding)
                ids.append(cur.lastrowid)
            if vectors:
                self.index.add_with_ids(np.array(vectors, dtype=np.float32), np.array(ids, dtype=np.uint64))

    def search_by_type(self, query_embedding, source_type, target_distinct, overfetch_k=None, category=None):
        if category:
            rows = self.conn.execute(
                "SELECT id FROM chunks WHERE source_type = ? AND category = ?", (source_type, category)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT id FROM chunks WHERE source_type = ?", (source_type,)).fetchall()
        type_ids = np.array([r[0] for r in rows], dtype=np.uint64)
        if len(type_ids) == 0:
            return {}

        k = min(overfetch_k or target_distinct, len(type_ids))
        scores, ids = self.index.search(query_embedding, k=k, allowlist=type_ids)
        ids = ids.flatten().tolist()
        scores = scores.flatten().tolist()

        meta_ids = [i for i in ids if i != 0]
        meta = {}
        if meta_ids:
            placeholders = ",".join("?" * len(meta_ids))
            meta_rows = self.conn.execute(
                f"SELECT id, chunk_id FROM chunks WHERE id IN ({placeholders})", meta_ids
            ).fetchall()
            meta = {r[0]: r[1] for r in meta_rows}

        best = {}
        for vec_id, score in zip(ids, scores):
            if vec_id == 0 or vec_id not in meta:
                continue
            chunk_id = meta[vec_id]
            if chunk_id not in best or score < best[chunk_id]:
                best[chunk_id] = score
        return dict(sorted(best.items(), key=lambda x: x[1])[:target_distinct])

    def get_full_text_for_chunks(self, chunk_ids):
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, content FROM chunks WHERE source_type IN ('question', 'raw_chunk') AND chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        texts = {r[0]: r[1] for r in rows}

        missing = [cid for cid in chunk_ids if cid not in texts]
        if missing:
            placeholders_m = ",".join("?" * len(missing))
            summary_rows = self.conn.execute(
                f"SELECT chunk_id, content FROM chunks WHERE source_type = 'summary' AND chunk_id IN ({placeholders_m})",
                missing,
            ).fetchall()
            for chunk_id, content in summary_rows:
                texts[chunk_id] = content
        return texts

    def get_all_chunk_texts(self):
        rows = self.conn.execute("SELECT DISTINCT chunk_id FROM chunks").fetchall()
        chunk_ids = [r[0] for r in rows]
        texts = self.get_full_text_for_chunks(chunk_ids)
        return [(cid, texts[cid]) for cid in chunk_ids if cid in texts]

    def finalize(self):
        with self._lock:
            self.index.write(str(self.index_path))

    def close(self):
        self.conn.close()
