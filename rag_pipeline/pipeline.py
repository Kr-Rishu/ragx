import time
from typing import Optional
import pandas as pd
from .config import DEFAULT_TOP_K
from .models import RetrievalConfig, QueryResult, EvalConfig, EvalRowResult, EvalReport, RetrievedChunk, IndexingResult
from .storage import get_storage_backend
from .storage.base import VectorStorageBackend
from .chunking import chunk_document, build_specs_for_chunk
from .retrieval import retrieve, build_bm25_index
from .generation import generate_answer
from .evaluation import score_correctness, score_faithfulness, score_answer_relevancy, estimate_cost_usd, build_report

class RAGPipeline:
    def __init__(self, segmenter, collection_name, storage_backend = "turbovec_sqlite", storage_kwargs = None, default_retrieval_config = None):
        self.segmenter = segmenter
        self.storage: VectorStorageBackend = get_storage_backend(storage_backend, collection_name, **(storage_kwargs or {}))
        self.default_retrieval_config = default_retrieval_config or RetrievalConfig(top_k=DEFAULT_TOP_K)
        self._bm25 = None
        self._bm25_corpus_ids = None
        self._bm25_stale = True
        self._cumulative_ingestion_cost_usd = 0.0

    # indexing
    def index_document(self, document_path, source_document_name, is_asset_category):
        new_records, skipped_chunk_ids, ingestion_llm_cost_usd = chunk_document(
            self.segmenter, self.storage, document_path, source_document_name, is_asset_category
        )

        if skipped_chunk_ids:
            self.storage.mark_chunks_unchanged(source_document_name, skipped_chunk_ids)

        for record in new_records:
            specs = build_specs_for_chunk(record)
            if not specs:
                continue
            embed_texts_list = [s[2] for s in specs]          # index 2 = embed_text
            embeddings = self.storage.embed_texts(embed_texts_list)
            full_specs = [
                (source_type, question_text, stored_text, embedding)
                for (source_type, question_text, embed_text, stored_text), embedding in zip(specs, embeddings)
            ]
            self.storage.upsert_vectors(record.chunk_id, record.source_document, record.category, full_specs)

        ingestion_embedding_cost_usd = self.storage.get_and_reset_embedding_cost_usd()
        ingestion_total_cost_usd = ingestion_llm_cost_usd + ingestion_embedding_cost_usd
        self._cumulative_ingestion_cost_usd += ingestion_total_cost_usd

        all_chunk_ids = skipped_chunk_ids + [r.chunk_id for r in new_records]
        self.storage.finalize_document(source_document_name, all_chunk_ids)

        self._bm25_stale = True
        return IndexingResult(
            document_name=source_document_name,
            new_chunks_indexed=len(new_records),
            skipped_chunks=len(skipped_chunk_ids),
            ingestion_llm_cost_usd=round(ingestion_llm_cost_usd, 6),
            ingestion_embedding_cost_usd=round(ingestion_embedding_cost_usd, 6),
            ingestion_total_cost_usd=round(ingestion_total_cost_usd, 6)
        )

    def get_cumulative_ingestion_cost_usd(self):
        return round(self._cumulative_ingestion_cost_usd, 6)

    def finalize_index(self):
        self.storage.finalize()
        self._ensure_bm25()

    def _ensure_bm25(self):
        if self._bm25_stale:
            self._bm25, self._bm25_corpus_ids = build_bm25_index(self.storage)
            self._bm25_stale = False

    # query
    def query(self, question, retrieval_config = None):
        cfg = retrieval_config or self.default_retrieval_config
        self._ensure_bm25()

        t0 = time.time()
        chunks = retrieve(self.storage, question, cfg, bm25=self._bm25, corpus_chunk_ids=self._bm25_corpus_ids)
        retrieval_latency = time.time() - t0

        t1 = time.time()
        gen, prompt_tokens, completion_tokens = generate_answer(question, chunks)
        generation_latency = time.time() - t1

        return QueryResult(
            question=question,
            answer=gen.answer,
            has_sufficient_context=gen.has_sufficient_context,
            retrieved_chunks=chunks,
            retrieval_latency_sec=retrieval_latency,
            generation_latency_sec=generation_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimate_cost_usd(prompt_tokens, completion_tokens),
        )

    # evaluation   
    def evaluate(self, df, cfg, question_col = 'Question', expected_answer_col = 'Expected Answer', retrieval_config = None):
        rows_to_run = df if cfg.n_questions is None else df.head(cfg.n_questions)
        results: list[EvalRowResult] = []

        for row_id, row in rows_to_run.iterrows():
            question = str(row[question_col])
            expected = str(row[expected_answer_col]) if cfg.mode == "qa_pairs" and expected_answer_col in row else None

            qr = self.query(question, retrieval_config=retrieval_config)
            retrieval_texts = [c.text for c in qr.retrieved_chunks]

            correctness_score = correctness_reason = None
            faith_score = faith_reason = None
            relevancy_score = relevancy_reason = None
            eval_cost_usd = 0.0

            if cfg.mode == 'qa_pairs':
                correctness_score, correctness_reason, judge_cost = score_correctness(
                    question, expected, qr.answer, threshold=cfg.correctness_threshold
                )
                eval_cost_usd += judge_cost

                passed = correctness_score is not None and correctness_score >= cfg.correctness_threshold
            else:
                faith_score, faith_reason, faith_cost = score_faithfulness(
                    question, qr.answer, retrieval_texts, threshold=cfg.faithfulness_threshold
                )
                relevancy_score, relevancy_reason, rel_cost = score_answer_relevancy(
                    question, qr.answer, threshold=cfg.answer_relevancy_threshold
                )
                eval_cost_usd += faith_cost + rel_cost
                faith_passed = faith_score is not None and faith_score >= cfg.faithfulness_threshold
                rel_passed = relevancy_score is not None and relevancy_score >= cfg.answer_relevancy_threshold
                passed = faith_passed and rel_passed

            results.append(EvalRowResult(
                row_id=row_id,
                question=question,
                expected_answer=expected,
                generated_answer=qr.answer,
                correctness_score=correctness_score,
                correctness_reason=correctness_reason,
                faithfulness_score=faith_score,
                faithfulness_reason=faith_reason,
                answer_relevancy_score=relevancy_score,
                answer_relevancy_reason=relevancy_reason,
                passed=passed,
                retrieval_latency_sec=qr.retrieval_latency_sec,
                generation_latency_sec=qr.generation_latency_sec,
                generation_cost_usd=qr.estimated_cost_usd,
                eval_cost_usd=eval_cost_usd,
            ))

        return build_report(cfg.mode, results)

    def close(self):
        self.storage.close()
