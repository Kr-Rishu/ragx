from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field

class ChunkRecord(BaseModel):
    """one chunk produced by chunking.py, ready to be embedded and stored"""
    chunk_id: str
    source: str
    summary: Optional[str] = None
    high_level_summary: Optional[list[str]] = None
    questions: list[str] = Field(default_factory=list)
    source_document: str
    category: str

class IndexingConfig(BaseModel):
    """what the user chose on the upload screen"""
    asset_category_documents: list[str] = Field(default_factory=list, description="Filenames the user flagged as asset-category (table-heavy) docs")
    storage_backend: Literal["turbovec_sqlite", "s3_vectors"] = "turbovec_sqlite"
    collection_name: str = "shared_knowledge_base"

class IndexingResult(BaseModel):
    document_name: str
    new_chunks_indexed: int
    skipped_chunks: int
    ingestion_llm_cost_usd: float = 0.0        # summary + questions + high-level-summary generation
    ingestion_embedding_cost_usd: float = 0.0   # 0.0 for turbovec+sqlite (local, free); Bedrock token cost for S3
    ingestion_total_cost_usd: float = 0.0

class RetrievalConfig(BaseModel):
    top_k: int = 5
    use_keyword: bool = True
    question_distinct: int = 15
    summary_distinct: int = 15
    raw_chunk_distinct: int = 15
    question_overfetch_k: int = 60
    keyword_k: int = 60
    keyword_min_score: float = 1.0
    fused_top_n: int = 30

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None

class QueryResult(BaseModel):
    question: str
    answer: str
    has_sufficient_context: bool
    retrieved_chunks: list[RetrievedChunk]
    retrieval_latency_sec: float
    generation_latency_sec: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

class EvalConfig(BaseModel):
    mode: Literal["qa_pairs", "questions_only"]
    n_questions: Optional[int] = None   # None = evaluate all rows provided
    correctness_threshold: float = 0.5           # qa_pairs mode: G-Eval correctness vs expected answer
    faithfulness_threshold: float = 0.5           # questions_only mode: grounded in retrieved context
    answer_relevancy_threshold: float = 0.5        # questions_only mode: actually addresses the question

class EvalRowResult(BaseModel):
    row_id: int
    question: str
    expected_answer: Optional[str] = None
    generated_answer: str
    correctness_score: Optional[float] = None
    correctness_reason: Optional[str] = None
    faithfulness_score: Optional[float] = None
    faithfulness_reason: Optional[str] = None
    answer_relevancy_score: Optional[float] = None
    answer_relevancy_reason: Optional[str] = None
    passed: bool = True
    retrieval_latency_sec: float = 0.0
    generation_latency_sec: float = 0.0
    generation_cost_usd: float = 0.0   # answer-LLM cost for this row
    eval_cost_usd: float = 0.0          # judge-LLM cost for this row (correctness OR faithfulness+relevancy)

class EvalReport(BaseModel):
    mode: str
    n_evaluated: int
    n_failed: int
    avg_correctness: Optional[float] = None
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None

    avg_retrieval_latency_sec: float
    avg_generation_latency_sec: float
    retrieval_latency_p50: float
    retrieval_latency_p95: float
    retrieval_latency_p99: float
    generation_latency_p50: float
    generation_latency_p95: float
    generation_latency_p99: float
    total_latency_p50: float
    total_latency_p95: float
    total_latency_p99: float

    total_generation_cost_usd: float
    total_eval_cost_usd: float
    total_cost_usd: float
    total_cost_inr: float

    failed_rows: list[EvalRowResult]
    all_rows: list[EvalRowResult]
    passed_qa_excel: bytes | None = None
