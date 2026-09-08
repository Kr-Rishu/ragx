from typing import Optional
from langchain_community.callbacks import get_openai_callback
from .config import (
    INGESTION_ASSET_INPUT_COST_PER_M_TOKENS, INGESTION_ASSET_OUTPUT_COST_PER_M_TOKENS,
    INGESTION_BASIC_INPUT_COST_PER_M_TOKENS, INGESTION_BASIC_OUTPUT_COST_PER_M_TOKENS,
)
from .models import ChunkRecord

ASSET_CATEGORY_LABEL = 'asset'  # generic label; caller decides which uploaded docs map to this

def chunk_document(segmenter, storage, document_path, source_document_name, is_asset_category):
    category = ASSET_CATEGORY_LABEL if is_asset_category else 'general'
    document = segmenter.generate_chunks(document_path=document_path)

    new_records = []
    skipped_chunk_ids = []
    previous_high_level = None
    total_ingestion_cost_usd = 0.0

    for chunk in document.chunks:
        if chunk is None:
            continue

        if storage.chunk_already_indexed(chunk.id, document_name=source_document_name):
            skipped_chunk_ids.append(chunk.id)
            continue  # identical content already indexed for this document; no re-extraction, no LLM call

        hint = previous_high_level if is_asset_category else None
        with get_openai_callback() as cb:
            updated = segmenter._update_chunk_entities(
                chunk, previous_high_level=hint, generate_high_level=is_asset_category
            )
        if is_asset_category:
            in_rate, out_rate = INGESTION_ASSET_INPUT_COST_PER_M_TOKENS, INGESTION_ASSET_OUTPUT_COST_PER_M_TOKENS
        else:
            in_rate, out_rate = INGESTION_BASIC_INPUT_COST_PER_M_TOKENS, INGESTION_BASIC_OUTPUT_COST_PER_M_TOKENS
        total_ingestion_cost_usd += (cb.prompt_tokens / 1_000_000) * in_rate + (cb.completion_tokens / 1_000_000) * out_rate

        if updated is None:
            continue
        chunk = updated

        if is_asset_category and getattr(chunk, 'high_level_summary', None):
            previous_high_level = chunk.high_level_summary

        new_records.append(ChunkRecord(
            chunk_id=chunk.id,
            source=chunk.source,
            summary=getattr(chunk, 'summary', None),
            high_level_summary=getattr(chunk, "high_level_summary", None),
            questions=getattr(chunk, "questions", None) or [],
            source_document=source_document_name,
            category=category
        ))

    return new_records, skipped_chunk_ids, total_ingestion_cost_usd

def build_specs_for_chunk(record):
    specs = []
    if record.category == ASSET_CATEGORY_LABEL:
        for question in record.questions:
            specs.append(('question', question, question, record.source))
        if record.summary:
            specs.append(('raw_chunk', None, record.summary, record.summary))
    else:
        if record.summary:
            specs.append(('summary', None, record.summary, record.summary))
        for question in record.questions:
            specs.append(('question', question, question, record.source))
    return specs
