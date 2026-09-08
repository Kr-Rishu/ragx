
chunk_entities_extraction_prompt = """
Extract high-quality retrieval metadata from the CURRENT CHUNK below.

## CONTEXT FROM PREVIOUS CHUNK (high-level facts carried forward - reference only)
{previous_high_level}

===== END OF PREVIOUS CONTEXT - CURRENT CHUNK STARTS BELOW =====

## CURRENT CHUNK TEXT
{text_chunk}

===== END OF CURRENT CHUNK =====

Everything you extract must come from the CURRENT CHUNK TEXT above. The previous context exists
only to help you correctly attribute unlabeled values in the current chunk (e.g. a parameter
table with no equipment name) to the right asset. Never state a fact from the previous context
as if it occurred in the current chunk.

## TASK

### 1. Summary
Concise, information-dense summary of the current chunk only. Preserve entities (equipment,
components, locations, standards), numerical values (limits, ranges, temperatures, pressures,
dimensions, capacities, ratings), relationships, operating conditions, alarms, actions,
procedures, responsibilities. Use the previous context only to correctly name the asset if the
current chunk itself doesn't name it. Optimize for semantic search retrieval.

### 2. High-Level Summary
A list of short fact strings (not a paragraph) — the asset's tag, name/type, and 1-3
distinguishing facts from the current chunk. Terse fragments, no filler words, no repeated phrasing.

FORGET RULE: If the current chunk introduces a DIFFERENT asset than the one in the previous
context, this list must describe ONLY the new asset - discard every previous-context fact, do
not merge old-asset facts in. If the current chunk continues the SAME asset (or names no asset
at all), build on the previous context: keep the established tag/name, add at most 1-2 new key
facts from this chunk, drop anything now superseded.

### 3. Questions
5-7 diverse questions answerable strictly from the current chunk text (What, Which, why, When, Where,
Who, How much, limit/range/value, "what action occurs when..."). Cover different parts of the
text, avoid duplicates. If the previous/current context identifies an asset the current chunk's
own text doesn't name, phrase at least one question using that asset's tag/name so it's
self-contained for retrieval.
"""


chunk_entities_extraction_prompt_basic = """
Extract high-quality retrieval metadata from the text.

## INPUT
{text_chunk}

## TASK

### 1. Summary
Write a concise but information-dense summary that preserves all important facts from the text.
Capture the main purpose/topic, entities, numerical values, relationships, operating conditions,
alarms, actions, procedures. Optimize for semantic search and embedding retrieval.

### 2. Questions
Generate 5-7 diverse questions answerable strictly from the text. Cover different parts of the
text, avoid duplicates.
"""
